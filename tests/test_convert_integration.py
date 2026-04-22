"""End-to-end integration test for popout convert.

Simulates a small cohort, runs the full popout pipeline with --probs,
converts to VCF, parses with popout's own FLARE parser, and checks that
per-site calls match simulated ground truth and ANP tuples sum to 1.0.
"""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pysam

from popout.simulate import simulate_admixed, evaluate_accuracy


def _write_simulated_vcf(chrom_data, sample_names, vcf_path):
    """Write simulated genotype data to a phased VCF."""
    header = pysam.VariantHeader()
    chrom = chrom_data.chrom
    header.add_line(f'##contig=<ID={chrom}>')
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    for s in sample_names:
        header.add_sample(s)

    with pysam.VariantFile(vcf_path, "wz", header=header) as vcf_out:
        for t in range(chrom_data.n_sites):
            rec = vcf_out.new_record()
            rec.contig = chrom
            rec.pos = int(chrom_data.pos_bp[t])
            rec.alleles = ("A", "T")
            for si, sample in enumerate(sample_names):
                h0 = int(chrom_data.geno[2 * si, t])
                h1 = int(chrom_data.geno[2 * si + 1, t])
                rec.samples[sample]["GT"] = (h0, h1)
                rec.samples[sample].phased = True
            vcf_out.write(rec)

    pysam.tabix_index(vcf_path, preset="vcf", force=True)


def test_convert_integration():
    """Full pipeline: simulate → popout → convert → parse_flare → verify."""
    from popout.em import run_em
    from popout.output import write_global_ancestry, write_model, write_ancestry_tracts, write_decode_parquet
    from popout.names import parse_ancestry_names
    from popout.convert import convert_to_vcf
    from popout.benchmark.parsers.flare import parse_flare

    n_samples = 50
    n_ancestries = 3
    n_sites = 500

    # 1. Simulate
    chrom_data, true_ancestry, true_params = simulate_admixed(
        n_samples=n_samples,
        n_sites=n_sites,
        n_ancestries=n_ancestries,
        gen_since_admix=20,
        rng_seed=42,
    )
    # Replace positions with reasonable VCF-compatible values (1-based, fits int32)
    chrom_data.pos_bp = np.arange(1, n_sites + 1, dtype=np.int64) * 1000

    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = str(Path(tmp) / "test")
        sample_names = [f"SAMPLE_{i}" for i in range(n_samples)]

        # 2. Run popout pipeline
        result = run_em(
            chrom_data,
            n_ancestries=n_ancestries,
            n_em_iter=3,
            gen_since_admix=20.0,
            write_dense_decode=True,
        )

        # 3. Write outputs
        ancestry_names = ["pop_A", "pop_B", "pop_C"]
        write_global_ancestry(
            [result], n_samples, sample_names,
            f"{out_prefix}.global.tsv",
        )
        write_model(
            result, f"{out_prefix}.model",
            chrom_data=chrom_data,
            ancestry_names=ancestry_names,
        )
        write_ancestry_tracts(
            [result], [chrom_data], n_samples, sample_names,
            f"{out_prefix}.tracts.tsv.gz",
            write_posteriors=True,
        )
        write_decode_parquet(
            result, chrom_data,
            f"{out_prefix}.chr{chrom_data.chrom}.decode.parquet",
            include_max_post=True,
        )

        # 4. Write input VCF
        vcf_path = str(Path(tmp) / "input.vcf.gz")
        _write_simulated_vcf(chrom_data, sample_names, vcf_path)

        # 5. Run convert
        out_vcf = str(Path(tmp) / "output.anc.vcf.gz")

        class Args:
            popout_prefix = out_prefix
            input_vcf = vcf_path
            out = out_vcf
            probs = True
            ancestry_names = None  # use model.npz names
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        convert_to_vcf(Args())

        # 6. Parse with FLARE parser
        ts = parse_flare(out_vcf, chrom=chrom_data.chrom)
        assert ts.n_sites == n_sites
        assert ts.n_haps == 2 * n_samples

        # Verify calls match the popout output
        np.testing.assert_array_equal(
            ts.calls, np.asarray(result.calls, dtype=np.uint16)
        )

        # Verify label_map from ancestry names
        assert ts.label_map == {0: "pop_A", 1: "pop_B", 2: "pop_C"}

        # 7. Check accuracy against simulated ground truth
        metrics = evaluate_accuracy(
            np.asarray(ts.calls, dtype=np.int8),
            true_ancestry,
            n_ancestries,
        )
        # Smoke-test: accuracy should be above chance (1/K = 33%)
        assert metrics["overall_accuracy"] > 0.40, (
            f"Accuracy {metrics['overall_accuracy']:.1%} too low"
        )

        # 8. Verify ANP1/ANP2 tuples sum to 1.0
        vcf = pysam.VariantFile(out_vcf)
        n_checked = 0
        for rec in vcf:
            for sample in sample_names:
                anp1 = rec.samples[sample]["ANP1"]
                anp2 = rec.samples[sample]["ANP2"]
                if anp1 is not None and not any(v is None for v in anp1):
                    np.testing.assert_almost_equal(sum(anp1), 1.0, decimal=2)
                    n_checked += 1
                if anp2 is not None and not any(v is None for v in anp2):
                    np.testing.assert_almost_equal(sum(anp2), 1.0, decimal=2)
                    n_checked += 1
        vcf.close()
        assert n_checked > 0

        # 9. Check global ancestry file exists and has correct header
        global_path = str(Path(tmp) / "output.global.anc.gz")
        with gzip.open(global_path, "rt") as f:
            header = f.readline().strip().split("\t")
        assert header[1:] == ancestry_names

        # 10. Cross-check global via _crosscheck_global
        from popout.benchmark.parsers.flare import _crosscheck_global
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _crosscheck_global(ts, Path(global_path), sample_names)
            # Warnings about divergence > 5pp are acceptable for this small sim
            # but should not error out
