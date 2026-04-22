"""Tests for popout convert module."""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pysam

import jax.numpy as jnp

from popout.convert import convert_to_vcf, _expand_max_post
from popout.datatypes import AncestryModel, AncestryResult, ChromData, DecodeResult
from popout.output import write_decode_parquet


def _create_synthetic_popout_outputs(
    tmp_dir: str,
    n_samples: int = 5,
    n_sites: int = 50,
    K: int = 3,
    chrom: str = "1",
    include_max_post: bool = True,
    ancestry_names: list[str] | None = None,
):
    """Create a minimal set of popout outputs for testing convert."""
    prefix = Path(tmp_dir) / "popout_test"
    rng = np.random.default_rng(42)
    n_haps = 2 * n_samples

    # Calls and max_post
    calls = rng.integers(0, K, size=(n_haps, n_sites)).astype(np.int8)
    pos_bp = np.arange(n_sites, dtype=np.int64) * 1000 + 100000

    max_post_arr = None
    if include_max_post:
        max_post_arr = (rng.random((n_haps, n_sites)) * 0.3 + 0.7).astype(np.float32)

    model = AncestryModel(
        n_ancestries=K,
        mu=jnp.array(rng.dirichlet(np.ones(K))),
        gen_since_admix=20.0,
        allele_freq=jnp.array(rng.random((K, n_sites)).astype(np.float32)),
    )
    decode = DecodeResult(calls=calls, max_post=max_post_arr) if max_post_arr is not None else None
    result = AncestryResult(calls=calls, model=model, chrom=chrom, decode=decode)
    chrom_data = ChromData(
        geno=rng.integers(0, 2, size=(n_haps, n_sites)).astype(np.uint8),
        pos_bp=pos_bp,
        pos_cm=np.linspace(0, 10, n_sites),
        chrom=chrom,
    )
    write_decode_parquet(
        result, chrom_data,
        str(prefix) + f".chr{chrom}.decode.parquet",
        include_max_post=include_max_post,
    )
    # Read back calls as uint8 for comparison (write_decode_parquet casts to uint8)
    calls = calls.astype(np.uint8)

    # Model npz
    model_dict = dict(
        n_ancestries=np.array(K),
        allele_freq=rng.random((K, n_sites)).astype(np.float32),
        mu=np.array(rng.dirichlet(np.ones(K))),
        gen_since_admix=np.array(20.0),
    )
    if ancestry_names is not None:
        model_dict["ancestry_names"] = np.array(ancestry_names, dtype=object)
    np.savez_compressed(str(prefix) + ".model.npz", **model_dict)

    # Global ancestry TSV
    sample_names = [f"SAMPLE_{i}" for i in range(n_samples)]
    with open(str(prefix) + ".global.tsv", "w") as f:
        header = "sample\t" + "\t".join(f"ancestry_{a}" for a in range(K))
        f.write(header + "\n")
        for s in sample_names:
            props = rng.dirichlet(np.ones(K))
            f.write(s + "\t" + "\t".join(f"{v:.4f}" for v in props) + "\n")

    # Create input VCF
    vcf_path = str(Path(tmp_dir) / "input.vcf.gz")
    _create_input_vcf(vcf_path, sample_names, pos_bp, chrom)

    return str(prefix), vcf_path, sample_names, calls, pos_bp


def _create_input_vcf(vcf_path, sample_names, pos_bp, chrom, extra_positions=None):
    """Create a minimal phased VCF for testing."""
    rng = np.random.default_rng(99)
    header = pysam.VariantHeader()
    header.add_line(f'##contig=<ID={chrom}>')
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    for s in sample_names:
        header.add_sample(s)

    positions = list(pos_bp)
    if extra_positions is not None:
        positions = sorted(set(positions) | set(extra_positions))

    with pysam.VariantFile(vcf_path, "wz", header=header) as vcf_out:
        for pos in positions:
            rec = vcf_out.new_record()
            rec.contig = chrom
            rec.pos = int(pos)
            rec.alleles = ("A", "T")
            for si, sample in enumerate(sample_names):
                rec.samples[sample]["GT"] = (rng.integers(0, 2), rng.integers(0, 2))
                rec.samples[sample].phased = True
            vcf_out.write(rec)

    # Index
    pysam.tabix_index(vcf_path, preset="vcf", force=True)


def test_convert_roundtrip_via_parse_flare():
    """Convert popout outputs to VCF, then parse with parse_flare."""
    from popout.benchmark.parsers.flare import parse_flare

    with tempfile.TemporaryDirectory() as tmp:
        names = ["afr", "eas", "eur"]
        prefix, vcf_path, sample_names, calls, pos_bp = _create_synthetic_popout_outputs(
            tmp, n_samples=5, n_sites=50, K=3, ancestry_names=names,
        )
        out_path = str(Path(tmp) / "output.anc.vcf.gz")

        class Args:
            popout_prefix = prefix
            input_vcf = vcf_path
            out = out_path
            probs = False
            ancestry_names = None
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        convert_to_vcf(Args())

        # Parse the output with popout's own FLARE parser
        ts = parse_flare(out_path, chrom="1")
        assert ts.tool_name == "flare"
        assert ts.n_sites == 50
        assert ts.n_haps == 10  # 5 samples × 2 haps

        # Verify calls match
        np.testing.assert_array_equal(ts.calls, calls)

        # Verify label_map was parsed from ##ANCESTRY header
        assert ts.label_map == {0: "afr", 1: "eas", 2: "eur"}


def test_convert_missing_decode_parquet_fails():
    """Convert with no decode.parquet files raises a clear error."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "nonexistent"
        # Create a minimal model.npz so it gets past that check
        np.savez_compressed(str(prefix) + ".model.npz", n_ancestries=np.array(3))

        class Args:
            popout_prefix = str(prefix)
            input_vcf = "dummy.vcf.gz"
            out = str(Path(tmp) / "out.anc.vcf.gz")
            probs = False
            ancestry_names = "a,b,c"
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        try:
            convert_to_vcf(Args())
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No decode.parquet files found" in str(e)


def test_convert_thinned_sites_skip_vs_fill():
    """skip drops non-popout sites; fill-missing includes them."""
    with tempfile.TemporaryDirectory() as tmp:
        names = ["a", "b"]
        prefix, vcf_path, sample_names, calls, pos_bp = _create_synthetic_popout_outputs(
            tmp, n_samples=3, n_sites=20, K=2, ancestry_names=names,
        )

        # Re-create input VCF with extra positions not in popout
        extra_positions = [50000, 60000, 70000]
        _create_input_vcf(vcf_path, sample_names, pos_bp, "1", extra_positions)

        # Test skip mode
        out_skip = str(Path(tmp) / "skip.anc.vcf.gz")

        class ArgsSkip:
            popout_prefix = prefix
            input_vcf = vcf_path
            out = out_skip
            probs = False
            ancestry_names = None
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        convert_to_vcf(ArgsSkip())
        from popout.benchmark.parsers.flare import parse_flare
        ts_skip = parse_flare(out_skip, chrom="1")
        assert ts_skip.n_sites == 20  # only popout sites

        # Test fill-missing mode
        out_fill = str(Path(tmp) / "fill.anc.vcf.gz")

        class ArgsFill:
            popout_prefix = prefix
            input_vcf = vcf_path
            out = out_fill
            probs = False
            ancestry_names = None
            thinned_sites = "fill-missing"
            chroms = None
            to = "vcf"

        convert_to_vcf(ArgsFill())

        # Count lines in output
        import gzip
        with gzip.open(out_fill, "rt") as f:
            data_lines = [l for l in f if not l.startswith("#")]
        assert len(data_lines) == 20 + 3  # popout sites + extra


def test_convert_ancestry_names_precedence():
    """CLI --ancestry-names overrides model.npz names."""
    with tempfile.TemporaryDirectory() as tmp:
        # model.npz has names ['afr', 'eas', 'eur']
        prefix, vcf_path, sample_names, calls, pos_bp = _create_synthetic_popout_outputs(
            tmp, n_samples=3, n_sites=10, K=3, ancestry_names=["afr", "eas", "eur"],
        )
        out_path = str(Path(tmp) / "output.anc.vcf.gz")

        # Override with CLI names
        class Args:
            popout_prefix = prefix
            input_vcf = vcf_path
            out = out_path
            probs = False
            ancestry_names = "pop_a,pop_b,pop_c"
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        convert_to_vcf(Args())

        from popout.benchmark.parsers.flare import parse_flare
        ts = parse_flare(out_path, chrom="1")
        assert ts.label_map == {0: "pop_a", 1: "pop_b", 2: "pop_c"}

    # Test mismatched length raises
    with tempfile.TemporaryDirectory() as tmp:
        prefix, vcf_path, _, _, _ = _create_synthetic_popout_outputs(
            tmp, n_samples=3, n_sites=10, K=3,
        )

        class ArgsBad:
            popout_prefix = prefix
            input_vcf = vcf_path
            out = str(Path(tmp) / "out.anc.vcf.gz")
            probs = False
            ancestry_names = "a,b"  # K=3 but only 2 names
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        try:
            convert_to_vcf(ArgsBad())
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "2 entries" in str(e)


def test_convert_anp_sums_to_one():
    """With --probs, ANP1/ANP2 tuples sum to 1.0 for every sample."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix, vcf_path, sample_names, calls, pos_bp = _create_synthetic_popout_outputs(
            tmp, n_samples=4, n_sites=30, K=3, include_max_post=True,
        )
        out_path = str(Path(tmp) / "output.anc.vcf.gz")

        class Args:
            popout_prefix = prefix
            input_vcf = vcf_path
            out = out_path
            probs = True
            ancestry_names = "a,b,c"
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        convert_to_vcf(Args())

        # Parse and check ANP sums
        vcf = pysam.VariantFile(out_path)
        for rec in vcf:
            for sample in sample_names:
                anp1 = rec.samples[sample]["ANP1"]
                anp2 = rec.samples[sample]["ANP2"]
                if anp1 is not None and not any(v is None for v in anp1):
                    np.testing.assert_almost_equal(sum(anp1), 1.0, decimal=3)
                if anp2 is not None and not any(v is None for v in anp2):
                    np.testing.assert_almost_equal(sum(anp2), 1.0, decimal=3)
        vcf.close()


def test_convert_global_anc_header():
    """Output global file has columns named from ancestry_names."""
    with tempfile.TemporaryDirectory() as tmp:
        names = ["pop_x", "pop_y", "pop_z"]
        prefix, vcf_path, _, _, _ = _create_synthetic_popout_outputs(
            tmp, n_samples=3, n_sites=10, K=3, ancestry_names=names,
        )
        out_path = str(Path(tmp) / "output.anc.vcf.gz")

        class Args:
            popout_prefix = prefix
            input_vcf = vcf_path
            out = out_path
            probs = False
            ancestry_names = None
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        convert_to_vcf(Args())

        # Check global ancestry file
        global_path = str(Path(tmp) / "output.global.anc.gz")
        with gzip.open(global_path, "rt") as f:
            header = f.readline().strip().split("\t")
        assert header[1:] == names


def test_expand_max_post():
    """_expand_max_post produces valid K-vector summing to 1."""
    probs = _expand_max_post(0.9, 1, 3)
    assert len(probs) == 3
    np.testing.assert_almost_equal(sum(probs), 1.0)
    assert probs[1] == 0.9
    assert probs[0] == probs[2]  # uniform off-ancestry

    # K=1 edge case
    probs_k1 = _expand_max_post(0.95, 0, 1)
    assert probs_k1 == (0.95,)
