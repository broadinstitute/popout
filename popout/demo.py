#!/usr/bin/env python3
"""Demo: run popout on simulated admixed data.

This exercises the full pipeline without needing a VCF or genetic map.
Includes an oracle benchmark (decode with true parameters) to establish
the accuracy ceiling.

Usage:
    python -m popout.demo --n-samples 500 --n-sites 2000 --n-ancestries 4
    python -m popout.demo --sweep
"""

from __future__ import annotations

import argparse
import logging
import time

import jax.numpy as jnp
import numpy as np

from .datatypes import AncestryModel
from .hmm import forward_backward_decode, forward_backward_em
from .em import (
    update_allele_freq_from_stats,
    update_mu_from_stats,
    update_generations_from_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("popout.demo")


def leaf_purity(calls, true_ancestry, n_true_ancestries):
    """Per-leaf purity: max fraction sharing the same true ancestry.

    Returns (mean_purity, min_purity, per_leaf_purity_array).
    """
    n_leaves = int(calls.max()) + 1
    purities = np.zeros(n_leaves)
    for leaf in range(n_leaves):
        in_leaf = (calls.flatten() == leaf) if calls.ndim == 1 else np.any(calls == leaf, axis=1)
        # Use per-haplotype majority ancestry
        if calls.ndim == 2:
            in_leaf_haps = np.where(in_leaf)[0]
            if len(in_leaf_haps) == 0:
                purities[leaf] = 1.0
                continue
            hap_majority = np.array([
                np.bincount(true_ancestry[h], minlength=n_true_ancestries).argmax()
                for h in in_leaf_haps
            ])
            counts = np.bincount(hap_majority, minlength=n_true_ancestries)
        else:
            mask = calls == leaf
            if mask.sum() == 0:
                purities[leaf] = 1.0
                continue
            counts = np.bincount(true_ancestry[mask], minlength=n_true_ancestries)
        purities[leaf] = counts.max() / counts.sum()
    return float(purities.mean()), float(purities.min()), purities


def run_demo(
    n_samples: int = 500,
    n_sites: int = 2000,
    n_ancestries: int = 4,
    gen_since_admix: int = 20,
    n_em_iter: int = 5,
    batch_size: int = 10_000,
    seed: int = 42,
    pure_fraction: float = 0.3,
    seed_method: str = "gmm",
    freeze_anchors_iters: int = 0,
):
    import jax
    log.info("JAX devices: %s", jax.devices())

    from .simulate import simulate_admixed, evaluate_accuracy
    from .em import run_em

    # --- Simulate ---
    t0 = time.perf_counter()
    chrom_data, true_ancestry, true_params = simulate_admixed(
        n_samples=n_samples,
        n_sites=n_sites,
        n_ancestries=n_ancestries,
        gen_since_admix=gen_since_admix,
        pure_fraction=pure_fraction,
        rng_seed=seed,
    )
    t_sim = time.perf_counter() - t0
    log.info("Simulation: %.1f seconds", t_sim)

    # --- Oracle benchmark: decode with true parameters ---
    log.info("=== Oracle benchmark ===")
    oracle_model = AncestryModel(
        n_ancestries=n_ancestries,
        mu=jnp.array(true_params["mu"]),
        gen_since_admix=float(true_params["gen_since_admix"]),
        allele_freq=jnp.array(true_params["pop_freq"]),
    )
    geno_j = jnp.array(chrom_data.geno)
    d_morgan_j = jnp.array(chrom_data.genetic_distances.astype(np.float64))

    t0 = time.perf_counter()
    oracle_decode = forward_backward_decode(
        geno_j, oracle_model, d_morgan_j, batch_size=batch_size,
    )
    t_oracle = time.perf_counter() - t0
    oracle_metrics = evaluate_accuracy(
        oracle_decode.calls, true_ancestry, n_ancestries,
    )
    log.info("Oracle accuracy: %.1f%% (%.1f sec)", 100 * oracle_metrics["overall_accuracy"], t_oracle)
    for a, acc in oracle_metrics["per_ancestry_accuracy"].items():
        log.info("  Ancestry %d: %.1f%%", a, 100 * acc)

    # --- Oracle T stability test ---
    log.info("=== Oracle T stability ===")
    oracle_stats = forward_backward_em(
        geno_j, oracle_model, d_morgan_j, batch_size=batch_size,
    )
    T_after = update_generations_from_stats(
        oracle_stats, d_morgan_j,
        current_T=float(true_params["gen_since_admix"]),
        mu=jnp.array(true_params["mu"]),
    )
    new_freq = update_allele_freq_from_stats(oracle_stats)
    new_mu = update_mu_from_stats(oracle_stats)
    freq_delta = float(jnp.abs(new_freq - jnp.array(true_params["pop_freq"])).max())
    mu_delta = float(jnp.abs(new_mu - jnp.array(true_params["mu"])).max())
    true_T = true_params["gen_since_admix"]
    log.info("  True T: %d → T after 1 oracle iter: %.1f (drift: %.1f%%)",
             true_T, T_after, 100 * abs(T_after - true_T) / true_T)
    log.info("  Freq max delta from true: %.6f", freq_delta)
    log.info("  Mu max delta from true: %.6f", mu_delta)
    del oracle_stats  # free memory

    # --- Run EM pipeline ---
    t0 = time.perf_counter()
    seed_resp = None
    em_n_ancestries = n_ancestries

    if seed_method == "recursive":
        from .recursive_seed import recursive_split_seed
        leaf_labels, leaf_info = recursive_split_seed(
            chrom_data.geno,
            min_cluster_size=max(100, n_samples * 2 // 20),
            bic_per_sample=0.01,
            max_depth=6,
            max_leaves=12,
            em_iter_per_split=3,
            rng_seed=seed,
            chrom_data=chrom_data,
            gen_since_admix=float(gen_since_admix),
        )
        n_leaves = len(leaf_info)
        log.info("Recursive seed: discovered K=%d (true K=%d)", n_leaves, n_ancestries)
        seed_resp = jnp.zeros((chrom_data.n_haps, n_leaves), dtype=jnp.float32)
        seed_resp = seed_resp.at[
            jnp.arange(chrom_data.n_haps), jnp.array(leaf_labels)
        ].set(1.0)
        em_n_ancestries = n_leaves

    result = run_em(
        chrom_data,
        n_ancestries=em_n_ancestries,
        n_em_iter=n_em_iter,
        gen_since_admix=float(gen_since_admix),
        batch_size=batch_size,
        rng_seed=seed,
        seed_responsibilities=seed_resp,
        freeze_anchors_iters=freeze_anchors_iters,
    )
    t_em = time.perf_counter() - t0
    log.info("EM pipeline (%s seed): %.1f seconds", seed_method, t_em)

    # --- Evaluate ---
    inferred_K = result.model.n_ancestries
    metrics = evaluate_accuracy(result.calls, true_ancestry, inferred_K)
    log.info("=" * 50)
    log.info("RESULTS (%s seed)", seed_method)
    log.info("=" * 50)
    if seed_method == "recursive":
        log.info("Discovered K:    %d (true K=%d)", inferred_K, n_ancestries)
    log.info("Oracle accuracy: %.1f%%", 100 * oracle_metrics["overall_accuracy"])
    log.info("EM accuracy:     %.1f%%", 100 * metrics["overall_accuracy"])
    log.info("Gap:             %.1f pp",
             100 * (oracle_metrics["overall_accuracy"] - metrics["overall_accuracy"]))
    log.info("Best permutation: %s", metrics["best_permutation"])
    for a, acc in metrics["per_ancestry_accuracy"].items():
        log.info("  Ancestry %d: %.1f%%", a, 100 * acc)

    # --- Per-leaf purity ---
    mean_pur, min_pur, per_leaf = leaf_purity(
        result.calls, true_ancestry, n_ancestries,
    )
    log.info("Per-leaf purity: mean=%.3f, min=%.3f", mean_pur, min_pur)
    for i, p in enumerate(per_leaf):
        log.info("  Leaf %d: purity=%.3f", i, p)

    # --- Model recovery ---
    log.info("True mu:     %s", np.array(true_params["mu"]).round(3))
    log.info("Inferred mu: %s", np.array(result.model.mu).round(3))
    log.info("True T:      %d generations", true_params["gen_since_admix"])
    log.info("Inferred T:  %.1f generations", result.model.gen_since_admix)
    log.info("Haplotypes:  %d", chrom_data.n_haps)
    log.info("Sites:       %d", chrom_data.n_sites)
    log.info(
        "Throughput:  %.0f haplotype-sites/sec",
        chrom_data.n_haps * chrom_data.n_sites / t_em,
    )

    return result, metrics, oracle_metrics


def run_sweep():
    """Run oracle vs EM comparison across two simulation regimes."""
    configs = [
        (500,   2000),
        (5000,  2000),
        (50000, 2000),
        (500,   10000),
        (5000,  10000),
    ]

    for pure_frac, label in [(0.3, "biobank-like (30% pure)"),
                              (0.0, "fully-admixed stress test")]:
        all_rows = []
        log.info("\n" + "=" * 60)
        log.info("REGIME: %s", label)
        log.info("=" * 60)

        for n_samples, n_sites in configs:
            log.info("\n--- n_samples=%d, n_sites=%d ---", n_samples, n_sites)
            result, metrics, oracle_metrics = run_demo(
                n_samples=n_samples,
                n_sites=n_sites,
                n_ancestries=4,
                gen_since_admix=20,
                n_em_iter=20,
                seed=42,
                pure_fraction=pure_frac,
            )
            all_rows.append({
                "samples": n_samples,
                "sites": n_sites,
                "oracle": oracle_metrics["overall_accuracy"],
                "em": metrics["overall_accuracy"],
                "gap": oracle_metrics["overall_accuracy"] - metrics["overall_accuracy"],
                "T_inf": float(result.model.gen_since_admix),
            })

        log.info("\n" + "=" * 60)
        log.info("SWEEP SUMMARY — %s (seed=42, A=4, T_true=20, n_em_iter=20)",
                 label)
        log.info("=" * 60)
        log.info("%-10s %-8s %-10s %-10s %-8s %-8s",
                 "samples", "sites", "oracle%", "em%", "gap_pp", "T_inf")
        log.info("-" * 54)
        for r in all_rows:
            log.info("%-10d %-8d %-10.1f %-10.1f %-8.1f %-8.1f",
                     r["samples"], r["sites"],
                     100 * r["oracle"], 100 * r["em"],
                     100 * r["gap"], r["T_inf"])


def run_convert_demo(
    n_samples: int = 50,
    n_sites: int = 500,
    n_ancestries: int = 3,
    gen_since_admix: int = 20,
    seed: int = 42,
):
    """Demo: simulate → popout → convert → parse_flare → verify."""
    import tempfile
    from pathlib import Path
    import pysam

    from .simulate import simulate_admixed, evaluate_accuracy
    from .em import run_em
    from .output import write_global_ancestry, write_model, write_ancestry_tracts, write_decode_npz
    from .convert import convert_to_vcf
    from .benchmark.parsers.flare import parse_flare

    log.info("=== Convert demo: %d samples, %d sites, K=%d ===",
             n_samples, n_sites, n_ancestries)

    chrom_data, true_ancestry, true_params = simulate_admixed(
        n_samples=n_samples, n_sites=n_sites,
        n_ancestries=n_ancestries, gen_since_admix=gen_since_admix,
        rng_seed=seed,
    )
    # VCF-compatible positions
    chrom_data.pos_bp = np.arange(1, n_sites + 1, dtype=np.int64) * 1000

    sample_names = [f"SAMPLE_{i}" for i in range(n_samples)]
    ancestry_names = [f"pop_{i}" for i in range(n_ancestries)]

    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "demo")

        # Run popout
        log.info("Running popout pipeline...")
        result = run_em(
            chrom_data, n_ancestries=n_ancestries,
            n_em_iter=3, gen_since_admix=float(gen_since_admix),
            write_dense_decode=True,
        )

        # Write outputs
        write_global_ancestry([result], n_samples, sample_names, f"{prefix}.global.tsv")
        write_model(result, f"{prefix}.model", chrom_data=chrom_data,
                     ancestry_names=ancestry_names)
        write_ancestry_tracts([result], [chrom_data], n_samples, sample_names,
                              f"{prefix}.tracts.tsv.gz", write_posteriors=True)
        write_decode_npz(result, chrom_data,
                         f"{prefix}.chr{chrom_data.chrom}.decode.npz",
                         include_max_post=True)

        # Write input VCF
        vcf_path = str(Path(tmp) / "input.vcf.gz")
        header = pysam.VariantHeader()
        header.add_line(f'##contig=<ID={chrom_data.chrom}>')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        for s in sample_names:
            header.add_sample(s)
        rng = np.random.default_rng(99)
        with pysam.VariantFile(vcf_path, "wz", header=header) as vcf_out:
            for t in range(n_sites):
                rec = vcf_out.new_record()
                rec.contig = chrom_data.chrom
                rec.pos = int(chrom_data.pos_bp[t])
                rec.alleles = ("A", "T")
                for si, s in enumerate(sample_names):
                    rec.samples[s]["GT"] = (int(chrom_data.geno[2*si, t]),
                                            int(chrom_data.geno[2*si+1, t]))
                    rec.samples[s].phased = True
                vcf_out.write(rec)
        pysam.tabix_index(vcf_path, preset="vcf", force=True)

        # Convert
        log.info("Running popout convert --to vcf...")
        out_vcf = str(Path(tmp) / "output.anc.vcf.gz")

        class Args:
            popout_prefix = prefix
            input_vcf = vcf_path
            out = out_vcf
            probs = True
            ancestry_names = None
            thinned_sites = "skip"
            chroms = None
            to = "vcf"

        convert_to_vcf(Args())

        # Parse and verify
        ts = parse_flare(out_vcf, chrom=chrom_data.chrom)
        metrics = evaluate_accuracy(
            np.asarray(ts.calls, dtype=np.int8),
            true_ancestry, n_ancestries,
        )

        log.info("=" * 50)
        log.info("CONVERT DEMO RESULTS")
        log.info("=" * 50)
        log.info("Sites in VCF:    %d", ts.n_sites)
        log.info("Haplotypes:      %d", ts.n_haps)
        log.info("Ancestry names:  %s", list(ts.label_map.values()))
        log.info("Accuracy vs sim: %.1f%%", 100 * metrics["overall_accuracy"])
        log.info("Best permutation: %s", metrics["best_permutation"])
        for a, acc in metrics["per_ancestry_accuracy"].items():
            log.info("  Ancestry %d: %.1f%%", a, 100 * acc)


def main():
    parser = argparse.ArgumentParser(description="popout demo on simulated data")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-sites", type=int, default=2000)
    parser.add_argument("--n-ancestries", type=int, default=4)
    parser.add_argument("--gen-since-admix", type=int, default=20)
    parser.add_argument("--n-em-iter", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pure-fraction", type=float, default=0.3,
                        help="Fraction of haplotypes that are single-ancestry (default: 0.3)")
    parser.add_argument("--seed-method", choices=["gmm", "recursive"], default="gmm",
                        help="Seeding strategy (default: gmm)")
    parser.add_argument("--freeze-anchors-iters", type=int, default=0,
                        help="Freeze seed responsibilities for first N EM iters (default: 0)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run multi-config sweep instead of single demo")
    parser.add_argument("--convert", action="store_true",
                        help="Run convert demo (simulate → popout → VCF → parse_flare)")
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    elif args.convert:
        run_convert_demo()
    else:
        run_demo(
            n_samples=args.n_samples,
            n_sites=args.n_sites,
            n_ancestries=args.n_ancestries,
            gen_since_admix=args.gen_since_admix,
            n_em_iter=args.n_em_iter,
            batch_size=args.batch_size,
            seed=args.seed,
            pure_fraction=args.pure_fraction,
            seed_method=args.seed_method,
            freeze_anchors_iters=args.freeze_anchors_iters,
        )


if __name__ == "__main__":
    main()
