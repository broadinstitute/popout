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


def run_demo(
    n_samples: int = 500,
    n_sites: int = 2000,
    n_ancestries: int = 4,
    gen_since_admix: int = 20,
    n_em_iter: int = 5,
    batch_size: int = 10_000,
    seed: int = 42,
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
    result = run_em(
        chrom_data,
        n_ancestries=n_ancestries,
        n_em_iter=n_em_iter,
        gen_since_admix=float(gen_since_admix),
        batch_size=batch_size,
        rng_seed=seed,
    )
    t_em = time.perf_counter() - t0
    log.info("EM pipeline: %.1f seconds", t_em)

    # --- Evaluate ---
    metrics = evaluate_accuracy(result.calls, true_ancestry, n_ancestries)
    log.info("=" * 50)
    log.info("RESULTS")
    log.info("=" * 50)
    log.info("Oracle accuracy: %.1f%%", 100 * oracle_metrics["overall_accuracy"])
    log.info("EM accuracy:     %.1f%%", 100 * metrics["overall_accuracy"])
    log.info("Gap:             %.1f pp",
             100 * (oracle_metrics["overall_accuracy"] - metrics["overall_accuracy"]))
    log.info("Best permutation: %s", metrics["best_permutation"])
    for a, acc in metrics["per_ancestry_accuracy"].items():
        log.info("  Ancestry %d: %.1f%%", a, 100 * acc)

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
    """Run oracle vs EM comparison across a grid of configurations."""
    configs = [
        (500,   2000),
        (5000,  2000),
        (50000, 2000),
        (500,   10000),
        (5000,  10000),
    ]

    rows = []
    for n_samples, n_sites in configs:
        log.info("\n" + "=" * 60)
        log.info("CONFIG: n_samples=%d, n_sites=%d", n_samples, n_sites)
        log.info("=" * 60)
        result, metrics, oracle_metrics = run_demo(
            n_samples=n_samples,
            n_sites=n_sites,
            n_ancestries=4,
            gen_since_admix=20,
            n_em_iter=20,
            seed=42,
        )
        rows.append({
            "samples": n_samples,
            "sites": n_sites,
            "oracle": oracle_metrics["overall_accuracy"],
            "em": metrics["overall_accuracy"],
            "gap": oracle_metrics["overall_accuracy"] - metrics["overall_accuracy"],
            "T_inf": float(result.model.gen_since_admix),
        })

    log.info("\n" + "=" * 60)
    log.info("SWEEP SUMMARY (seed=42, A=4, T_true=20, n_em_iter=20)")
    log.info("=" * 60)
    log.info("%-10s %-8s %-10s %-10s %-8s %-8s",
             "samples", "sites", "oracle%", "em%", "gap_pp", "T_inf")
    log.info("-" * 54)
    for r in rows:
        log.info("%-10d %-8d %-10.1f %-10.1f %-8.1f %-8.1f",
                 r["samples"], r["sites"],
                 100 * r["oracle"], 100 * r["em"],
                 100 * r["gap"], r["T_inf"])


def main():
    parser = argparse.ArgumentParser(description="popout demo on simulated data")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-sites", type=int, default=2000)
    parser.add_argument("--n-ancestries", type=int, default=4)
    parser.add_argument("--gen-since-admix", type=int, default=20)
    parser.add_argument("--n-em-iter", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true",
                        help="Run multi-config sweep instead of single demo")
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        run_demo(
            n_samples=args.n_samples,
            n_sites=args.n_sites,
            n_ancestries=args.n_ancestries,
            gen_since_admix=args.gen_since_admix,
            n_em_iter=args.n_em_iter,
            batch_size=args.batch_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
