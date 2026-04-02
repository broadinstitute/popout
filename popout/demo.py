#!/usr/bin/env python3
"""Demo: run popout on simulated admixed data.

This exercises the full pipeline without needing a VCF or genetic map.

Usage:
    python -m popout.demo --n-samples 500 --n-sites 2000 --n-ancestries 4
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

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
    n_em_iter: int = 3,
    batch_size: int = 10_000,
    seed: int = 42,
):
    import jax
    log.info("JAX devices: %s", jax.devices())

    from .simulate import simulate_admixed, evaluate_accuracy
    from .em import run_em

    # --- Simulate ---
    t0 = time.perf_counter()
    chrom_data, true_ancestry = simulate_admixed(
        n_samples=n_samples,
        n_sites=n_sites,
        n_ancestries=n_ancestries,
        gen_since_admix=gen_since_admix,
        rng_seed=seed,
    )
    t_sim = time.perf_counter() - t0
    log.info("Simulation: %.1f seconds", t_sim)

    # --- Run pipeline ---
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
    log.info("Overall accuracy: %.1f%%", 100 * metrics["overall_accuracy"])
    log.info("Best permutation: %s", metrics["best_permutation"])
    for a, acc in metrics["per_ancestry_accuracy"].items():
        log.info("  Ancestry %d: %.1f%%", a, 100 * acc)

    # --- Model recovery ---
    log.info("Inferred mu: %s", np.array(result.model.mu).round(3))
    log.info("Inferred T:  %.1f generations", result.model.gen_since_admix)
    log.info("Haplotypes:  %d", chrom_data.n_haps)
    log.info("Sites:       %d", chrom_data.n_sites)
    log.info(
        "Throughput:  %.0f haplotype-sites/sec",
        chrom_data.n_haps * chrom_data.n_sites / t_em,
    )

    return result, metrics


def main():
    parser = argparse.ArgumentParser(description="popout demo on simulated data")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-sites", type=int, default=2000)
    parser.add_argument("--n-ancestries", type=int, default=4)
    parser.add_argument("--gen-since-admix", type=int, default=20)
    parser.add_argument("--n-em-iter", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
