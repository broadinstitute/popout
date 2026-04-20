#!/usr/bin/env python3
"""Smoke test: recursive seeding at half-biobank scale.

Verifies no OOM in the merge phase and the pre-merge dump works.

Usage:
    python scripts/smoke_test_recursive_biobank.py

Expects ~8-12 GB peak RSS. If it exceeds 32 GB, the merge caching
is broken and needs another pass.
"""

import logging
import resource
import tempfile
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test")


def main():
    from popout.simulate import simulate_admixed
    from popout.recursive_seed import recursive_split_seed

    # --- Generate synthetic data: 250K samples (500K haps) x 5K sites ---
    log.info("Generating synthetic data: 250K samples x 5K sites, K=6")
    t0 = time.perf_counter()
    chrom_data, _true_ancestry, _true_params = simulate_admixed(
        n_samples=250_000,
        n_sites=5_000,
        n_ancestries=6,
        gen_since_admix=20,
        pure_fraction=0.3,
        rng_seed=42,
    )
    t_sim = time.perf_counter() - t0
    log.info("Simulation: %.1f seconds, geno shape %s (%.1f GB)",
             t_sim, chrom_data.geno.shape,
             chrom_data.geno.nbytes / 1e9)

    # --- Run recursive seeding ---
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = f"{tmpdir}/pre_merge"

        log.info("Running recursive_split_seed (max_leaves=20, merge_hellinger=0.04)")
        t0 = time.perf_counter()
        leaf_labels, leaf_info = recursive_split_seed(
            chrom_data.geno,
            min_cluster_size=5000,
            max_leaves=20,
            merge_hellinger_threshold=0.04,
            rng_seed=42,
            chrom_data=chrom_data,
            dump_pre_merge_path=dump_path,
        )
        t_total = time.perf_counter() - t0

        # --- Report ---
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KB
        import sys
        if sys.platform == "darwin":
            peak_gb = peak_rss / 1e9
        else:
            peak_gb = peak_rss / 1e6

        log.info("=== Results ===")
        log.info("Total time: %.1f seconds", t_total)
        log.info("Discovered K: %d (post-merge)", len(leaf_info))
        log.info("Peak RSS: %.1f GB", peak_gb)

        # Verify dump files
        import os
        for ext in [".leaves.tsv", ".leaf_meta.tsv", ".leaf_freqs.npz"]:
            fpath = f"{dump_path}{ext}"
            exists = os.path.exists(fpath)
            size_mb = os.path.getsize(fpath) / 1e6 if exists else 0
            log.info("Dump file %s: %s (%.1f MB)",
                     ext, "OK" if exists else "MISSING", size_mb)
            assert exists, f"Missing dump file: {fpath}"

        # Verify freqs
        data = np.load(f"{dump_path}.leaf_freqs.npz", allow_pickle=True)
        log.info("Pre-merge leaf freqs shape: %s", data["allele_freq"].shape)

        # Pass/fail
        if peak_gb > 32:
            log.error("FAIL: peak RSS %.1f GB exceeds 32 GB limit", peak_gb)
            sys.exit(1)
        else:
            log.info("PASS: peak RSS %.1f GB (< 32 GB)", peak_gb)


if __name__ == "__main__":
    main()
