"""Memory profile for window_init_allele_freq streaming.

Usage:
    python tests/profile_window_init.py [--H 10000] [--T 5000] [--A 4]

Prints peak device memory and wall clock time.
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from popout.spectral import window_init_allele_freq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--H", type=int, default=10_000)
    parser.add_argument("--T", type=int, default=5_000)
    parser.add_argument("--A", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--hap-batch", type=int, default=50_000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    geno = rng.integers(0, 2, size=(args.H, args.T), dtype=np.uint8)
    freq = jnp.array(rng.uniform(0.05, 0.95, (args.A, args.T)).astype(np.float32))

    print(f"geno: ({args.H}, {args.T}) = {geno.nbytes / 1e6:.1f} MB")
    print(f"hap_batch={args.hap_batch}, window_size={args.window_size}")

    _ = jnp.ones(1).block_until_ready()

    t0 = time.perf_counter()
    result = window_init_allele_freq(
        geno, freq, args.A,
        window_size=args.window_size,
        hap_batch=args.hap_batch,
    )
    _ = np.asarray(result)
    elapsed = time.perf_counter() - t0

    print(f"Result shape: {result.shape}")
    print(f"Wall clock: {elapsed:.2f}s")

    try:
        stats = jax.devices()[0].memory_stats()
        peak = stats.get("peak_bytes_in_use", 0)
        print(f"Peak device memory: {peak / 1e9:.2f} GB")
    except Exception:
        print("(Device memory stats not available)")


if __name__ == "__main__":
    main()
