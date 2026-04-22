"""Memory profile for streaming decode parquet writer.

Usage:
    python tests/profile_decode_parquet.py

Verifies that transient memory overhead during write stays bounded.
"""
from __future__ import annotations

import resource
import sys
import tempfile

import jax.numpy as jnp
import numpy as np

from popout.datatypes import AncestryModel, AncestryResult, ChromData, DecodeResult
from popout.output import write_decode_parquet


def peak_rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB
    if sys.platform == "darwin":
        return ru / (1024 ** 3)
    return ru / (1024 ** 2)


def main():
    H, T, A = 100_000, 10_000, 20
    rng = np.random.default_rng(0)
    calls = rng.integers(0, A, size=(H, T), dtype=np.int8)
    max_post = rng.uniform(0.3, 0.99, size=(H, T)).astype(np.float32)

    decode = DecodeResult(calls=calls, max_post=max_post)
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.ones(A) / A,
        gen_since_admix=20.0,
        allele_freq=jnp.zeros((A, T)),
    )
    result = AncestryResult(
        calls=calls, model=model, chrom="1", decode=decode,
    )
    cdata = ChromData(
        geno=np.zeros((H, T), dtype=np.uint8),
        pos_bp=np.arange(T, dtype=np.int64) * 1000,
        pos_cm=np.arange(T, dtype=np.float64) * 0.001,
        chrom="1",
    )

    baseline = peak_rss_gb()
    print(f"RSS before write: {baseline:.2f} GB")
    print(f"  calls: {calls.nbytes / 1e9:.2f} GB")
    print(f"  max_post: {max_post.nbytes / 1e9:.2f} GB")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
        write_decode_parquet(result, cdata, tmp.name, include_max_post=True)

    peak = peak_rss_gb()
    overhead = peak - baseline
    print(f"RSS peak: {peak:.2f} GB")
    print(f"Transient overhead: {overhead:.2f} GB")

    if overhead > 2.0:
        print(f"FAIL: transient overhead {overhead:.2f} GB exceeds 2 GB limit")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
