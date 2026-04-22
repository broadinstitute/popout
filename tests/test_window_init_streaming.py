"""Tests for the streaming (hap-batched) window_init_allele_freq.

Verifies that batching over haplotypes produces numerically identical
results to the non-batched path, and that the function does not retain
lazy references to the input genotype array.
"""
from __future__ import annotations

import gc

import jax.numpy as jnp
import numpy as np
import pytest

from popout.spectral import window_init_allele_freq


def _synthetic(H=400, T=200, A=3, seed=42):
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    freq = jnp.array(rng.uniform(0.05, 0.95, size=(A, T)).astype(np.float32))
    return geno, freq, A


@pytest.mark.parametrize("hap_batch,window_size", [
    (50, 25),
    (100, 50),
    (200, 100),
    (1000, 50),   # batch larger than H
    (50, 37),     # window size doesn't divide T
])
def test_batching_equivalence(hap_batch, window_size):
    """Batched and single-pass window init must produce identical results."""
    geno, freq, A = _synthetic(H=400, T=200, A=3)

    ref = window_init_allele_freq(
        geno, freq, A, window_size=window_size, hap_batch=10_000,
    )
    result = window_init_allele_freq(
        geno, freq, A, window_size=window_size, hap_batch=hap_batch,
    )
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(ref),
        rtol=1e-5, atol=1e-5,
        err_msg=f"Mismatch at hap_batch={hap_batch}, window_size={window_size}",
    )


def test_numpy_geno_accepted():
    """Must work when geno is host numpy, not device jax array."""
    geno, freq, A = _synthetic()
    assert isinstance(geno, np.ndarray)
    result = window_init_allele_freq(geno, freq, A, window_size=50, hap_batch=100)
    assert result.shape == (A, 200)
    assert np.all(np.isfinite(np.asarray(result)))


def test_jax_geno_accepted():
    """Must also accept jax device arrays."""
    geno, freq, A = _synthetic()
    geno_jax = jnp.array(geno)
    result = window_init_allele_freq(geno_jax, freq, A, window_size=50, hap_batch=100)
    assert result.shape == (A, 200)
    assert np.all(np.isfinite(np.asarray(result)))


def test_lazy_reference_independence():
    """Returned array must not hold a lazy reference to geno.

    This is the core OOM-prevention regression test: if the returned
    array lazily references geno, deleting geno and forcing GC would
    either crash or silently upload the full array later.
    """
    geno, freq, A = _synthetic(H=200, T=100)
    result = window_init_allele_freq(
        geno, freq, A, window_size=25, hap_batch=50,
    )
    result_np = np.asarray(result)
    del geno
    gc.collect()
    result_np2 = np.asarray(result)
    np.testing.assert_array_equal(result_np, result_np2)
    assert np.all(np.isfinite(result_np))


def test_output_shape_and_range():
    """Output shape matches (A, T) and values are valid frequencies."""
    geno, freq, A = _synthetic(H=300, T=150, A=4)
    result = window_init_allele_freq(geno, freq, A, window_size=50)
    assert result.shape == (4, 150)
    result_np = np.asarray(result)
    assert result_np.min() > 0
    assert result_np.max() < 1


def test_single_window():
    """Edge case: window_size >= T means only one window."""
    geno, freq, A = _synthetic(H=100, T=50, A=2)
    result = window_init_allele_freq(
        geno, freq, A, window_size=100, hap_batch=30,
    )
    assert result.shape == (2, 50)
    assert np.all(np.isfinite(np.asarray(result)))
