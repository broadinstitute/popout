"""Tests for EM module."""

import numpy as np
import jax.numpy as jnp

from popout.em import (
    compute_bucket_centers,
    assign_buckets,
    update_generations_per_hap,
)


def test_bucket_centers():
    """Bucket centers are geometrically spaced."""
    centers = compute_bucket_centers(n_buckets=20)
    assert centers.shape == (20,)
    assert float(centers[0]) >= 1.0
    assert float(centers[-1]) <= 1000.0
    # Check geometric spacing: ratios between consecutive should be equal
    ratios = np.array(centers[1:] / centers[:-1])
    np.testing.assert_allclose(ratios, ratios[0], atol=1e-5)


def test_assign_buckets():
    """Bucket assignments pick the nearest center in log-space."""
    centers = jnp.array([1.0, 10.0, 100.0, 1000.0])
    T_per_hap = jnp.array([1.5, 8.0, 15.0, 200.0, 999.0])
    assignments = assign_buckets(T_per_hap, centers)
    expected = jnp.array([0, 1, 1, 2, 3])
    np.testing.assert_array_equal(np.array(assignments), np.array(expected))


def test_per_hap_T_zero_switches():
    """Haplotypes with zero switches regularize toward global T."""
    H, T, A = 10, 50, 3
    # Make gamma that gives identical ancestry everywhere (no switches)
    gamma = jnp.zeros((H, T, A))
    gamma = gamma.at[:, :, 0].set(1.0)  # all ancestry 0

    d_morgan = jnp.full(T - 1, 0.001)
    mu = jnp.array([0.5, 0.3, 0.2])
    centers = compute_bucket_centers(10)

    T_per_hap, _, T_global = update_generations_per_hap(
        gamma, d_morgan, 20.0, mu, centers,
    )
    # With zero switches, T_raw = 0 → regularized toward global T=20
    T_arr = np.array(T_per_hap)
    # Should be pulled toward 20 (the current global), clamped to >= 1
    assert T_arr.min() >= 1.0
    assert T_arr.max() <= 1000.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
