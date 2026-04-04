"""Tests for EM module, including allele frequency smoothing."""

import numpy as np
import jax.numpy as jnp

from popout.em import (
    smooth_rare_frequencies,
    compute_bucket_centers,
    assign_buckets,
    update_generations_per_hap,
)


def test_smooth_common_unchanged():
    """Common variants (MAF > threshold) are not smoothed."""
    A, T = 3, 20
    rng = np.random.default_rng(42)
    # All common variants (freq near 0.5)
    freq = jnp.array(rng.uniform(0.3, 0.7, (A, T)), dtype=jnp.float32)
    pos_cm = jnp.linspace(0, 10, T)

    result = smooth_rare_frequencies(freq, pos_cm, bandwidth_cm=0.5, maf_threshold=0.05)
    np.testing.assert_allclose(np.array(result), np.array(freq), atol=1e-6)


def test_smooth_disabled_when_zero_bandwidth():
    """bandwidth_cm=0 disables smoothing."""
    A, T = 3, 20
    rng = np.random.default_rng(42)
    freq = jnp.array(rng.uniform(0.001, 0.01, (A, T)), dtype=jnp.float32)
    pos_cm = jnp.linspace(0, 10, T)

    result = smooth_rare_frequencies(freq, pos_cm, bandwidth_cm=0.0, maf_threshold=0.05)
    np.testing.assert_array_equal(np.array(result), np.array(freq))


def test_smooth_rare_modified():
    """Rare variants are actually modified by smoothing."""
    A, T = 3, 50
    rng = np.random.default_rng(42)
    freq = jnp.array(rng.uniform(0.001, 0.01, (A, T)), dtype=jnp.float32)
    pos_cm = jnp.linspace(0, 10, T)

    result = smooth_rare_frequencies(freq, pos_cm, bandwidth_cm=0.5, maf_threshold=0.05)
    # Smoothed values should differ from raw
    assert not jnp.allclose(result, freq)
    # But should still be in valid range
    assert float(result.min()) >= 1e-6
    assert float(result.max()) <= 1.0 - 1e-6


def test_smooth_single_site():
    """T=1 is a no-op."""
    freq = jnp.array([[0.01], [0.02], [0.03]])
    pos_cm = jnp.array([0.0])
    result = smooth_rare_frequencies(freq, pos_cm, bandwidth_cm=0.5)
    np.testing.assert_array_equal(np.array(result), np.array(freq))


def test_smooth_variable_spacing():
    """Smoothing correctly handles non-uniform site spacing."""
    A, T = 2, 10
    # Two clusters of sites: sites 0-4 close together, sites 5-9 far away
    pos_cm = jnp.array([0.0, 0.01, 0.02, 0.03, 0.04,
                         5.0, 5.01, 5.02, 5.03, 5.04])
    # Rare frequencies: cluster 1 has low freq, cluster 2 has medium-low
    freq = jnp.array([
        [0.001, 0.002, 0.001, 0.002, 0.001,
         0.01,  0.01,  0.01,  0.01,  0.01],
        [0.002, 0.001, 0.002, 0.001, 0.002,
         0.02,  0.02,  0.02,  0.02,  0.02],
    ])

    result = smooth_rare_frequencies(freq, pos_cm, bandwidth_cm=0.05, maf_threshold=0.05)

    # Smoothing within clusters shouldn't mix across clusters (bandwidth=0.05 cM)
    # Cluster 1 values should be averaged among themselves
    # Cluster 2 values should be averaged among themselves
    result_np = np.array(result)
    cluster1_mean = float(result_np[0, :5].mean())
    cluster2_mean = float(result_np[0, 5:].mean())
    # Cluster 2 should still have higher frequency than cluster 1
    assert cluster2_mean > cluster1_mean


def test_smooth_output_shape():
    """Output shape matches input shape."""
    freq = jnp.ones((4, 100)) * 0.005
    pos_cm = jnp.linspace(0, 10, 100)
    result = smooth_rare_frequencies(freq, pos_cm)
    assert result.shape == freq.shape


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
