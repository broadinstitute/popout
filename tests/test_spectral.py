"""Tests for spectral initialization, including hierarchical detection."""

import numpy as np
import jax
import jax.numpy as jnp

from popout.spectral import (
    _detect_n_ancestries_recursive,
    _detect_n_ancestries_eigenvalue_gap,
    _bic_split_test,
    _sub_pca,
    seed_ancestry_soft,
)


def _make_clustered_data(n_clusters, n_per_cluster=200, d=5, separation=5.0, seed=42):
    """Generate well-separated clusters in d dimensions."""
    rng = np.random.default_rng(seed)
    points = []
    for i in range(n_clusters):
        center = np.zeros(d)
        center[i % d] = separation * (i + 1)
        pts = rng.normal(center, 1.0, size=(n_per_cluster, d))
        points.append(pts)
    return jnp.array(np.vstack(points))


def test_bic_split_unimodal():
    """BIC test should NOT split a single Gaussian."""
    rng = np.random.default_rng(42)
    X = jnp.array(rng.normal(0, 1, size=(500, 3)))
    key = jax.random.PRNGKey(0)
    should_split, labels = _bic_split_test(X, key, per_sample_threshold=0.01)
    assert not should_split


def test_bic_split_bimodal():
    """BIC test should split two well-separated clusters."""
    X = _make_clustered_data(2, n_per_cluster=300, d=3, separation=10.0)
    key = jax.random.PRNGKey(0)
    should_split, labels = _bic_split_test(X, key, per_sample_threshold=0.01)
    assert should_split
    assert labels is not None
    assert labels.shape == (600,)


def test_recursive_detects_known_structure():
    """Recursive detector finds the right number of well-separated clusters."""
    for n_clusters in [2, 3, 4]:
        X = _make_clustered_data(n_clusters, n_per_cluster=200, d=5, separation=8.0)
        key = jax.random.PRNGKey(42)
        n_detected = _detect_n_ancestries_recursive(X, max_a=8, key=key)
        assert n_detected == n_clusters, (
            f"Expected {n_clusters}, detected {n_detected}"
        )


def test_recursive_minimum_is_two():
    """Even a single cluster returns at least 2."""
    rng = np.random.default_rng(42)
    X = jnp.array(rng.normal(0, 1, size=(500, 5)))
    key = jax.random.PRNGKey(0)
    n_detected = _detect_n_ancestries_recursive(X, max_a=8, key=key)
    assert n_detected >= 2


def test_recursive_respects_max_a():
    """Recursive detector caps at max_a."""
    X = _make_clustered_data(6, n_per_cluster=200, d=5, separation=10.0)
    key = jax.random.PRNGKey(0)
    n_detected = _detect_n_ancestries_recursive(X, max_a=4, key=key)
    assert n_detected <= 4


def test_sub_pca():
    """Sub-PCA produces the expected shape."""
    X = jnp.array(np.random.default_rng(42).normal(size=(100, 10)))
    proj = _sub_pca(X, n_components=2)
    assert proj.shape == (100, 2)


def test_seed_ancestry_soft_recursive():
    """Integration: seed_ancestry_soft with recursive detection runs without error."""
    from popout.simulate import simulate_admixed
    chrom_data, _, _ = simulate_admixed(n_samples=200, n_sites=500, n_ancestries=3)
    labels, resp, n_anc, _proj = seed_ancestry_soft(
        chrom_data.geno, detection_method="recursive", rng_seed=42,
    )
    assert n_anc >= 2
    assert labels.shape == (400,)  # 200 samples * 2 haplotypes
    assert resp.shape == (400, n_anc)


def test_seed_ancestry_soft_eigenvalue_gap():
    """Integration: seed_ancestry_soft with eigenvalue-gap detection."""
    from popout.simulate import simulate_admixed
    chrom_data, _, _ = simulate_admixed(n_samples=200, n_sites=500, n_ancestries=3)
    labels, resp, n_anc, _proj = seed_ancestry_soft(
        chrom_data.geno, detection_method="eigenvalue-gap", rng_seed=42,
    )
    assert n_anc >= 2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
