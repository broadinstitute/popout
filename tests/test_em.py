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


def test_run_em_per_hap_T_with_block_emissions_uses_buckets(monkeypatch):
    """When per_hap_T=True and use_block_emissions=True, the block-emissions
    E-step must dispatch across bucket centers once buckets are set.

    Iter 0: T held fixed; E-step runs once per chunk at global T.
    Iter 1: M-step sets bucket_assignments on the model.
    Iter 2: E-step must run forward_backward_blocks once per bucket per chunk,
    each call carrying that bucket's center as gen_since_admix.

    Bucket assignments are forced (via spy on the M-step) to span multiple
    buckets so dispatch is exercised even on small synthetic data.
    """
    import popout.em as em_mod
    import jax.numpy as jnp_local
    from popout.simulate import simulate_admixed

    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=240, n_ancestries=3,
        gen_since_admix=20, chrom_length_cm=80.0,
        pure_fraction=0.0, rng_seed=0,
    )

    captured_T_per_iter: list[list[float]] = []
    iter_buf: list[float] = []
    real_fb_blocks = em_mod.forward_backward_blocks

    def spy(model, bd, *, compute_soft_switches=False):
        iter_buf.append(float(model.gen_since_admix))
        return real_fb_blocks(model, bd, compute_soft_switches=compute_soft_switches)

    monkeypatch.setattr(em_mod, "forward_backward_blocks", spy)

    real_update = em_mod.update_generations_per_hap_from_stats

    def update_spy(stats, *args, **kwargs):
        captured_T_per_iter.append(list(iter_buf))
        iter_buf.clear()
        T_per_hap, _, T_global = real_update(stats, *args, **kwargs)
        H = stats.n_haps
        forced = jnp_local.array(np.arange(H) % 3, dtype=jnp_local.int32)
        return T_per_hap, forced, T_global

    monkeypatch.setattr(em_mod, "update_generations_per_hap_from_stats", update_spy)

    em_mod.run_em(
        chrom_data,
        n_ancestries=3,
        n_em_iter=3,
        gen_since_admix=20.0,
        per_hap_T=True,
        n_T_buckets=3,
        use_block_emissions=True,
        block_size=8,
        skip_decode=True,
        rng_seed=0,
    )

    assert len(captured_T_per_iter) >= 2, (
        f"expected ≥2 M-step calls (iter 1, iter 2); got {len(captured_T_per_iter)}"
    )
    iter1_T = set(captured_T_per_iter[0])
    iter2_T = set(captured_T_per_iter[1])
    assert len(iter1_T) == 1, (
        f"iter 1 should run with one global T, got {sorted(iter1_T)}"
    )
    assert len(iter2_T) > 1, (
        f"iter 2 should dispatch across multiple bucket centers; "
        f"got only {sorted(iter2_T)}"
    )


def test_block_em_soft_switches_are_density_invariant_not_hard_cast(monkeypatch):
    """The block-emissions E-step must populate EMStats.soft_switches_per_hap
    with xi-based density-invariant values, not hard-call integer counts cast
    to float (the bug fixed by this commit). Captured stats from iter 1's
    M-step must contain at least some non-integer soft-switch values.
    """
    import popout.em as em_mod
    from popout.simulate import simulate_admixed

    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=240, n_ancestries=3,
        gen_since_admix=20, chrom_length_cm=80.0,
        pure_fraction=0.0, rng_seed=1,
    )

    captured: list = []
    real_update = em_mod.update_generations_per_hap_from_stats

    def update_spy(stats, *args, **kwargs):
        captured.append(np.array(stats.soft_switches_per_hap))
        return real_update(stats, *args, **kwargs)

    monkeypatch.setattr(em_mod, "update_generations_per_hap_from_stats", update_spy)

    em_mod.run_em(
        chrom_data,
        n_ancestries=3,
        n_em_iter=2,
        gen_since_admix=20.0,
        per_hap_T=True,
        n_T_buckets=5,
        use_block_emissions=True,
        block_size=8,
        skip_decode=True,
        rng_seed=1,
    )

    assert captured, "M-step was not called"
    soft = captured[0]
    # Hard-switch-cast-to-float would produce integer values everywhere.
    # xi-based soft switches yield fractional values for haplotypes whose
    # block-boundary posteriors aren't perfectly concentrated.
    n_non_integer = int(np.sum(np.abs(soft - np.round(soft)) > 1e-4))
    assert n_non_integer > 0, (
        f"soft_switches_per_hap looks like hard-count cast (all integers); "
        f"min={soft.min():.3f}, max={soft.max():.3f}"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
