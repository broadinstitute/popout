"""Tests for update_generations_per_component_from_stats."""

from __future__ import annotations

import math
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from popout.datatypes import EMStats
from popout.em import update_generations_per_component_from_stats
from popout.priors import load_priors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MPS = 1.2e-4  # representative per-step distance used throughout these tests


def _make_stats(switches_per_comp, d_weighted_occupancy):
    """Build a minimal EMStats with only the per-component fields that the
    estimator reads. Other fields are zeros of the right rank.
    """
    A = len(switches_per_comp)
    return EMStats(
        weighted_counts=jnp.zeros((A, 1)),
        total_weights=jnp.zeros((A, 1)),
        mu_sum=jnp.zeros((A,)),
        switch_sum=jnp.zeros((1,)),
        switches_per_hap=np.zeros((1,), dtype=np.float32),
        soft_switches_per_hap=np.zeros((1,), dtype=np.float32),
        n_haps=1,
        n_sites=1,
        switches_per_comp=np.asarray(switches_per_comp, dtype=np.float64),
        d_weighted_occupancy=np.asarray(d_weighted_occupancy, dtype=np.float64),
    )


def _write_priors(tmp_path, body: str):
    p = tmp_path / "priors.yaml"
    p.write_text(textwrap.dedent(body).lstrip())
    return load_priors(p)


# Trivial Priors stub for the no-prior case in tests where every
# component is unprimed.  We still need a Priors object because the
# estimator reads morgans_per_step from it.
def _empty_priors(tmp_path, mps=MPS):
    return _write_priors(tmp_path, f"""
        morgans_per_step: {mps}
        components:
          - {{component_idx: 99, gen_mean: 5, gen_lo: 1, gen_hi: 20}}
    """)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_prior_recovers_mle(tmp_path):
    """Unprimed components: the MAP estimator equals the MLE.

    With strong data (saturated suff stats), the log-space blend with
    the previous estimate dominates only weakly — the headline behavior
    is still that the data-implied T is what the function returns
    (after the 30% step toward the new value).
    """
    priors = _empty_priors(tmp_path)

    # Data perfectly consistent with T=10, mu=[0.4, 0.4, 0.2].
    # With (1 - mu[k]) correction:
    # T = sw / ((1 - mu[k]) * occ).
    mu = jnp.array([0.4, 0.4, 0.2])
    target_T = np.array([10.0, 10.0, 10.0])
    occ = np.array([1.0, 1.0, 0.5])  # Morgans of d-weighted occupancy
    sw = target_T * (1.0 - np.asarray(mu)) * occ  # exactly target_T

    stats = _make_stats(sw, occ)
    # Start "current" at target_T so log-space blend is a no-op.
    current = jnp.array(target_T)

    new_T = np.asarray(
        update_generations_per_component_from_stats(stats, current, mu, priors)
    )

    # Components 0,1,2 are all unprimed (priors only has idx 99).
    np.testing.assert_allclose(new_T, target_T, rtol=0.02)


def test_strong_prior_pulls_weak_data(tmp_path):
    """Headline test from the spec.

    Data weakly prefers T=10 (just a handful of switches over a small
    Morgan extent). Prior centered T=2 with [1, 4]. MAP for the primed
    component is < 4. The unprimed companions ride the MLE.

    The estimator applies a 30% log-space blend toward the previous T,
    so the steady state is the MAP. We iterate to convergence to read
    that off.
    """
    priors = _write_priors(tmp_path, f"""
        morgans_per_step: {MPS}
        components:
          - {{component_idx: 0, gen_mean: 2, gen_lo: 1, gen_hi: 4}}
    """)

    mu = jnp.array([0.5, 0.5])  # 2 components

    # Component 0: weak data — 1 switch over 0.1 Morgans (data implies high T,
    # but absolute counts are tiny so prior should dominate).
    sw_0 = 1.0
    occ_0 = 0.1

    # Component 1 (no prior): strong data preferring T=8
    sw_1 = 8.0 * 0.5 * 2.0
    occ_1 = 2.0

    stats = _make_stats([sw_0, sw_1], [occ_0, occ_1])
    current = jnp.array([10.0, 8.0])

    # Iterate to steady state (blend has α=0.3, so 50 iters more than enough).
    for _ in range(50):
        current = update_generations_per_component_from_stats(
            stats, current, mu, priors
        )
    new_T = np.asarray(current)

    assert new_T[0] < 4.0, f"primed comp T={new_T[0]} not pulled below 4 at convergence"
    assert new_T[0] > 1.0
    assert abs(new_T[1] - 8.0) < 0.5


def test_mixed_primed_unprimed(tmp_path):
    """K=8 with priors on a subset; unprimed indices skip the Beta shift."""
    priors = _write_priors(tmp_path, f"""
        morgans_per_step: {MPS}
        components:
          - {{component_idx: 0, gen_mean: 5, gen_lo: 3, gen_hi: 8}}
          - {{component_idx: 3, gen_mean: 5, gen_lo: 3, gen_hi: 8}}
          - {{component_idx: 5, gen_mean: 5, gen_lo: 3, gen_hi: 8}}
    """)

    K = 8
    mu = jnp.full((K,), 1.0 / K)
    # Weak data preferring T=12 across all components (small occ → prior pulls).
    target_T = 12.0
    occ = np.full((K,), 0.05)  # weak data
    sw = target_T * (1.0 - 1.0 / K) * occ
    stats = _make_stats(sw, occ)
    current = jnp.full((K,), target_T)

    # Iterate to steady state.
    for _ in range(50):
        current = update_generations_per_component_from_stats(
            stats, current, mu, priors
        )
    new_T = np.asarray(current)

    primed = {0, 3, 5}
    for k in primed:
        # Should be between prior gen_mean (5) and MLE (12), pulled toward prior.
        assert 4.0 < new_T[k] < 11.0, f"primed comp {k}: T={new_T[k]}"
    # Unprimed: stays at MLE.
    for k in range(K):
        if k not in primed:
            assert abs(new_T[k] - 12.0) < 0.5, f"unprimed comp {k}: T={new_T[k]}"


def test_prior_matches_mle_when_consistent(tmp_path):
    """When the data agrees with the prior, MAP = both ≈ prior mean."""
    priors = _write_priors(tmp_path, f"""
        morgans_per_step: {MPS}
        components:
          - {{component_idx: 0, gen_mean: 7, gen_lo: 4, gen_hi: 12}}
    """)

    mu = jnp.array([0.5, 0.5])
    occ = np.array([1.0, 1.0])
    target_T = 7.0
    sw = np.array([target_T * 0.5 * occ[0], target_T * 0.5 * occ[1]])

    stats = _make_stats(sw, occ)
    current = jnp.array([target_T, target_T])

    new_T = np.asarray(
        update_generations_per_component_from_stats(stats, current, mu, priors)
    )
    # Both components should land near 7.
    np.testing.assert_allclose(new_T, [7.0, 7.0], rtol=0.05)


def test_raises_when_per_comp_stats_missing():
    """If the xi-with-transitions branch wasn't run, function raises."""
    A = 3
    stats = EMStats(
        weighted_counts=jnp.zeros((A, 1)),
        total_weights=jnp.zeros((A, 1)),
        mu_sum=jnp.zeros((A,)),
        switch_sum=jnp.zeros((1,)),
        switches_per_hap=np.zeros((1,), dtype=np.float32),
        soft_switches_per_hap=np.zeros((1,), dtype=np.float32),
        n_haps=1, n_sites=1,
        # switches_per_comp / d_weighted_occupancy left as None
    )

    class _StubPriors:
        morgans_per_step = MPS

        def get(self, idx):
            return None

    with pytest.raises(ValueError, match="per-component switch stats"):
        update_generations_per_component_from_stats(
            stats,
            current_T_per_comp=jnp.full((A,), 10.0),
            mu=jnp.full((A,), 1.0 / A),
            priors=_StubPriors(),
        )


def test_log_space_blend_dampens_movement(tmp_path):
    """When current_T is far from MLE, the new T is between current and MLE
    in log-space — a single iteration doesn't fully move there."""
    priors = _empty_priors(tmp_path)
    mu = jnp.array([0.5, 0.5])
    occ = np.array([1.0, 1.0])
    # Data implies T=20, current is at T=2.
    sw = np.array([20.0 * 0.5 * 1.0, 20.0 * 0.5 * 1.0])
    stats = _make_stats(sw, occ)
    current = jnp.array([2.0, 2.0])

    new_T = np.asarray(
        update_generations_per_component_from_stats(stats, current, mu, priors)
    )
    # Should be between 2 and 20, log-blended (~30% step).
    # log-blend with α=0.3: log(T_new) = 0.7*log(2) + 0.3*log(20) ≈ 1.39
    # T_new ≈ exp(1.39) ≈ 4.0
    assert 3.0 < new_T[0] < 5.5, f"T_new={new_T[0]}"
    assert 3.0 < new_T[1] < 5.5
