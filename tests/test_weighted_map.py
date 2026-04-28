"""Tests for popout.em.update_generations_with_priors — the
similarity-weighted MAP T-estimator.

This replaces the old ``test_per_component_t_map.py`` (deleted with the
v1 priors module). The math:

    α_eff[k] = 1 + Σ_p assignment[p, k] * (α_p - 1)
    β_eff[k] = 1 + Σ_p assignment[p, k] * (β_p - 1)
    r_MAP[k] = (successes_eff[k] + α_eff[k] - 1)
             / (trials_eff[k] + α_eff[k] + β_eff[k] - 2)

with successes_eff = switches_per_comp / (1 - mu),
trials_eff = d_weighted_occupancy / morgans_per_step.
"""

from __future__ import annotations

import numpy as np
import pytest

from popout.datatypes import EMStats
from popout.em import update_generations_with_priors
from popout.identity import AIMPanel, AIMSignature
from popout.prior_spec import (
    LinearAnnealingSchedule,
    Prior,
    Priors,
    prior_to_beta,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _trivial_panel() -> AIMPanel:
    return AIMPanel(
        chrom=np.array(["1"], dtype=object),
        pos_bp=np.array([100], dtype=np.int64),
        expected_freq=np.array([0.5]),
        marker_weight=np.array([1.0]),
    )


def _make_priors(specs: list[tuple[float, float, float]], mps: float = 1e-4) -> Priors:
    """Build a Priors object with one prior per (gen_mean, gen_lo, gen_hi)
    spec. Each prior carries a placeholder identity signature."""
    sig = AIMSignature(panel=_trivial_panel())
    plist = []
    for i, (mean, lo, hi) in enumerate(specs):
        a, b = prior_to_beta(mean, lo, hi, mps)
        plist.append(
            Prior(
                name=f"P{i}",
                identity_signatures=(sig,),
                gen_mean=mean, gen_lo=lo, gen_hi=hi,
                alpha=a, beta=b,
            )
        )
    return Priors(
        priors=tuple(plist),
        morgans_per_step=mps,
        annealing=LinearAnnealingSchedule(1.0, 0.1, 10),
        fingerprint="x" * 64,
        source_path="<test>",
    )


def _stats(switches: np.ndarray, occ: np.ndarray) -> EMStats:
    A = len(switches)
    return EMStats(
        weighted_counts=np.zeros((A, 1), dtype=np.float32),
        total_weights=np.ones((A, 1), dtype=np.float32),
        mu_sum=np.full((A,), 1.0 / A, dtype=np.float32),
        switch_sum=np.array(0.0, dtype=np.float32),
        switches_per_hap=np.array(0.0, dtype=np.float32),
        soft_switches_per_hap=np.array(0.0, dtype=np.float32),
        n_haps=np.array(1, dtype=np.int64),
        n_sites=np.array(1, dtype=np.int64),
        switches_per_comp=switches.astype(np.float64),
        d_weighted_occupancy=occ.astype(np.float64),
    )


def _compute_r(switches: float, occ: float, mu_k: float, mps: float, alpha: float, beta: float) -> float:
    """Closed-form MAP for one component."""
    succ = switches / max(1 - mu_k, 1e-3)
    trials = occ / mps
    num = succ + alpha - 1
    den = trials + alpha + beta - 2
    return num / den


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


def test_zero_assignment_collapses_to_mle():
    """Components with zero total assignment use bare MLE (Beta(1,1))."""
    K = 3
    switches = np.array([5.0, 8.0, 12.0])
    occ = np.array([1000.0, 2000.0, 3000.0])
    mu = np.array([0.3, 0.3, 0.4])
    cur_T = np.array([20.0, 20.0, 20.0])
    priors = _make_priors([(7, 4, 12), (2, 1, 4)])

    # Zero assignment → all priors at zero weight on every component.
    assignment = np.zeros((2, K))

    new_T = update_generations_with_priors(
        _stats(switches, occ), cur_T, mu, priors, assignment,
    )
    new_T = np.asarray(new_T)

    # MLE r_k = succ / trials with successes = sw/(1-mu), trials = occ/mps
    mps = priors.morgans_per_step
    expected_T = []
    for k in range(K):
        succ = switches[k] / (1 - mu[k])
        trials = occ[k] / mps
        r = succ / trials
        T_est = -np.log1p(-r) / mps
        # Apply log-space blend toward cur_T[k]
        T_est = max(1.0, min(T_est, 1000.0))
        log_T = 0.7 * np.log(max(cur_T[k], 1.0)) + 0.3 * np.log(T_est)
        expected_T.append(max(1.0, min(np.exp(log_T), 1000.0)))

    np.testing.assert_allclose(new_T, expected_T, rtol=1e-4)


def test_full_assignment_matches_standard_map():
    """assignment[p, k] = 1 for exactly one p reproduces standard MAP."""
    K = 2
    switches = np.array([10.0, 20.0])
    occ = np.array([5000.0, 10_000.0])
    mu = np.array([0.4, 0.6])
    cur_T = np.array([10.0, 10.0])
    priors = _make_priors([(7, 4, 12), (15, 8, 25)])

    # Comp 0 fully assigned to prior 0; comp 1 fully assigned to prior 1.
    assignment = np.array([[1.0, 0.0], [0.0, 1.0]])

    new_T = update_generations_with_priors(
        _stats(switches, occ), cur_T, mu, priors, assignment,
    )
    new_T = np.asarray(new_T)

    mps = priors.morgans_per_step
    p0 = priors.priors[0]
    p1 = priors.priors[1]

    r0 = _compute_r(switches[0], occ[0], mu[0], mps, p0.alpha, p0.beta)
    r1 = _compute_r(switches[1], occ[1], mu[1], mps, p1.alpha, p1.beta)
    T0_est = max(1.0, min(-np.log1p(-r0) / mps, 1000.0))
    T1_est = max(1.0, min(-np.log1p(-r1) / mps, 1000.0))

    expected_0 = max(1.0, min(
        np.exp(0.7 * np.log(max(cur_T[0], 1.0)) + 0.3 * np.log(T0_est)),
        1000.0,
    ))
    expected_1 = max(1.0, min(
        np.exp(0.7 * np.log(max(cur_T[1], 1.0)) + 0.3 * np.log(T1_est)),
        1000.0,
    ))

    np.testing.assert_allclose(new_T, [expected_0, expected_1], rtol=1e-4)


def test_half_assignment_interpolates():
    """assignment[p, k] = 0.5 sits strictly between zero and full
    assignment. Uses a small-occupancy regime so the prior pseudocount
    actually moves the estimate (large occ → MLE dominates → all three
    settings return the same value)."""
    K = 1
    switches = np.array([3.0])   # successes_eff = 3 / 0.5 = 6
    occ = np.array([0.5])         # trials_eff = 0.5 / 1e-4 = 5000
    mu = np.array([0.5])
    cur_T = np.array([5.0])
    priors = _make_priors([(50, 30, 80)])  # prior pulls T upward

    T_zero = float(update_generations_with_priors(
        _stats(switches, occ), cur_T, mu, priors,
        np.array([[0.0]]),
    )[0])
    T_half = float(update_generations_with_priors(
        _stats(switches, occ), cur_T, mu, priors,
        np.array([[0.5]]),
    )[0])
    T_full = float(update_generations_with_priors(
        _stats(switches, occ), cur_T, mu, priors,
        np.array([[1.0]]),
    )[0])

    assert T_zero < T_half < T_full, (
        f"interpolation failed: zero={T_zero}, half={T_half}, full={T_full}"
    )


# --------------------------------------------------------------------------
# Multi-prior, mixed primed / unprimed
# --------------------------------------------------------------------------


def test_k8_mixed_primed_unprimed_gets_correct_treatment():
    """K=8: priors fully attached to k ∈ {0, 3, 5}; the rest receive
    bare MLE (zero assignment row sum). Uses small-occupancy regime so
    the prior actually moves the MAP."""
    K = 8
    switches = np.full(K, 0.5)
    occ = np.full(K, 0.05)
    mu = np.full(K, 1.0 / K)
    cur_T = np.full(K, 20.0)
    priors = _make_priors([(7, 4, 12), (2, 1, 4), (15, 8, 25)])

    # 3 priors. Each fully assigned to one target component.
    P = 3
    assignment = np.zeros((P, K))
    assignment[0, 0] = 1.0
    assignment[1, 3] = 1.0
    assignment[2, 5] = 1.0

    new_T = update_generations_with_priors(
        _stats(switches, occ), cur_T, mu, priors, assignment,
    )
    new_T = np.asarray(new_T)

    # Components 0/3/5 are pulled by their respective priors; 1/2/4/6/7
    # are at MLE (no influence from priors).
    mps = priors.morgans_per_step
    succ = switches[0] / (1 - mu[0])
    trials = occ[0] / mps
    r_mle = succ / trials
    T_mle = max(1.0, min(-np.log1p(-r_mle) / mps, 1000.0))
    log_T_mle = 0.7 * np.log(max(cur_T[0], 1.0)) + 0.3 * np.log(T_mle)
    T_mle_blended = max(1.0, min(np.exp(log_T_mle), 1000.0))

    for k in (1, 2, 4, 6, 7):
        np.testing.assert_allclose(new_T[k], T_mle_blended, rtol=1e-4)

    # Primed components must differ from the MLE value.
    for k in (0, 3, 5):
        assert not np.isclose(new_T[k], T_mle_blended, rtol=1e-3), (
            f"primed component {k} got the MLE value {new_T[k]} == {T_mle_blended}"
        )


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------


def test_raises_when_per_comp_stats_missing():
    K = 2
    priors = _make_priors([(7, 4, 12)])
    stats = EMStats(
        weighted_counts=np.zeros((K, 1), dtype=np.float32),
        total_weights=np.ones((K, 1), dtype=np.float32),
        mu_sum=np.full((K,), 0.5, dtype=np.float32),
        switch_sum=np.array(0.0, dtype=np.float32),
        switches_per_hap=np.array(0.0, dtype=np.float32),
        soft_switches_per_hap=np.array(0.0, dtype=np.float32),
        n_haps=np.array(1, dtype=np.int64),
        n_sites=np.array(1, dtype=np.int64),
        # No switches_per_comp / d_weighted_occupancy
    )
    with pytest.raises(ValueError, match="per-component switch stats"):
        update_generations_with_priors(
            stats, np.array([10.0, 10.0]), np.array([0.5, 0.5]),
            priors, np.zeros((1, K)),
        )


def test_rejects_assignment_shape_mismatch():
    K = 3
    switches = np.full(K, 5.0)
    occ = np.full(K, 1000.0)
    priors = _make_priors([(7, 4, 12), (2, 1, 4)])  # P=2

    with pytest.raises(ValueError, match="shape"):
        update_generations_with_priors(
            _stats(switches, occ),
            np.full(K, 10.0),
            np.full(K, 1.0 / K),
            priors,
            np.zeros((1, K)),  # wrong P
        )
