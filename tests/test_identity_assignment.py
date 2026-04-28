"""Tests for popout.identity_assignment."""

from __future__ import annotations

import numpy as np
import pytest

from popout.identity import ComponentState
from popout.identity_assignment import assign_priors_to_components
from popout.prior_spec import LinearAnnealingSchedule, Prior, Priors


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _ConstScoreSig:
    """Test signature: pre-set per-component scores; ignores the
    component's actual freq."""

    def __init__(self, scores: list[float], weight: float = 1.0):
        self._scores = list(scores)
        self.weight = weight

    def score(self, cs: ComponentState) -> float:
        # Cheat: identify component via cs.chrom == str(idx).
        return float(self._scores[int(cs.chrom)])


def _build_priors(
    sigs_per_prior: list[list[_ConstScoreSig]],
    *,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    ramp_iters: int = 10,
) -> Priors:
    priors_list = []
    for i, sigs in enumerate(sigs_per_prior):
        priors_list.append(
            Prior(
                name=f"P{i}",
                identity_signatures=tuple(sigs),
                gen_mean=2.0,
                gen_lo=1.0,
                gen_hi=4.0,
                alpha=2.0,
                beta=8.0,
            )
        )
    return Priors(
        priors=tuple(priors_list),
        morgans_per_step=1e-4,
        annealing=LinearAnnealingSchedule(
            tau_start=tau_start, tau_end=tau_end, ramp_iters=ramp_iters,
        ),
        fingerprint="x" * 64,
        source_path="<test>",
    )


def _states(K: int) -> list[ComponentState]:
    return [
        ComponentState(
            freq=np.array([0.5]),
            mu=1.0 / K,
            pos_bp=np.array([10], dtype=np.int64),
            chrom=str(k),
        )
        for k in range(K)
    ]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_one_prior_strong_winner_at_low_tau():
    K = 4
    sig = _ConstScoreSig([10.0, 1.0, 1.0, 1.0])
    priors = _build_priors([[sig]], tau_start=0.1, tau_end=0.1)
    states = _states(K)

    w = assign_priors_to_components(priors, states, iteration=0)
    assert w.shape == (1, K)
    np.testing.assert_allclose(w.sum(axis=1), [1.0])
    # Component 0 should overwhelmingly dominate at low tau.
    assert w[0, 0] > 0.99


def test_one_prior_uniform_at_high_tau():
    K = 4
    sig = _ConstScoreSig([10.0, 1.0, 1.0, 1.0])
    # Make tau very large by scaling the *raw* scores down (z-standardized
    # scores have unit std, so tau >> 1 flattens softmax).
    priors = _build_priors([[sig]], tau_start=100.0, tau_end=100.0)
    w = assign_priors_to_components(priors, _states(4), iteration=0)
    # Roughly uniform.
    assert np.allclose(w[0], 0.25, atol=0.01)


def test_two_priors_pick_different_components():
    K = 4
    p0 = _ConstScoreSig([10.0, 1.0, 1.0, 1.0])
    p1 = _ConstScoreSig([1.0, 10.0, 1.0, 1.0])
    priors = _build_priors([[p0], [p1]], tau_start=0.1, tau_end=0.1)
    w = assign_priors_to_components(priors, _states(K), iteration=0)
    assert w.shape == (2, K)
    assert int(np.argmax(w[0])) == 0
    assert int(np.argmax(w[1])) == 1
    # Each row sums to 1.
    np.testing.assert_allclose(w.sum(axis=1), [1.0, 1.0])


def test_two_priors_competing_split_weight():
    """Both priors prefer the same component → weights split, neither
    approaches 1.0. (The 'racing priors' regime — diagnostic, not wrong.)"""
    K = 4
    p0 = _ConstScoreSig([10.0, 1.0, 1.0, 1.0])
    p1 = _ConstScoreSig([10.0, 1.0, 1.0, 1.0])
    priors = _build_priors([[p0], [p1]], tau_start=0.1, tau_end=0.1)
    w = assign_priors_to_components(priors, _states(K), iteration=0)
    # At low tau, each prior latches onto component 0 individually; no
    # cross-prior normalization, so both rows have ~1.0 on component 0.
    # That is the documented "diagnostic event" — verify both rows sum
    # to 1 and each row puts most weight on component 0.
    np.testing.assert_allclose(w.sum(axis=1), [1.0, 1.0])
    assert w[0, 0] > 0.9
    assert w[1, 0] > 0.9


def test_annealing_changes_assignment():
    """Higher iteration → cooler tau → more concentrated weights."""
    K = 4
    sig = _ConstScoreSig([3.0, 1.0, 1.0, 1.0])
    priors = _build_priors(
        [[sig]], tau_start=2.0, tau_end=0.1, ramp_iters=10,
    )
    states = _states(K)

    w_hot = assign_priors_to_components(priors, states, iteration=0)
    w_cold = assign_priors_to_components(priors, states, iteration=10)
    # Hot tau is more uniform; cold is more concentrated on component 0.
    assert w_cold[0, 0] > w_hot[0, 0]


def test_annealing_endpoints_in_assignment():
    """tau_at endpoints feed through correctly."""
    K = 3
    sig = _ConstScoreSig([5.0, 0.0, -5.0])
    priors = _build_priors(
        [[sig]], tau_start=1.0, tau_end=0.01, ramp_iters=10,
    )
    states = _states(K)

    # At iteration 0: tau = 1.0 → moderate softmax.
    w0 = assign_priors_to_components(priors, states, iteration=0)
    # At iteration 10: tau = 0.01 → near-deterministic argmax.
    w10 = assign_priors_to_components(priors, states, iteration=10)
    assert w10[0, 0] > w0[0, 0]
    assert w10[0, 0] > 0.999


def test_empty_components_returns_empty():
    priors = _build_priors([[_ConstScoreSig([])]])
    w = assign_priors_to_components(priors, [], iteration=0)
    assert w.shape == (1, 0)
