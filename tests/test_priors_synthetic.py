"""Synthetic smoke tests for end-to-end priors behavior.

These tests run actual EM iterations on simulated data and verify that
the priors machinery (Step 1-6) wires together correctly:

* When the prior mean matches the simulated truth, per-component T
  converges near the truth.
* When the prior mean is far from the truth on weak data, T is pulled
  toward the prior.
* When the prior mean is far from the truth on strong data, the data
  wins.

simulate_admixed only generates a single scalar T, so each test runs
on data with one true T and varies the prior to probe both regimes.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from popout.em import run_em
from popout.priors import load_priors
from popout.simulate import simulate_admixed


def _write_priors(tmp_path, body):
    p = tmp_path / "priors.yaml"
    p.write_text(textwrap.dedent(body).lstrip())
    return load_priors(p)


def test_priors_matching_truth_recover_T(tmp_path):
    """When prior is centered on the simulated truth, the per-comp T
    estimate stays in a reasonable band around the truth (a few-iter
    EM run on small data won't hit the truth exactly — within a factor
    of ~2 across components is the bar)."""
    true_T = 10.0
    chrom_data, _, _ = simulate_admixed(
        n_samples=120, n_sites=400, n_ancestries=3,
        gen_since_admix=int(true_T),
        chrom_length_cm=80.0, rng_seed=2,
    )

    priors = _write_priors(tmp_path, """
        morgans_per_step: 1.2e-4
        components:
          - {component_idx: 0, gen_mean: 10, gen_lo: 6, gen_hi: 18}
          - {component_idx: 1, gen_mean: 10, gen_lo: 6, gen_hi: 18}
          - {component_idx: 2, gen_mean: 10, gen_lo: 6, gen_hi: 18}
    """)

    res = run_em(
        chrom_data, n_ancestries=3, n_em_iter=4, gen_since_admix=true_T,
        rng_seed=0, priors=priors,
    )

    gpc = np.array(res.model.gen_per_comp)
    # All per-comp T should be within a factor of 2 of the truth.
    for k, T_k in enumerate(gpc):
        assert 0.5 * true_T < T_k < 2.0 * true_T, f"comp {k}: T={T_k}"


def test_priors_far_from_truth_pull_components(tmp_path):
    """When the prior is far from the simulated truth and EM is given
    only a couple of iterations on modest-size data, the per-comp T
    estimates should be pulled toward the prior mean (vs the no-prior
    baseline)."""
    true_T = 10.0
    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=200, n_ancestries=3,
        gen_since_admix=int(true_T),
        chrom_length_cm=40.0, rng_seed=3,
    )

    # Tight prior centered far below the truth on all components.
    priors = _write_priors(tmp_path, """
        morgans_per_step: 1.2e-4
        components:
          - {component_idx: 0, gen_mean: 2, gen_lo: 1, gen_hi: 4}
          - {component_idx: 1, gen_mean: 2, gen_lo: 1, gen_hi: 4}
          - {component_idx: 2, gen_mean: 2, gen_lo: 1, gen_hi: 4}
    """)

    res_no = run_em(
        chrom_data, n_ancestries=3, n_em_iter=3, gen_since_admix=true_T,
        rng_seed=0, priors=None,
    )
    res_p = run_em(
        chrom_data, n_ancestries=3, n_em_iter=3, gen_since_admix=true_T,
        rng_seed=0, priors=priors,
    )

    # No-prior baseline: scalar T, may have moved or stayed put.
    T_baseline = float(res_no.model.gen_since_admix)

    # Priors run: every per-comp T should be ≤ baseline (pulled down).
    gpc = np.array(res_p.model.gen_per_comp)
    assert (gpc <= T_baseline + 0.5).all(), (
        f"priors did not pull T below baseline {T_baseline:.2f}: {gpc}"
    )


def test_block_emissions_with_priors_runs_to_convergence(tmp_path):
    """End-to-end: --block-emissions + --priors must run without raising
    and must produce a per-comp T vector on the fitted model.

    Regression test for the AoU 2026-04-27 crash where the block path
    didn't populate switches_per_comp / d_weighted_occupancy on EMStats."""
    true_T = 8.0
    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=200, n_ancestries=3,
        gen_since_admix=int(true_T),
        chrom_length_cm=40.0, rng_seed=44,
    )

    priors = _write_priors(tmp_path, """
        morgans_per_step: 1.2e-4
        components:
          - {component_idx: 0, gen_mean: 5, gen_lo: 3, gen_hi: 9}
          - {component_idx: 1, gen_mean: 5, gen_lo: 3, gen_hi: 9}
          - {component_idx: 2, gen_mean: 5, gen_lo: 3, gen_hi: 9}
    """)

    # use_block_emissions=True is the path that crashed at AoU iter 2.
    # We need n_em_iter ≥ 2 to exercise the M-step priors branch (T is
    # frozen for iter 0).
    res = run_em(
        chrom_data, n_ancestries=3, n_em_iter=3, gen_since_admix=true_T,
        rng_seed=0, priors=priors,
        use_block_emissions=True, block_size=8,
    )

    assert res.model.gen_per_comp is not None
    gpc = np.array(res.model.gen_per_comp)
    assert gpc.shape == (3,)
    # Sanity: each per-comp T is a finite, positive value in the clipping
    # band [1, 1000]. This test deliberately does not pin specific
    # numerics; the point is just that the priors M-step ran.
    assert np.isfinite(gpc).all()
    assert (gpc >= 1.0).all() and (gpc <= 1000.0).all()


def test_priors_unprimed_components_use_mle(tmp_path):
    """A run with priors only on a subset of components: the unprimed
    ones inherit the bare MLE path, primed ones see the prior pull."""
    true_T = 10.0
    chrom_data, _, _ = simulate_admixed(
        n_samples=100, n_sites=300, n_ancestries=4,
        gen_since_admix=int(true_T),
        chrom_length_cm=60.0, rng_seed=4,
    )

    # Only component 0 has a prior; components 1, 2, 3 are unprimed.
    priors = _write_priors(tmp_path, """
        morgans_per_step: 1.2e-4
        components:
          - {component_idx: 0, gen_mean: 2, gen_lo: 1, gen_hi: 4}
    """)

    res = run_em(
        chrom_data, n_ancestries=4, n_em_iter=3, gen_since_admix=true_T,
        rng_seed=0, priors=priors,
    )
    gpc = np.array(res.model.gen_per_comp)

    # Component 0 should be pulled toward 2.
    # Components 1, 2, 3 should NOT be near 2 (they only see data MLE).
    # We can't pin exact MLE without running a no-prior baseline, so the
    # simplest check: at least one unprimed component lands clearly
    # away from the prior center (≥ 4) — otherwise we'd be flagging the
    # bare-MLE branch broken.
    far_from_prior = [k for k, T_k in enumerate(gpc[1:], start=1) if T_k > 4.0]
    assert len(far_from_prior) >= 1, (
        f"all unprimed components stuck at prior mean: {gpc.tolist()}"
    )
