"""Synthetic smoke tests for end-to-end priors behavior under v2 schema.

The v2 priors framework uses soft assignment: each prior is matched
against components by an identity signature, and the resulting (P, K)
weight matrix scales the Beta(α, β) pseudocounts.

These tests use a uniform-scoring identity signature (see
``tests/conftest.UniformScoreSig``) so the assignment is uniform 1/K
across components. That isolates M-step + EM-loop behavior from
identity-scoring details (covered separately in ``test_identity.py`` /
the upcoming ``test_identity_synthetic.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from popout.em import run_em
from popout.simulate import simulate_admixed
from tests.conftest import make_priors_uniform


def test_priors_matching_truth_produces_finite_gen_per_comp(tmp_path):
    """Prior centered on the simulated truth — per-comp T runs cleanly,
    is finite, and lands inside the prior's plausibility band."""
    true_T = 10.0
    chrom_data, _, _ = simulate_admixed(
        n_samples=120, n_sites=400, n_ancestries=3,
        gen_since_admix=int(true_T),
        chrom_length_cm=80.0, rng_seed=2,
    )

    priors = make_priors_uniform([(10, 6, 18), (10, 6, 18), (10, 6, 18)])

    res = run_em(
        chrom_data, n_ancestries=3, n_em_iter=4, gen_since_admix=true_T,
        rng_seed=0, priors=priors,
    )

    gpc = np.array(res.model.gen_per_comp)
    # Each per-comp T should be near the prior + truth band.
    for k, T_k in enumerate(gpc):
        assert 0.5 * true_T < T_k < 2.0 * true_T, f"comp {k}: T={T_k}"


def test_priors_far_from_truth_pull_components(tmp_path):
    """Priors centered far below the simulated truth pull T downward
    relative to the no-prior baseline (which estimates a much higher
    scalar T from the noisy small-sample data)."""
    true_T = 10.0
    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=200, n_ancestries=3,
        gen_since_admix=int(true_T),
        chrom_length_cm=40.0, rng_seed=3,
    )

    priors = make_priors_uniform([(2, 1, 4), (2, 1, 4), (2, 1, 4)])

    res_no = run_em(
        chrom_data, n_ancestries=3, n_em_iter=3, gen_since_admix=true_T,
        rng_seed=0, priors=None,
    )
    res_p = run_em(
        chrom_data, n_ancestries=3, n_em_iter=3, gen_since_admix=true_T,
        rng_seed=0, priors=priors,
    )

    T_baseline = float(res_no.model.gen_since_admix)
    gpc = np.array(res_p.model.gen_per_comp)
    # Priors pull every component below the no-prior scalar T.
    assert (gpc <= T_baseline + 0.5).all(), (
        f"priors did not pull T below baseline {T_baseline:.2f}: {gpc}"
    )


def test_block_emissions_with_priors_runs_to_convergence(tmp_path):
    """End-to-end: --block-emissions + --priors must run without raising
    and must produce a per-comp T vector on the fitted model.

    Regression test for the AoU 2026-04-27 crash where the block path
    didn't populate switches_per_comp / d_weighted_occupancy on EMStats.
    """
    true_T = 8.0
    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=200, n_ancestries=3,
        gen_since_admix=int(true_T),
        chrom_length_cm=40.0, rng_seed=44,
    )

    priors = make_priors_uniform([(5, 3, 9), (5, 3, 9), (5, 3, 9)])

    res = run_em(
        chrom_data, n_ancestries=3, n_em_iter=3, gen_since_admix=true_T,
        rng_seed=0, priors=priors,
        use_block_emissions=True, block_size=8,
    )

    assert res.model.gen_per_comp is not None
    gpc = np.array(res.model.gen_per_comp)
    assert gpc.shape == (3,)
    assert np.isfinite(gpc).all()
    assert (gpc >= 1.0).all() and (gpc <= 1000.0).all()
