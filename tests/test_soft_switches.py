"""Tests for xi-based soft transition posteriors (density-invariant T estimator).

Verifies that:
1. Soft switches are non-negative and sum to ≤ T-1 per haplotype
2. Soft switches are density-invariant (same total across 1x, 2x, 4x site density)
3. forward_backward with compute_transitions=True matches without
4. Checkpointed and non-checkpointed paths agree on soft switches
5. The soft-switch T estimator recovers known T from simulated data
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from popout.simulate import simulate_admixed
from popout.datatypes import AncestryModel, ChromData
from popout.hmm import (
    forward_backward,
    forward_backward_em,
    forward_backward_bucketed_em,
    _compute_soft_switches,
    forward as hmm_forward,
    backward as hmm_backward,
)
from popout.em import (
    init_model_soft,
    update_generations_from_stats,
    update_generations_per_hap_from_stats,
    compute_bucket_centers,
)
from popout.spectral import seed_ancestry_soft


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(n_samples=100, n_sites=100, n_ancestries=3,
                gen_since_admix=20, chrom_length_cm=100.0, rng_seed=42):
    """Build a small model + data for testing."""
    chrom_data, true_ancestry = simulate_admixed(
        n_samples=n_samples,
        n_sites=n_sites,
        n_ancestries=n_ancestries,
        gen_since_admix=gen_since_admix,
        chrom_length_cm=chrom_length_cm,
        rng_seed=rng_seed,
    )
    geno = jnp.array(chrom_data.geno)
    d_morgan = jnp.array(chrom_data.genetic_distances)

    _, resp, n_anc, _proj = seed_ancestry_soft(
        chrom_data.geno, n_ancestries=n_ancestries, rng_seed=rng_seed,
    )
    model = init_model_soft(geno, resp, n_anc, gen_since_admix=gen_since_admix)
    return chrom_data, geno, model, d_morgan


def _make_known_model(n_haps=200, n_sites=100, n_ancestries=3,
                      gen_since_admix=20, chrom_length_cm=100.0, rng_seed=42):
    """Build a model with KNOWN parameters (not estimated from data).

    Returns the model used to generate the data, so T is exactly known.
    """
    chrom_data, true_ancestry = simulate_admixed(
        n_samples=n_haps // 2,
        n_sites=n_sites,
        n_ancestries=n_ancestries,
        gen_since_admix=gen_since_admix,
        chrom_length_cm=chrom_length_cm,
        rng_seed=rng_seed,
    )
    geno = jnp.array(chrom_data.geno)
    d_morgan = jnp.array(chrom_data.genetic_distances)

    # Build model from true ancestry labels (ground truth frequencies)
    pop_freq = np.zeros((n_ancestries, n_sites))
    for a in range(n_ancestries):
        mask = true_ancestry == a
        counts = mask.sum(axis=0)
        geno_at_a = (chrom_data.geno * mask).sum(axis=0)
        pop_freq[a] = np.where(counts > 0, geno_at_a / counts, 0.5)

    # True mu from true ancestry counts
    mu = np.array([(true_ancestry == a).mean() for a in range(n_ancestries)])
    mu = mu / mu.sum()

    model = AncestryModel(
        n_ancestries=n_ancestries,
        mu=jnp.array(mu, dtype=jnp.float32),
        gen_since_admix=float(gen_since_admix),
        allele_freq=jnp.array(np.clip(pop_freq, 1e-3, 1 - 1e-3), dtype=jnp.float32),
        mismatch=jnp.zeros(n_ancestries),
    )
    return chrom_data, geno, model, d_morgan, true_ancestry


# ---------------------------------------------------------------------------
# Test 1: Basic properties of soft switches
# ---------------------------------------------------------------------------

def test_soft_switches_basic_properties():
    """Soft switches are non-negative and bounded by T-1."""
    _, geno, model, d_morgan = _make_model(n_samples=50, n_sites=80)
    H, T = geno.shape

    gamma, soft_sw = forward_backward(
        geno, model, d_morgan, compute_transitions=True,
    )

    soft_sw_np = np.array(soft_sw)
    assert soft_sw_np.shape == (H,)
    assert np.all(np.isfinite(soft_sw_np))
    assert np.all(soft_sw_np >= 0)
    assert np.all(soft_sw_np <= T - 1 + 0.01)  # small tolerance for numerics


def test_soft_switches_vs_hard_with_good_model():
    """With a well-fitted model, soft switches should track hard switches.

    Note: with uncertain posteriors (e.g. fresh init), hard switches can
    be LOWER than soft because argmax is "sticky" — it doesn't flip
    unless a different ancestry clearly dominates. Soft switches correctly
    capture the moderate transition probability even when posteriors are
    uncertain. We test with a ground-truth model to get confident posteriors.
    """
    _, geno, model, d_morgan, _ = _make_known_model(
        n_haps=200, n_sites=100, n_ancestries=3,
        gen_since_admix=20, rng_seed=42,
    )

    gamma, soft_sw = forward_backward(
        geno, model, d_morgan, compute_transitions=True,
    )

    calls = jnp.argmax(gamma, axis=2)
    hard_sw = (calls[:, 1:] != calls[:, :-1]).sum(axis=1).astype(jnp.float32)

    # With a good model: both should be in the same ballpark
    mean_soft = float(jnp.mean(soft_sw))
    mean_hard = float(jnp.mean(hard_sw))
    assert mean_soft > 0, "Expected some soft switches"
    assert mean_hard > 0, "Expected some hard switches"
    # They won't match exactly, but should be within 5x of each other
    ratio = max(mean_soft, mean_hard) / (min(mean_soft, mean_hard) + 1e-10)
    assert ratio < 5.0, f"Soft ({mean_soft:.1f}) and hard ({mean_hard:.1f}) too far apart"


# ---------------------------------------------------------------------------
# Test 2: Gamma output is unchanged by compute_transitions flag
# ---------------------------------------------------------------------------

def test_gamma_unchanged_by_transitions_flag():
    """Posteriors must be identical whether or not soft switches are computed."""
    _, geno, model, d_morgan = _make_model(n_samples=30, n_sites=50)

    gamma_plain = forward_backward(geno, model, d_morgan, compute_transitions=False)
    gamma_with, soft_sw = forward_backward(geno, model, d_morgan, compute_transitions=True)

    np.testing.assert_allclose(
        np.array(gamma_plain), np.array(gamma_with), atol=1e-6,
        err_msg="compute_transitions should not change gamma",
    )


# ---------------------------------------------------------------------------
# Test 3: Checkpointed vs non-checkpointed agree on soft switches
# ---------------------------------------------------------------------------

def test_checkpointed_matches_non_checkpointed():
    """Both paths should produce identical soft switches."""
    _, geno, model, d_morgan = _make_model(n_samples=30, n_sites=50)

    gamma_nc, soft_nc = forward_backward(
        geno, model, d_morgan,
        use_checkpointing=False, compute_transitions=True,
    )
    gamma_ck, soft_ck = forward_backward(
        geno, model, d_morgan,
        use_checkpointing=True, compute_transitions=True,
    )

    np.testing.assert_allclose(
        np.array(gamma_nc), np.array(gamma_ck), atol=1e-5,
        err_msg="gamma mismatch between checkpointed and non-checkpointed",
    )
    np.testing.assert_allclose(
        np.array(soft_nc), np.array(soft_ck), atol=1e-3,
        err_msg="soft switches mismatch between checkpointed and non-checkpointed",
    )


# ---------------------------------------------------------------------------
# Test 4: DENSITY INVARIANCE — the core property
# ---------------------------------------------------------------------------

def test_density_invariance():
    """Soft switches must be approximately invariant to site density.

    Generate data at 1x, 2x, and 4x site density over the same genetic
    length. Total soft switches per haplotype should stay roughly constant.
    Hard switches will inflate with density — that's the bug we're fixing.
    """
    n_samples = 100
    n_ancestries = 3
    gen_since_admix = 20
    chrom_cm = 50.0
    rng_seed = 123

    densities = [100, 200, 400]
    mean_soft = []
    mean_hard = []

    for n_sites in densities:
        chrom_data, true_ancestry = simulate_admixed(
            n_samples=n_samples,
            n_sites=n_sites,
            n_ancestries=n_ancestries,
            gen_since_admix=gen_since_admix,
            chrom_length_cm=chrom_cm,
            rng_seed=rng_seed,
        )
        geno = jnp.array(chrom_data.geno)
        d_morgan = jnp.array(chrom_data.genetic_distances)

        # Use ground-truth-ish model to isolate the switch-counting effect
        _, resp, n_anc, _proj = seed_ancestry_soft(
            chrom_data.geno, n_ancestries=n_ancestries, rng_seed=rng_seed,
        )
        model = init_model_soft(geno, resp, n_anc, gen_since_admix=gen_since_admix)

        gamma, soft_sw = forward_backward(
            geno, model, d_morgan, compute_transitions=True,
        )
        calls = jnp.argmax(gamma, axis=2)
        hard_sw = (calls[:, 1:] != calls[:, :-1]).sum(axis=1).astype(jnp.float32)

        mean_soft.append(float(jnp.mean(soft_sw)))
        mean_hard.append(float(jnp.mean(hard_sw)))

    # Soft switches: ratio between 4x and 1x density should be close to 1
    soft_ratio_4x = mean_soft[2] / (mean_soft[0] + 1e-10)
    # Hard switches: ratio between 4x and 1x should be notably > 1
    hard_ratio_4x = mean_hard[2] / (mean_hard[0] + 1e-10)

    # Soft switches should stay within 50% of original (density-invariant)
    assert 0.5 < soft_ratio_4x < 1.5, (
        f"Soft switches not density-invariant: 1x={mean_soft[0]:.1f}, "
        f"4x={mean_soft[2]:.1f}, ratio={soft_ratio_4x:.2f}"
    )

    # Hard switches should increase meaningfully with density (the old bug)
    # This is a sanity check that the test setup actually exercises the problem
    assert hard_ratio_4x > 1.2, (
        f"Hard switches didn't increase with density as expected: "
        f"1x={mean_hard[0]:.1f}, 4x={mean_hard[2]:.1f}, ratio={hard_ratio_4x:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 5: T recovery — soft switches recover known T
# ---------------------------------------------------------------------------

def test_T_recovery_from_soft_switches():
    """The soft-switch T estimator should recover a T close to the true value."""
    true_T = 20
    _, geno, model, d_morgan, _ = _make_known_model(
        n_haps=400, n_sites=200, n_ancestries=3,
        gen_since_admix=true_T, chrom_length_cm=100.0, rng_seed=77,
    )

    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=200)

    T_est = update_generations_from_stats(
        em_stats, d_morgan, current_T=float(true_T), mu=model.mu,
    )

    # Should be within a factor of 3 of true T (generous for small synthetic data)
    assert true_T / 3 < T_est < true_T * 3, (
        f"T estimate {T_est:.1f} too far from true T={true_T}"
    )


def test_T_recovery_density_stable():
    """T estimated from soft switches should be stable across site densities.

    The same underlying population (same T, same mu, same chromosome)
    should yield similar T estimates regardless of site density.
    """
    true_T = 25
    n_ancestries = 3
    chrom_cm = 80.0

    T_estimates = []
    for n_sites in [100, 200, 400]:
        _, geno, model, d_morgan, _ = _make_known_model(
            n_haps=400, n_sites=n_sites, n_ancestries=n_ancestries,
            gen_since_admix=true_T, chrom_length_cm=chrom_cm, rng_seed=55,
        )
        em_stats = forward_backward_em(geno, model, d_morgan, batch_size=200)
        T_est = update_generations_from_stats(
            em_stats, d_morgan, current_T=float(true_T), mu=model.mu,
        )
        T_estimates.append(T_est)

    # T estimates at different densities should be within 50% of each other
    T_min, T_max = min(T_estimates), max(T_estimates)
    ratio = T_max / (T_min + 1e-10)
    assert ratio < 1.5, (
        f"T estimates not density-stable: {T_estimates}, ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Per-haplotype T from soft switches
# ---------------------------------------------------------------------------

def test_per_hap_T_from_soft_switches():
    """Per-hap T estimation from soft switches produces valid assignments."""
    _, geno, model, d_morgan = _make_model(n_samples=100, n_sites=100)

    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=100)
    bucket_centers = compute_bucket_centers(20)

    T_per_hap, bucket_assignments, T_global = update_generations_per_hap_from_stats(
        em_stats, d_morgan, model.gen_since_admix, model.mu, bucket_centers,
    )

    T_hap_np = np.array(T_per_hap)
    ba_np = np.array(bucket_assignments)

    assert T_hap_np.shape == (geno.shape[0],)
    assert np.all(np.isfinite(T_hap_np))
    assert np.all(T_hap_np >= 1.0)
    assert np.all(T_hap_np <= 1000.0)

    assert np.all(ba_np >= 0)
    assert np.all(ba_np < 20)

    assert 1.0 <= T_global <= 1000.0


# ---------------------------------------------------------------------------
# Test 7: Streaming EM stats include soft switches
# ---------------------------------------------------------------------------

def test_em_stats_have_soft_switches():
    """forward_backward_em populates soft_switches_per_hap."""
    _, geno, model, d_morgan = _make_model(n_samples=50, n_sites=80)

    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=50)

    assert hasattr(em_stats, 'soft_switches_per_hap')
    assert em_stats.soft_switches_per_hap.shape == (geno.shape[0],)
    assert em_stats.soft_switches_per_hap.dtype == np.float32
    assert np.all(np.isfinite(em_stats.soft_switches_per_hap))
    assert np.all(em_stats.soft_switches_per_hap >= 0)


def test_bucketed_em_stats_have_soft_switches():
    """forward_backward_bucketed_em populates soft_switches_per_hap."""
    _, geno, model, d_morgan = _make_model(n_samples=50, n_sites=80)
    H = geno.shape[0]

    # Set up bucketed model
    bucket_centers = compute_bucket_centers(5)
    rng = np.random.default_rng(42)
    bucket_assignments = jnp.array(rng.integers(0, 5, size=H), dtype=jnp.int32)

    bucketed_model = AncestryModel(
        n_ancestries=model.n_ancestries,
        mu=model.mu,
        gen_since_admix=model.gen_since_admix,
        allele_freq=model.allele_freq,
        mismatch=model.mismatch,
        bucket_centers=bucket_centers,
        bucket_assignments=bucket_assignments,
    )

    em_stats = forward_backward_bucketed_em(
        geno, bucketed_model, d_morgan, batch_size=50,
    )

    assert em_stats.soft_switches_per_hap.shape == (H,)
    assert em_stats.soft_switches_per_hap.dtype == np.float32
    assert np.all(np.isfinite(em_stats.soft_switches_per_hap))
    assert np.all(em_stats.soft_switches_per_hap >= 0)


# ---------------------------------------------------------------------------
# Test 8: _compute_soft_switches helper directly
# ---------------------------------------------------------------------------

def test_compute_soft_switches_trivial():
    """With identity transitions (T=0), soft switches should be ~0."""
    H, T, A = 10, 20, 3
    rng = np.random.default_rng(99)

    geno = jnp.array(rng.integers(0, 2, size=(H, T)), dtype=jnp.uint8)
    mu = jnp.array([1/A] * A)

    # Model with T ≈ 0 → nearly identity transitions → no switches
    model = AncestryModel(
        n_ancestries=A,
        mu=mu,
        gen_since_admix=0.001,  # nearly zero
        allele_freq=jnp.array(rng.uniform(0.1, 0.9, (A, T)), dtype=jnp.float32),
        mismatch=jnp.zeros(A),
    )
    d_morgan = jnp.ones(T - 1) * 0.01  # 1 cM per interval

    log_alpha, _ = hmm_forward(geno, model, d_morgan)
    log_beta = hmm_backward(geno, model, d_morgan)

    soft_sw = _compute_soft_switches(log_alpha, log_beta, model, geno, d_morgan)
    soft_sw_np = np.array(soft_sw)

    # With T ≈ 0, switches should be very small
    assert np.all(soft_sw_np < 1.0), (
        f"Expected near-zero soft switches with T≈0, got mean={soft_sw_np.mean():.3f}"
    )


def test_compute_soft_switches_high_T():
    """With very high T, many switches expected."""
    H, T, A = 10, 50, 3
    rng = np.random.default_rng(99)

    geno = jnp.array(rng.integers(0, 2, size=(H, T)), dtype=jnp.uint8)
    mu = jnp.array([1/A] * A)

    # Model with very high T → many recombinations
    model = AncestryModel(
        n_ancestries=A,
        mu=mu,
        gen_since_admix=500.0,
        allele_freq=jnp.array(rng.uniform(0.1, 0.9, (A, T)), dtype=jnp.float32),
        mismatch=jnp.zeros(A),
    )
    d_morgan = jnp.ones(T - 1) * 0.01

    log_alpha, _ = hmm_forward(geno, model, d_morgan)
    log_beta = hmm_backward(geno, model, d_morgan)

    soft_sw = _compute_soft_switches(log_alpha, log_beta, model, geno, d_morgan)
    soft_sw_np = np.array(soft_sw)

    # With high T, expect many switches (close to p_diff per interval)
    p_diff = 1.0 - float((mu ** 2).sum())
    # Expected per haplotype ≈ (T-1) * p_diff (saturated recombination)
    expected_max = (T - 1) * p_diff
    mean_sw = float(soft_sw_np.mean())
    assert mean_sw > expected_max * 0.3, (
        f"Expected many soft switches with T=500, got mean={mean_sw:.1f}, "
        f"expected_max≈{expected_max:.1f}"
    )
