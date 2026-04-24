"""Tests for streaming M-step (forward_backward_em / forward_backward_decode).

Verifies that the streaming path produces numerically identical results
to the full-gamma path at small scale, where both paths fit in memory.
"""

import numpy as np
import jax.numpy as jnp

from popout.simulate import simulate_admixed
from popout.hmm import (
    forward_backward,
    forward_backward_batched,
    forward_backward_em,
    forward_backward_decode,
    forward_backward_ancestry_sums,
)
from popout.em import (
    init_model_soft,
    update_allele_freq,
    update_mu,
    update_generations,
    update_generations_per_hap,
    update_allele_freq_from_stats,
    update_mu_from_stats,
    update_generations_from_stats,
    update_generations_per_hap_from_stats,
    compute_bucket_centers,
    run_em,
)
from popout.spectral import seed_ancestry_soft


def _make_model(n_samples=200, n_sites=100, n_ancestries=3, rng_seed=42):
    """Build a small model + data for testing."""
    chrom_data, true_ancestry, _ = simulate_admixed(
        n_samples=n_samples,
        n_sites=n_sites,
        n_ancestries=n_ancestries,
        rng_seed=rng_seed,
    )
    geno = jnp.array(chrom_data.geno)
    d_morgan = jnp.array(chrom_data.genetic_distances)

    _, resp, n_anc, _proj = seed_ancestry_soft(
        chrom_data.geno, n_ancestries=n_ancestries, rng_seed=rng_seed,
    )
    model = init_model_soft(geno, resp, n_anc)
    return chrom_data, geno, model, d_morgan


# -----------------------------------------------------------------------
# EMStats vs full gamma
# -----------------------------------------------------------------------

def test_em_stats_match_full_gamma():
    """Streaming EMStats must match reductions over full gamma."""
    _, geno, model, d_morgan = _make_model(n_samples=100, n_sites=150)

    # Full gamma (ground truth) — use full forward_backward since
    # forward_backward_batched refuses to materialize at large H
    gamma = forward_backward(geno, model, d_morgan)

    # Streaming stats with small batch to exercise multiple iterations
    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=50)

    # Check weighted_counts
    geno_f = geno.astype(jnp.float32)
    expected_wc = jnp.einsum('hta,ht->at', gamma, geno_f)
    np.testing.assert_allclose(
        np.array(em_stats.weighted_counts), np.array(expected_wc),
        atol=1e-4, err_msg="weighted_counts mismatch",
    )

    # Check total_weights
    expected_tw = gamma.sum(axis=0).T
    np.testing.assert_allclose(
        np.array(em_stats.total_weights), np.array(expected_tw),
        atol=1e-4, err_msg="total_weights mismatch",
    )

    # Check mu_sum — rtol needed because float32 accumulation order differs
    expected_mu = gamma.sum(axis=(0, 1))
    np.testing.assert_allclose(
        np.array(em_stats.mu_sum), np.array(expected_mu),
        rtol=1e-4, err_msg="mu_sum mismatch",
    )

    # Hard switch diagnostics (switch_sum, switches_per_hap) are zeroed in
    # the streaming path — they require materializing gamma and are only
    # used for QC output. Verify they're properly zeroed.
    np.testing.assert_array_equal(
        np.array(em_stats.switch_sum), np.zeros(geno.shape[1] - 1),
        err_msg="switch_sum should be zero in streaming path",
    )
    np.testing.assert_array_equal(
        em_stats.switches_per_hap, np.zeros(geno.shape[0], dtype=np.int32),
        err_msg="switches_per_hap should be zero in streaming path",
    )

    assert em_stats.n_haps == geno.shape[0]
    assert em_stats.n_sites == geno.shape[1]


def test_em_stats_single_batch():
    """With batch_size >= H, streaming stats should still match."""
    _, geno, model, d_morgan = _make_model(n_samples=50, n_sites=80)

    gamma = forward_backward(geno, model, d_morgan)
    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=10_000)

    expected_wc = jnp.einsum('hta,ht->at', gamma, geno.astype(jnp.float32))
    np.testing.assert_allclose(
        np.array(em_stats.weighted_counts), np.array(expected_wc), atol=1e-5,
    )


# -----------------------------------------------------------------------
# DecodeResult vs full gamma
# -----------------------------------------------------------------------

def test_decode_result_matches_full():
    """DecodeResult calls/max_post/global_sums must match full gamma."""
    _, geno, model, d_morgan = _make_model(n_samples=100, n_sites=150)

    gamma = forward_backward(geno, model, d_morgan)
    decode = forward_backward_decode(geno, model, d_morgan, batch_size=50)

    # Hard calls
    expected_calls = np.array(jnp.argmax(gamma, axis=2), dtype=np.int8)
    np.testing.assert_array_equal(decode.calls, expected_calls)

    # Max posterior (fp16 storage — atol matches half-precision granularity)
    expected_max = np.array(gamma.max(axis=2))
    np.testing.assert_allclose(
        decode.max_post, expected_max, atol=5e-4,
        err_msg="max_post mismatch",
    )

    # Global sums
    expected_gs = np.array(gamma.sum(axis=1))
    np.testing.assert_allclose(
        decode.global_sums, expected_gs, atol=1e-4,
        err_msg="global_sums mismatch",
    )


def test_ancestry_sums_matches_decode():
    """forward_backward_ancestry_sums must match decode.global_sums."""
    _, geno, model, d_morgan = _make_model(n_samples=100, n_sites=150)

    gs = forward_backward_ancestry_sums(geno, model, d_morgan, batch_size=50)
    decode = forward_backward_decode(geno, model, d_morgan, batch_size=50)

    np.testing.assert_allclose(gs, decode.global_sums, atol=1e-6)


# -----------------------------------------------------------------------
# *_from_stats M-step variants
# -----------------------------------------------------------------------

def test_allele_freq_from_stats():
    """update_allele_freq_from_stats matches update_allele_freq."""
    _, geno, model, d_morgan = _make_model(n_samples=100, n_sites=100)

    gamma = forward_backward(geno, model, d_morgan)
    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=50)

    freq_full = update_allele_freq(geno, gamma)
    freq_streaming = update_allele_freq_from_stats(em_stats)

    np.testing.assert_allclose(
        np.array(freq_streaming), np.array(freq_full), atol=1e-5,
        err_msg="allele freq mismatch",
    )


def test_mu_from_stats():
    """update_mu_from_stats matches update_mu."""
    _, geno, model, d_morgan = _make_model(n_samples=100, n_sites=100)

    gamma = forward_backward(geno, model, d_morgan)
    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=50)

    mu_full = update_mu(gamma)
    mu_streaming = update_mu_from_stats(em_stats)

    np.testing.assert_allclose(
        np.array(mu_streaming), np.array(mu_full), atol=1e-5,
        err_msg="mu mismatch",
    )


def test_generations_from_stats():
    """update_generations_from_stats produces reasonable T from soft switches."""
    _, geno, model, d_morgan = _make_model(n_samples=100, n_sites=100)

    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=50)

    # Soft switches should be non-negative and finite
    assert np.all(np.isfinite(em_stats.soft_switches_per_hap))
    assert np.all(em_stats.soft_switches_per_hap >= 0)

    T_streaming = update_generations_from_stats(
        em_stats, d_morgan, model.gen_since_admix, model.mu,
    )

    # T should be in a reasonable range
    assert 1.0 <= T_streaming <= 1000.0
    assert np.isfinite(T_streaming)


def test_generations_per_hap_from_stats():
    """update_generations_per_hap_from_stats produces reasonable per-hap T from soft switches."""
    _, geno, model, d_morgan = _make_model(n_samples=100, n_sites=100)

    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=50)

    bucket_centers = compute_bucket_centers(20)

    T_hap_stream, ba_stream, Tg_stream = update_generations_per_hap_from_stats(
        em_stats, d_morgan, model.gen_since_admix, model.mu, bucket_centers,
    )

    T_hap_np = np.array(T_hap_stream)
    assert np.all(np.isfinite(T_hap_np))
    assert np.all(T_hap_np >= 1.0)
    assert np.all(T_hap_np <= 1000.0)

    assert 1.0 <= Tg_stream <= 1000.0

    # Bucket assignments should be valid indices
    ba_np = np.array(ba_stream)
    assert np.all(ba_np >= 0)
    assert np.all(ba_np < len(bucket_centers))


# -----------------------------------------------------------------------
# Full EM: streaming path vs old path
# -----------------------------------------------------------------------

def test_streaming_em_matches_full():
    """Full run_em with streaming M-step must produce same model and calls.

    Since run_em now uses the streaming path internally, we compare to a
    manually-computed ground truth using the old (full-gamma) functions
    for a single EM iteration.
    """
    chrom_data, geno, model, d_morgan = _make_model(
        n_samples=100, n_sites=100, n_ancestries=3,
    )

    # Run one EM iteration with full gamma (ground truth)
    gamma = forward_backward(geno, model, d_morgan)
    freq_expected = update_allele_freq(geno, gamma)
    mu_expected = update_mu(gamma)

    # Run one EM iteration with streaming
    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=50)
    freq_streaming = update_allele_freq_from_stats(em_stats)
    mu_streaming = update_mu_from_stats(em_stats)

    np.testing.assert_allclose(
        np.array(freq_streaming), np.array(freq_expected), atol=1e-5,
        err_msg="allele freq after 1 EM iter",
    )
    np.testing.assert_allclose(
        np.array(mu_streaming), np.array(mu_expected), atol=1e-5,
        err_msg="mu after 1 EM iter",
    )


# -----------------------------------------------------------------------
# Moderate scale: multi-batch exercise
# -----------------------------------------------------------------------

def test_moderate_scale():
    """Test at moderate scale (10K haps) to exercise multi-batch streaming."""
    chrom_data, _true, _ = simulate_admixed(
        n_samples=5000, n_sites=200, n_ancestries=4, rng_seed=99,
    )
    geno = jnp.array(chrom_data.geno)
    d_morgan = jnp.array(chrom_data.genetic_distances)

    _, resp, n_anc, _proj = seed_ancestry_soft(
        chrom_data.geno, n_ancestries=4, rng_seed=99,
    )
    model = init_model_soft(geno, resp, n_anc)

    # Streaming with batch_size=2000 → 5 batches
    em_stats = forward_backward_em(geno, model, d_morgan, batch_size=2000)
    decode = forward_backward_decode(geno, model, d_morgan, batch_size=2000)

    # Basic sanity checks
    assert em_stats.n_haps == 10000
    assert em_stats.n_sites == 200
    assert em_stats.weighted_counts.shape == (4, 200)
    assert em_stats.total_weights.shape == (4, 200)
    assert em_stats.mu_sum.shape == (4,)
    assert em_stats.switch_sum.shape == (199,)
    assert em_stats.switches_per_hap.shape == (10000,)

    assert decode.calls.shape == (10000, 200)
    assert decode.calls.dtype == np.int8
    assert decode.max_post.shape == (10000, 200)
    assert decode.global_sums.shape == (10000, 4)

    # M-step from stats should produce valid results
    freq = update_allele_freq_from_stats(em_stats)
    mu = update_mu_from_stats(em_stats)

    assert freq.shape == (4, 200)
    assert float(freq.min()) > 0
    assert float(freq.max()) < 1
    assert mu.shape == (4,)
    np.testing.assert_allclose(float(mu.sum()), 1.0, atol=1e-6)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
