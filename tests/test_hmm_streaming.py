"""Numerical equivalence tests for streaming forward-backward primitives.

Validates that _streaming_em_checkpointed and _streaming_decode_checkpointed
produce identical results to the gamma-materializing reference paths.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from popout.datatypes import AncestryModel
from popout.hmm import (
    forward_backward,
    forward_backward_checkpointed,
    forward_backward_em,
    forward_backward_decode,
    _streaming_em_checkpointed,
    _streaming_decode_checkpointed,
    _precompute_streaming_tensors,
)


def _call_streaming_em(geno, model, d_morgan, checkpoint_interval=None):
    """Test helper: wraps the JIT'd _streaming_em_checkpointed with the old interface."""
    T = geno.shape[1]
    pc = _precompute_streaming_tensors(model, d_morgan, T, checkpoint_interval)
    emit_pad = pc["emit_pad"]
    geno_j = geno if isinstance(geno, jnp.ndarray) else jnp.array(geno)
    if emit_pad > 0:
        geno_j = jnp.concatenate(
            [geno_j, jnp.zeros((geno_j.shape[0], emit_pad), dtype=geno_j.dtype)],
            axis=1,
        )
    return _streaming_em_checkpointed(
        geno_j, pc["log_f0"], pc["log_odds"], pc["seg_trans"],
        pc["site_idx"], pc["gamma_site_idx"], pc["log_prior"],
        C=pc["C"], S=pc["S"], n_fwd_steps=pc["n_fwd_steps"],
        emit_pad=emit_pad,
    )


def _call_streaming_decode(geno, model, d_morgan, checkpoint_interval=None,
                           compute_max_post=True):
    """Test helper: wraps the new _streaming_decode_checkpointed interface."""
    T = geno.shape[1]
    pc = _precompute_streaming_tensors(model, d_morgan, T, checkpoint_interval)
    emit_pad = pc["emit_pad"]
    geno_j = geno if isinstance(geno, jnp.ndarray) else jnp.array(geno)
    if emit_pad > 0:
        geno_j = jnp.concatenate(
            [geno_j, jnp.zeros((geno_j.shape[0], emit_pad), dtype=geno_j.dtype)],
            axis=1,
        )
    return _streaming_decode_checkpointed(
        geno_j, pc["log_f0"], pc["log_odds"], pc["seg_trans"],
        pc["site_idx"], pc["log_prior"],
        C=pc["C"], S=pc["S"], n_fwd_steps=pc["n_fwd_steps"],
        emit_pad=emit_pad,
        compute_max_post=compute_max_post,
    )


def _tiny_setup(H=64, T=500, A=4, seed=0):
    """Build a small synthetic model for testing."""
    rng = np.random.default_rng(seed)
    geno = jnp.array(rng.integers(0, 2, size=(H, T), dtype=np.uint8))
    freq = rng.uniform(0.05, 0.95, size=(A, T)).astype(np.float32)
    mu = rng.dirichlet(np.ones(A)).astype(np.float32)
    d_morgan = np.diff(np.sort(rng.uniform(0, 1, size=T))).astype(np.float64)
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.array(mu),
        gen_since_admix=10.0,
        allele_freq=jnp.array(freq),
    )
    return geno, model, jnp.array(d_morgan)


# -- EM streaming tests --

def test_streaming_em_weighted_counts():
    """Streaming EM weighted_counts match reference gamma-based computation."""
    geno, model, d_morgan = _tiny_setup()
    gamma, soft_sw_ref = forward_backward_checkpointed(
        geno, model, d_morgan, compute_transitions=True,
    )
    geno_f = geno.astype(jnp.float32)
    wc_ref = jnp.einsum('hta,ht->at', gamma, geno_f)

    wc, tw, mu_sum, soft_sw = _call_streaming_em(geno, model, d_morgan)
    np.testing.assert_allclose(np.array(wc), np.array(wc_ref), atol=1e-4, rtol=1e-4)


def test_streaming_em_total_weights():
    """Streaming EM total_weights match reference."""
    geno, model, d_morgan = _tiny_setup()
    gamma = forward_backward_checkpointed(geno, model, d_morgan)
    tw_ref = gamma.sum(axis=0).T  # (T, A).T → (A, T)

    wc, tw, mu_sum, soft_sw = _call_streaming_em(geno, model, d_morgan)
    np.testing.assert_allclose(np.array(tw), np.array(tw_ref), atol=1e-4, rtol=1e-4)


def test_streaming_em_mu_sum():
    """Streaming EM mu_sum matches reference."""
    geno, model, d_morgan = _tiny_setup()
    gamma = forward_backward_checkpointed(geno, model, d_morgan)
    mu_ref = gamma.sum(axis=(0, 1))  # (A,)

    wc, tw, mu_sum, soft_sw = _call_streaming_em(geno, model, d_morgan)
    np.testing.assert_allclose(np.array(mu_sum), np.array(mu_ref), atol=1e-4, rtol=1e-4)


def test_streaming_em_soft_switches():
    """Streaming EM soft_switches match reference checkpointed path."""
    geno, model, d_morgan = _tiny_setup()
    _, soft_sw_ref = forward_backward_checkpointed(
        geno, model, d_morgan, compute_transitions=True,
    )

    wc, tw, mu_sum, soft_sw = _call_streaming_em(geno, model, d_morgan)
    np.testing.assert_allclose(
        np.array(soft_sw), np.array(soft_sw_ref), atol=1e-4, rtol=1e-4,
    )


# -- Decode streaming tests --

def test_streaming_decode_calls():
    """Streaming decode calls match reference argmax."""
    geno, model, d_morgan = _tiny_setup(H=32, T=300, A=3)
    gamma = forward_backward_checkpointed(geno, model, d_morgan)
    calls_ref = np.array(jnp.argmax(gamma, axis=2)).astype(np.int8)

    result = _call_streaming_decode(geno, model, d_morgan)
    np.testing.assert_array_equal(result.calls, calls_ref)


def test_streaming_decode_max_post():
    """Streaming decode max_post matches reference."""
    geno, model, d_morgan = _tiny_setup(H=32, T=300, A=3)
    gamma = forward_backward_checkpointed(geno, model, d_morgan)
    max_post_ref = np.array(gamma.max(axis=2))

    result = _call_streaming_decode(geno, model, d_morgan)
    np.testing.assert_allclose(result.max_post, max_post_ref, atol=1e-5)


def test_streaming_decode_global_sums():
    """Streaming decode global_sums matches reference."""
    geno, model, d_morgan = _tiny_setup(H=32, T=300, A=3)
    gamma = forward_backward_checkpointed(geno, model, d_morgan)
    global_sums_ref = np.array(gamma.sum(axis=1))

    result = _call_streaming_decode(geno, model, d_morgan)
    np.testing.assert_allclose(result.global_sums, global_sums_ref, atol=1e-4)


# -- Parametric tests --

@pytest.mark.parametrize("T,A", [
    (10, 2), (50, 3), (100, 2), (500, 4), (1024, 8), (2000, 6),
])
def test_streaming_em_parametric(T, A):
    """Streaming EM matches reference across various T and A values.

    Includes non-perfect-square T to stress the padding logic.
    """
    geno, model, d_morgan = _tiny_setup(H=32, T=T, A=A)
    gamma, soft_sw_ref = forward_backward_checkpointed(
        geno, model, d_morgan, compute_transitions=True,
    )
    geno_f = geno.astype(jnp.float32)
    wc_ref = jnp.einsum('hta,ht->at', gamma, geno_f)
    tw_ref = gamma.sum(axis=0).T
    mu_ref = gamma.sum(axis=(0, 1))

    wc, tw, mu_sum, soft_sw = _call_streaming_em(geno, model, d_morgan)

    np.testing.assert_allclose(np.array(wc), np.array(wc_ref), atol=1e-4, rtol=1e-4,
                               err_msg=f"wc mismatch at T={T}, A={A}")
    np.testing.assert_allclose(np.array(tw), np.array(tw_ref), atol=1e-4, rtol=1e-4,
                               err_msg=f"tw mismatch at T={T}, A={A}")
    np.testing.assert_allclose(np.array(mu_sum), np.array(mu_ref), atol=1e-4, rtol=1e-4,
                               err_msg=f"mu_sum mismatch at T={T}, A={A}")
    np.testing.assert_allclose(np.array(soft_sw), np.array(soft_sw_ref), atol=1e-4, rtol=1e-4,
                               err_msg=f"soft_sw mismatch at T={T}, A={A}")


@pytest.mark.parametrize("T,A", [
    (10, 2), (50, 3), (100, 4), (300, 3),
])
def test_streaming_decode_parametric(T, A):
    """Streaming decode matches reference across various T and A."""
    geno, model, d_morgan = _tiny_setup(H=32, T=T, A=A)
    gamma = forward_backward_checkpointed(geno, model, d_morgan)
    calls_ref = np.array(jnp.argmax(gamma, axis=2)).astype(np.int8)
    max_post_ref = np.array(gamma.max(axis=2))
    global_sums_ref = np.array(gamma.sum(axis=1))

    result = _call_streaming_decode(geno, model, d_morgan)

    np.testing.assert_array_equal(result.calls, calls_ref,
                                  err_msg=f"calls mismatch at T={T}, A={A}")
    np.testing.assert_allclose(result.max_post, max_post_ref, atol=1e-5,
                               err_msg=f"max_post mismatch at T={T}, A={A}")
    np.testing.assert_allclose(result.global_sums, global_sums_ref, atol=1e-4,
                               err_msg=f"global_sums mismatch at T={T}, A={A}")


# -- Edge cases --

def test_streaming_em_small_T():
    """Streaming EM works correctly for very small T values."""
    for T in [2, 3, 4, 5]:
        geno, model, d_morgan = _tiny_setup(H=16, T=T, A=2, seed=T)
        gamma, soft_sw_ref = forward_backward_checkpointed(
            geno, model, d_morgan, compute_transitions=True,
        )
        geno_f = geno.astype(jnp.float32)
        wc_ref = jnp.einsum('hta,ht->at', gamma, geno_f)
        tw_ref = gamma.sum(axis=0).T
        mu_ref = gamma.sum(axis=(0, 1))

        wc, tw, mu_sum, soft_sw = _call_streaming_em(geno, model, d_morgan)

        np.testing.assert_allclose(np.array(wc), np.array(wc_ref), atol=1e-4,
                                   err_msg=f"wc mismatch at T={T}")
        np.testing.assert_allclose(np.array(tw), np.array(tw_ref), atol=1e-4,
                                   err_msg=f"tw mismatch at T={T}")
        np.testing.assert_allclose(np.array(mu_sum), np.array(mu_ref), atol=1e-4,
                                   err_msg=f"mu_sum mismatch at T={T}")
        np.testing.assert_allclose(np.array(soft_sw), np.array(soft_sw_ref), atol=1e-4,
                                   err_msg=f"soft_sw mismatch at T={T}")


def test_streaming_em_explicit_checkpoint_intervals():
    """Streaming EM produces correct results with various checkpoint intervals."""
    geno, model, d_morgan = _tiny_setup(H=32, T=100, A=3)

    for C in [2, 3, 5, 10, 50, 99, 100]:
        # Reference with the SAME checkpoint interval (floating-point
        # arithmetic order depends on C)
        gamma, soft_sw_ref = forward_backward_checkpointed(
            geno, model, d_morgan, checkpoint_interval=C,
            compute_transitions=True,
        )
        geno_f = geno.astype(jnp.float32)
        wc_ref = jnp.einsum('hta,ht->at', gamma, geno_f)

        wc, tw, mu_sum, soft_sw = _call_streaming_em(
            geno, model, d_morgan, checkpoint_interval=C,
        )
        np.testing.assert_allclose(
            np.array(wc), np.array(wc_ref), atol=1e-4, rtol=1e-4,
            err_msg=f"wc mismatch at checkpoint_interval={C}",
        )
        np.testing.assert_allclose(
            np.array(soft_sw), np.array(soft_sw_ref), atol=1e-4, rtol=1e-4,
            err_msg=f"soft_sw mismatch at checkpoint_interval={C}",
        )


# -- Host-resident (numpy) geno tests --

def test_em_accepts_host_geno():
    """Streaming EM path must work with numpy-array geno."""
    geno, model, d_morgan = _tiny_setup(H=128, T=500, A=3)
    geno_np = np.asarray(geno)  # host numpy

    stats_host = forward_backward_em(geno_np, model, d_morgan, batch_size=32)
    stats_dev = forward_backward_em(geno, model, d_morgan, batch_size=32)

    np.testing.assert_allclose(stats_host.weighted_counts,
                               stats_dev.weighted_counts, atol=1e-4)
    np.testing.assert_allclose(stats_host.mu_sum,
                               stats_dev.mu_sum, atol=1e-4)


def test_decode_accepts_host_geno():
    """Streaming decode path must work with numpy-array geno."""
    geno, model, d_morgan = _tiny_setup(H=64, T=300, A=3)
    geno_np = np.asarray(geno)

    result_host = forward_backward_decode(geno_np, model, d_morgan, batch_size=32)
    result_dev = forward_backward_decode(geno, model, d_morgan, batch_size=32)

    np.testing.assert_array_equal(result_host.calls, result_dev.calls)
    np.testing.assert_allclose(result_host.global_sums,
                               result_dev.global_sums, atol=1e-4)


def test_init_model_soft_accepts_host_geno():
    """init_model_soft must work with numpy-array geno."""
    from popout.em import init_model_soft
    rng = np.random.default_rng(42)
    H, T, A = 200, 500, 3
    geno_np = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    geno_jx = jnp.array(geno_np)
    resp = jnp.array(rng.dirichlet(np.ones(A), size=H).astype(np.float32))

    model_host = init_model_soft(geno_np, resp, A, window_refine=False)
    model_dev = init_model_soft(geno_jx, resp, A, window_refine=False)

    np.testing.assert_allclose(model_host.allele_freq,
                               model_dev.allele_freq, atol=1e-4)


def test_run_k2_em_split_respects_budget(monkeypatch):
    """K=2 EM must not upload full geno when budget is tight."""
    from popout import _device
    from popout.recursive_seed import _run_k2_em_split
    from popout.datatypes import ChromData
    monkeypatch.setattr(_device, "_HEADROOM_BYTES", 10**18)  # force host path

    rng = np.random.default_rng(7)
    H, T = 500, 200
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    pos_bp = np.arange(T, dtype=np.int64) * 1000
    pos_cm = pos_bp / 1e6

    chrom_data = ChromData(geno=geno, pos_bp=pos_bp, pos_cm=pos_cm, chrom="chr1")
    resp = jnp.array(rng.dirichlet(np.ones(2), size=H).astype(np.float32))

    labels = _run_k2_em_split(
        geno, np.arange(H), chrom_data,
        n_iter=1, seed_resp=resp,
    )
    assert labels.shape == (H,)
    assert set(np.unique(labels)).issubset({0, 1})
