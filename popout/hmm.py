"""Forward-backward HMM on GPU via JAX.

All haplotypes run the A-state HMM simultaneously — one site at a time,
sequential across T sites, parallel across all H haplotypes.

The forward state is (H, A) — typically 1M × 8 = 32 MB.  This fits in
L2 cache on an A100, making the algorithm memory-bandwidth bound at
~2-3 seconds per chromosome for 1M haplotypes.

Checkpointing reduces memory from O(H·T·A) to O(H·√T·A) at the cost
of recomputing forward values from the nearest checkpoint during the
backward pass (~2× forward compute).
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

import numpy as _np

from .datatypes import AncestryModel, DecodeResult, EMStats


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    checkpoint_interval: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the forward algorithm for all haplotypes simultaneously.

    Stores full (H, T, A) alpha array.  For memory-constrained settings,
    use ``forward_backward_checkpointed`` instead.

    Parameters
    ----------
    geno : array (H, T), uint8 — binary allele matrix
    model : AncestryModel with allele_freq (A, T), mu (A,), gen_since_admix
    d_morgan : array (T-1,) — inter-site genetic distances

    Returns
    -------
    log_alpha : array (H, T, A) — log forward probabilities
    checkpoints : array (n_ckpt, H, A) — saved states for backward pass
    """
    H, T = geno.shape
    A = model.n_ancestries

    if checkpoint_interval is None:
        checkpoint_interval = max(1, int(math.isqrt(T)))

    log_emit = model.log_emission(geno)
    log_trans = model.log_transition_matrix(d_morgan)

    log_prior = jnp.log(model.mu)  # (A,)
    log_alpha_0 = log_prior[None, :] + log_emit[:, 0, :]  # (H, A)

    checkpoint_indices = jnp.arange(0, T, checkpoint_interval)

    log_alphas = _forward_scan(log_alpha_0, log_emit, log_trans)

    checkpoints = log_alphas[:, checkpoint_indices, :]

    return log_alphas, checkpoints


def _forward_scan(
    log_alpha_0: jnp.ndarray,
    log_emit: jnp.ndarray,
    log_trans: jnp.ndarray,
) -> jnp.ndarray:
    """Sequential forward scan across T sites, storing all states.

    Parameters
    ----------
    log_alpha_0 : (H, A) — initial log forward state
    log_emit : (H, T, A) — log emissions
    log_trans : (T-1, A, A) — log transition matrices

    Returns
    -------
    log_alphas : (H, T, A) — all forward states
    """
    H, T, A = log_emit.shape

    def step(log_alpha_prev, t):
        lt = log_trans[t - 1]
        log_alpha_pred = _log_matvec_batch(log_alpha_prev, lt)
        log_alpha_new = log_alpha_pred + log_emit[:, t, :]
        return log_alpha_new, log_alpha_new

    _, log_alphas_rest = jax.lax.scan(step, log_alpha_0, jnp.arange(1, T))

    log_alphas = jnp.concatenate(
        [log_alpha_0[None, :, :], log_alphas_rest],
        axis=0,
    )
    return jnp.transpose(log_alphas, (1, 0, 2))  # (H, T, A)


def _log_matvec_batch(log_v: jnp.ndarray, log_M: jnp.ndarray) -> jnp.ndarray:
    """Batched log-space matrix-vector product.

    log_v : (H, A) — log probability vectors
    log_M : (A, A) — log transition matrix, log_M[i, j] = log P(j | i)

    Returns (H, A) where result[h, j] = logsumexp_i(log_v[h, i] + log_M[i, j])
    """
    combined = log_v[:, :, None] + log_M[None, :, :]  # (H, A, A)
    return jax.nn.logsumexp(combined, axis=1)  # (H, A)


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def backward(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
) -> jnp.ndarray:
    """Run the backward algorithm for all haplotypes simultaneously.

    Parameters
    ----------
    geno : array (H, T)
    model : AncestryModel
    d_morgan : array (T-1,)

    Returns
    -------
    log_beta : array (H, T, A) — log backward probabilities
    """
    H, T = geno.shape
    A = model.n_ancestries

    log_emit = model.log_emission(geno)
    log_trans = model.log_transition_matrix(d_morgan)

    log_beta_T = jnp.zeros((H, A))

    def step(log_beta_next, t):
        lt = log_trans[t]
        log_integrand = log_emit[:, t + 1, :] + log_beta_next
        log_beta_new = _log_matvec_batch_transpose(log_integrand, lt)
        return log_beta_new, log_beta_new

    _, log_betas_rev = jax.lax.scan(
        step, log_beta_T, jnp.arange(T - 2, -1, -1)
    )

    log_betas = jnp.concatenate(
        [jnp.flip(log_betas_rev, axis=0), log_beta_T[None, :, :]],
        axis=0,
    )
    return jnp.transpose(log_betas, (1, 0, 2))  # (H, T, A)


def _log_matvec_batch_transpose(
    log_v: jnp.ndarray, log_M: jnp.ndarray
) -> jnp.ndarray:
    """Batched log-space product with transpose of M.

    result[h, i] = logsumexp_j(log_M[i, j] + log_v[h, j])
    """
    combined = log_v[:, None, :] + log_M[None, :, :]  # (H, A_from, A_to)
    return jax.nn.logsumexp(combined, axis=2)  # (H, A)


# ---------------------------------------------------------------------------
# Posteriors
# ---------------------------------------------------------------------------

def posteriors(
    log_alpha: jnp.ndarray,
    log_beta: jnp.ndarray,
) -> jnp.ndarray:
    """Compute posterior ancestry probabilities from forward-backward.

    Parameters
    ----------
    log_alpha : (H, T, A)
    log_beta : (H, T, A)

    Returns
    -------
    gamma : (H, T, A) — P(ancestry = a | data) at each site
    """
    log_gamma = log_alpha + log_beta
    log_norm = jax.nn.logsumexp(log_gamma, axis=2, keepdims=True)
    gamma = jnp.exp(log_gamma - log_norm)
    return gamma


# ---------------------------------------------------------------------------
# Checkpointed forward-backward (memory-efficient)
# ---------------------------------------------------------------------------

def forward_backward_checkpointed(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    checkpoint_interval: int | None = None,
    compute_transitions: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Memory-efficient forward-backward using gradient checkpointing.

    Instead of storing the full (H, T, A) forward state, stores only
    √T checkpoint alphas and recomputes segment-level forward states
    from the nearest checkpoint during the backward pass.

    Memory: O(H · √T · A)  instead of O(H · T · A).
    Compute: ~2× forward pass (one full + one segment recompute).

    Results are numerically equivalent to ``forward_backward()``.

    Parameters
    ----------
    geno : (H, T) uint8
    model : AncestryModel
    d_morgan : (T-1,)
    checkpoint_interval : sites between checkpoints (default: √T)
    compute_transitions : if True, also return xi-based soft switches (H,)

    Returns
    -------
    gamma : (H, T, A) posterior probabilities
        — or (gamma, soft_switches) when compute_transitions=True.
    """
    H, T = geno.shape
    A = model.n_ancestries

    if checkpoint_interval is None:
        checkpoint_interval = max(1, int(math.isqrt(T)))
    C = checkpoint_interval

    # Precompute emissions and transitions
    log_emit = model.log_emission(geno)               # (H, T, A)
    log_trans = model.log_transition_matrix(d_morgan)  # (T-1, A, A)
    log_prior = jnp.log(model.mu)                      # (A,)
    log_alpha_0 = log_prior[None, :] + log_emit[:, 0, :]  # (H, A)

    # Pad so that T_pad is a multiple of C.
    # Each posterior group covers exactly C sites; padding is sliced off.
    S = (T + C - 1) // C          # number of posterior groups
    T_pad = S * C                  # padded site count

    # The forward checkpoint scan needs S segments of C steps each = S*C steps.
    # Forward step i uses transition[i] and emission at site i+1.
    # So we need emissions at sites 1..S*C and transitions at intervals 0..S*C-1.
    # Pad emissions to S*C + 1 sites (0..S*C) and transitions to S*C.
    n_fwd_steps = S * C
    emit_pad = n_fwd_steps + 1 - T  # extra emission sites needed
    trans_pad = n_fwd_steps - (T - 1)
    log_eye = jnp.where(jnp.eye(A, dtype=bool), 0.0, -1e30)

    if emit_pad > 0:
        log_emit = jnp.concatenate(
            [log_emit, jnp.zeros((H, emit_pad, A))], axis=1
        )  # (H, S*C + 1, A)

    if trans_pad > 0:
        log_trans_fwd = jnp.concatenate(
            [log_trans, jnp.broadcast_to(log_eye, (trans_pad, A, A))],
            axis=0,
        )  # (n_fwd_steps, A, A)
    else:
        log_trans_fwd = log_trans[:n_fwd_steps]

    # Build step-indexed segment arrays.
    # Forward step i (0-indexed): transition[i], emission at site i+1.
    step_emit = log_emit[:, 1:n_fwd_steps + 1, :].transpose(1, 0, 2)  # (n_fwd_steps, H, A)
    del log_emit  # free ~10 GB; step_emit has all we need
    step_trans = log_trans_fwd                                          # (n_fwd_steps, A, A)

    seg_emit = step_emit.reshape(S, C, H, A)    # (S, C, H, A)
    seg_trans = step_trans.reshape(S, C, A, A)   # (S, C, A, A)

    # ------------------------------------------------------------------
    # Phase 1: Checkpointed forward — only store segment-end alphas
    # ------------------------------------------------------------------

    def _ckpt_fwd_body(alpha, xs):
        se, st = xs  # (C, H, A), (C, A, A)
        def _inner(a, x):
            e, t = x
            return _log_matvec_batch(a, t) + e, None
        final, _ = jax.lax.scan(_inner, alpha, (se, st))
        return final, final  # carry, output

    _, ckpt_ends = jax.lax.scan(
        _ckpt_fwd_body, log_alpha_0, (seg_emit, seg_trans)
    )
    # ckpt_ends: (S, H, A) — alpha at sites C, 2C, …, S*C
    checkpoints = jnp.concatenate([log_alpha_0[None], ckpt_ends], axis=0)
    # checkpoints: (S+1, H, A) at sites 0, C, 2C, …, S*C
    # Posterior groups use checkpoints[0..S-1].

    # ------------------------------------------------------------------
    # Phase 2: Reverse scan — recompute forward + backward + posteriors
    # ------------------------------------------------------------------
    # Process groups S-1 → 0 via a forward scan over reversed data.
    #
    # Group g covers sites [g*C, (g+1)*C).
    #   Forward recompute: C-1 steps from checkpoints[g].
    #   Backward: C steps from beta_right using reversed segment data.
    #     seg_emit[g] contains emissions at sites g*C+1..(g+1)*C
    #     seg_trans[g] contains transitions at intervals g*C..(g+1)*C-1
    #     The backward step at site t uses trans[t] and emit[t+1].
    #     Reversed segment data gives the correct pairing.

    seg_emit_rev = jnp.flip(seg_emit, axis=0)         # (S, C, H, A)
    del step_emit, seg_emit  # free ~10 GB; reversed copy is all we need
    seg_trans_rev = jnp.flip(seg_trans, axis=0)        # (S, C, A, A)
    ckpt_rev = jnp.flip(checkpoints[:S], axis=0)       # (S, H, A)

    # Shared forward-recompute + backward logic for each segment
    def _segment_fwd_bwd(ckpt, se, st, beta_right):
        """Recompute forward + backward within a segment, return alphas, betas, gamma."""
        # Forward recompute from checkpoint: C-1 steps
        def _fwd(a, x):
            e, t = x
            a_new = _log_matvec_batch(a, t) + e
            return a_new, a_new

        _, alphas_inner = jax.lax.scan(_fwd, ckpt, (se[:C - 1], st[:C - 1]))
        alphas = jnp.concatenate([ckpt[None], alphas_inner], axis=0)  # (C, H, A)

        # Backward: C steps from beta_right
        se_bwd = jnp.flip(se, axis=0)
        st_bwd = jnp.flip(st, axis=0)

        def _bwd(beta, x):
            e, t = x
            b_new = _log_matvec_batch_transpose(e + beta, t)
            return b_new, b_new

        beta_left, betas_rev = jax.lax.scan(_bwd, beta_right, (se_bwd, st_bwd))
        betas = jnp.flip(betas_rev, axis=0)  # (C, H, A) forward site order

        # Posteriors
        log_gamma = alphas + betas
        log_norm = jax.nn.logsumexp(log_gamma, axis=2, keepdims=True)
        gamma_seg = jnp.exp(log_gamma - log_norm)  # (C, H, A)

        return alphas, betas, beta_left, gamma_seg

    def _bwd_group_body(beta_right, xs):
        se, st, ckpt = xs  # (C, H, A), (C, A, A), (H, A)
        _, _, beta_left, gamma_seg = _segment_fwd_bwd(ckpt, se, st, beta_right)
        return beta_left, gamma_seg

    def _bwd_group_body_with_xi(carry, xs):
        beta_right, soft_sw = carry
        se, st, ckpt = xs  # (C, H, A), (C, A, A), (H, A)

        alphas, betas, beta_left, gamma_seg = _segment_fwd_bwd(
            ckpt, se, st, beta_right,
        )

        # --- Xi-based soft switches for this segment ---
        # betas_ext[k] = backward at site (g*C + k + 1):
        #   k < C-1: betas[k+1]  (intra-segment)
        #   k = C-1: beta_right  (boundary to next segment)
        betas_ext = jnp.concatenate(
            [betas[1:], beta_right[None]], axis=0,
        )  # (C, H, A)

        # P(data) — constant across t, from first position
        log_P = jax.nn.logsumexp(alphas[0] + betas[0], axis=1)  # (H,)

        # Diagonal of transition matrices for this segment
        st_diag = jnp.diagonal(st, axis1=1, axis2=2)  # (C, A)

        # ξ_diag[k,h,a] = α[k,h,a] · trans_diag[k,a] · emit[k,h,a] · β_ext[k,h,a] / P
        log_xi_diag = (
            alphas + st_diag[:, None, :] + se + betas_ext
        )  # (C, H, A)
        xi_diag = jnp.exp(log_xi_diag - log_P[None, :, None])  # (C, H, A)
        stay = xi_diag.sum(axis=2)                                # (C, H)
        seg_soft = jnp.clip(1.0 - stay, 0.0, 1.0).sum(axis=0)   # (H,)

        return (beta_left, soft_sw + seg_soft), gamma_seg

    if compute_transitions:
        (_, soft_switches), gammas_rev = jax.lax.scan(
            _bwd_group_body_with_xi,
            (jnp.zeros((H, A)), jnp.zeros(H)),
            (seg_emit_rev, seg_trans_rev, ckpt_rev),
        )
    else:
        _, gammas_rev = jax.lax.scan(
            _bwd_group_body, jnp.zeros((H, A)),
            (seg_emit_rev, seg_trans_rev, ckpt_rev),
        )
        soft_switches = None

    # gammas_rev: (S, C, H, A) in reverse group order
    del seg_emit_rev, seg_trans_rev, ckpt_rev, checkpoints  # free scan inputs

    gammas = jnp.flip(gammas_rev, axis=0)            # (S, C, H, A)
    del gammas_rev  # free the unflipped copy
    gamma_flat = gammas.reshape(S * C, H, A)[:T]     # (T, H, A) — trim padding
    del gammas  # gamma_flat may share buffer, but del allows GC if not
    gamma = jnp.transpose(gamma_flat, (1, 0, 2))     # (H, T, A)

    if compute_transitions:
        return gamma, soft_switches
    return gamma


# ---------------------------------------------------------------------------
# Streaming forward-backward (no full-tensor emission or gamma)
# ---------------------------------------------------------------------------

def _precompute_streaming_tensors(
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    T: int,
    checkpoint_interval: int | None = None,
) -> dict:
    """Precompute model-derived tensors shared across all batches.

    Avoids redundant recomputation of log-emission tables, transition
    matrices, and index arrays on every batch call.
    """
    A = model.n_ancestries
    if checkpoint_interval is None:
        checkpoint_interval = max(1, int(math.isqrt(T)))
    C = checkpoint_interval
    S = (T + C - 1) // C
    n_fwd_steps = S * C

    log_trans = model.log_transition_matrix(d_morgan)
    log_prior = jnp.log(model.mu)

    freq = jnp.clip(model.allele_freq, 1e-6, 1.0 - 1e-6)
    log_f0 = jnp.log(1.0 - freq)
    log_odds = jnp.log(freq) - log_f0

    emit_pad = n_fwd_steps + 1 - T
    if emit_pad > 0:
        log_f0 = jnp.concatenate(
            [log_f0, jnp.zeros((A, emit_pad))], axis=1,
        )
        log_odds = jnp.concatenate(
            [log_odds, jnp.zeros((A, emit_pad))], axis=1,
        )

    trans_pad = n_fwd_steps - (T - 1)
    log_eye = jnp.where(jnp.eye(A, dtype=bool), 0.0, -1e30)
    if trans_pad > 0:
        log_trans_fwd = jnp.concatenate(
            [log_trans, jnp.broadcast_to(log_eye, (trans_pad, A, A))],
            axis=0,
        )
    else:
        log_trans_fwd = log_trans[:n_fwd_steps]
    seg_trans = log_trans_fwd.reshape(S, C, A, A)

    site_idx = jnp.arange(1, n_fwd_steps + 1).reshape(S, C)
    gamma_site_idx = jnp.arange(0, n_fwd_steps).reshape(S, C)

    return {
        "log_f0": log_f0,
        "log_odds": log_odds,
        "seg_trans": seg_trans,
        "site_idx": site_idx,
        "gamma_site_idx": gamma_site_idx,
        "log_prior": log_prior,
        "C": C,
        "S": S,
        "n_fwd_steps": n_fwd_steps,
        "emit_pad": emit_pad,
    }


@partial(jax.jit, static_argnames=("C", "S", "n_fwd_steps", "emit_pad"))
def _streaming_em_checkpointed(
    geno_j, log_f0, log_odds, seg_trans, site_idx, gamma_site_idx, log_prior,
    *, C, S, n_fwd_steps, emit_pad,
):
    """JIT-compiled checkpointed forward-backward returning sufficient statistics.

    Never materializes (B, T, A). Emissions are computed on the fly from
    (geno, allele_freq) slices; posteriors are reduced to M-step accumulators
    per segment inside the scan carry.

    All model-derived tensors are precomputed by _precompute_streaming_tensors
    and passed in directly. Shape-sensitive Python ints are static.

    Parameters
    ----------
    geno_j : (B, T_padded) uint8 — already padded to n_fwd_steps + 1 sites
    log_f0 : (A, T_padded) float32
    log_odds : (A, T_padded) float32
    seg_trans : (S, C, A, A) float32
    site_idx : (S, C) int32
    gamma_site_idx : (S, C) int32
    log_prior : (A,) float32
    C, S, n_fwd_steps, emit_pad : static ints

    Returns
    -------
    weighted_counts : (A, T) float32
    total_weights   : (A, T) float32
    mu_sum          : (A,) float32
    soft_switches   : (B,) float32
    """
    B = geno_j.shape[0]
    T = geno_j.shape[1] - emit_pad
    A = log_prior.shape[0]

    # Helper: compute emissions at a batch of sites → (C, B, A)
    def _emit_batch(sites):
        g = geno_j[:, sites].astype(jnp.float32)  # (B, C)
        f0 = log_f0[:, sites]                       # (A, C)
        lo = log_odds[:, sites]                     # (A, C)
        return f0.T[:, None, :] + g.T[:, :, None] * lo.T[:, None, :]

    # Initial alpha at site 0
    e0 = log_f0[:, 0][None, :] + geno_j[:, 0].astype(jnp.float32)[:, None] * log_odds[:, 0][None, :]
    log_alpha_0 = log_prior[None, :] + e0  # (B, A)

    # Phase 1: Checkpointed forward
    def _ckpt_fwd_body(alpha, xs):
        st, idx = xs
        se = _emit_batch(idx)
        def _inner(a, x):
            e, t = x
            return _log_matvec_batch(a, t) + e, None
        final, _ = jax.lax.scan(_inner, alpha, (se, st))
        return final, final

    _, ckpt_ends = jax.lax.scan(
        _ckpt_fwd_body, log_alpha_0, (seg_trans, site_idx),
    )
    checkpoints = jnp.concatenate([log_alpha_0[None], ckpt_ends], axis=0)

    # Phase 2: Reverse backward scan + streaming M-step accumulation
    seg_trans_rev = jnp.flip(seg_trans, axis=0)
    ckpt_rev = jnp.flip(checkpoints[:S], axis=0)
    site_idx_rev = jnp.flip(site_idx, axis=0)
    gamma_idx_rev = jnp.flip(gamma_site_idx, axis=0)

    def _bwd_segment(carry, xs):
        beta_right, soft_sw, wc, tw = carry
        st, ckpt, e_sites, g_sites = xs

        se = _emit_batch(e_sites)

        def _fwd(a, x):
            e, t = x
            a_new = _log_matvec_batch(a, t) + e
            return a_new, a_new
        _, alphas_inner = jax.lax.scan(_fwd, ckpt, (se[:C - 1], st[:C - 1]))
        alphas = jnp.concatenate([ckpt[None], alphas_inner], axis=0)

        se_bwd = jnp.flip(se, axis=0)
        st_bwd = jnp.flip(st, axis=0)
        def _bwd(beta, x):
            e, t = x
            b_new = _log_matvec_batch_transpose(e + beta, t)
            return b_new, b_new
        beta_left, betas_rev = jax.lax.scan(_bwd, beta_right, (se_bwd, st_bwd))
        betas = jnp.flip(betas_rev, axis=0)

        log_gamma = alphas + betas
        log_norm = jax.nn.logsumexp(log_gamma, axis=2, keepdims=True)
        gamma_seg = jnp.exp(log_gamma - log_norm)

        betas_ext = jnp.concatenate(
            [betas[1:], beta_right[None]], axis=0,
        )
        log_P = jax.nn.logsumexp(alphas[0] + betas[0], axis=1)
        st_diag = jnp.diagonal(st, axis1=1, axis2=2)
        log_xi_diag = (
            alphas + st_diag[:, None, :] + se + betas_ext
        )
        xi_diag = jnp.exp(log_xi_diag - log_P[None, :, None])
        stay = xi_diag.sum(axis=2)
        seg_soft = jnp.clip(1.0 - stay, 0.0, 1.0).sum(axis=0)

        geno_at_gamma = geno_j[:, g_sites].astype(jnp.float32)
        wc_seg = jnp.einsum('cba,bc->ac', gamma_seg, geno_at_gamma)
        tw_seg = gamma_seg.sum(axis=1).T

        wc = wc.at[:, g_sites].add(wc_seg)
        tw = tw.at[:, g_sites].add(tw_seg)

        return (beta_left, soft_sw + seg_soft, wc, tw), None

    init_carry = (
        jnp.zeros((B, A)),
        jnp.zeros(B),
        jnp.zeros((A, n_fwd_steps), jnp.float32),
        jnp.zeros((A, n_fwd_steps), jnp.float32),
    )
    (_, soft_switches, wc_pad, tw_pad), _ = jax.lax.scan(
        _bwd_segment, init_carry,
        (seg_trans_rev, ckpt_rev, site_idx_rev, gamma_idx_rev),
    )

    weighted_counts = wc_pad[:, :T]
    total_weights = tw_pad[:, :T]
    mu_sum = total_weights.sum(axis=1)

    return weighted_counts, total_weights, mu_sum, soft_switches


@partial(jax.jit, static_argnames=("C", "S", "n_fwd_steps", "emit_pad"))
def _streaming_decode_fwd_checkpoints(
    geno_j, log_f0, log_odds, seg_trans, site_idx, log_prior,
    *, C, S, n_fwd_steps, emit_pad,
):
    """JIT-compiled forward checkpoint pass for decode (shared with EM)."""
    e0 = log_f0[:, 0][None, :] + geno_j[:, 0].astype(jnp.float32)[:, None] * log_odds[:, 0][None, :]
    log_alpha_0 = log_prior[None, :] + e0

    def _emit_batch(sites):
        g = geno_j[:, sites].astype(jnp.float32)
        f0 = log_f0[:, sites]
        lo = log_odds[:, sites]
        return f0.T[:, None, :] + g.T[:, :, None] * lo.T[:, None, :]

    def _ckpt_fwd_body(alpha, xs):
        st, idx = xs
        se = _emit_batch(idx)
        def _inner(a, x):
            e, t = x
            return _log_matvec_batch(a, t) + e, None
        final, _ = jax.lax.scan(_inner, alpha, (se, st))
        return final, final

    _, ckpt_ends = jax.lax.scan(
        _ckpt_fwd_body, log_alpha_0, (seg_trans, site_idx),
    )
    return jnp.concatenate([log_alpha_0[None], ckpt_ends], axis=0)


@partial(jax.jit, static_argnames=("C",))
def _streaming_decode_segment(
    geno_j, log_f0, log_odds, ckpt, st, e_sites, beta_right,
    *, C,
):
    """JIT-compiled per-segment backward + posterior for decode."""
    def _emit_batch(sites):
        g = geno_j[:, sites].astype(jnp.float32)
        f0 = log_f0[:, sites]
        lo = log_odds[:, sites]
        return f0.T[:, None, :] + g.T[:, :, None] * lo.T[:, None, :]

    se = _emit_batch(e_sites)

    def _fwd(a, x):
        e, t = x
        a_new = _log_matvec_batch(a, t) + e
        return a_new, a_new
    _, alphas_inner = jax.lax.scan(_fwd, ckpt, (se[:C - 1], st[:C - 1]))
    alphas = jnp.concatenate([ckpt[None], alphas_inner], axis=0)

    se_bwd = jnp.flip(se, axis=0)
    st_bwd = jnp.flip(st, axis=0)
    def _bwd(beta, x):
        e, t = x
        b_new = _log_matvec_batch_transpose(e + beta, t)
        return b_new, b_new
    beta_left, betas_rev = jax.lax.scan(_bwd, beta_right, (se_bwd, st_bwd))
    betas = jnp.flip(betas_rev, axis=0)

    log_gamma = alphas + betas
    log_norm = jax.nn.logsumexp(log_gamma, axis=2, keepdims=True)
    gamma_seg = jnp.exp(log_gamma - log_norm)

    return gamma_seg, beta_left


def _streaming_decode_checkpointed(
    geno_j, log_f0, log_odds, seg_trans, site_idx, log_prior,
    *, C, S, n_fwd_steps, emit_pad,
    compute_max_post=True,
    sums_only=False,
) -> DecodeResult:
    """Checkpointed forward-backward that writes decode outputs per segment.

    Never materializes the full (B, T, A) posterior tensor. The forward
    checkpoint pass and per-segment backward are JIT-compiled; the outer
    segment loop stays in Python for D2H transfers.

    Parameters
    ----------
    geno_j : (B, T_padded) uint8 — already padded
    log_f0 : (A, T_padded) float32
    log_odds : (A, T_padded) float32
    seg_trans : (S, C, A, A) float32
    site_idx : (S, C) int32
    log_prior : (A,) float32
    C, S, n_fwd_steps, emit_pad : static ints
    compute_max_post : whether to compute max_post and global_sums
    sums_only : when True, skip calls/max_post and return only global_sums

    Returns
    -------
    DecodeResult
    """
    B = geno_j.shape[0]
    T = geno_j.shape[1] - emit_pad
    A = log_prior.shape[0]

    checkpoints = _streaming_decode_fwd_checkpoints(
        geno_j, log_f0, log_odds, seg_trans, site_idx, log_prior,
        C=C, S=S, n_fwd_steps=n_fwd_steps, emit_pad=emit_pad,
    )

    calls = None if sums_only else _np.zeros((B, T), dtype=_np.int8)
    max_post = None if sums_only else (
        _np.zeros((B, T), dtype=_np.float32) if compute_max_post else None
    )
    global_sums = jnp.zeros((B, A))

    beta_right = jnp.zeros((B, A))

    for g_rev in range(S):
        g = S - 1 - g_rev

        gamma_seg, beta_left = _streaming_decode_segment(
            geno_j, log_f0, log_odds,
            checkpoints[g], seg_trans[g], site_idx[g], beta_right,
            C=C,
        )

        t_start = g * C
        t_end = min(t_start + C, T)
        n_valid = t_end - t_start
        if n_valid > 0:
            if sums_only:
                global_sums = global_sums + gamma_seg[:n_valid].sum(axis=0)
            else:
                gamma_np = _np.array(gamma_seg[:n_valid])
                calls[:, t_start:t_end] = gamma_np.argmax(axis=2).astype(_np.int8).T
                if compute_max_post:
                    max_post[:, t_start:t_end] = gamma_np.max(axis=2).T
                    global_sums = global_sums + gamma_seg[:n_valid].sum(axis=0)

        beta_right = beta_left

    return DecodeResult(
        calls=calls,
        max_post=max_post,
        global_sums=_np.array(global_sums, dtype=_np.float64) if (
            sums_only or compute_max_post
        ) else None,
    )


# ---------------------------------------------------------------------------
# Block-level forward-backward
# ---------------------------------------------------------------------------

def forward_backward_blocks(
    model: AncestryModel,
    block_data,
    *,
    compute_soft_switches: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-backward operating on block-level emissions.

    The scan iterates over blocks instead of sites. Emissions come from
    pattern frequency tables; transitions use aggregate block distances.

    Parameters
    ----------
    model : AncestryModel with pattern_freq set
    block_data : BlockData
    compute_soft_switches : if True, also return per-haplotype expected
        transition counts (soft switches) from block-boundary xi.

    Returns
    -------
    gamma_block : (H, n_blocks, A) — block-level posteriors
        — or (gamma_block, soft_switches) when compute_soft_switches=True,
        where soft_switches is (H,) float32.
    """
    log_emit = model.log_emission_block(block_data)  # (H, n_blocks, A)
    d_block = jnp.array(block_data.block_distances)   # (n_blocks-1,)
    log_trans = model.log_transition_matrix(d_block)   # (n_blocks-1, A, A)

    H, n_blocks, A = log_emit.shape
    log_prior = jnp.log(model.mu)
    log_alpha_0 = log_prior[None, :] + log_emit[:, 0, :]

    # Forward scan (reusing the same scan structure)
    def fwd_step(alpha, t):
        lt = log_trans[t - 1]
        alpha_new = _log_matvec_batch(alpha, lt) + log_emit[:, t, :]
        return alpha_new, alpha_new

    _, alphas_rest = jax.lax.scan(fwd_step, log_alpha_0, jnp.arange(1, n_blocks))
    log_alphas = jnp.concatenate([log_alpha_0[None], alphas_rest], axis=0)
    log_alphas = jnp.transpose(log_alphas, (1, 0, 2))  # (H, n_blocks, A)

    # Backward scan
    log_beta_last = jnp.zeros((H, A))

    def bwd_step(beta_next, t):
        lt = log_trans[t]
        log_integrand = log_emit[:, t + 1, :] + beta_next
        beta_new = _log_matvec_batch_transpose(log_integrand, lt)
        return beta_new, beta_new

    _, betas_rev = jax.lax.scan(
        bwd_step, log_beta_last, jnp.arange(n_blocks - 2, -1, -1)
    )
    log_betas = jnp.concatenate(
        [jnp.flip(betas_rev, axis=0), log_beta_last[None]], axis=0
    )
    log_betas = jnp.transpose(log_betas, (1, 0, 2))  # (H, n_blocks, A)

    gamma_block = posteriors(log_alphas, log_betas)
    if compute_soft_switches:
        soft_sw = _compute_soft_switches_block(
            log_alphas, log_betas, log_emit, log_trans,
        )
        return gamma_block, soft_sw
    return gamma_block


def forward_backward_blocks_batched(
    model: AncestryModel,
    block_data,
    batch_size: int,
) -> jnp.ndarray:
    """Batched forward_backward_blocks: chunks over haplotypes.

    At large K the full block posterior tensor (H, n_blocks, A) may
    exceed GPU memory. This runs the existing scan on chunks of
    batch_size haplotypes and concatenates the results.

    Returns
    -------
    gamma_block : (H, n_blocks, A)
    """
    H = block_data.pattern_indices.shape[0]
    if batch_size >= H:
        return forward_backward_blocks(model, block_data)

    from .blocks import BlockData
    chunks = []
    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        bd_chunk = BlockData(
            pattern_indices=block_data.pattern_indices[start:end],
            block_starts=block_data.block_starts,
            block_ends=block_data.block_ends,
            block_distances=block_data.block_distances,
            pattern_counts=block_data.pattern_counts,
            max_patterns=block_data.max_patterns,
            block_size=block_data.block_size,
        )
        gamma_chunk = forward_backward_blocks(model, bd_chunk)
        chunks.append(_np.array(gamma_chunk))

    return jnp.array(_np.concatenate(chunks, axis=0))


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _compute_soft_switches(
    log_alpha: jnp.ndarray,
    log_beta: jnp.ndarray,
    model: AncestryModel,
    geno: jnp.ndarray,
    d_morgan: jnp.ndarray,
) -> jnp.ndarray:
    """Expected number of ancestry transitions per haplotype from xi.

    Uses the diagonal of the pairwise posterior ξ[t,a,a] to compute
    P(z_t ≠ z_{t+1} | data) at each interval — density-invariant
    unlike hard-call switch counting.

    Parameters
    ----------
    log_alpha : (H, T, A)
    log_beta : (H, T, A)
    model : AncestryModel (for emissions + transitions)
    geno : (H, T) uint8
    d_morgan : (T-1,)

    Returns
    -------
    soft_switches : (H,) float32 — expected transition count per haplotype
    """
    H, T, A = log_alpha.shape
    if T <= 1:
        return jnp.zeros(H)

    log_emit = model.log_emission(geno)               # (H, T, A)
    log_trans = model.log_transition_matrix(d_morgan)  # (T-1, A, A)

    # P(data) — constant across t, compute from position 0
    log_P = jax.nn.logsumexp(
        log_alpha[:, 0, :] + log_beta[:, 0, :], axis=1
    )  # (H,)

    # Diagonal of transition matrices: log((1-p) + p·μ[a])
    log_trans_diag = jnp.diagonal(log_trans, axis1=1, axis2=2)  # (T-1, A)

    # ξ_diag[h,t,a] = α[h,t,a] · trans_diag[t,a] · emit[h,t+1,a] · β[h,t+1,a] / P(data)
    log_xi_diag = (
        log_alpha[:, :-1, :]           # (H, T-1, A)
        + log_trans_diag[None, :, :]   # (1, T-1, A)
        + log_emit[:, 1:, :]           # (H, T-1, A)
        + log_beta[:, 1:, :]           # (H, T-1, A)
    )  # (H, T-1, A)

    xi_diag = jnp.exp(log_xi_diag - log_P[:, None, None])  # (H, T-1, A)
    stay = xi_diag.sum(axis=2)                               # (H, T-1)
    return jnp.clip(1.0 - stay, 0.0, 1.0).sum(axis=1)       # (H,)


def _compute_soft_switches_block(
    log_alpha_block: jnp.ndarray,   # (H, n_blocks, A)
    log_beta_block: jnp.ndarray,    # (H, n_blocks, A)
    log_emit_block: jnp.ndarray,    # (H, n_blocks, A)
    log_trans_block: jnp.ndarray,   # (n_blocks-1, A, A)
) -> jnp.ndarray:
    """Expected ancestry transitions per haplotype at block boundaries.

    Block analog of :func:`_compute_soft_switches`. Same xi-diagonal trick:
    soft_sw[h] = sum over block boundaries t of P(z_t != z_{t+1} | data, h).

    Density-invariant: block widths enter through ``log_trans_block``,
    so longer block intervals contribute proportionally more transition
    mass per boundary than shorter ones.

    Parameters
    ----------
    log_alpha_block : (H, n_blocks, A)
    log_beta_block : (H, n_blocks, A)
    log_emit_block : (H, n_blocks, A) — block-level emissions
    log_trans_block : (n_blocks-1, A, A) — block-interval transitions

    Returns
    -------
    soft_switches : (H,) float32 — expected transition count per haplotype
    """
    H, n_blocks, A = log_alpha_block.shape
    if n_blocks <= 1:
        return jnp.zeros(H)

    log_P = jax.nn.logsumexp(
        log_alpha_block[:, 0, :] + log_beta_block[:, 0, :], axis=1,
    )  # (H,)
    log_trans_diag = jnp.diagonal(log_trans_block, axis1=1, axis2=2)  # (n_blocks-1, A)

    log_xi_diag = (
        log_alpha_block[:, :-1, :]
        + log_trans_diag[None, :, :]
        + log_emit_block[:, 1:, :]
        + log_beta_block[:, 1:, :]
    )  # (H, n_blocks-1, A)
    xi_diag = jnp.exp(log_xi_diag - log_P[:, None, None])
    stay = xi_diag.sum(axis=2)                                 # (H, n_blocks-1)
    return jnp.clip(1.0 - stay, 0.0, 1.0).sum(axis=1)         # (H,)


def forward_backward(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    use_checkpointing: bool | None = None,
    compute_transitions: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Full forward-backward, returning posteriors.

    This is the main entry point for the HMM.  For T > 64,
    automatically uses gradient checkpointing to reduce memory.

    Parameters
    ----------
    geno : (H, T) uint8
    model : AncestryModel
    d_morgan : (T-1,)
    use_checkpointing : force checkpointing on/off (default: auto)
    compute_transitions : if True, also return per-haplotype expected
        transition counts (soft switches) from xi posteriors

    Returns
    -------
    gamma : (H, T, A) posterior probabilities
        — or (gamma, soft_switches) when compute_transitions=True,
        where soft_switches is (H,) float32.
    """
    T = geno.shape[1]
    if use_checkpointing is None:
        use_checkpointing = T > 64

    if use_checkpointing:
        return forward_backward_checkpointed(
            geno, model, d_morgan,
            compute_transitions=compute_transitions,
        )

    log_alpha, _ = forward(geno, model, d_morgan)
    log_beta = backward(geno, model, d_morgan)
    gamma = posteriors(log_alpha, log_beta)

    if compute_transitions:
        soft_sw = _compute_soft_switches(
            log_alpha, log_beta, model, geno, d_morgan,
        )
        return gamma, soft_sw

    return gamma


# ---------------------------------------------------------------------------
# Batched forward-backward for memory management
# ---------------------------------------------------------------------------

def forward_backward_batched(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
) -> jnp.ndarray:
    """Forward-backward in batches over haplotypes.

    For very large H where (H, T, A) doesn't fit in GPU memory at once,
    process in chunks.  The HMM is independent across haplotypes so this
    is exact.

    Parameters
    ----------
    geno : (H, T)
    model : AncestryModel
    d_morgan : (T-1,)
    batch_size : max haplotypes per batch

    Returns
    -------
    gamma : (H, T, A) posterior probabilities
    """
    H = geno.shape[0]

    # Guard: this function materializes the full (H, T, A) gamma. At biobank
    # scale use forward_backward_em (sufficient-statistics streaming) or
    # forward_backward_decode (hard-call streaming) instead.
    if H > 2 * batch_size:
        raise RuntimeError(
            f"forward_backward_batched would materialize a ({H}, T, A) tensor "
            f"for H={H} > 2*batch_size={2*batch_size}. Use forward_backward_em "
            f"for EM or forward_backward_decode for decoding."
        )

    if H <= batch_size:
        return forward_backward(geno, model, d_morgan)

    gammas = []
    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        batch_geno = jnp.asarray(geno[start:end])
        gamma_batch = forward_backward(batch_geno, model, d_morgan)
        gammas.append(gamma_batch)

    return jnp.concatenate(gammas, axis=0)


# ---------------------------------------------------------------------------
# Bucketed forward-backward (per-haplotype T)
# ---------------------------------------------------------------------------

def forward_backward_bucketed(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
) -> jnp.ndarray:
    """Forward-backward with per-bucket transition matrices.

    Partitions haplotypes by their T-bucket assignment, runs
    forward-backward for each bucket with its own transition
    matrices, then reassembles the output.

    Falls back to standard forward-backward when no bucket
    assignments are set on the model.

    Parameters
    ----------
    geno : (H, T)
    model : AncestryModel (with bucket_assignments set)
    d_morgan : (T-1,)
    batch_size : max haplotypes per bucket batch

    Returns
    -------
    gamma : (H, T, A)
    """
    if model.bucket_assignments is None:
        return forward_backward_batched(geno, model, d_morgan, batch_size)

    import numpy as _np

    H, T = geno.shape
    A = model.n_ancestries
    B = len(model.bucket_centers)

    # Precompute per-bucket transition matrices: (B, T-1, A, A)
    all_log_trans = model.log_transition_matrices_bucketed(d_morgan)

    # Precompute emissions once (shared): (H, T, A)
    log_emit = model.log_emission(geno)
    log_prior = jnp.log(model.mu)

    bucket_np = _np.array(model.bucket_assignments)

    # Allocate output on host; fill per bucket
    gamma_out = _np.zeros((H, T, A), dtype=_np.float32)

    for b in range(B):
        mask = bucket_np == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue

        hap_idx = _np.where(mask)[0]
        b_trans = all_log_trans[b]            # (T-1, A, A)

        # Build a temporary scalar-T model for this bucket
        b_model = AncestryModel(
            n_ancestries=A,
            mu=model.mu,
            gen_since_admix=float(model.bucket_centers[b]),
            allele_freq=model.allele_freq,
        )

        # Run in sub-batches if n_b is large
        b_gammas = []
        for s in range(0, n_b, batch_size):
            e = min(s + batch_size, n_b)
            g = forward_backward(geno[hap_idx[s:e]], b_model, d_morgan)
            b_gammas.append(_np.array(g))

        gamma_out[hap_idx] = _np.concatenate(b_gammas, axis=0)

    return jnp.array(gamma_out)


# ---------------------------------------------------------------------------
# Streaming batched forward-backward (no full-tensor materialisation)
# ---------------------------------------------------------------------------

def forward_backward_em(
    geno,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
) -> EMStats:
    """Batched forward-backward with streaming M-step accumulation.

    Uses _streaming_em_checkpointed to avoid materializing the (B, T, A)
    posterior tensor.  Peak GPU memory is O(batch_size * sqrt(T) * A),
    independent of total H and linear in sqrt(T) rather than T.

    Parameters
    ----------
    geno : (H, T)
    model : AncestryModel
    d_morgan : (T-1,)
    batch_size : max haplotypes per batch

    Returns
    -------
    EMStats with accumulated sufficient statistics.
    """
    H, T = geno.shape
    A = model.n_ancestries

    # Precompute model-derived tensors once (shared across all batches).
    pc = _precompute_streaming_tensors(model, d_morgan, T)
    emit_pad = pc["emit_pad"]

    weighted_counts = jnp.zeros((A, T))
    total_weights = jnp.zeros((A, T))
    mu_sum = jnp.zeros((A,))
    soft_switches_per_hap = _np.zeros(H, dtype=_np.float32)

    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        batch_geno = jnp.asarray(geno[start:end])
        if emit_pad > 0:
            batch_geno = jnp.concatenate(
                [batch_geno, jnp.zeros((end - start, emit_pad), dtype=batch_geno.dtype)],
                axis=1,
            )
        wc_b, tw_b, mu_b, sw_b = _streaming_em_checkpointed(
            batch_geno, pc["log_f0"], pc["log_odds"], pc["seg_trans"],
            pc["site_idx"], pc["gamma_site_idx"], pc["log_prior"],
            C=pc["C"], S=pc["S"], n_fwd_steps=pc["n_fwd_steps"],
            emit_pad=emit_pad,
        )
        weighted_counts = weighted_counts + wc_b
        total_weights = total_weights + tw_b
        mu_sum = mu_sum + mu_b
        soft_switches_per_hap[start:end] = _np.array(sw_b)

    return EMStats(
        weighted_counts=weighted_counts,
        total_weights=total_weights,
        mu_sum=mu_sum,
        switch_sum=jnp.zeros((T - 1,)) if T > 1 else jnp.zeros((0,)),
        switches_per_hap=_np.zeros(H, dtype=_np.int32),
        soft_switches_per_hap=soft_switches_per_hap,
        n_haps=H,
        n_sites=T,
    )


def forward_backward_bucketed_em(
    geno,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
) -> EMStats:
    """Bucketed forward-backward with streaming M-step accumulation.

    Like forward_backward_em but uses per-bucket transition matrices
    when model.bucket_assignments is set (per-haplotype T).

    Falls back to forward_backward_em when no bucket assignments exist.
    """
    if model.bucket_assignments is None:
        return forward_backward_em(geno, model, d_morgan, batch_size)

    H, T = geno.shape
    A = model.n_ancestries
    B = len(model.bucket_centers)

    weighted_counts = jnp.zeros((A, T))
    total_weights = jnp.zeros((A, T))
    mu_sum = jnp.zeros((A,))
    soft_switches_per_hap = _np.zeros(H, dtype=_np.float32)

    bucket_np = _np.array(model.bucket_assignments)

    for b in range(B):
        mask = bucket_np == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue

        hap_idx = _np.where(mask)[0]

        b_model = AncestryModel(
            n_ancestries=A,
            mu=model.mu,
            gen_since_admix=float(model.bucket_centers[b]),
            allele_freq=model.allele_freq,
        )
        b_pc = _precompute_streaming_tensors(b_model, d_morgan, T)
        b_emit_pad = b_pc["emit_pad"]

        for s in range(0, n_b, batch_size):
            e = min(s + batch_size, n_b)
            batch_hap_idx = hap_idx[s:e]
            batch_geno = jnp.asarray(geno[batch_hap_idx])
            if b_emit_pad > 0:
                batch_geno = jnp.concatenate(
                    [batch_geno, jnp.zeros((e - s, b_emit_pad), dtype=batch_geno.dtype)],
                    axis=1,
                )
            wc_b, tw_b, mu_b, sw_b = _streaming_em_checkpointed(
                batch_geno, b_pc["log_f0"], b_pc["log_odds"], b_pc["seg_trans"],
                b_pc["site_idx"], b_pc["gamma_site_idx"], b_pc["log_prior"],
                C=b_pc["C"], S=b_pc["S"], n_fwd_steps=b_pc["n_fwd_steps"],
                emit_pad=b_emit_pad,
            )
            weighted_counts = weighted_counts + wc_b
            total_weights = total_weights + tw_b
            mu_sum = mu_sum + mu_b
            soft_switches_per_hap[batch_hap_idx] = _np.array(sw_b)

    return EMStats(
        weighted_counts=weighted_counts,
        total_weights=total_weights,
        mu_sum=mu_sum,
        switch_sum=jnp.zeros((T - 1,)) if T > 1 else jnp.zeros((0,)),
        switches_per_hap=_np.zeros(H, dtype=_np.int32),
        soft_switches_per_hap=soft_switches_per_hap,
        n_haps=H,
        n_sites=T,
    )


def forward_backward_decode(
    geno,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
    compute_max_post: bool = True,
) -> DecodeResult:
    """Batched forward-backward for final decode pass.

    Uses _streaming_decode_checkpointed to avoid materializing the full
    (B, T, A) posterior tensor.  Peak GPU memory per batch is
    O(batch_size * sqrt(T) * A).

    Parameters
    ----------
    geno : (H, T)
    model : AncestryModel
    d_morgan : (T-1,)
    batch_size : max haplotypes per batch
    compute_max_post : whether to compute max_post (H, T) and global_sums (H, A)

    Returns
    -------
    DecodeResult
    """
    H, T = geno.shape
    A = model.n_ancestries

    pc = _precompute_streaming_tensors(model, d_morgan, T)
    emit_pad = pc["emit_pad"]

    calls = _np.zeros((H, T), dtype=_np.int8)
    max_post = _np.zeros((H, T), dtype=_np.float16) if compute_max_post else None
    global_sums = _np.zeros((H, A), dtype=_np.float64) if compute_max_post else None

    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        batch_geno = jnp.asarray(geno[start:end])
        if emit_pad > 0:
            batch_geno = jnp.concatenate(
                [batch_geno, jnp.zeros((end - start, emit_pad), dtype=batch_geno.dtype)],
                axis=1,
            )
        result = _streaming_decode_checkpointed(
            batch_geno, pc["log_f0"], pc["log_odds"], pc["seg_trans"],
            pc["site_idx"], pc["log_prior"],
            C=pc["C"], S=pc["S"], n_fwd_steps=pc["n_fwd_steps"],
            emit_pad=emit_pad,
            compute_max_post=compute_max_post,
        )
        calls[start:end] = result.calls
        if compute_max_post:
            max_post[start:end] = result.max_post
            global_sums[start:end] = result.global_sums

    return DecodeResult(calls=calls, max_post=max_post, global_sums=global_sums)


def forward_backward_bucketed_decode(
    geno,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
    compute_max_post: bool = True,
) -> DecodeResult:
    """Bucketed forward-backward for final decode with per-haplotype T.

    Falls back to forward_backward_decode when no bucket assignments exist.
    """
    if model.bucket_assignments is None:
        return forward_backward_decode(
            geno, model, d_morgan, batch_size, compute_max_post,
        )

    H, T = geno.shape
    A = model.n_ancestries
    B = len(model.bucket_centers)

    calls = _np.zeros((H, T), dtype=_np.int8)
    max_post = _np.zeros((H, T), dtype=_np.float16) if compute_max_post else None
    global_sums = _np.zeros((H, A), dtype=_np.float64) if compute_max_post else None

    bucket_np = _np.array(model.bucket_assignments)

    for b in range(B):
        mask = bucket_np == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue

        hap_idx = _np.where(mask)[0]

        b_model = AncestryModel(
            n_ancestries=A,
            mu=model.mu,
            gen_since_admix=float(model.bucket_centers[b]),
            allele_freq=model.allele_freq,
        )
        b_pc = _precompute_streaming_tensors(b_model, d_morgan, T)
        b_emit_pad = b_pc["emit_pad"]

        for s in range(0, n_b, batch_size):
            e = min(s + batch_size, n_b)
            batch_hap_idx = hap_idx[s:e]
            batch_geno = jnp.asarray(geno[batch_hap_idx])
            if b_emit_pad > 0:
                batch_geno = jnp.concatenate(
                    [batch_geno, jnp.zeros((e - s, b_emit_pad), dtype=batch_geno.dtype)],
                    axis=1,
                )
            result = _streaming_decode_checkpointed(
                batch_geno, b_pc["log_f0"], b_pc["log_odds"], b_pc["seg_trans"],
                b_pc["site_idx"], b_pc["log_prior"],
                C=b_pc["C"], S=b_pc["S"], n_fwd_steps=b_pc["n_fwd_steps"],
                emit_pad=b_emit_pad,
                compute_max_post=compute_max_post,
            )
            calls[batch_hap_idx] = result.calls
            if compute_max_post:
                max_post[batch_hap_idx] = result.max_post
                global_sums[batch_hap_idx] = result.global_sums

    return DecodeResult(calls=calls, max_post=max_post, global_sums=global_sums)


# ---------------------------------------------------------------------------
# Lightweight ancestry-sums-only streaming decode
# ---------------------------------------------------------------------------

def forward_backward_ancestry_sums(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
) -> _np.ndarray:
    """Streaming decode returning only per-haplotype ancestry sums.

    Equivalent to ``forward_backward_decode(...).global_sums`` but never
    allocates the (H, T) ``calls`` or ``max_post`` arrays, avoiding
    ~135 GB of host memory at biobank scale.

    Returns
    -------
    global_sums : (H, A) float64
        Sum_t gamma[h, t, a] per haplotype.
    """
    H, T = geno.shape
    A = model.n_ancestries

    pc = _precompute_streaming_tensors(model, d_morgan, T)
    emit_pad = pc["emit_pad"]

    global_sums = _np.zeros((H, A), dtype=_np.float64)

    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        batch_geno = jnp.asarray(geno[start:end])
        if emit_pad > 0:
            batch_geno = jnp.concatenate(
                [batch_geno, jnp.zeros((end - start, emit_pad), dtype=batch_geno.dtype)],
                axis=1,
            )
        result = _streaming_decode_checkpointed(
            batch_geno, pc["log_f0"], pc["log_odds"], pc["seg_trans"],
            pc["site_idx"], pc["log_prior"],
            C=pc["C"], S=pc["S"], n_fwd_steps=pc["n_fwd_steps"],
            emit_pad=emit_pad,
            sums_only=True,
        )
        global_sums[start:end] = result.global_sums

    return global_sums


def forward_backward_bucketed_ancestry_sums(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
) -> _np.ndarray:
    """Bucketed streaming decode returning only per-haplotype ancestry sums.

    Falls back to ``forward_backward_ancestry_sums`` when no bucket
    assignments exist.

    Returns
    -------
    global_sums : (H, A) float64
    """
    if model.bucket_assignments is None:
        return forward_backward_ancestry_sums(geno, model, d_morgan, batch_size)

    H, T = geno.shape
    A = model.n_ancestries
    B = len(model.bucket_centers)

    global_sums = _np.zeros((H, A), dtype=_np.float64)

    bucket_np = _np.array(model.bucket_assignments)

    for b in range(B):
        mask = bucket_np == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue

        hap_idx = _np.where(mask)[0]

        b_model = AncestryModel(
            n_ancestries=A,
            mu=model.mu,
            gen_since_admix=float(model.bucket_centers[b]),
            allele_freq=model.allele_freq,
        )
        b_pc = _precompute_streaming_tensors(b_model, d_morgan, T)
        b_emit_pad = b_pc["emit_pad"]

        for s in range(0, n_b, batch_size):
            e = min(s + batch_size, n_b)
            batch_hap_idx = hap_idx[s:e]
            batch_geno = geno[batch_hap_idx]
            if b_emit_pad > 0:
                batch_geno = jnp.concatenate(
                    [batch_geno, jnp.zeros((e - s, b_emit_pad), dtype=batch_geno.dtype)],
                    axis=1,
                )
            result = _streaming_decode_checkpointed(
                batch_geno, b_pc["log_f0"], b_pc["log_odds"], b_pc["seg_trans"],
                b_pc["site_idx"], b_pc["log_prior"],
                C=b_pc["C"], S=b_pc["S"], n_fwd_steps=b_pc["n_fwd_steps"],
                emit_pad=b_emit_pad,
                sums_only=True,
            )
            global_sums[batch_hap_idx] = result.global_sums

    return global_sums
