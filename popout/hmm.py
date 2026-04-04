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

from .datatypes import AncestryModel


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
) -> jnp.ndarray:
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

    Returns
    -------
    gamma : (H, T, A) posterior probabilities
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
    seg_trans_rev = jnp.flip(seg_trans, axis=0)        # (S, C, A, A)
    ckpt_rev = jnp.flip(checkpoints[:S], axis=0)       # (S, H, A)

    def _bwd_group_body(beta_right, xs):
        se, st, ckpt = xs  # (C, H, A), (C, A, A), (H, A)

        # --- Forward recompute from checkpoint ---
        # C-1 steps produce alpha at sites ckpt+1 … ckpt+C-1
        def _fwd(a, x):
            e, t = x
            a_new = _log_matvec_batch(a, t) + e
            return a_new, a_new

        _, alphas_inner = jax.lax.scan(_fwd, ckpt, (se[:C - 1], st[:C - 1]))
        # alphas_inner: (C-1, H, A)
        alphas = jnp.concatenate([ckpt[None], alphas_inner], axis=0)  # (C, H, A)

        # --- Backward: C steps from beta_right ---
        se_bwd = jnp.flip(se, axis=0)  # reversed within segment
        st_bwd = jnp.flip(st, axis=0)

        def _bwd(beta, x):
            e, t = x
            b_new = _log_matvec_batch_transpose(e + beta, t)
            return b_new, b_new

        beta_left, betas_rev = jax.lax.scan(_bwd, beta_right, (se_bwd, st_bwd))
        betas = jnp.flip(betas_rev, axis=0)  # (C, H, A) forward site order

        # --- Posteriors ---
        log_gamma = alphas + betas
        log_norm = jax.nn.logsumexp(log_gamma, axis=2, keepdims=True)
        gamma_seg = jnp.exp(log_gamma - log_norm)  # (C, H, A)

        return beta_left, gamma_seg

    _, gammas_rev = jax.lax.scan(
        _bwd_group_body, jnp.zeros((H, A)),  # beta at T_pad = 0 (log 1)
        (seg_emit_rev, seg_trans_rev, ckpt_rev),
    )
    # gammas_rev: (S, C, H, A) in reverse group order

    gammas = jnp.flip(gammas_rev, axis=0)            # (S, C, H, A)
    gamma_flat = gammas.reshape(S * C, H, A)[:T]     # (T, H, A) — trim padding
    return jnp.transpose(gamma_flat, (1, 0, 2))      # (H, T, A)


# ---------------------------------------------------------------------------
# Block-level forward-backward
# ---------------------------------------------------------------------------

def forward_backward_blocks(
    model: AncestryModel,
    block_data,
) -> jnp.ndarray:
    """Forward-backward operating on block-level emissions.

    The scan iterates over blocks instead of sites. Emissions come from
    pattern frequency tables; transitions use aggregate block distances.

    Parameters
    ----------
    model : AncestryModel with pattern_freq set
    block_data : BlockData

    Returns
    -------
    gamma_block : (H, n_blocks, A) — block-level posteriors
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

    return posteriors(log_alphas, log_betas)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def forward_backward(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    use_checkpointing: bool | None = None,
) -> jnp.ndarray:
    """Full forward-backward, returning posteriors.

    This is the main entry point for the HMM.  For T > 64,
    automatically uses gradient checkpointing to reduce memory.

    Parameters
    ----------
    geno : (H, T) uint8
    model : AncestryModel
    d_morgan : (T-1,)
    use_checkpointing : force checkpointing on/off (default: auto)

    Returns
    -------
    gamma : (H, T, A) posterior probabilities
    """
    T = geno.shape[1]
    if use_checkpointing is None:
        use_checkpointing = T > 64

    if use_checkpointing:
        return forward_backward_checkpointed(geno, model, d_morgan)

    log_alpha, _ = forward(geno, model, d_morgan)
    log_beta = backward(geno, model, d_morgan)
    return posteriors(log_alpha, log_beta)


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
    if H <= batch_size:
        return forward_backward(geno, model, d_morgan)

    gammas = []
    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        batch_geno = geno[start:end]
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
        b_emit = log_emit[hap_idx]           # (n_b, T, A)
        b_geno = geno[hap_idx]               # (n_b, T)
        b_trans = all_log_trans[b]            # (T-1, A, A)

        # Build a temporary scalar-T model for this bucket
        b_model = AncestryModel(
            n_ancestries=A,
            mu=model.mu,
            gen_since_admix=float(model.bucket_centers[b]),
            allele_freq=model.allele_freq,
            mismatch=model.mismatch,
        )

        # Run in sub-batches if n_b is large
        b_gammas = []
        for s in range(0, n_b, batch_size):
            e = min(s + batch_size, n_b)
            g = forward_backward(b_geno[s:e], b_model, d_morgan)
            b_gammas.append(_np.array(g))

        gamma_out[hap_idx] = _np.concatenate(b_gammas, axis=0)

    return jnp.array(gamma_out)
