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
            g = forward_backward(geno[hap_idx[s:e]], b_model, d_morgan)
            b_gammas.append(_np.array(g))

        gamma_out[hap_idx] = _np.concatenate(b_gammas, axis=0)

    return jnp.array(gamma_out)


# ---------------------------------------------------------------------------
# Streaming batched forward-backward (no full-tensor materialisation)
# ---------------------------------------------------------------------------

def forward_backward_em(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
) -> EMStats:
    """Batched forward-backward with streaming M-step accumulation.

    Instead of concatenating all batch gammas into (H, T, A), accumulates
    the sufficient statistics needed by the M-step per batch.  Peak GPU
    memory is O(batch_size * T * A), independent of total H.

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

    weighted_counts = jnp.zeros((A, T))
    total_weights = jnp.zeros((A, T))
    mu_sum = jnp.zeros((A,))
    switch_sum = jnp.zeros((T - 1,)) if T > 1 else jnp.zeros((0,))
    switches_per_hap = _np.zeros(H, dtype=_np.int32)
    soft_switches_per_hap = _np.zeros(H, dtype=_np.float32)

    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        batch_geno = geno[start:end]
        gamma, batch_soft_sw = forward_backward(
            batch_geno, model, d_morgan, compute_transitions=True,
        )

        # Allele freq stats: einsum('hta,ht->at', gamma, geno_f)
        geno_f = batch_geno.astype(jnp.float32)
        weighted_counts += jnp.einsum('hta,ht->at', gamma, geno_f)
        total_weights += gamma.sum(axis=0).T  # (T, A).T → (A, T)

        # Mu stats
        mu_sum += gamma.sum(axis=(0, 1))  # (A,)

        # Soft switches (xi-based, density-invariant)
        soft_switches_per_hap[start:end] = _np.array(batch_soft_sw)

        # Hard switch stats (kept for diagnostics)
        calls = jnp.argmax(gamma, axis=2)  # (B, T)
        del gamma, geno_f, batch_soft_sw  # free GPU before switch computation

        if T > 1:
            switches = (calls[:, 1:] != calls[:, :-1])  # (B, T-1) bool
            switch_sum += switches.sum(axis=0).astype(jnp.float32)
            switches_per_hap[start:end] = _np.array(
                switches.sum(axis=1), dtype=_np.int32,
            )
            del calls, switches  # free GPU before next batch

    return EMStats(
        weighted_counts=weighted_counts,
        total_weights=total_weights,
        mu_sum=mu_sum,
        switch_sum=switch_sum,
        switches_per_hap=switches_per_hap,
        soft_switches_per_hap=soft_switches_per_hap,
        n_haps=H,
        n_sites=T,
    )


def forward_backward_bucketed_em(
    geno: jnp.ndarray,
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
    switch_sum = jnp.zeros((T - 1,)) if T > 1 else jnp.zeros((0,))
    switches_per_hap = _np.zeros(H, dtype=_np.int32)
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
            mismatch=model.mismatch,
        )

        for s in range(0, n_b, batch_size):
            e = min(s + batch_size, n_b)
            batch_hap_idx = hap_idx[s:e]
            batch_geno = geno[batch_hap_idx]  # index from original, no per-bucket copy
            gamma, batch_soft_sw = forward_backward(
                batch_geno, b_model, d_morgan, compute_transitions=True,
            )

            geno_f = batch_geno.astype(jnp.float32)
            weighted_counts += jnp.einsum('hta,ht->at', gamma, geno_f)
            total_weights += gamma.sum(axis=0).T
            mu_sum += gamma.sum(axis=(0, 1))

            # Soft switches (xi-based, density-invariant)
            soft_switches_per_hap[batch_hap_idx] = _np.array(batch_soft_sw)

            # Hard switch stats (kept for diagnostics)
            calls = jnp.argmax(gamma, axis=2)
            del gamma, geno_f, batch_geno, batch_soft_sw  # free GPU

            if T > 1:
                switches = (calls[:, 1:] != calls[:, :-1])
                switch_sum += switches.sum(axis=0).astype(jnp.float32)
                switches_per_hap[batch_hap_idx] = _np.array(
                    switches.sum(axis=1), dtype=_np.int32,
                )
                del calls, switches  # free GPU before next batch

    return EMStats(
        weighted_counts=weighted_counts,
        total_weights=total_weights,
        mu_sum=mu_sum,
        switch_sum=switch_sum,
        switches_per_hap=switches_per_hap,
        soft_switches_per_hap=soft_switches_per_hap,
        n_haps=H,
        n_sites=T,
    )


def forward_backward_decode(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
    batch_size: int = 50_000,
    compute_max_post: bool = True,
) -> DecodeResult:
    """Batched forward-backward for final decode pass.

    Returns hard calls and optional pre-computed reductions on CPU,
    without materialising the full (H, T, A) posterior tensor.

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

    calls = _np.zeros((H, T), dtype=_np.int8)
    max_post = _np.zeros((H, T), dtype=_np.float32) if compute_max_post else None
    global_sums = _np.zeros((H, A), dtype=_np.float64) if compute_max_post else None

    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        gamma = forward_backward(geno[start:end], model, d_morgan)  # (B, T, A)

        calls[start:end] = _np.array(jnp.argmax(gamma, axis=2), dtype=_np.int8)
        if compute_max_post:
            max_post[start:end] = _np.array(gamma.max(axis=2))
            global_sums[start:end] = _np.array(gamma.sum(axis=1))
        del gamma  # free GPU before next batch

    return DecodeResult(calls=calls, max_post=max_post, global_sums=global_sums)


def forward_backward_bucketed_decode(
    geno: jnp.ndarray,
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
    max_post = _np.zeros((H, T), dtype=_np.float32) if compute_max_post else None
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
            mismatch=model.mismatch,
        )

        for s in range(0, n_b, batch_size):
            e = min(s + batch_size, n_b)
            batch_hap_idx = hap_idx[s:e]
            gamma = forward_backward(geno[batch_hap_idx], b_model, d_morgan)

            calls[batch_hap_idx] = _np.array(
                jnp.argmax(gamma, axis=2), dtype=_np.int8,
            )
            if compute_max_post:
                max_post[batch_hap_idx] = _np.array(gamma.max(axis=2))
                global_sums[batch_hap_idx] = _np.array(gamma.sum(axis=1))
            del gamma  # free GPU before next batch

    return DecodeResult(calls=calls, max_post=max_post, global_sums=global_sums)
