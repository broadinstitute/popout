"""Forward-backward HMM on GPU via JAX.

All haplotypes run the A-state HMM simultaneously — one site at a time,
sequential across T sites, parallel across all H haplotypes.

The forward state is (H, A) — typically 1M × 8 = 32 MB.  This fits in
L2 cache on an A100, making the algorithm memory-bandwidth bound at
~2-3 seconds per chromosome for 1M haplotypes.

Checkpointing reduces backward-pass memory from O(T) to O(√T) at the
cost of recomputing forward values from the nearest checkpoint.
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

    # Precompute log emission table: (H, T, A)
    log_emit = model.log_emission(geno)

    # Precompute transition matrices: (T-1, A, A)
    log_trans = model.log_transition_matrix(d_morgan)

    # Initial state: log(mu) + log_emit[:, 0, :]
    log_prior = jnp.log(model.mu)  # (A,)
    log_alpha_0 = log_prior[None, :] + log_emit[:, 0, :]  # (H, A)

    # Storage
    # Full alpha storage for posterior computation
    # We store checkpoints and reconstruct between them during backward
    n_checkpoints = (T - 1) // checkpoint_interval + 1
    checkpoint_indices = jnp.arange(0, T, checkpoint_interval)

    # Run forward with scan, storing everything
    # For the first draft, store all alphas (memory permitting)
    # TODO: implement checkpointed version for very long chromosomes
    log_alphas = _forward_scan(log_alpha_0, log_emit, log_trans)

    # Extract checkpoints
    checkpoints = log_alphas[:, checkpoint_indices, :]

    return log_alphas, checkpoints


def _forward_scan(
    log_alpha_0: jnp.ndarray,
    log_emit: jnp.ndarray,
    log_trans: jnp.ndarray,
) -> jnp.ndarray:
    """Sequential forward scan across T sites.

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
        """One forward step: transition + emission for all haplotypes."""
        lt = log_trans[t - 1]  # (A, A) — transition at this interval

        # Transition: for each haplotype, compute
        #   log_alpha_pred[h, j] = logsumexp_i(log_alpha_prev[h, i] + lt[i, j])
        # This is a batched log-space matrix-vector product.
        # log_alpha_prev is (H, A), lt is (A, A)
        # We want: result[h, j] = logsumexp over i of (alpha[h, i] + lt[i, j])
        log_alpha_pred = _log_matvec_batch(log_alpha_prev, lt)  # (H, A)

        # Emission
        log_alpha_new = log_alpha_pred + log_emit[:, t, :]  # (H, A)

        return log_alpha_new, log_alpha_new

    # Scan over t = 1 .. T-1
    _, log_alphas_rest = jax.lax.scan(step, log_alpha_0, jnp.arange(1, T))

    # Prepend t=0
    # log_alphas_rest is (T-1, H, A), need (H, T, A)
    log_alphas = jnp.concatenate(
        [log_alpha_0[None, :, :], log_alphas_rest],  # (T, H, A)
        axis=0,
    )
    return jnp.transpose(log_alphas, (1, 0, 2))  # (H, T, A)


def _log_matvec_batch(log_v: jnp.ndarray, log_M: jnp.ndarray) -> jnp.ndarray:
    """Batched log-space matrix-vector product.

    log_v : (H, A) — log probability vectors
    log_M : (A, A) — log transition matrix, log_M[i, j] = log P(j | i)

    Returns (H, A) where result[h, j] = logsumexp_i(log_v[h, i] + log_M[i, j])
    """
    # Expand: log_v[:, :, None] + log_M[None, :, :] → (H, A_from, A_to)
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

    # Beta at last site is 0 (log(1))
    log_beta_T = jnp.zeros((H, A))

    def step(log_beta_next, t):
        """One backward step."""
        # t indexes from T-2 down to 0
        lt = log_trans[t]  # (A, A) transition at interval (t, t+1)

        # beta_pred[h, i] = logsumexp_j(lt[i,j] + log_emit[h, t+1, j] + log_beta_next[h, j])
        log_integrand = log_emit[:, t + 1, :] + log_beta_next  # (H, A)
        log_beta_new = _log_matvec_batch_transpose(log_integrand, lt)

        return log_beta_new, log_beta_new

    # Scan backward: t from T-2 down to 0
    _, log_betas_rev = jax.lax.scan(
        step, log_beta_T, jnp.arange(T - 2, -1, -1)
    )
    # log_betas_rev is (T-1, H, A) in reverse order

    log_betas = jnp.concatenate(
        [jnp.flip(log_betas_rev, axis=0), log_beta_T[None, :, :]],  # (T, H, A)
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
    # Normalize over ancestries at each (haplotype, site)
    log_norm = jax.nn.logsumexp(log_gamma, axis=2, keepdims=True)
    gamma = jnp.exp(log_gamma - log_norm)
    return gamma


def forward_backward(
    geno: jnp.ndarray,
    model: AncestryModel,
    d_morgan: jnp.ndarray,
) -> jnp.ndarray:
    """Full forward-backward, returning posteriors.

    This is the main entry point for the HMM.

    Parameters
    ----------
    geno : (H, T) uint8
    model : AncestryModel
    d_morgan : (T-1,)

    Returns
    -------
    gamma : (H, T, A) posterior probabilities
    """
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
