"""Linear-chain Conditional Random Field output layer.

Adds pairwise transition modeling on top of CNN logits using the same
``jax.lax.scan`` + ``logsumexp`` patterns as the HMM forward-backward
in ``hmm.py``.

The CRF score for a label sequence z given unary potentials φ is::

    score(z | φ) = Σ_t φ[t, z_t]  +  Σ_t W[z_{t-1}, z_t]

where W is a learnable (A × A) transition matrix.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CRFParams(NamedTuple):
    """CRF transition parameters."""
    W: jnp.ndarray  # (A, A) — pairwise transition scores


def init_crf_params(n_ancestries: int) -> CRFParams:
    """Initialize CRF with identity-biased transitions.

    Self-transitions start higher, encouraging smooth tracts.
    """
    W = jnp.eye(n_ancestries) * 2.0
    return CRFParams(W=W)


# ---------------------------------------------------------------------------
# Forward algorithm (partition function)
# ---------------------------------------------------------------------------

def crf_log_partition(
    logits: jnp.ndarray,
    W: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log partition function via the forward algorithm.

    Parameters
    ----------
    logits : (batch, T, A) — unary potentials from CNN
    W : (A, A) — pairwise transition scores

    Returns
    -------
    log_Z : (batch,) — log partition function per sequence
    """
    B, T, A = logits.shape

    # Initialize: alpha_0[a] = logits[:, 0, a]
    alpha_0 = logits[:, 0, :]  # (B, A)

    def _step(alpha, t):
        # alpha: (B, A) — log-space forward scores at previous position
        # transition: alpha[b, a'] + W[a', a] for all a', then logsumexp over a'
        # result[b, a] = logsumexp_{a'}(alpha[b, a'] + W[a', a]) + logits[b, t, a]
        combined = alpha[:, :, None] + W[None, :, :]  # (B, A_from, A_to)
        alpha_new = jax.nn.logsumexp(combined, axis=1) + logits[:, t, :]  # (B, A)
        return alpha_new, None

    alpha_final, _ = jax.lax.scan(_step, alpha_0, jnp.arange(1, T))
    # log_Z = logsumexp over final states
    return jax.nn.logsumexp(alpha_final, axis=-1)  # (B,)


# ---------------------------------------------------------------------------
# Backward algorithm
# ---------------------------------------------------------------------------

def _crf_backward(
    logits: jnp.ndarray,
    W: jnp.ndarray,
) -> jnp.ndarray:
    """Backward pass for CRF marginals.

    Parameters
    ----------
    logits : (batch, T, A)
    W : (A, A)

    Returns
    -------
    log_beta : (batch, T, A) — log backward scores
    """
    B, T, A = logits.shape

    beta_T = jnp.zeros((B, A))  # (B, A)

    def _step(beta, t):
        # beta: (B, A) at position t+1
        # Need: beta_new[b, a] = logsumexp_{a'}(W[a, a'] + logits[b, t+1, a'] + beta[b, a'])
        integrand = logits[:, t + 1, :] + beta  # (B, A)
        combined = integrand[:, None, :] + W[None, :, :]  # (B, A_from, A_to)
        beta_new = jax.nn.logsumexp(combined, axis=2)  # (B, A)
        return beta_new, beta_new

    # Scan backwards: t from T-2 down to 0
    beta_0, betas_rev = jax.lax.scan(
        _step, beta_T, jnp.arange(T - 2, -1, -1),
    )
    # betas_rev: (T-1, B, A) in reverse order (T-2, T-3, ..., 0)
    # Flip to get (0, 1, ..., T-2), then append beta_T
    betas = jnp.flip(betas_rev, axis=0)  # (T-1, B, A) for positions 0..T-2
    # Add beta for position T-1
    log_beta = jnp.concatenate([betas, beta_T[None, :, :]], axis=0)  # (T, B, A)
    return jnp.transpose(log_beta, (1, 0, 2))  # (B, T, A)


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------

def crf_marginals(
    logits: jnp.ndarray,
    W: jnp.ndarray,
) -> jnp.ndarray:
    """Compute CRF marginal posteriors via forward-backward.

    Parameters
    ----------
    logits : (batch, T, A) — unary potentials
    W : (A, A) — transition scores

    Returns
    -------
    marginals : (batch, T, A) — P(z_t = a | x) at each position
    """
    B, T, A = logits.shape

    # Forward pass — collect all alphas
    alpha_0 = logits[:, 0, :]  # (B, A)

    def _fwd_step(alpha, t):
        combined = alpha[:, :, None] + W[None, :, :]  # (B, A_from, A_to)
        alpha_new = jax.nn.logsumexp(combined, axis=1) + logits[:, t, :]
        return alpha_new, alpha_new

    _, alphas_rest = jax.lax.scan(_fwd_step, alpha_0, jnp.arange(1, T))
    # alphas_rest: (T-1, B, A)
    log_alpha = jnp.concatenate([alpha_0[None, :, :], alphas_rest], axis=0)  # (T, B, A)
    log_alpha = jnp.transpose(log_alpha, (1, 0, 2))  # (B, T, A)

    # Backward pass
    log_beta = _crf_backward(logits, W)  # (B, T, A)

    # Marginals: alpha * beta, normalized
    log_gamma = log_alpha + log_beta
    log_Z = jax.nn.logsumexp(log_gamma, axis=-1, keepdims=True)
    return jnp.exp(log_gamma - log_Z)


# ---------------------------------------------------------------------------
# Log-likelihood (for training with hard labels)
# ---------------------------------------------------------------------------

def crf_log_likelihood(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    W: jnp.ndarray,
) -> jnp.ndarray:
    """Log-likelihood of hard label sequences under the CRF.

    Parameters
    ----------
    logits : (batch, T, A)
    labels : (batch, T) int — hard ancestry labels
    W : (A, A)

    Returns
    -------
    ll : (batch,) — log P(labels | logits, W) per sequence
    """
    B, T, A = logits.shape

    # Unary score: sum_t logits[b, t, labels[b, t]]
    unary = jnp.take_along_axis(
        logits, labels[:, :, None], axis=2,
    ).squeeze(-1).sum(axis=1)  # (B,)

    # Pairwise score: sum_t W[labels[b, t], labels[b, t+1]]
    prev_labels = labels[:, :-1]  # (B, T-1)
    next_labels = labels[:, 1:]   # (B, T-1)
    pairwise = W[prev_labels, next_labels].sum(axis=1)  # (B,)

    # Log partition function
    log_Z = crf_log_partition(logits, W)  # (B,)

    return unary + pairwise - log_Z


# ---------------------------------------------------------------------------
# CRF loss for soft pseudo-labels (used during training)
# ---------------------------------------------------------------------------

def crf_soft_loss(
    logits: jnp.ndarray,
    pseudo_labels: jnp.ndarray,
    crf_params: CRFParams,
) -> jnp.ndarray:
    """CRF regularization loss encouraging smooth transitions.

    Uses the expected pairwise score under pseudo-labels as a penalty
    for rapid ancestry switching.

    Parameters
    ----------
    logits : (batch, T, A)
    pseudo_labels : (batch, T, A)
    crf_params : CRFParams

    Returns
    -------
    loss : scalar — negative expected transition score (to be minimized)
    """
    # Expected pairwise score: Σ_t Σ_{i,j} p[t,i] * W[i,j] * p[t+1,j]
    p_prev = pseudo_labels[:, :-1, :]  # (B, T-1, A)
    p_next = pseudo_labels[:, 1:, :]   # (B, T-1, A)
    # Outer product across ancestries, weighted by transition scores
    expected_transition = jnp.einsum("bti,ij,btj->", p_prev, crf_params.W, p_next)
    n_elements = pseudo_labels.shape[0] * (pseudo_labels.shape[1] - 1)
    return -expected_transition / n_elements
