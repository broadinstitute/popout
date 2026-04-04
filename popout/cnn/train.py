"""Training loop and optimizer for the CNN refinement backend.

Implements:
- Adam optimizer (pure JAX, no external dependency)
- Cosine learning rate schedule with linear warmup
- Confidence-weighted KL divergence loss
- Mini-batch training loop with optional CRF loss
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from .model import CNNConfig, CNNParams, cnn_forward, cnn_forward_checkpointed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

class AdamState(NamedTuple):
    """Adam optimizer state — pytrees matching the parameter structure."""
    step: int
    m: CNNParams   # first moment estimates
    v: CNNParams   # second moment estimates


def init_adam(params: CNNParams) -> AdamState:
    """Initialize Adam state with zeros matching parameter shapes."""
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return AdamState(step=0, m=m, v=v)


def adam_step(
    params: CNNParams,
    grads: CNNParams,
    state: AdamState,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[CNNParams, AdamState]:
    """Single Adam update step."""
    step = state.step + 1

    # Update moments
    m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, state.m, grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g ** 2, state.v, grads)

    # Bias correction
    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step
    m_hat = jax.tree.map(lambda m: m / bc1, m)
    v_hat = jax.tree.map(lambda v: v / bc2, v)

    # Parameter update
    new_params = jax.tree.map(
        lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps),
        params, m_hat, v_hat,
    )
    return new_params, AdamState(step=step, m=m, v=v)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def cosine_lr(
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 100,
) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return base_lr * 0.5 * (1.0 + float(jnp.cos(jnp.pi * progress)))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def kl_loss(
    pseudo_labels: jnp.ndarray,
    logits: jnp.ndarray,
    confidence_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Confidence-weighted cross-entropy loss (equivalent to KL up to constant).

    Computes  -Σ_{h,t,a}  w[h,t] · p[h,t,a] · log_softmax(logits)[h,t,a]

    where p = pseudo_labels and w = confidence_weights.

    Parameters
    ----------
    pseudo_labels : (batch, T, A) — target soft labels
    logits : (batch, T, A) — CNN output logits
    confidence_weights : (batch, T) or None — per-site confidence

    Returns
    -------
    loss : scalar — mean cross-entropy per site
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, T, A)
    # Per-site cross-entropy: -sum_a p * log q
    ce = -(pseudo_labels * log_probs).sum(axis=-1)    # (B, T)

    if confidence_weights is not None:
        ce = ce * confidence_weights

    return ce.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnn(
    params: CNNParams,
    config: CNNConfig,
    features: jnp.ndarray,
    pseudo_labels: jnp.ndarray,
    n_epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    key: Optional[jax.Array] = None,
    use_confidence_weights: bool = True,
    crf_params=None,
    crf_loss_fn=None,
    stats=None,
    use_checkpointing: bool = False,
) -> tuple[CNNParams, object]:
    """Train CNN on pseudo-labels with mini-batch Adam.

    Parameters
    ----------
    params : CNNParams — initial model parameters
    config : CNNConfig
    features : (H, T, C_in) — precomputed input features
    pseudo_labels : (H, T, A) — target posteriors
    n_epochs : training epochs per pseudo-label round
    batch_size : haplotypes per mini-batch
    lr : base learning rate
    key : PRNG key for shuffling
    use_confidence_weights : weight loss by max pseudo-label confidence
    crf_params : optional CRFParams for joint training
    crf_loss_fn : optional callable(logits, pseudo_labels, crf_params) → scalar
    stats : optional StatsCollector
    use_checkpointing : use gradient checkpointing for memory efficiency

    Returns
    -------
    (params, crf_params) — updated parameters
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    H = features.shape[0]
    n_batches = max(1, (H + batch_size - 1) // batch_size)
    total_steps = n_epochs * n_batches

    # Precompute confidence weights
    confidence_weights = None
    if use_confidence_weights:
        confidence_weights = pseudo_labels.max(axis=-1)  # (H, T)

    # Choose forward function
    fwd_fn = cnn_forward_checkpointed if use_checkpointing else cnn_forward

    # Build loss function
    def loss_fn(params, crf_params, batch_features, batch_labels, batch_weights):
        logits = fwd_fn(params, config, batch_features)
        loss = kl_loss(batch_labels, logits, batch_weights)
        if crf_params is not None and crf_loss_fn is not None:
            loss = loss + crf_loss_fn(logits, batch_labels, crf_params)
        return loss

    # Initialize optimizer
    opt_state = init_adam(params)
    if crf_params is not None:
        crf_opt_state = init_adam(crf_params)

    step = 0
    for epoch in range(n_epochs):
        # Shuffle haplotype indices
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, H)

        epoch_loss = 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, H)
            idx = perm[start:end]

            batch_feat = features[idx]
            batch_labels = pseudo_labels[idx]
            batch_weights = confidence_weights[idx] if confidence_weights is not None else None

            # Compute gradients
            if crf_params is not None:
                grad_fn = jax.grad(loss_fn, argnums=(0, 1))
                (p_grads, c_grads) = grad_fn(
                    params, crf_params, batch_feat, batch_labels, batch_weights,
                )
            else:
                grad_fn = jax.grad(loss_fn, argnums=0)
                p_grads = grad_fn(
                    params, crf_params, batch_feat, batch_labels, batch_weights,
                )

            # Learning rate
            current_lr = cosine_lr(step, lr, total_steps, warmup_steps=min(100, total_steps // 10))

            # Update CNN params
            params, opt_state = adam_step(params, p_grads, opt_state, current_lr)

            # Update CRF params
            if crf_params is not None:
                crf_params, crf_opt_state = adam_step(crf_params, c_grads, crf_opt_state, current_lr)

            # Track loss (evaluate without grad for logging)
            batch_loss = float(loss_fn(params, crf_params, batch_feat, batch_labels, batch_weights))
            epoch_loss += batch_loss

            step += 1

        mean_loss = epoch_loss / n_batches
        log.info("  CNN epoch %d/%d: loss=%.4f, lr=%.2e", epoch + 1, n_epochs, mean_loss, current_lr)
        if stats is not None:
            stats.emit("cnn/train_loss", mean_loss, epoch=epoch)

    return params, crf_params
