"""Dilated residual 1D CNN for local ancestry inference.

Architecture: WaveNet / Temporal Convolutional Network (TCN) with
exponentially increasing dilation rates.  Each residual block applies
LayerNorm → dilated conv1d → GELU → skip connection.  A stem projects
input channels to the hidden dimension and a 1×1 head projects to
per-ancestry logits.

All operations use pure JAX (no external NN frameworks).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Config & parameter containers
# ---------------------------------------------------------------------------

class CNNConfig(NamedTuple):
    """Architecture hyperparameters."""
    n_layers: int = 12
    hidden_dim: int = 64
    kernel_size: int = 3
    n_ancestries: int = 0   # set at runtime
    c_in: int = 0           # set at runtime: A + 2


class CNNParams(NamedTuple):
    """All learnable parameters for the dilated CNN.

    Weights are stored as stacked arrays so that they are valid JAX
    pytrees and work with ``jax.grad`` / ``jax.tree.map`` out of the box.
    """
    # Stem: projects C_in → hidden_dim  (kernel_size=1)
    stem_weight: jnp.ndarray    # (hidden_dim, C_in, 1)
    stem_bias: jnp.ndarray      # (hidden_dim,)
    # Residual blocks
    block_weights: jnp.ndarray  # (n_layers, hidden_dim, hidden_dim, kernel_size)
    block_biases: jnp.ndarray   # (n_layers, hidden_dim)
    block_ln_scales: jnp.ndarray   # (n_layers, hidden_dim)
    block_ln_biases: jnp.ndarray   # (n_layers, hidden_dim)
    # Head: projects hidden_dim → n_ancestries  (kernel_size=1)
    head_weight: jnp.ndarray    # (n_ancestries, hidden_dim, 1)
    head_bias: jnp.ndarray      # (n_ancestries,)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_cnn_params(config: CNNConfig, key: jax.Array) -> CNNParams:
    """Initialize CNN parameters with He (Kaiming) initialization."""
    keys = jax.random.split(key, 2 + config.n_layers)
    ki = 0

    # Stem
    fan_in = config.c_in * 1
    stem_weight = jax.random.normal(keys[ki], (config.hidden_dim, config.c_in, 1)) * (2.0 / fan_in) ** 0.5
    stem_bias = jnp.zeros(config.hidden_dim)
    ki += 1

    # Residual blocks
    fan_in_block = config.hidden_dim * config.kernel_size
    block_weights = jnp.stack([
        jax.random.normal(keys[ki + i], (config.hidden_dim, config.hidden_dim, config.kernel_size))
        * (2.0 / fan_in_block) ** 0.5
        for i in range(config.n_layers)
    ])
    block_biases = jnp.zeros((config.n_layers, config.hidden_dim))
    block_ln_scales = jnp.ones((config.n_layers, config.hidden_dim))
    block_ln_biases = jnp.zeros((config.n_layers, config.hidden_dim))
    ki += config.n_layers

    # Head
    fan_in_head = config.hidden_dim * 1
    head_weight = jax.random.normal(keys[ki], (config.n_ancestries, config.hidden_dim, 1)) * (2.0 / fan_in_head) ** 0.5
    head_bias = jnp.zeros(config.n_ancestries)

    return CNNParams(
        stem_weight=stem_weight,
        stem_bias=stem_bias,
        block_weights=block_weights,
        block_biases=block_biases,
        block_ln_scales=block_ln_scales,
        block_ln_biases=block_ln_biases,
        head_weight=head_weight,
        head_bias=head_bias,
    )


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def conv1d(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    dilation: int = 1,
) -> jnp.ndarray:
    """1D convolution with SAME padding.

    Parameters
    ----------
    x : (batch, T, C_in)
    weight : (C_out, C_in, K)
    bias : (C_out,)
    dilation : dilation rate

    Returns
    -------
    y : (batch, T, C_out)
    """
    # jax.lax.conv_general_dilated expects:
    #   lhs: (batch, spatial..., channels)  with dimension_numbers NTC
    #   rhs: (out_channels, in_channels, spatial...) with OIT
    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=weight,
        window_strides=(1,),
        padding="SAME",
        lhs_dilation=None,
        rhs_dilation=(dilation,),
        dimension_numbers=("NTC", "OIT", "NTC"),
    )
    return y + bias[None, None, :]


def layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Layer normalization over the channel (last) dimension.

    Parameters
    ----------
    x : (batch, T, C)
    scale : (C,)
    bias : (C,)
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return x_norm * scale[None, None, :] + bias[None, None, :]


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def cnn_forward(
    params: CNNParams,
    config: CNNConfig,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Full CNN forward pass.

    Parameters
    ----------
    params : CNNParams
    config : CNNConfig
    x : (batch, T, C_in) — input features

    Returns
    -------
    logits : (batch, T, A) — raw logits before softmax
    """
    # Stem: project input channels to hidden dimension
    h = conv1d(x, params.stem_weight, params.stem_bias)  # (B, T, hidden)

    # Residual blocks with exponentially increasing dilation
    for i in range(config.n_layers):
        dilation = 2 ** i
        residual = h
        h = layer_norm(h, params.block_ln_scales[i], params.block_ln_biases[i])
        h = conv1d(h, params.block_weights[i], params.block_biases[i], dilation=dilation)
        h = jax.nn.gelu(h)
        h = h + residual

    # Head: project to per-ancestry logits
    logits = conv1d(h, params.head_weight, params.head_bias)  # (B, T, A)
    return logits


def cnn_forward_checkpointed(
    params: CNNParams,
    config: CNNConfig,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Forward pass with gradient checkpointing per residual block.

    Reduces activation memory from O(n_layers * batch * T * hidden) to
    O(batch * T * hidden) at the cost of recomputing each block's
    activations during backpropagation.
    """
    h = conv1d(x, params.stem_weight, params.stem_bias)

    for i in range(config.n_layers):
        dilation = 2 ** i

        @jax.checkpoint
        def _block(h_in, w, b, ln_s, ln_b):
            residual = h_in
            h_out = layer_norm(h_in, ln_s, ln_b)
            h_out = conv1d(h_out, w, b, dilation=dilation)
            h_out = jax.nn.gelu(h_out)
            return h_out + residual

        h = _block(
            h,
            params.block_weights[i],
            params.block_biases[i],
            params.block_ln_scales[i],
            params.block_ln_biases[i],
        )

    logits = conv1d(h, params.head_weight, params.head_bias)
    return logits


def cnn_posteriors(
    params: CNNParams,
    config: CNNConfig,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """CNN forward pass → softmax posteriors.

    Returns
    -------
    posteriors : (batch, T, A) — ancestry probabilities summing to 1
    """
    logits = cnn_forward(params, config, x)
    return jax.nn.softmax(logits, axis=-1)
