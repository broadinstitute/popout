"""Tests for the CNN model primitives."""

import jax
import jax.numpy as jnp
import pytest

from popout.cnn.model import (
    CNNConfig,
    CNNParams,
    cnn_forward,
    cnn_forward_checkpointed,
    cnn_posteriors,
    conv1d,
    init_cnn_params,
    layer_norm,
)
from popout.cnn.features import build_cnn_features


# ---------------------------------------------------------------------------
# conv1d
# ---------------------------------------------------------------------------

class TestConv1d:
    def test_shape(self):
        x = jnp.ones((4, 100, 10))  # (batch, T, C_in)
        w = jnp.ones((32, 10, 3))   # (C_out, C_in, K)
        b = jnp.zeros(32)
        y = conv1d(x, w, b)
        assert y.shape == (4, 100, 32)

    def test_same_padding_preserves_length(self):
        """SAME padding should keep T unchanged regardless of dilation."""
        x = jnp.ones((2, 50, 8))
        w = jnp.ones((16, 8, 3))
        b = jnp.zeros(16)
        for dilation in [1, 2, 4, 8, 16]:
            y = conv1d(x, w, b, dilation=dilation)
            assert y.shape == (2, 50, 16), f"failed for dilation={dilation}"

    def test_kernel_1(self):
        """Kernel size 1 should be a pointwise linear transform."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 20, 4))
        w = jax.random.normal(key, (8, 4, 1))
        b = jnp.zeros(8)
        y = conv1d(x, w, b)
        # Equivalent to x @ w[:, :, 0].T + b
        expected = x @ w[:, :, 0].T + b[None, None, :]
        assert jnp.allclose(y, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# layer_norm
# ---------------------------------------------------------------------------

class TestLayerNorm:
    def test_output_normalized(self):
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (4, 50, 64)) * 5.0 + 3.0
        scale = jnp.ones(64)
        bias = jnp.zeros(64)
        y = layer_norm(x, scale, bias)
        # Mean should be ~0, std ~1 along channel dim
        assert jnp.allclose(y.mean(axis=-1), 0.0, atol=1e-4)
        assert jnp.allclose(y.std(axis=-1), 1.0, atol=1e-2)

    def test_affine_transform(self):
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (2, 10, 8))
        scale = jnp.full(8, 2.0)
        bias = jnp.full(8, 1.0)
        y = layer_norm(x, scale, bias)
        # After affine: mean ≈ 1, std ≈ 2
        assert jnp.allclose(y.mean(axis=-1), 1.0, atol=1e-4)
        assert jnp.allclose(y.std(axis=-1), 2.0, atol=1e-2)


# ---------------------------------------------------------------------------
# CNN forward
# ---------------------------------------------------------------------------

class TestCNNForward:
    @pytest.fixture
    def small_config(self):
        return CNNConfig(n_layers=3, hidden_dim=16, kernel_size=3,
                         n_ancestries=4, c_in=6)

    @pytest.fixture
    def small_params(self, small_config):
        return init_cnn_params(small_config, jax.random.PRNGKey(42))

    def test_logits_shape(self, small_config, small_params):
        x = jnp.ones((8, 100, small_config.c_in))
        logits = cnn_forward(small_params, small_config, x)
        assert logits.shape == (8, 100, small_config.n_ancestries)

    def test_posteriors_sum_to_one(self, small_config, small_params):
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 50, small_config.c_in))
        post = cnn_posteriors(small_params, small_config, x)
        assert post.shape == (4, 50, small_config.n_ancestries)
        sums = post.sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_posteriors_non_negative(self, small_config, small_params):
        x = jax.random.normal(jax.random.PRNGKey(1), (4, 50, small_config.c_in))
        post = cnn_posteriors(small_params, small_config, x)
        assert (post >= 0).all()

    def test_checkpointed_matches_forward(self, small_config, small_params):
        x = jax.random.normal(jax.random.PRNGKey(3), (4, 50, small_config.c_in))
        logits_std = cnn_forward(small_params, small_config, x)
        logits_ckpt = cnn_forward_checkpointed(small_params, small_config, x)
        assert jnp.allclose(logits_std, logits_ckpt, atol=1e-5)

    def test_gradient_flows(self, small_config, small_params):
        """jax.grad through cnn_forward produces finite gradients."""
        x = jax.random.normal(jax.random.PRNGKey(4), (2, 30, small_config.c_in))

        def loss_fn(params):
            logits = cnn_forward(params, small_config, x)
            return logits.sum()

        grads = jax.grad(loss_fn)(small_params)
        for leaf in jax.tree.leaves(grads):
            assert jnp.isfinite(leaf).all(), "gradient contains non-finite values"
            assert not jnp.allclose(leaf, 0.0), "gradient is all zeros"


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

class TestParams:
    def test_param_count_default(self):
        """Default config (~600KB) parameter count."""
        config = CNNConfig(n_layers=12, hidden_dim=64, kernel_size=3,
                           n_ancestries=8, c_in=10)
        params = init_cnn_params(config, jax.random.PRNGKey(0))
        total = sum(p.size for p in jax.tree.leaves(params))
        # Stem: 64*10*1 + 64 = 704
        # Blocks: 12*(64*64*3 + 64 + 64 + 64) = 12*12480 = 149760
        # Head: 8*64*1 + 8 = 520
        expected = 704 + 149760 + 520
        assert total == expected

    def test_init_shapes(self):
        config = CNNConfig(n_layers=4, hidden_dim=32, kernel_size=3,
                           n_ancestries=6, c_in=8)
        params = init_cnn_params(config, jax.random.PRNGKey(1))
        assert params.stem_weight.shape == (32, 8, 1)
        assert params.stem_bias.shape == (32,)
        assert params.block_weights.shape == (4, 32, 32, 3)
        assert params.block_biases.shape == (4, 32)
        assert params.block_ln_scales.shape == (4, 32)
        assert params.block_ln_biases.shape == (4, 32)
        assert params.head_weight.shape == (6, 32, 1)
        assert params.head_bias.shape == (6,)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_feature_shape(self):
        H, T, A = 100, 500, 6
        geno = jnp.zeros((H, T), dtype=jnp.uint8)
        freq = jnp.ones((A, T)) * 0.5
        d_morgan = jnp.ones(T - 1) * 0.001
        features = build_cnn_features(geno, freq, d_morgan)
        assert features.shape == (H, T, A + 2)

    def test_allele_channel(self):
        H, T, A = 10, 20, 3
        geno = jnp.ones((H, T), dtype=jnp.uint8)
        freq = jnp.ones((A, T)) * 0.5
        d_morgan = jnp.ones(T - 1) * 0.001
        features = build_cnn_features(geno, freq, d_morgan)
        # Channel 0 should be allele values
        assert jnp.allclose(features[:, :, 0], 1.0)

    def test_distance_channel_last_site_zero(self):
        H, T, A = 10, 20, 3
        geno = jnp.zeros((H, T), dtype=jnp.uint8)
        freq = jnp.ones((A, T)) * 0.5
        d_morgan = jnp.ones(T - 1) * 0.05
        features = build_cnn_features(geno, freq, d_morgan)
        # Last channel is genetic distance; last site should be 0
        assert features[0, -1, -1] == 0.0
        assert features[0, 0, -1] == 0.05
