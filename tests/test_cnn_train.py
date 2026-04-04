"""Tests for CNN training infrastructure."""

import jax
import jax.numpy as jnp
import pytest

from popout.cnn.model import CNNConfig, cnn_forward, init_cnn_params
from popout.cnn.train import (
    AdamState,
    adam_step,
    cosine_lr,
    init_adam,
    kl_loss,
    train_cnn,
)


@pytest.fixture
def small_setup():
    config = CNNConfig(n_layers=2, hidden_dim=8, kernel_size=3,
                       n_ancestries=3, c_in=5)
    params = init_cnn_params(config, jax.random.PRNGKey(0))
    return config, params


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

class TestAdam:
    def test_init_shapes(self, small_setup):
        _, params = small_setup
        state = init_adam(params)
        assert state.step == 0
        for m_leaf, p_leaf in zip(jax.tree.leaves(state.m), jax.tree.leaves(params)):
            assert m_leaf.shape == p_leaf.shape
            assert jnp.allclose(m_leaf, 0.0)

    def test_step_changes_params(self, small_setup):
        _, params = small_setup
        state = init_adam(params)
        # Fake gradients
        grads = jax.tree.map(jnp.ones_like, params)
        new_params, new_state = adam_step(params, grads, state, lr=0.01)
        assert new_state.step == 1
        # Params should have changed
        for p_old, p_new in zip(jax.tree.leaves(params), jax.tree.leaves(new_params)):
            assert not jnp.allclose(p_old, p_new)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

class TestCosineLR:
    def test_warmup(self):
        assert cosine_lr(0, 1.0, 1000, warmup_steps=100) == 0.0
        assert abs(cosine_lr(50, 1.0, 1000, warmup_steps=100) - 0.5) < 1e-6
        assert abs(cosine_lr(100, 1.0, 1000, warmup_steps=100) - 1.0) < 1e-6

    def test_decay(self):
        # At the end, lr should be ~0
        lr_end = cosine_lr(999, 1.0, 1000, warmup_steps=0)
        assert lr_end < 0.01

    def test_midpoint(self):
        # At halfway, cosine decay gives ~0.5 * base_lr
        lr_mid = cosine_lr(500, 1.0, 1000, warmup_steps=0)
        assert abs(lr_mid - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestKLLoss:
    def test_zero_for_perfect_match(self):
        """When logits produce probs matching pseudo-labels, loss is minimal."""
        # Create pseudo-labels that are one-hot
        labels = jnp.array([[[1, 0, 0], [0, 1, 0]]]).astype(jnp.float32)
        # Logits that will produce near-one-hot after softmax
        logits = jnp.array([[[10.0, -10, -10], [-10, 10.0, -10]]])
        loss = kl_loss(labels, logits)
        assert loss < 0.001

    def test_positive_for_mismatch(self):
        labels = jnp.array([[[1, 0, 0], [0, 1, 0]]]).astype(jnp.float32)
        logits = jnp.array([[[-10.0, 10, -10], [10.0, -10, -10]]])
        loss = kl_loss(labels, logits)
        assert loss > 1.0

    def test_confidence_weighting(self):
        """High-confidence sites should contribute more to loss."""
        labels = jnp.array([[[0.9, 0.05, 0.05], [0.34, 0.33, 0.33]]]).astype(jnp.float32)
        logits = jnp.zeros_like(labels)  # uniform predictions

        # Without weighting
        loss_unweighted = kl_loss(labels, logits)

        # With weighting (second site has low confidence, should be downweighted)
        weights = jnp.array([[0.9, 0.34]])
        loss_weighted = kl_loss(labels, logits, confidence_weights=weights)

        # Weighted loss should differ from unweighted
        assert not jnp.allclose(loss_weighted, loss_unweighted)

    def test_gradient_finite(self):
        labels = jax.random.dirichlet(jax.random.PRNGKey(0), jnp.ones(4), shape=(8, 50))
        logits = jax.random.normal(jax.random.PRNGKey(1), (8, 50, 4))
        grad = jax.grad(kl_loss, argnums=1)(labels, logits)
        assert jnp.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class TestTrainCNN:
    def test_loss_decreases(self, small_setup):
        """Training should reduce loss over a few epochs."""
        config, params = small_setup
        key = jax.random.PRNGKey(42)

        H, T = 32, 20
        features = jax.random.normal(key, (H, T, config.c_in))
        # Create pseudo-labels: softmax of random logits
        pseudo_labels = jax.nn.softmax(
            jax.random.normal(jax.random.PRNGKey(1), (H, T, config.n_ancestries)),
            axis=-1,
        )

        # Measure initial loss
        logits_before = cnn_forward(params, config, features)
        loss_before = float(kl_loss(pseudo_labels, logits_before))

        # Train
        trained_params, _ = train_cnn(
            params, config, features, pseudo_labels,
            n_epochs=5, batch_size=16, lr=1e-3,
            key=jax.random.PRNGKey(2),
        )

        # Measure final loss
        logits_after = cnn_forward(trained_params, config, features)
        loss_after = float(kl_loss(pseudo_labels, logits_after))

        assert loss_after < loss_before, f"Loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"

    def test_runs_with_small_data(self, small_setup):
        """Training loop completes on tiny dataset."""
        config, params = small_setup
        features = jnp.ones((4, 10, config.c_in))
        labels = jax.nn.softmax(jnp.ones((4, 10, config.n_ancestries)), axis=-1)

        trained_params, _ = train_cnn(
            params, config, features, labels,
            n_epochs=1, batch_size=4, lr=1e-3,
        )
        # Just check it returns valid params
        for leaf in jax.tree.leaves(trained_params):
            assert jnp.isfinite(leaf).all()
