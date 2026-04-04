"""Tests for the CRF output layer."""

import itertools

import jax
import jax.numpy as jnp
import pytest

from popout.cnn.crf import (
    CRFParams,
    crf_log_likelihood,
    crf_log_partition,
    crf_marginals,
    crf_soft_loss,
    init_crf_params,
)


class TestCRFPartition:
    def test_matches_brute_force(self):
        """For small T and A, compare CRF partition to brute-force enumeration."""
        T, A = 4, 3
        key = jax.random.PRNGKey(0)
        logits = jax.random.normal(key, (1, T, A))
        W = jax.random.normal(jax.random.PRNGKey(1), (A, A))

        # Brute force: enumerate all A^T sequences
        log_Z_bf = -jnp.inf
        for seq in itertools.product(range(A), repeat=T):
            seq = list(seq)
            score = sum(logits[0, t, seq[t]] for t in range(T))
            score += sum(W[seq[t], seq[t + 1]] for t in range(T - 1))
            log_Z_bf = jnp.logaddexp(log_Z_bf, score)

        log_Z = crf_log_partition(logits, W)
        assert jnp.allclose(log_Z[0], log_Z_bf, atol=1e-4), \
            f"partition mismatch: {float(log_Z[0]):.4f} vs {float(log_Z_bf):.4f}"

    def test_batched(self):
        """Partition function should work on batched inputs."""
        B, T, A = 4, 10, 3
        logits = jax.random.normal(jax.random.PRNGKey(0), (B, T, A))
        W = jax.random.normal(jax.random.PRNGKey(1), (A, A))
        log_Z = crf_log_partition(logits, W)
        assert log_Z.shape == (B,)
        assert jnp.isfinite(log_Z).all()


class TestCRFMarginals:
    def test_sum_to_one(self):
        """CRF marginals should sum to 1 at each position."""
        B, T, A = 4, 20, 5
        logits = jax.random.normal(jax.random.PRNGKey(0), (B, T, A))
        W = jax.random.normal(jax.random.PRNGKey(1), (A, A)) * 0.5
        marginals = crf_marginals(logits, W)
        assert marginals.shape == (B, T, A)
        sums = marginals.sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-4)

    def test_non_negative(self):
        logits = jax.random.normal(jax.random.PRNGKey(2), (2, 15, 4))
        W = jax.random.normal(jax.random.PRNGKey(3), (4, 4))
        marginals = crf_marginals(logits, W)
        assert (marginals >= -1e-6).all()

    def test_matches_brute_force(self):
        """For small T/A, compare CRF marginals to brute-force."""
        T, A = 4, 3
        key = jax.random.PRNGKey(10)
        logits = jax.random.normal(key, (1, T, A))
        W = jax.random.normal(jax.random.PRNGKey(11), (A, A))

        # Brute force marginals
        bf_marginals = jnp.zeros((T, A))
        for seq in itertools.product(range(A), repeat=T):
            seq = list(seq)
            score = sum(logits[0, t, seq[t]] for t in range(T))
            score += sum(W[seq[t], seq[t + 1]] for t in range(T - 1))
            prob = jnp.exp(score)
            for t in range(T):
                bf_marginals = bf_marginals.at[t, seq[t]].add(prob)

        bf_marginals = bf_marginals / bf_marginals.sum(axis=-1, keepdims=True)

        marginals = crf_marginals(logits, W)
        assert jnp.allclose(marginals[0], bf_marginals, atol=1e-4)


class TestCRFLogLikelihood:
    def test_bounded_above_by_zero(self):
        """Log-likelihood should be <= 0."""
        B, T, A = 4, 20, 5
        logits = jax.random.normal(jax.random.PRNGKey(0), (B, T, A))
        labels = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, A)
        W = jax.random.normal(jax.random.PRNGKey(2), (A, A))
        ll = crf_log_likelihood(logits, labels, W)
        assert (ll <= 1e-5).all()

    def test_gradient_flows(self):
        """Gradients through CRF log-likelihood should be finite."""
        B, T, A = 2, 10, 3
        logits = jax.random.normal(jax.random.PRNGKey(0), (B, T, A))
        labels = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, A)
        W = jax.random.normal(jax.random.PRNGKey(2), (A, A))

        def loss(W):
            return -crf_log_likelihood(logits, labels, W).mean()

        grad = jax.grad(loss)(W)
        assert jnp.isfinite(grad).all()


class TestCRFSoftLoss:
    def test_finite(self):
        B, T, A = 4, 20, 5
        logits = jax.random.normal(jax.random.PRNGKey(0), (B, T, A))
        pseudo = jax.nn.softmax(logits, axis=-1)
        crf_params = init_crf_params(A)
        loss = crf_soft_loss(logits, pseudo, crf_params)
        assert jnp.isfinite(loss)

    def test_gradient_flows(self):
        B, T, A = 2, 10, 3
        logits = jax.random.normal(jax.random.PRNGKey(0), (B, T, A))
        pseudo = jax.nn.softmax(logits, axis=-1)
        crf_params = init_crf_params(A)

        def loss(W):
            return crf_soft_loss(logits, pseudo, CRFParams(W=W))

        grad = jax.grad(loss)(crf_params.W)
        assert jnp.isfinite(grad).all()
