"""Integration tests for the CNN refinement pipeline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from popout.simulate import simulate_admixed
from popout.datatypes import AncestryResult
from popout.em import update_allele_freq, update_mu


class TestCNNPosteriorsFeedMStep:
    """CNN posteriors should be valid input for the M-step functions."""

    def test_update_allele_freq(self):
        from popout.cnn.model import CNNConfig, cnn_posteriors, init_cnn_params

        H, T, A = 50, 100, 4
        config = CNNConfig(n_layers=2, hidden_dim=8, kernel_size=3,
                           n_ancestries=A, c_in=A + 2)
        params = init_cnn_params(config, jax.random.PRNGKey(0))

        features = jax.random.normal(jax.random.PRNGKey(1), (H, T, config.c_in))
        gamma = cnn_posteriors(params, config, features)

        geno = jax.random.bernoulli(jax.random.PRNGKey(2), 0.3, (H, T)).astype(jnp.uint8)
        freq = update_allele_freq(geno, gamma)

        assert freq.shape == (A, T)
        assert (freq >= 0).all() and (freq <= 1).all()

    def test_update_mu(self):
        from popout.cnn.model import CNNConfig, cnn_posteriors, init_cnn_params

        H, T, A = 50, 100, 4
        config = CNNConfig(n_layers=2, hidden_dim=8, kernel_size=3,
                           n_ancestries=A, c_in=A + 2)
        params = init_cnn_params(config, jax.random.PRNGKey(0))
        features = jax.random.normal(jax.random.PRNGKey(1), (H, T, config.c_in))
        gamma = cnn_posteriors(params, config, features)

        mu = update_mu(gamma)
        assert mu.shape == (A,)
        assert jnp.allclose(mu.sum(), 1.0, atol=1e-5)


class TestRunCNNSimulated:
    """Full CNN pipeline on simulated data."""

    def test_produces_valid_result(self):
        chrom_data, true_ancestry = simulate_admixed(
            n_samples=50, n_sites=200, n_ancestries=3,
            gen_since_admix=20, rng_seed=42,
        )

        from popout.cnn.refine import run_cnn

        result, params, crf_params = run_cnn(
            chrom_data,
            n_ancestries=3,
            gen_since_admix=20.0,
            hmm_batch_size=1000,
            rng_seed=42,
            n_layers=3,
            hidden_dim=8,
            n_epochs=2,
            n_pseudo_rounds=1,
            cnn_lr=1e-3,
            cnn_batch_size=50,
            use_crf=False,
        )

        assert isinstance(result, AncestryResult)
        assert result.calls.shape == (chrom_data.n_haps, chrom_data.n_sites)
        assert result.calls.dtype == np.int8
        assert set(np.unique(result.calls)).issubset({0, 1, 2})

        # Decode result should have pre-computed reductions
        assert result.decode is not None
        assert result.decode.max_post.shape == (chrom_data.n_haps, chrom_data.n_sites)
        assert result.decode.global_sums.shape == (chrom_data.n_haps, 3)

        # Model should have valid parameters
        assert result.model.n_ancestries == 3
        assert result.model.allele_freq.shape == (3, chrom_data.n_sites)
        assert jnp.allclose(result.model.mu.sum(), 1.0, atol=1e-4)

    def test_cnn_crf_runs(self):
        """CNN-CRF pipeline completes on simulated data."""
        chrom_data, _ = simulate_admixed(
            n_samples=30, n_sites=100, n_ancestries=3,
            gen_since_admix=20, rng_seed=99,
        )

        from popout.cnn.refine import run_cnn

        result, params, crf_params = run_cnn(
            chrom_data,
            n_ancestries=3,
            gen_since_admix=20.0,
            hmm_batch_size=1000,
            rng_seed=99,
            n_layers=2,
            hidden_dim=8,
            n_epochs=1,
            n_pseudo_rounds=1,
            cnn_lr=1e-3,
            cnn_batch_size=30,
            use_crf=True,
        )

        assert isinstance(result, AncestryResult)
        assert result.calls.shape == (chrom_data.n_haps, chrom_data.n_sites, )
        assert result.decode is not None
        assert crf_params is not None


class TestOutputFormatMatch:
    """CNN and HMM outputs should have compatible shapes/dtypes."""

    def test_matches_hmm_output_format(self):
        chrom_data, _ = simulate_admixed(
            n_samples=30, n_sites=100, n_ancestries=3,
            gen_since_admix=20, rng_seed=42,
        )

        from popout.em import run_em
        from popout.cnn.refine import run_cnn

        hmm_result = run_em(
            chrom_data, n_ancestries=3, n_em_iter=1,
            gen_since_admix=20.0, batch_size=1000, rng_seed=42,
        )

        cnn_result, _, _ = run_cnn(
            chrom_data, n_ancestries=3, gen_since_admix=20.0,
            hmm_batch_size=1000, rng_seed=42,
            n_layers=2, hidden_dim=8, n_epochs=1,
            n_pseudo_rounds=1, cnn_batch_size=30,
        )

        # Same shapes
        assert hmm_result.calls.shape == cnn_result.calls.shape
        assert hmm_result.calls.dtype == cnn_result.calls.dtype

        # Both should have decode results
        assert hmm_result.decode is not None
        assert cnn_result.decode is not None

        # Same model structure
        assert hmm_result.model.n_ancestries == cnn_result.model.n_ancestries
        assert hmm_result.model.allele_freq.shape == cnn_result.model.allele_freq.shape
        assert hmm_result.model.mu.shape == cnn_result.model.mu.shape
