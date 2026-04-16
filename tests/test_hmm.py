"""Tests for the HMM module, including gradient checkpointing equivalence."""

import numpy as np
import jax.numpy as jnp

from popout.simulate import simulate_admixed
from popout.hmm import (
    forward,
    backward,
    posteriors,
    forward_backward,
    forward_backward_checkpointed,
    forward_backward_batched,
)
from popout.em import init_model_soft
from popout.spectral import seed_ancestry_soft


def _make_model(n_samples=200, n_sites=100, n_ancestries=3, rng_seed=42):
    """Build a small model + data for testing."""
    chrom_data, true_ancestry = simulate_admixed(
        n_samples=n_samples,
        n_sites=n_sites,
        n_ancestries=n_ancestries,
        rng_seed=rng_seed,
    )
    geno = jnp.array(chrom_data.geno)
    d_morgan = jnp.array(chrom_data.genetic_distances)

    _, resp, n_anc, _proj = seed_ancestry_soft(
        chrom_data.geno, n_ancestries=n_ancestries, rng_seed=rng_seed,
    )
    model = init_model_soft(geno, resp, n_anc)
    return geno, model, d_morgan, true_ancestry


def test_checkpointed_matches_full():
    """Checkpointed forward-backward produces identical posteriors."""
    geno, model, d_morgan, _ = _make_model(n_samples=100, n_sites=200)

    # Full (non-checkpointed) path
    gamma_full = forward_backward(geno, model, d_morgan, use_checkpointing=False)

    # Checkpointed path
    gamma_ckpt = forward_backward_checkpointed(geno, model, d_morgan)

    np.testing.assert_allclose(
        np.array(gamma_full), np.array(gamma_ckpt), atol=1e-5,
        err_msg="Checkpointed posteriors differ from full forward-backward",
    )


def test_checkpointed_various_intervals():
    """Checkpointed path works with different checkpoint intervals."""
    geno, model, d_morgan, _ = _make_model(n_samples=50, n_sites=100)

    gamma_ref = forward_backward(geno, model, d_morgan, use_checkpointing=False)

    for C in [2, 3, 5, 10, 50, 99, 100]:
        gamma_ckpt = forward_backward_checkpointed(
            geno, model, d_morgan, checkpoint_interval=C,
        )
        np.testing.assert_allclose(
            np.array(gamma_ref), np.array(gamma_ckpt), atol=1e-5,
            err_msg=f"Mismatch at checkpoint_interval={C}",
        )


def test_checkpointed_small_T():
    """Edge cases: various small T values."""
    from popout.datatypes import AncestryModel

    # Build a synthetic model directly (spectral init fails for very small T)
    A = 3
    H = 40
    rng = np.random.default_rng(123)

    for T in [2, 3, 4, 5, 10, 20]:
        geno = jnp.array(rng.integers(0, 2, size=(H, T), dtype=np.uint8))
        d_morgan = jnp.array(rng.uniform(0.0001, 0.01, size=(T - 1,)))
        freq = jnp.array(rng.uniform(0.05, 0.95, size=(A, T)), dtype=jnp.float32)
        mu = jnp.array([0.4, 0.35, 0.25])
        model = AncestryModel(
            n_ancestries=A, mu=mu, gen_since_admix=20.0,
            allele_freq=freq,
        )

        gamma_full = forward_backward(geno, model, d_morgan, use_checkpointing=False)
        gamma_ckpt = forward_backward_checkpointed(geno, model, d_morgan)

        np.testing.assert_allclose(
            np.array(gamma_full), np.array(gamma_ckpt), atol=1e-5,
            err_msg=f"Mismatch at T={T}",
        )


def test_checkpointed_T_exact_multiple_of_C():
    """T exactly divisible by C (no padding needed for groups)."""
    geno, model, d_morgan, _ = _make_model(n_samples=50, n_sites=60)

    for C in [2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]:
        gamma_ref = forward_backward(geno, model, d_morgan, use_checkpointing=False)
        gamma_ckpt = forward_backward_checkpointed(
            geno, model, d_morgan, checkpoint_interval=C,
        )
        np.testing.assert_allclose(
            np.array(gamma_ref), np.array(gamma_ckpt), atol=1e-5,
            err_msg=f"Mismatch at T=60, C={C}",
        )


def test_posteriors_sum_to_one():
    """Posteriors from checkpointed path sum to 1 across ancestries."""
    geno, model, d_morgan, _ = _make_model(n_samples=50, n_sites=100)
    gamma = forward_backward_checkpointed(geno, model, d_morgan)

    sums = np.array(gamma.sum(axis=2))
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_batched_uses_checkpointing():
    """Batched forward-backward uses checkpointing and matches."""
    geno, model, d_morgan, _ = _make_model(n_samples=100, n_sites=200)

    gamma_single = forward_backward(geno, model, d_morgan)
    # batch_size=150: H=200 <= 2*150=300, so guard passes; still batches (200 > 150)
    gamma_batched = forward_backward_batched(geno, model, d_morgan, batch_size=150)

    np.testing.assert_allclose(
        np.array(gamma_single), np.array(gamma_batched), atol=1e-5,
    )


def test_hard_calls_match():
    """Hard ancestry calls (argmax) are identical for both paths."""
    geno, model, d_morgan, _ = _make_model(n_samples=100, n_sites=200)

    gamma_full = forward_backward(geno, model, d_morgan, use_checkpointing=False)
    gamma_ckpt = forward_backward_checkpointed(geno, model, d_morgan)

    calls_full = np.array(jnp.argmax(gamma_full, axis=2))
    calls_ckpt = np.array(jnp.argmax(gamma_ckpt, axis=2))

    np.testing.assert_array_equal(calls_full, calls_ckpt)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
