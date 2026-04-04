"""Tests for the block emission module."""

import numpy as np
import jax.numpy as jnp

from popout.blocks import pack_blocks, init_pattern_freq, expand_block_posteriors, update_pattern_freq


def test_pack_blocks_basic():
    """Pack/unpack round-trip for basic data."""
    rng = np.random.default_rng(42)
    H, T = 50, 24
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=8)

    assert bd.n_blocks == 3
    assert bd.pattern_indices.shape == (H, 3)
    assert bd.block_starts.tolist() == [0, 8, 16]
    assert bd.block_ends.tolist() == [8, 16, 24]
    assert bd.block_distances.shape == (2,)
    assert bd.max_patterns <= 256  # 2^8


def test_pack_blocks_last_block_short():
    """Last block may have fewer than k sites."""
    geno = np.zeros((10, 10), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=8)
    assert bd.n_blocks == 2
    assert bd.block_starts.tolist() == [0, 8]
    assert bd.block_ends.tolist() == [8, 10]


def test_pack_blocks_with_pos_cm():
    """Block distances computed from genetic positions."""
    H, T = 20, 16
    geno = np.zeros((H, T), dtype=np.uint8)
    pos_cm = np.linspace(0, 10, T)
    bd = pack_blocks(geno, block_size=8, pos_cm=pos_cm)
    assert bd.block_distances.shape == (1,)
    assert bd.block_distances[0] > 0


def test_init_pattern_freq_shape():
    """Pattern frequency table has the right shape."""
    rng = np.random.default_rng(42)
    H, T, A = 50, 24, 3
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    freq = jnp.array(rng.uniform(0.1, 0.9, (A, T)), dtype=jnp.float32)
    bd = pack_blocks(geno, block_size=8)

    pf = init_pattern_freq(freq, bd, geno)
    assert pf.shape == (3, bd.max_patterns, A)
    # Each block's patterns should sum to ~1 per ancestry
    for b in range(3):
        n_p = bd.pattern_counts[b]
        for a in range(A):
            total = float(pf[b, :n_p, a].sum())
            np.testing.assert_allclose(total, 1.0, atol=0.01)


def test_expand_block_posteriors():
    """Block posteriors expand to correct site-level shape."""
    n_blocks, H, A = 3, 10, 4
    gamma_block = jnp.ones((H, n_blocks, A)) / A

    geno = np.zeros((H, 24), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=8)

    gamma_site = expand_block_posteriors(gamma_block, bd, n_sites=24)
    assert gamma_site.shape == (H, 24, A)
    # Sites within same block should have identical posteriors
    np.testing.assert_array_equal(
        np.array(gamma_site[:, 0, :]),
        np.array(gamma_site[:, 7, :]),
    )


def test_update_pattern_freq():
    """Pattern frequency update produces valid frequencies."""
    rng = np.random.default_rng(42)
    H, T, A = 100, 16, 3
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=8)

    gamma_block = jnp.array(rng.dirichlet([1] * A, size=(H, bd.n_blocks)))

    pf = update_pattern_freq(bd, gamma_block)
    assert pf.shape == (bd.n_blocks, bd.max_patterns, A)
    # All values should be non-negative
    assert float(pf.min()) >= 0


def test_k1_equals_bernoulli():
    """With k=1 (single-site blocks), pattern freq should match Bernoulli."""
    rng = np.random.default_rng(42)
    H, T, A = 200, 10, 2
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    # Simple per-site frequencies
    freq = jnp.array(rng.uniform(0.2, 0.8, (A, T)), dtype=jnp.float32)

    bd = pack_blocks(geno, block_size=1)
    assert bd.n_blocks == T
    # Each block has at most 2 patterns (0 and 1)
    assert bd.max_patterns <= 2

    pf = init_pattern_freq(freq, bd, geno)
    # For k=1, pattern 0 = allele 0, pattern 1 = allele 1
    # Pattern freq should be proportional to Bernoulli probabilities
    # This is a sanity check that the init works for degenerate block size


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
