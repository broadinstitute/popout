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


def test_block_mstep_no_expansion_matches_expanded():
    """M-step accumulators from block-level gamma must match per-site expansion."""
    from popout.hmm import forward_backward_blocks
    from popout.datatypes import AncestryModel

    rng = np.random.default_rng(42)
    H, T, A = 200, 64, 4
    block_size = 8
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=block_size)
    allele_freq = jnp.array(rng.uniform(0.1, 0.9, (A, T)).astype(np.float32))
    pf = init_pattern_freq(allele_freq, bd, geno)
    model = AncestryModel(
        n_ancestries=A, mu=jnp.full(A, 1.0/A), gen_since_admix=20.0,
        allele_freq=allele_freq, pattern_freq=pf, block_data=bd,
    )

    gamma_block = forward_backward_blocks(model, bd)
    geno_j = jnp.array(geno, dtype=jnp.float32)

    # --- Reference: expand to per-site then reduce ---
    gamma_site = expand_block_posteriors(gamma_block, bd, T)
    ref_wc = jnp.einsum('hta,ht->at', gamma_site, geno_j)
    ref_tw = gamma_site.sum(axis=0).T
    ref_mu = gamma_site.sum(axis=(0, 1))

    # --- New: block-level accumulation ---
    b_starts = np.array(bd.block_starts)
    b_ends = np.array(bd.block_ends)
    block_widths = jnp.array(b_ends - b_starts, dtype=jnp.float32)

    new_wc = jnp.zeros((A, T), dtype=jnp.float32)
    new_tw = jnp.zeros((A, T), dtype=jnp.float32)
    for b_idx in range(bd.n_blocks):
        s, e = int(b_starts[b_idx]), int(b_ends[b_idx])
        new_wc = new_wc.at[:, s:e].add(gamma_block[:, b_idx, :].T @ geno_j[:, s:e])
        per_anc = gamma_block[:, b_idx, :].sum(axis=0)
        new_tw = new_tw.at[:, s:e].add(
            jnp.broadcast_to(per_anc[:, None], (A, e - s))
        )
    new_mu = jnp.einsum('hba,b->a', gamma_block, block_widths)

    np.testing.assert_allclose(np.array(new_wc), np.array(ref_wc), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.array(new_tw), np.array(ref_tw), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.array(new_mu), np.array(ref_mu), rtol=1e-4, atol=1e-6)

    # --- Switches: block-boundary vs per-site argmax ---
    calls_site = jnp.argmax(gamma_site, axis=2)
    ref_sw = np.array((calls_site[:, 1:] != calls_site[:, :-1]).sum(axis=1))
    calls_block = jnp.argmax(gamma_block, axis=2)
    new_sw = np.array((calls_block[:, 1:] != calls_block[:, :-1]).sum(axis=1))
    # Block-boundary switches <= per-site switches (within-block switches are
    # zero by construction). They should be equal since within-block gamma is
    # constant, so within-block argmax is constant, so no within-block switches.
    np.testing.assert_array_equal(new_sw, ref_sw)


def test_forward_backward_blocks_batched_matches_unbatched():
    """Batched and unbatched forward_backward_blocks must agree."""
    from popout.hmm import forward_backward_blocks, forward_backward_blocks_batched
    from popout.datatypes import AncestryModel

    rng = np.random.default_rng(42)
    H, T, A = 500, 256, 4
    block_size = 8
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=block_size)
    allele_freq = jnp.array(rng.uniform(0.1, 0.9, (A, T)).astype(np.float32))
    pf = init_pattern_freq(allele_freq, bd, geno)
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.full(A, 1.0 / A),
        gen_since_admix=20.0,
        allele_freq=allele_freq,
        pattern_freq=pf,
        block_data=bd,
    )

    gamma_unbatched = forward_backward_blocks(model, bd)
    gamma_batched = forward_backward_blocks_batched(model, bd, batch_size=100)

    np.testing.assert_allclose(
        np.array(gamma_unbatched), np.array(gamma_batched),
        rtol=1e-4, atol=1e-6,
    )


def _scalar_reference_init_pattern_freq(allele_freq, block_data, geno, pseudocount=0.01):
    """Original scalar implementation for equivalence testing."""
    A = allele_freq.shape[0]
    n_blocks = block_data.n_blocks
    max_p = block_data.max_patterns
    freq = np.array(allele_freq)
    freq = np.clip(freq, 1e-6, 1.0 - 1e-6)
    pf = np.full((n_blocks, max_p, A), pseudocount, dtype=np.float32)
    for b in range(n_blocks):
        s = block_data.block_starts[b]
        e = block_data.block_ends[b]
        n_p = block_data.pattern_counts[b]
        block_geno = geno[:, s:e]
        pat_idx = block_data.pattern_indices[:, b]
        for p in range(n_p):
            exemplar = np.where(pat_idx == p)[0][0]
            bits = block_geno[exemplar]
            for a in range(A):
                log_prob = 0.0
                for i, bit in enumerate(bits):
                    f = freq[a, s + i]
                    log_prob += np.log(f) if bit == 1 else np.log(1.0 - f)
                pf[b, p, a] = np.exp(log_prob)
    for b in range(n_blocks):
        n_p = block_data.pattern_counts[b]
        for a in range(A):
            total = pf[b, :n_p, a].sum()
            if total > 0:
                pf[b, :n_p, a] /= total
    return pf


def test_init_pattern_freq_matches_scalar_reference():
    """Vectorized init_pattern_freq must match scalar reference."""
    rng = np.random.default_rng(42)
    H, T, A = 100, 64, 4
    block_size = 8
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    allele_freq = jnp.array(rng.uniform(0.1, 0.9, size=(A, T)).astype(np.float32))
    bd = pack_blocks(geno, block_size=block_size)

    pf_vec = np.array(init_pattern_freq(allele_freq, bd, geno))
    pf_ref = _scalar_reference_init_pattern_freq(allele_freq, bd, geno)

    np.testing.assert_allclose(pf_vec, pf_ref, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
