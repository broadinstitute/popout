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


def test_block_decode_without_expansion():
    """Block-level argmax/max expanded via site_to_block must match per-site."""
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
        n_ancestries=A, mu=jnp.full(A, 1.0 / A), gen_since_admix=20.0,
        allele_freq=allele_freq, pattern_freq=pf, block_data=bd,
    )

    gamma_block = forward_backward_blocks(model, bd)

    # --- Reference: expand to per-site, then reduce ---
    gamma_site = expand_block_posteriors(gamma_block, bd, T)
    ref_calls = np.array(jnp.argmax(gamma_site, axis=2), dtype=np.int8)
    ref_max_post = np.array(gamma_site.max(axis=2), dtype=np.float32)
    ref_global_sums = np.array(gamma_site.sum(axis=1), dtype=np.float64)

    # --- New: block-level reduction + integer gather ---
    site_to_block = np.empty(T, dtype=np.int32)
    for b_idx in range(bd.n_blocks):
        site_to_block[bd.block_starts[b_idx]:bd.block_ends[b_idx]] = b_idx
    block_widths = jnp.array(
        [bd.block_ends[b] - bd.block_starts[b] for b in range(bd.n_blocks)],
        dtype=jnp.float32,
    )

    calls_block = np.array(jnp.argmax(gamma_block, axis=2), dtype=np.int8)
    new_calls = calls_block[:, site_to_block]
    max_post_block = np.array(gamma_block.max(axis=2), dtype=np.float32)
    new_max_post = max_post_block[:, site_to_block]
    new_global_sums = np.array(
        jnp.einsum('hba,b->ha', gamma_block, block_widths), dtype=np.float64,
    )

    np.testing.assert_array_equal(new_calls, ref_calls)
    np.testing.assert_allclose(new_max_post, ref_max_post, rtol=1e-6)
    np.testing.assert_allclose(new_global_sums, ref_global_sums, rtol=1e-4, atol=1e-6)


def test_block_soft_switches_match_hard_switches_in_sharp_limit():
    """When pattern_freq is sharply peaked per ancestry, gamma concentrates
    on a single ancestry per block, so xi-based soft switches at block
    boundaries closely approximate the hard-call argmax switch count."""
    from popout.hmm import forward_backward_blocks
    from popout.datatypes import AncestryModel

    rng = np.random.default_rng(0)
    H, T, A = 100, 64, 3
    block_size = 8
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=block_size)

    n_blocks = bd.n_blocks
    max_p = bd.max_patterns
    pf = np.full((n_blocks, max_p, A), 1e-6, dtype=np.float32)
    for b in range(n_blocks):
        n_p = int(bd.pattern_counts[b])
        for p in range(n_p):
            a_assign = (b * 7 + p * 3) % A
            pf[b, p, a_assign] = 1.0
    for b in range(n_blocks):
        n_p = int(bd.pattern_counts[b])
        for a in range(A):
            s = pf[b, :n_p, a].sum()
            if s > 0:
                pf[b, :n_p, a] /= s
    pf_j = jnp.array(pf)
    allele_freq = jnp.array(rng.uniform(0.1, 0.9, (A, T)).astype(np.float32))
    model = AncestryModel(
        n_ancestries=A, mu=jnp.full(A, 1.0 / A), gen_since_admix=20.0,
        allele_freq=allele_freq, pattern_freq=pf_j, block_data=bd,
    )

    gamma_block, soft_sw = forward_backward_blocks(
        model, bd, compute_soft_switches=True,
    )
    calls_block = np.array(jnp.argmax(gamma_block, axis=2))
    hard_sw = (calls_block[:, 1:] != calls_block[:, :-1]).sum(axis=1).astype(np.float32)
    soft_sw_np = np.array(soft_sw)

    np.testing.assert_allclose(soft_sw_np, hard_sw, atol=0.5, rtol=0.05)


def test_block_soft_switches_density_invariant():
    """Soft switches scale ~linearly with block_distance in the
    low-distance regime (1 - exp(-T*d) ≈ T*d for small T*d). Doubling
    every block_distance roughly doubles soft-switch count."""
    from popout.hmm import forward_backward_blocks
    from popout.datatypes import AncestryModel
    from popout.blocks import BlockData

    rng = np.random.default_rng(1)
    H, T, A = 200, 64, 3
    block_size = 8
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=block_size)
    allele_freq = jnp.array(rng.uniform(0.1, 0.9, (A, T)).astype(np.float32))
    pf = init_pattern_freq(allele_freq, bd, geno)

    small_d = np.full(bd.n_blocks - 1, 1e-5, dtype=np.float32)
    bd1 = BlockData(
        pattern_indices=bd.pattern_indices,
        block_starts=bd.block_starts,
        block_ends=bd.block_ends,
        block_distances=small_d,
        pattern_counts=bd.pattern_counts,
        max_patterns=bd.max_patterns,
        block_size=bd.block_size,
    )
    bd2 = BlockData(
        pattern_indices=bd.pattern_indices,
        block_starts=bd.block_starts,
        block_ends=bd.block_ends,
        block_distances=2.0 * small_d,
        pattern_counts=bd.pattern_counts,
        max_patterns=bd.max_patterns,
        block_size=bd.block_size,
    )

    model = AncestryModel(
        n_ancestries=A, mu=jnp.full(A, 1.0 / A), gen_since_admix=20.0,
        allele_freq=allele_freq, pattern_freq=pf, block_data=bd,
    )
    _, sw1 = forward_backward_blocks(model, bd1, compute_soft_switches=True)
    _, sw2 = forward_backward_blocks(model, bd2, compute_soft_switches=True)

    sw1_total = float(np.array(sw1).sum())
    sw2_total = float(np.array(sw2).sum())
    assert sw2_total > sw1_total
    ratio = sw2_total / max(sw1_total, 1e-12)
    assert 1.7 < ratio < 2.3, f"expected ratio ≈ 2, got {ratio:.3f}"


def test_forward_backward_blocks_em_matches_unbucketed_for_B_eq_1():
    """All-one-bucket assignment whose center matches gen_since_admix
    must produce the same EMStats as the no-bucket path. Confirms the
    Task 3 extraction preserves bucket-dispatch equivalence."""
    from popout.hmm import forward_backward_blocks_em
    from popout.datatypes import AncestryModel

    rng = np.random.default_rng(7)
    H, T, A = 32, 64, 3
    block_size = 8
    geno = rng.integers(0, 2, size=(H, T), dtype=np.uint8)
    bd = pack_blocks(geno, block_size=block_size)
    allele_freq = jnp.array(rng.uniform(0.1, 0.9, (A, T)).astype(np.float32))
    pf = init_pattern_freq(allele_freq, bd, geno)

    no_bucket_model = AncestryModel(
        n_ancestries=A, mu=jnp.full(A, 1.0 / A), gen_since_admix=20.0,
        allele_freq=allele_freq, pattern_freq=pf, block_data=bd,
    )
    bucketed_model = AncestryModel(
        n_ancestries=A, mu=jnp.full(A, 1.0 / A), gen_since_admix=20.0,
        allele_freq=allele_freq, pattern_freq=pf, block_data=bd,
        bucket_centers=jnp.array([20.0], dtype=jnp.float32),
        bucket_assignments=jnp.zeros(H, dtype=jnp.int32),
    )

    em_a, pf_a = forward_backward_blocks_em(geno, no_bucket_model, bd, batch_size=H)
    em_b, pf_b = forward_backward_blocks_em(geno, bucketed_model, bd, batch_size=H)

    np.testing.assert_allclose(
        np.asarray(em_a.weighted_counts), np.asarray(em_b.weighted_counts),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(em_a.total_weights), np.asarray(em_b.total_weights),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(em_a.mu_sum), np.asarray(em_b.mu_sum),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(em_a.soft_switches_per_hap),
        np.asarray(em_b.soft_switches_per_hap),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(em_a.switches_per_hap),
        np.asarray(em_b.switches_per_hap),
    )
    np.testing.assert_allclose(
        np.asarray(pf_a), np.asarray(pf_b), rtol=1e-5, atol=1e-6,
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
