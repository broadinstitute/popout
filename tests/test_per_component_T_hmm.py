"""Tests for per-component T in the HMM transition matrix and the
per-component sufficient statistics accumulated by forward_backward_em.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from popout.datatypes import AncestryModel
from popout.hmm import forward_backward_em
from popout.simulate import simulate_admixed


def _make_model(A, T_sites, n_blocks_unused=None, gen_per_comp=None,
                gen_since_admix=10.0, mu=None):
    if mu is None:
        mu = jnp.full((A,), 1.0 / A)
    freq = jnp.full((A, T_sites), 0.5)
    return AncestryModel(
        n_ancestries=A,
        mu=mu,
        gen_since_admix=gen_since_admix,
        allele_freq=freq,
        gen_per_comp=gen_per_comp,
    )


# ---------------------------------------------------------------------------
# Transition matrix: per-comp branch reduces to scalar when T_i are equal
# ---------------------------------------------------------------------------

def test_transition_matrix_scalar_equals_uniform_per_comp():
    A = 4
    T_val = 12.0
    model_scalar = _make_model(A, 10, gen_since_admix=T_val)
    model_per = _make_model(
        A, 10, gen_since_admix=T_val,
        gen_per_comp=jnp.full((A,), T_val),
    )
    d = jnp.array([0.001, 0.002, 0.003, 0.004])

    lt_s = model_scalar.log_transition_matrix(d)
    lt_p = model_per.log_transition_matrix(d)

    assert lt_s.shape == lt_p.shape == (4, A, A)
    np.testing.assert_allclose(np.array(lt_s), np.array(lt_p), atol=1e-6)


def test_transition_matrix_differing_T_changes_diagonals():
    A = 3
    model = _make_model(
        A, 10,
        gen_per_comp=jnp.array([2.0, 10.0, 50.0]),
    )
    d = jnp.array([0.01])  # one interval
    lt = np.array(model.log_transition_matrix(d))[0]

    # Diagonal stay-probabilities should differ by T_i.
    # exp(diag) = (1 - p_i) + p_i * mu[i]
    diag = np.exp(np.diag(lt))
    # Component with low T (2): high stay prob; component with high T (50): low stay
    assert diag[0] > diag[1] > diag[2]


def test_transition_matrix_rows_sum_to_one_per_comp():
    A = 4
    model = _make_model(
        A, 10,
        gen_per_comp=jnp.array([3.0, 8.0, 15.0, 25.0]),
        mu=jnp.array([0.1, 0.3, 0.4, 0.2]),
    )
    d = jnp.array([0.001, 0.005, 0.01])
    lt = np.array(model.log_transition_matrix(d))
    row_sums = np.exp(lt).sum(axis=2)  # (n_intervals, A)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_mutex_gen_per_comp_with_gen_per_hap_raises():
    with pytest.raises(ValueError, match="gen_per_comp"):
        AncestryModel(
            n_ancestries=2,
            mu=jnp.array([0.5, 0.5]),
            gen_since_admix=10.0,
            allele_freq=jnp.zeros((2, 5)),
            gen_per_comp=jnp.array([5.0, 10.0]),
            gen_per_hap=jnp.full((100,), 10.0),
        )


# ---------------------------------------------------------------------------
# EMStats per-component sufficient stats: shape, sanity, and consistency
# ---------------------------------------------------------------------------

def test_em_stats_have_per_comp_fields_after_forward_backward():
    """After forward_backward_em, EMStats carries (A,) per-comp stats."""
    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=200, n_ancestries=3,
        gen_since_admix=8.0, chrom_length_cm=50.0, rng_seed=0,
    )
    A = 3
    H = chrom_data.geno.shape[0]
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.full((A,), 1.0 / A),
        gen_since_admix=8.0,
        allele_freq=jnp.full((A, chrom_data.n_sites), 0.5),
    )
    d_morgan = jnp.asarray(chrom_data.genetic_distances)

    stats = forward_backward_em(
        chrom_data.geno, model, d_morgan, batch_size=H,
    )

    assert stats.switches_per_comp is not None
    assert stats.switches_per_comp.shape == (A,)
    assert stats.d_weighted_occupancy is not None
    assert stats.d_weighted_occupancy.shape == (A,)

    # Sanity: per-comp occupancy sums (across components) to roughly the
    # total chromosome length × H × ratio, since γ rows sum to 1.
    # d_weighted_occupancy[a] = Σ_{h, n} d_n · γ[h,n,a]
    # Σ_a d_weighted_occupancy[a] = Σ_{h, n} d_n  (only n in 0..T-2; padding zeroed)
    total_d = float(jnp.sum(d_morgan))
    observed_total = float(stats.d_weighted_occupancy.sum())
    expected = total_d * H
    np.testing.assert_allclose(observed_total, expected, rtol=1e-3)


def test_em_stats_per_comp_switches_nonneg():
    """switches_per_comp[k] = Σ (γ - ξ_diag)[a=k] is always ≥ 0 (γ ≥ ξ_diag)."""
    chrom_data, _, _ = simulate_admixed(
        n_samples=60, n_sites=180, n_ancestries=2,
        gen_since_admix=15.0, chrom_length_cm=40.0, rng_seed=1,
    )
    A = 2
    H = chrom_data.geno.shape[0]
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.full((A,), 0.5),
        gen_since_admix=15.0,
        allele_freq=jnp.full((A, chrom_data.n_sites), 0.5),
    )
    d_morgan = jnp.asarray(chrom_data.genetic_distances)

    stats = forward_backward_em(
        chrom_data.geno, model, d_morgan, batch_size=H,
    )

    assert (stats.switches_per_comp >= -1e-5).all()


def _setup_block_em(n_samples, n_sites, A, gen, cm, rng_seed, block_size=8):
    """Build geno, BlockData, and an initial AncestryModel with pattern_freq.

    Mirrors the run_em block-emissions setup just enough to drive
    forward_backward_blocks_em directly from a test.
    """
    from popout.blocks import pack_blocks, init_pattern_freq
    chrom_data, _, _ = simulate_admixed(
        n_samples=n_samples, n_sites=n_sites, n_ancestries=A,
        gen_since_admix=gen, chrom_length_cm=cm, rng_seed=rng_seed,
    )
    geno = chrom_data.geno
    bd = pack_blocks(geno, block_size=block_size, pos_cm=chrom_data.pos_cm)
    allele_freq = jnp.full((A, chrom_data.n_sites), 0.5)
    pf = init_pattern_freq(allele_freq, bd, geno)
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.full((A,), 1.0 / A),
        gen_since_admix=float(gen),
        allele_freq=allele_freq,
        pattern_freq=pf,
        block_data=bd,
    )
    return geno, model, bd, chrom_data


def test_block_em_stats_have_per_comp_fields():
    """forward_backward_blocks_em must populate switches_per_comp /
    d_weighted_occupancy on EMStats. Total occupancy across components
    equals (sum of block_distances) × H within rounding."""
    from popout.hmm import forward_backward_blocks_em
    geno, model, bd, chrom_data = _setup_block_em(
        n_samples=80, n_sites=200, A=3, gen=8, cm=40.0, rng_seed=20,
        block_size=8,
    )
    H = geno.shape[0]
    A = model.n_ancestries

    stats, _ = forward_backward_blocks_em(geno, model, bd, batch_size=H)

    assert stats.switches_per_comp is not None
    assert stats.switches_per_comp.shape == (A,)
    assert stats.d_weighted_occupancy is not None
    assert stats.d_weighted_occupancy.shape == (A,)

    # γ at each block sums to 1 over A; pair with block_distances and sum
    # across (h, b in 0..n_blocks-1) gives total_distance × H.
    total_d = float(np.asarray(bd.block_distances).sum())
    observed_total = float(stats.d_weighted_occupancy.sum())
    expected = total_d * H
    np.testing.assert_allclose(observed_total, expected, rtol=2e-3)


def test_block_em_stats_per_comp_switches_nonneg():
    """switches_per_comp[k] = Σ (γ - ξ_diag)[a=k] is always ≥ 0 in the
    block path too."""
    from popout.hmm import forward_backward_blocks_em
    geno, model, bd, _ = _setup_block_em(
        n_samples=60, n_sites=160, A=2, gen=15, cm=30.0, rng_seed=21,
    )
    stats, _ = forward_backward_blocks_em(geno, model, bd, batch_size=geno.shape[0])
    assert (stats.switches_per_comp >= -1e-5).all()


def test_block_em_stats_per_comp_invariant_to_batching():
    """Splitting H into batches must give identical per-comp stats in the
    block path (within float rounding)."""
    from popout.hmm import forward_backward_blocks_em
    geno, model, bd, _ = _setup_block_em(
        n_samples=64, n_sites=120, A=3, gen=10, cm=30.0, rng_seed=22,
        block_size=8,
    )
    H = geno.shape[0]

    stats_full, _ = forward_backward_blocks_em(geno, model, bd, batch_size=H)
    stats_split, _ = forward_backward_blocks_em(geno, model, bd, batch_size=16)

    np.testing.assert_allclose(
        stats_full.switches_per_comp, stats_split.switches_per_comp,
        rtol=1e-3, atol=1e-3,
    )
    np.testing.assert_allclose(
        stats_full.d_weighted_occupancy, stats_split.d_weighted_occupancy,
        rtol=1e-3, atol=1e-3,
    )


def test_em_stats_per_comp_invariant_to_batching():
    """Splitting H into batches must give identical per-comp stats."""
    chrom_data, _, _ = simulate_admixed(
        n_samples=64, n_sites=120, n_ancestries=3,
        gen_since_admix=10.0, chrom_length_cm=30.0, rng_seed=7,
    )
    A = 3
    H = chrom_data.geno.shape[0]
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.full((A,), 1.0 / A),
        gen_since_admix=10.0,
        allele_freq=jnp.full((A, chrom_data.n_sites), 0.5),
    )
    d_morgan = jnp.asarray(chrom_data.genetic_distances)

    stats_full = forward_backward_em(chrom_data.geno, model, d_morgan, batch_size=H)
    stats_split = forward_backward_em(chrom_data.geno, model, d_morgan, batch_size=16)

    np.testing.assert_allclose(
        stats_full.switches_per_comp, stats_split.switches_per_comp,
        rtol=1e-3, atol=1e-3,
    )
    np.testing.assert_allclose(
        stats_full.d_weighted_occupancy, stats_split.d_weighted_occupancy,
        rtol=1e-3, atol=1e-3,
    )
