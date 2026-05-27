"""Tests for EM module."""

import numpy as np
import jax.numpy as jnp

from popout.em import (
    compute_bucket_centers,
    assign_buckets,
    update_generations_per_hap,
)


def test_bucket_centers():
    """Bucket centers are geometrically spaced."""
    centers = compute_bucket_centers(n_buckets=20)
    assert centers.shape == (20,)
    assert float(centers[0]) >= 1.0
    assert float(centers[-1]) <= 1000.0
    # Check geometric spacing: ratios between consecutive should be equal
    ratios = np.array(centers[1:] / centers[:-1])
    np.testing.assert_allclose(ratios, ratios[0], atol=1e-5)


def test_assign_buckets():
    """Bucket assignments pick the nearest center in log-space."""
    centers = jnp.array([1.0, 10.0, 100.0, 1000.0])
    T_per_hap = jnp.array([1.5, 8.0, 15.0, 200.0, 999.0])
    assignments = assign_buckets(T_per_hap, centers)
    expected = jnp.array([0, 1, 1, 2, 3])
    np.testing.assert_array_equal(np.array(assignments), np.array(expected))


def test_per_hap_T_zero_switches():
    """Haplotypes with zero switches regularize toward global T."""
    H, T, A = 10, 50, 3
    # Make gamma that gives identical ancestry everywhere (no switches)
    gamma = jnp.zeros((H, T, A))
    gamma = gamma.at[:, :, 0].set(1.0)  # all ancestry 0

    d_morgan = jnp.full(T - 1, 0.001)
    mu = jnp.array([0.5, 0.3, 0.2])
    centers = compute_bucket_centers(10)

    T_per_hap, _, T_global = update_generations_per_hap(
        gamma, d_morgan, 20.0, mu, centers,
    )
    # With zero switches, T_raw = 0 → regularized toward global T=20
    T_arr = np.array(T_per_hap)
    # Should be pulled toward 20 (the current global), clamped to >= 1
    assert T_arr.min() >= 1.0
    assert T_arr.max() <= 1000.0


def test_run_em_per_hap_T_with_block_emissions_uses_buckets(monkeypatch):
    """When per_hap_T=True and use_block_emissions=True, the block-emissions
    E-step must dispatch across bucket centers once buckets are set.

    Iter 0: T held fixed; E-step runs once per chunk at global T.
    Iter 1: M-step sets bucket_assignments on the model.
    Iter 2: E-step must run forward_backward_blocks once per bucket per chunk,
    each call carrying that bucket's center as gen_since_admix.

    Bucket assignments are forced (via spy on the M-step) to span multiple
    buckets so dispatch is exercised even on small synthetic data.
    """
    import popout.em as em_mod
    import jax.numpy as jnp_local
    from popout.simulate import simulate_admixed

    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=240, n_ancestries=3,
        gen_since_admix=20, chrom_length_cm=80.0,
        pure_fraction=0.0, rng_seed=0,
    )

    captured_T_per_iter: list[list[float]] = []
    iter_buf: list[float] = []
    real_fb_blocks = em_mod.forward_backward_blocks

    def spy(model, bd, *, compute_soft_switches=False):
        iter_buf.append(float(model.gen_since_admix))
        return real_fb_blocks(model, bd, compute_soft_switches=compute_soft_switches)

    monkeypatch.setattr(em_mod, "forward_backward_blocks", spy)

    real_update = em_mod.update_generations_per_hap_from_stats

    def update_spy(stats, *args, **kwargs):
        captured_T_per_iter.append(list(iter_buf))
        iter_buf.clear()
        T_per_hap, _, T_global = real_update(stats, *args, **kwargs)
        H = stats.n_haps
        forced = jnp_local.array(np.arange(H) % 3, dtype=jnp_local.int32)
        return T_per_hap, forced, T_global

    monkeypatch.setattr(em_mod, "update_generations_per_hap_from_stats", update_spy)

    em_mod.run_em(
        chrom_data,
        n_ancestries=3,
        n_em_iter=3,
        gen_since_admix=20.0,
        per_hap_T=True,
        n_T_buckets=3,
        use_block_emissions=True,
        block_size=8,
        skip_decode=True,
        rng_seed=0,
        em_t_policy="every-iter",
    )

    assert len(captured_T_per_iter) >= 2, (
        f"expected ≥2 M-step calls (iter 1, iter 2); got {len(captured_T_per_iter)}"
    )
    iter1_T = set(captured_T_per_iter[0])
    iter2_T = set(captured_T_per_iter[1])
    assert len(iter1_T) == 1, (
        f"iter 1 should run with one global T, got {sorted(iter1_T)}"
    )
    assert len(iter2_T) > 1, (
        f"iter 2 should dispatch across multiple bucket centers; "
        f"got only {sorted(iter2_T)}"
    )


def test_block_em_soft_switches_are_density_invariant_not_hard_cast(monkeypatch):
    """The block-emissions E-step must populate EMStats.soft_switches_per_hap
    with xi-based density-invariant values, not hard-call integer counts cast
    to float (the bug fixed by this commit). Captured stats from iter 1's
    M-step must contain at least some non-integer soft-switch values.
    """
    import popout.em as em_mod
    from popout.simulate import simulate_admixed

    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=240, n_ancestries=3,
        gen_since_admix=20, chrom_length_cm=80.0,
        pure_fraction=0.0, rng_seed=1,
    )

    captured: list = []
    real_update = em_mod.update_generations_per_hap_from_stats

    def update_spy(stats, *args, **kwargs):
        captured.append(np.array(stats.soft_switches_per_hap))
        return real_update(stats, *args, **kwargs)

    monkeypatch.setattr(em_mod, "update_generations_per_hap_from_stats", update_spy)

    em_mod.run_em(
        chrom_data,
        n_ancestries=3,
        n_em_iter=2,
        gen_since_admix=20.0,
        per_hap_T=True,
        n_T_buckets=5,
        use_block_emissions=True,
        block_size=8,
        skip_decode=True,
        rng_seed=1,
        em_t_policy="every-iter",
    )

    assert captured, "M-step was not called"
    soft = captured[0]
    # Hard-switch-cast-to-float would produce integer values everywhere.
    # xi-based soft switches yield fractional values for haplotypes whose
    # block-boundary posteriors aren't perfectly concentrated.
    n_non_integer = int(np.sum(np.abs(soft - np.round(soft)) > 1e-4))
    assert n_non_integer > 0, (
        f"soft_switches_per_hap looks like hard-count cast (all integers); "
        f"min={soft.min():.3f}, max={soft.max():.3f}"
    )


def _make_block_model_for_decode(n_haps, n_sites, n_anc, block_size, rng_seed):
    """Build a (chrom_data, model) pair with block_data + pattern_freq set,
    suitable for invoking decode_chromosome's block-emissions branch."""
    import jax.numpy as jnp_local
    from popout.simulate import simulate_admixed
    from popout.blocks import pack_blocks, init_pattern_freq
    from popout.datatypes import AncestryModel

    chrom_data, _, true = simulate_admixed(
        n_samples=n_haps // 2, n_sites=n_sites, n_ancestries=n_anc,
        gen_since_admix=20, chrom_length_cm=80.0, pure_fraction=0.0,
        rng_seed=rng_seed,
    )
    bd = pack_blocks(chrom_data.geno, block_size=block_size, pos_cm=chrom_data.pos_cm)
    allele_freq = jnp_local.array(true["pop_freq"], dtype=jnp_local.float32)
    pf = init_pattern_freq(allele_freq, bd, chrom_data.geno)
    model = AncestryModel(
        n_ancestries=n_anc,
        mu=jnp_local.array(true["mu"], dtype=jnp_local.float32),
        gen_since_admix=20.0,
        allele_freq=allele_freq,
        pattern_freq=pf,
        block_data=bd,
    )
    return chrom_data, model


def test_decode_chromosome_bucketed_blocks_memmap(tmp_path):
    """End-to-end: decode_chromosome with block_emissions and explicit
    bucket_assignments produces a memmapped .calls.dat and a merged
    decode_parquet, with no in-memory (H, T) max_post."""
    import jax.numpy as jnp_local
    from popout.em import decode_chromosome, compute_bucket_centers
    from popout.datatypes import AncestryModel

    H, T_sites, A = 40, 80, 3
    chrom_data, model = _make_block_model_for_decode(
        H, T_sites, A, block_size=8, rng_seed=2,
    )

    bucket_centers = compute_bucket_centers(4)
    rng = np.random.default_rng(3)
    bucket_assignments = jnp_local.array(
        rng.integers(0, 4, size=H), dtype=jnp_local.int32,
    )
    bucketed_model = AncestryModel(
        n_ancestries=model.n_ancestries,
        mu=model.mu,
        gen_since_admix=model.gen_since_admix,
        allele_freq=model.allele_freq,
        pattern_freq=model.pattern_freq,
        block_data=model.block_data,
        bucket_centers=bucket_centers,
        bucket_assignments=bucket_assignments,
    )

    out_path = str(tmp_path / "decode.parquet")
    decode = decode_chromosome(
        chrom_data, bucketed_model,
        write_dense_decode=True,
        decode_parquet_path=out_path,
    )

    from pathlib import Path as _P
    assert _P(out_path).exists(), "merged decode parquet missing"
    assert _P(out_path + ".calls.dat").exists() or \
           _P(str(_P(out_path).with_suffix(".calls.dat"))).exists(), \
           "calls memmap missing"
    assert decode.max_post is None, (
        "max_post must not be allocated when streaming to parquet"
    )
    assert decode.calls.shape == (H, T_sites)
    assert decode.parquet_path == out_path

    for b in range(4):
        assert not _P(str(_P(out_path).with_suffix(f".bucket{b}.parquet"))).exists(), (
            f"per-bucket parquet for bucket {b} should have been removed after merge"
        )


def test_bucketed_blocks_decode_matches_unbucketed_when_B_eq_1(tmp_path):
    """All-one-bucket assignment whose center matches model.gen_since_admix
    must produce bit-identical decode output to the no-bucket path."""
    import jax.numpy as jnp_local
    from popout.em import decode_chromosome
    from popout.datatypes import AncestryModel

    H, T_sites, A = 24, 64, 3
    chrom_data, model = _make_block_model_for_decode(
        H, T_sites, A, block_size=8, rng_seed=4,
    )

    no_bucket_decode = decode_chromosome(
        chrom_data, model,
        write_dense_decode=True,
        decode_parquet_path=str(tmp_path / "no_bucket.parquet"),
    )

    bucketed_model = AncestryModel(
        n_ancestries=model.n_ancestries,
        mu=model.mu,
        gen_since_admix=model.gen_since_admix,
        allele_freq=model.allele_freq,
        pattern_freq=model.pattern_freq,
        block_data=model.block_data,
        bucket_centers=jnp_local.array([model.gen_since_admix], dtype=jnp_local.float32),
        bucket_assignments=jnp_local.zeros(H, dtype=jnp_local.int32),
    )
    bucketed_decode = decode_chromosome(
        chrom_data, bucketed_model,
        write_dense_decode=True,
        decode_parquet_path=str(tmp_path / "bucketed.parquet"),
    )

    np.testing.assert_array_equal(
        np.asarray(no_bucket_decode.calls),
        np.asarray(bucketed_decode.calls),
    )
    np.testing.assert_array_equal(
        np.asarray(no_bucket_decode.global_sums),
        np.asarray(bucketed_decode.global_sums),
    )


def test_run_em_seeded_init_skips_window_refine():
    """When seed_responsibilities is supplied, init_model_soft must NOT
    run window-refinement — the iter-0 allele_freq must equal the
    closed-form seed-derived value (pseudocount-smoothed, clipped).

    Pre-fix: window_init_allele_freq's uniform-prior softmax perturbs
    freq and this assertion fails. Post-fix: window_refine is gated on
    seed_responsibilities is None, so freq matches the closed form.
    """
    from popout.em import run_em
    from popout.simulate import simulate_admixed

    chrom_data, _, _ = simulate_admixed(
        n_samples=200, n_sites=400, n_ancestries=3,
        gen_since_admix=20, chrom_length_cm=80.0,
        pure_fraction=0.3, rng_seed=42,
    )

    H, _T = chrom_data.geno.shape
    A = 3
    rng = np.random.default_rng(0)
    labels = rng.integers(0, A, size=H)
    seed_resp_np = np.zeros((H, A), dtype=np.float32)
    seed_resp_np[np.arange(H), labels] = 1.0
    seed_resp = jnp.array(seed_resp_np)

    result = run_em(
        chrom_data,
        n_ancestries=A,
        n_em_iter=0,
        gen_since_admix=20.0,
        rng_seed=42,
        seed_responsibilities=seed_resp,
    )

    geno_f32 = jnp.asarray(chrom_data.geno).astype(jnp.float32)
    expected_wc = seed_resp.T @ geno_f32
    expected_totals = seed_resp.sum(axis=0)[:, None]
    expected_freq = (expected_wc + 0.5) / (expected_totals + 1.0)
    expected_freq = jnp.clip(expected_freq, 1e-4, 1.0 - 1e-4)

    np.testing.assert_allclose(
        np.array(result.model.allele_freq),
        np.array(expected_freq),
        rtol=1e-5, atol=1e-5,
    )


def _run_em_with_seeded_fixture(em_t_policy, n_em_iter=3, monkeypatch=None,
                                spy_target=None):
    """Shared fixture for em_t_policy tests. Returns (result, n_T_calls)
    when spy_target is provided, else (result, None)."""
    from popout.em import run_em
    import popout.em as em_mod
    from popout.simulate import simulate_admixed

    chrom_data, _, _ = simulate_admixed(
        n_samples=200, n_sites=400, n_ancestries=3,
        gen_since_admix=20, chrom_length_cm=80.0,
        pure_fraction=0.3, rng_seed=42,
    )
    H = chrom_data.geno.shape[0]
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=H)
    seed_resp_np = np.zeros((H, 3), dtype=np.float32)
    seed_resp_np[np.arange(H), labels] = 1.0
    seed_resp = jnp.array(seed_resp_np)

    n_T_calls = [0]
    if spy_target is not None and monkeypatch is not None:
        real_fn = getattr(em_mod, spy_target)
        def spy(*args, **kwargs):
            n_T_calls[0] += 1
            return real_fn(*args, **kwargs)
        monkeypatch.setattr(em_mod, spy_target, spy)

    result = run_em(
        chrom_data,
        n_ancestries=3,
        n_em_iter=n_em_iter,
        gen_since_admix=20.0,
        rng_seed=42,
        seed_responsibilities=seed_resp,
        em_t_policy=em_t_policy,
    )
    return result, n_T_calls[0] if spy_target is not None else None


def test_em_t_policy_hold_keeps_T_fixed(monkeypatch):
    """--em-t-policy hold: T is never updated; update_generations_* is
    never called."""
    result, n_T_calls = _run_em_with_seeded_fixture(
        "hold", n_em_iter=3, monkeypatch=monkeypatch,
        spy_target="update_generations_from_stats",
    )
    assert n_T_calls == 0, f"hold policy must not call T update; got {n_T_calls}"
    assert float(result.model.gen_since_admix) == 20.0, (
        f"hold policy must keep T at 20.0; got {result.model.gen_since_admix}"
    )


def test_em_t_policy_gated_fires_at_most_once(monkeypatch):
    """--em-t-policy gated: update_generations_* is called at most once
    across the run."""
    result, n_T_calls = _run_em_with_seeded_fixture(
        "gated", n_em_iter=5, monkeypatch=monkeypatch,
        spy_target="update_generations_from_stats",
    )
    assert n_T_calls <= 1, (
        f"gated policy must fire at most once; got {n_T_calls} calls"
    )


def test_em_t_policy_every_iter_updates_every_iter_after_zero(monkeypatch):
    """--em-t-policy every-iter: update_generations_* is called
    n_em_iter - 1 times (held at iter 0, updated thereafter)."""
    n_iter = 3
    _, n_T_calls = _run_em_with_seeded_fixture(
        "every-iter", n_em_iter=n_iter, monkeypatch=monkeypatch,
        spy_target="update_generations_from_stats",
    )
    assert n_T_calls == n_iter - 1, (
        f"every-iter policy must call T update {n_iter - 1} times for "
        f"n_em_iter={n_iter}; got {n_T_calls}"
    )


def test_em_t_policy_rejects_unknown():
    """Unknown em_t_policy values must raise ValueError fast."""
    import pytest
    from popout.em import run_em
    from popout.simulate import simulate_admixed

    chrom_data, _, _ = simulate_admixed(
        n_samples=100, n_sites=200, n_ancestries=2,
        gen_since_admix=20, chrom_length_cm=80.0,
        pure_fraction=0.3, rng_seed=42,
    )
    with pytest.raises(ValueError, match="em_t_policy"):
        run_em(
            chrom_data, n_ancestries=2, n_em_iter=1,
            gen_since_admix=20.0, rng_seed=42,
            em_t_policy="bogus",
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
