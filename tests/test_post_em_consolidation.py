"""Tests for post-EM ancestry consolidation."""

import numpy as np
import jax.numpy as jnp
import pytest

from popout.datatypes import AncestryModel, AncestryResult, DecodeResult
from popout.post_em_consolidation import consolidate, _pairwise_fst


def _make_result(
    A: int,
    T: int = 200,
    H: int = 1000,
    mu: np.ndarray | None = None,
    freq: np.ndarray | None = None,
    max_post: np.ndarray | None = None,
    calls: np.ndarray | None = None,
) -> AncestryResult:
    """Build a synthetic AncestryResult for testing."""
    rng = np.random.default_rng(42)

    if mu is None:
        mu = np.ones(A, dtype=np.float32) / A
    if freq is None:
        freq = rng.beta(0.3, 0.3, size=(A, T)).astype(np.float32)
    if calls is None:
        calls = rng.integers(0, A, size=(H, T), dtype=np.int8)
    if max_post is None:
        max_post = rng.uniform(0.5, 1.0, size=(H, T)).astype(np.float16)

    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.array(mu),
        gen_since_admix=20.0,
        allele_freq=jnp.array(freq),
    )
    decode = DecodeResult(
        calls=calls,
        max_post=max_post,
        global_sums=np.ones((H, A), dtype=np.float64) * T / A,
    )
    return AncestryResult(
        calls=calls,
        model=model,
        chrom="1",
        decode=decode,
    )


def test_pairwise_fst():
    """F_ST between identical frequencies is 0; between distinct is positive."""
    rng = np.random.default_rng(42)
    freq = rng.beta(0.3, 0.3, size=500).astype(np.float64)

    assert _pairwise_fst(freq, freq) == pytest.approx(0.0, abs=1e-10)

    freq2 = np.clip(freq + 0.2, 0, 1)
    fst = _pairwise_fst(freq, freq2)
    assert fst > 0.01


def test_no_consolidation_when_all_healthy():
    """No merges when all ancestries have strong support."""
    result = _make_result(A=4, H=2000, T=200)
    # Override calls so each ancestry gets ~500 haps × 200 sites = 100k high-post sites
    calls = np.zeros((2000, 200), dtype=np.int8)
    for a in range(4):
        calls[a * 500:(a + 1) * 500] = a
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, 200), 0.95, dtype=np.float16),
        global_sums=result.decode.global_sums,
    )
    # Set mu well above threshold
    result.model = AncestryModel(
        n_ancestries=4,
        mu=jnp.array([0.25, 0.25, 0.25, 0.25]),
        gen_since_admix=20.0,
        allele_freq=result.model.allele_freq,
    )

    out = consolidate([result])
    assert out[0].model.n_ancestries == 4


def test_consolidation_low_mu():
    """Ancestry with mu < 0.005 gets merged."""
    # 4 ancestries, one with tiny mu
    mu = np.array([0.49, 0.49, 0.002, 0.018], dtype=np.float32)
    result = _make_result(A=4, mu=mu, H=2000)
    # Give ancestry 2 almost no calls
    calls = np.zeros((2000, 200), dtype=np.int8)
    calls[:980] = 0
    calls[980:1960] = 1
    calls[1960:1970] = 2  # only 10 haps
    calls[1970:] = 3
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, 200), 0.9, dtype=np.float16),
        global_sums=result.decode.global_sums,
    )

    out = consolidate([result])
    assert out[0].model.n_ancestries < 4
    # Ancestry 2 should be gone
    new_mu = np.array(out[0].model.mu)
    assert new_mu.sum() == pytest.approx(1.0, abs=1e-5)


def test_consolidation_low_high_post():
    """Ancestry with < 1000 high-posterior sites gets merged."""
    mu = np.array([0.4, 0.4, 0.1, 0.1], dtype=np.float32)
    result = _make_result(A=4, mu=mu, H=2000, T=200)
    # Give ancestry 2 very low posteriors
    calls = np.zeros((2000, 200), dtype=np.int8)
    calls[:800] = 0
    calls[800:1600] = 1
    calls[1600:1800] = 2
    calls[1800:] = 3
    max_post = np.full((2000, 200), 0.95, dtype=np.float16)
    # Ancestry 2 sites have low posterior (below 0.8)
    max_post[1600:1800] = 0.5
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=max_post,
        global_sums=result.decode.global_sums,
    )

    out = consolidate([result])
    assert out[0].model.n_ancestries < 4


def test_consolidation_sibling_fst():
    """Sibling pair with F_ST < 0.008 gets merged when leaf paths provided."""
    rng = np.random.default_rng(42)
    T = 500
    base_freq = rng.beta(0.3, 0.3, size=T).astype(np.float32)
    # Ancestry 0 and 1 are siblings (L00, L01) with nearly identical freqs
    freq = np.stack([
        base_freq,
        base_freq + rng.normal(0, 0.01, T).astype(np.float32),  # near-dup of 0
        np.clip(base_freq + 0.3, 0, 1),  # distinct
        np.clip(base_freq + 0.5, 0, 1),  # distinct
    ])
    freq = np.clip(freq, 0.01, 0.99).astype(np.float32)
    mu = np.array([0.3, 0.2, 0.3, 0.2], dtype=np.float32)

    result = _make_result(A=4, T=T, mu=mu, freq=freq, H=2000)
    calls = np.zeros((2000, T), dtype=np.int8)
    calls[:600] = 0
    calls[600:1000] = 1
    calls[1000:1600] = 2
    calls[1600:] = 3
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, T), 0.95, dtype=np.float16),
        global_sums=result.decode.global_sums,
    )

    # Leaf paths: 0 and 1 are siblings (L00, L01)
    leaf_paths = ["L00", "L01", "L10", "L11"]
    fst_01 = _pairwise_fst(freq[0], freq[1])
    assert fst_01 < 0.008, f"Test setup: F_ST(0,1)={fst_01:.4f} should be < 0.008"

    out = consolidate([result], leaf_paths=leaf_paths)
    assert out[0].model.n_ancestries < 4


def test_consolidation_preserves_non_sibling():
    """Non-sibling pair with low F_ST is NOT merged (structural protection)."""
    rng = np.random.default_rng(42)
    T = 500
    base_freq = rng.beta(0.3, 0.3, size=T).astype(np.float32)
    # Ancestry 0 and 1 are near-duplicates but NOT siblings
    freq = np.stack([
        base_freq,
        base_freq + rng.normal(0, 0.01, T).astype(np.float32),  # near-dup of 0
        np.clip(base_freq + 0.3, 0, 1),
        np.clip(base_freq + 0.5, 0, 1),
    ])
    freq = np.clip(freq, 0.01, 0.99).astype(np.float32)
    mu = np.array([0.3, 0.2, 0.3, 0.2], dtype=np.float32)

    result = _make_result(A=4, T=T, mu=mu, freq=freq, H=2000)
    calls = np.zeros((2000, T), dtype=np.int8)
    calls[:600] = 0
    calls[600:1000] = 1
    calls[1000:1600] = 2
    calls[1600:] = 3
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, T), 0.95, dtype=np.float16),
        global_sums=result.decode.global_sums,
    )

    # 0 (L00) and 1 (L10) have low F_ST but are NOT siblings (different parents)
    leaf_paths = ["L00", "L10", "L01", "L11"]
    out = consolidate([result], leaf_paths=leaf_paths)
    # Should NOT merge — non-siblings are protected by sibling check
    assert out[0].model.n_ancestries == 4


def test_consolidation_report(tmp_path):
    """Consolidation writes audit TSV."""
    mu = np.array([0.49, 0.49, 0.002, 0.018], dtype=np.float32)
    result = _make_result(A=4, mu=mu, H=2000)
    calls = np.zeros((2000, 200), dtype=np.int8)
    calls[:980] = 0
    calls[980:1960] = 1
    calls[1960:1970] = 2
    calls[1970:] = 3
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, 200), 0.9, dtype=np.float16),
        global_sums=result.decode.global_sums,
    )

    prefix = str(tmp_path / "test")
    consolidate([result], out_prefix=prefix)

    report_path = f"{prefix}.post_em_consolidation.tsv"
    import os
    assert os.path.exists(report_path)
    with open(report_path) as f:
        lines = f.readlines()
    assert lines[0].startswith("source_idx")
    assert len(lines) >= 2  # header + at least one action


def test_mu_renormalized():
    """After consolidation, mu sums to 1."""
    mu = np.array([0.48, 0.48, 0.001, 0.039], dtype=np.float32)
    result = _make_result(A=4, mu=mu, H=2000)
    calls = np.zeros((2000, 200), dtype=np.int8)
    calls[:960] = 0
    calls[960:1920] = 1
    calls[1920:1922] = 2
    calls[1922:] = 3
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, 200), 0.9, dtype=np.float16),
        global_sums=result.decode.global_sums,
    )

    out = consolidate([result])
    new_mu = np.array(out[0].model.mu)
    assert new_mu.sum() == pytest.approx(1.0, abs=1e-5)
    assert all(new_mu > 0)


def test_consolidate_new_calls_is_int8():
    """Remap must not widen calls dtype from int8 to int32."""
    mu = np.array([0.49, 0.49, 0.001, 0.019], dtype=np.float32)
    result = _make_result(A=4, mu=mu, H=2000)
    calls = np.zeros((2000, 200), dtype=np.int8)
    calls[:960] = 0
    calls[960:1920] = 1
    calls[1920:1922] = 2
    calls[1922:] = 3
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, 200), 0.9, dtype=np.float16),
        global_sums=result.decode.global_sums,
    )

    out = consolidate([result])
    assert out[0].calls.dtype == np.int8


def test_consolidate_nulls_parquet_path_on_merge(tmp_path):
    """After consolidation with merges, parquet_path must be None so
    cli.py re-writes with new ancestry labels."""
    pq_path = tmp_path / "stale.parquet"
    pq_path.write_bytes(b"fake")

    mu = np.array([0.49, 0.49, 0.001, 0.019], dtype=np.float32)
    result = _make_result(A=4, mu=mu, H=2000)
    calls = np.zeros((2000, 200), dtype=np.int8)
    calls[:960] = 0
    calls[960:1920] = 1
    calls[1920:1922] = 2
    calls[1922:] = 3
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, 200), 0.9, dtype=np.float16),
        global_sums=result.decode.global_sums,
        parquet_path=str(pq_path),
    )

    out = consolidate([result])
    assert out[0].decode.parquet_path is None
    assert not pq_path.exists()


def test_consolidate_preserves_parquet_path_without_merges():
    """If consolidation finds no merges, parquet_path passes through."""
    result = _make_result(A=4, H=2000, T=200)
    calls = np.zeros((2000, 200), dtype=np.int8)
    for a in range(4):
        calls[a * 500:(a + 1) * 500] = a
    result.calls = calls
    result.decode = DecodeResult(
        calls=calls,
        max_post=np.full((2000, 200), 0.95, dtype=np.float16),
        global_sums=result.decode.global_sums,
        parquet_path="/some/path.parquet",
    )
    result.model = AncestryModel(
        n_ancestries=4,
        mu=jnp.array([0.25, 0.25, 0.25, 0.25]),
        gen_since_admix=20.0,
        allele_freq=result.model.allele_freq,
    )

    out = consolidate([result])
    assert out[0].decode.parquet_path == "/some/path.parquet"
