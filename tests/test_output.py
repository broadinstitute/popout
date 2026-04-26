"""Tests for output module."""

import tempfile
from pathlib import Path

import numpy as np

from popout.datatypes import AncestryModel, AncestryResult, ChromData, DecodeResult
from popout.output import (
    write_model,
    write_ancestry_tracts,
    write_decode_parquet,
    read_decode_parquet,
    DecodeParquetWriter,
    _merge_bucket_parquets,
)


def _make_minimal_result(n_ancestries=3, n_haps=10, n_sites=100):
    """Build a minimal AncestryResult for testing output functions."""
    import jax.numpy as jnp

    rng = np.random.default_rng(42)
    model = AncestryModel(
        n_ancestries=n_ancestries,
        mu=jnp.array(rng.dirichlet(np.ones(n_ancestries))),
        gen_since_admix=20.0,
        allele_freq=jnp.array(rng.random((n_ancestries, n_sites)).astype(np.float32)),
    )
    calls = rng.integers(0, n_ancestries, size=(n_haps, n_sites)).astype(np.int8)
    chrom_data = ChromData(
        geno=rng.integers(0, 2, size=(n_haps, n_sites)).astype(np.uint8),
        pos_bp=np.arange(n_sites, dtype=np.int64) * 1000 + 100000,
        pos_cm=np.linspace(0, 10, n_sites),
        chrom="chr1",
    )
    result = AncestryResult(calls=calls, model=model, chrom="chr1")
    return result, chrom_data


def test_write_model_with_names():
    """ancestry_names stored in model.npz as dtype object array of length K."""
    result, chrom_data = _make_minimal_result(n_ancestries=3)
    names = ["afr", "eas", "eur"]

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test.model")
        write_model(result, out_path, chrom_data=chrom_data, ancestry_names=names)

        data = np.load(f"{out_path}.npz", allow_pickle=True)
        assert "ancestry_names" in data
        loaded = data["ancestry_names"]
        assert loaded.dtype == object
        assert len(loaded) == 3
        assert list(loaded) == names


def test_write_model_without_names():
    """When ancestry_names is None, key is absent from npz."""
    result, chrom_data = _make_minimal_result(n_ancestries=3)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test.model")
        write_model(result, out_path, chrom_data=chrom_data)

        data = np.load(f"{out_path}.npz", allow_pickle=True)
        assert "ancestry_names" not in data


def test_write_model_names_length_mismatch():
    """Mismatched ancestry_names length raises ValueError."""
    result, chrom_data = _make_minimal_result(n_ancestries=3)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test.model")
        try:
            write_model(result, out_path, chrom_data=chrom_data, ancestry_names=["a", "b"])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "2 entries" in str(e)
            assert "3 ancestries" in str(e)


def test_tracts_mean_posterior_from_max_post():
    """mean_posterior column computed from decode.max_post matches expected values."""
    import jax.numpy as jnp
    import gzip

    n_ancestries, n_haps, n_sites = 3, 4, 20
    rng = np.random.default_rng(99)
    calls = np.zeros((n_haps, n_sites), dtype=np.int8)
    # Set up known tracts: hap0 = all ancestry 0, hap1 = first 10 anc 1 then anc 2
    calls[0, :] = 0
    calls[1, :] = 0
    calls[2, :10] = 1
    calls[2, 10:] = 2
    calls[3, :10] = 1
    calls[3, 10:] = 2

    max_post = rng.random((n_haps, n_sites)).astype(np.float32) * 0.3 + 0.7

    model = AncestryModel(
        n_ancestries=n_ancestries,
        mu=jnp.array([0.4, 0.3, 0.3]),
        gen_since_admix=20.0,
        allele_freq=jnp.array(rng.random((n_ancestries, n_sites)).astype(np.float32)),
    )
    decode = DecodeResult(calls=calls, max_post=max_post)
    result = AncestryResult(calls=calls, model=model, chrom="chr1", decode=decode)
    chrom_data = ChromData(
        geno=rng.integers(0, 2, size=(n_haps, n_sites)).astype(np.uint8),
        pos_bp=np.arange(n_sites, dtype=np.int64) * 1000 + 100000,
        pos_cm=np.linspace(0, 10, n_sites),
        chrom="chr1",
    )

    sample_names = ["S0", "S1"]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test.tracts.tsv.gz")
        write_ancestry_tracts(
            [result], [chrom_data], 2, sample_names, out_path,
            write_posteriors=True,
        )
        with gzip.open(out_path, "rt") as f:
            lines = f.readlines()

    # Parse mean_posterior from the output
    header = lines[0].strip().split("\t")
    assert "mean_posterior" in header
    mp_col = header.index("mean_posterior")

    for line in lines[1:]:
        parts = line.strip().split("\t")
        mean_val = float(parts[mp_col])
        # Verify it's in a reasonable range
        assert 0.0 < mean_val <= 1.0

    # Check specific tract: hap 0 of sample S0 is all ancestry 0, all 20 sites
    # mean_posterior should be mean(max_post[0, 0:20])
    data_lines = [l.strip().split("\t") for l in lines[1:]]
    hap0_lines = [l for l in data_lines if l[3] == "S0" and l[4] == "0"]
    assert len(hap0_lines) == 1  # single tract
    expected = float(max_post[0, :].mean())
    actual = float(hap0_lines[0][mp_col])
    np.testing.assert_almost_equal(actual, expected, decimal=3)


def test_tracts_posteriors_fallback():
    """Legacy fallback: posteriors tensor produces same output as max_post path."""
    import jax.numpy as jnp
    import gzip

    n_ancestries, n_haps, n_sites = 2, 2, 10
    rng = np.random.default_rng(77)
    calls = np.zeros((n_haps, n_sites), dtype=np.int8)
    calls[0, :] = 0
    calls[1, :] = 1

    # Build posteriors tensor where argmax matches calls
    posteriors = np.zeros((n_haps, n_sites, n_ancestries), dtype=np.float32)
    for h in range(n_haps):
        for t in range(n_sites):
            anc = calls[h, t]
            posteriors[h, t, anc] = 0.8 + rng.random() * 0.15
            for a in range(n_ancestries):
                if a != anc:
                    posteriors[h, t, a] = (1 - posteriors[h, t, anc]) / max(n_ancestries - 1, 1)

    model = AncestryModel(
        n_ancestries=n_ancestries,
        mu=jnp.array([0.5, 0.5]),
        gen_since_admix=20.0,
        allele_freq=jnp.array(rng.random((n_ancestries, n_sites)).astype(np.float32)),
    )
    # No decode — fallback to posteriors
    result = AncestryResult(
        calls=calls, model=model, chrom="chr1",
        posteriors=jnp.array(posteriors),
    )
    chrom_data = ChromData(
        geno=rng.integers(0, 2, size=(n_haps, n_sites)).astype(np.uint8),
        pos_bp=np.arange(n_sites, dtype=np.int64) * 1000 + 100000,
        pos_cm=np.linspace(0, 10, n_sites),
        chrom="chr1",
    )

    sample_names = ["S0"]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test.tracts.tsv.gz")
        write_ancestry_tracts(
            [result], [chrom_data], 1, sample_names, out_path,
            write_posteriors=True,
        )
        with gzip.open(out_path, "rt") as f:
            lines = f.readlines()

    header = lines[0].strip().split("\t")
    mp_col = header.index("mean_posterior")

    # For hap 0 (all anc 0), mean_post = posteriors[0,:,0].mean() = max_post[0,:].mean()
    expected_max_post = posteriors.max(axis=2)  # (H, T)
    data_lines = [l.strip().split("\t") for l in lines[1:]]
    hap0_line = [l for l in data_lines if l[4] == "0"][0]
    expected = float(expected_max_post[0, :].mean())
    actual = float(hap0_line[mp_col])
    np.testing.assert_almost_equal(actual, expected, decimal=3)


def test_block_emissions_max_post():
    """Block-emissions decode computes max_post when write_dense_decode=True."""
    from popout.simulate import simulate_admixed
    from popout.em import run_em

    chrom_data, true_ancestry, true_params = simulate_admixed(
        n_samples=50, n_sites=200, n_ancestries=3,
        gen_since_admix=20, rng_seed=42,
    )
    result = run_em(
        chrom_data,
        n_ancestries=3,
        n_em_iter=2,
        gen_since_admix=20.0,
        use_block_emissions=True,
        block_size=8,
        write_dense_decode=True,
    )
    assert result.decode is not None
    assert result.decode.max_post is not None
    assert result.decode.max_post.shape == result.calls.shape
    # max_post values should be valid probabilities
    assert result.decode.max_post.min() >= 0.0
    assert result.decode.max_post.max() <= 1.0 + 1e-6


def test_block_emissions_no_max_post_by_default():
    """Block-emissions decode omits max_post when write_dense_decode=False."""
    from popout.simulate import simulate_admixed
    from popout.em import run_em

    chrom_data, true_ancestry, true_params = simulate_admixed(
        n_samples=50, n_sites=200, n_ancestries=3,
        gen_since_admix=20, rng_seed=42,
    )
    result = run_em(
        chrom_data,
        n_ancestries=3,
        n_em_iter=2,
        gen_since_admix=20.0,
        use_block_emissions=True,
        block_size=8,
        write_dense_decode=False,
    )
    assert result.decode is not None
    assert result.decode.max_post is None


def test_write_decode_parquet_roundtrip():
    """write_decode_parquet round-trips calls, pos_bp, max_post, T, K, chrom."""
    import jax.numpy as jnp

    n_ancestries, n_haps, n_sites = 3, 6, 50
    rng = np.random.default_rng(55)
    calls = rng.integers(0, n_ancestries, size=(n_haps, n_sites)).astype(np.int8)
    max_post_orig = rng.random((n_haps, n_sites)).astype(np.float32) * 0.3 + 0.7
    pos_bp = np.arange(n_sites, dtype=np.int64) * 1000 + 100000

    model = AncestryModel(
        n_ancestries=n_ancestries,
        mu=jnp.array([0.4, 0.3, 0.3]),
        gen_since_admix=20.0,
        allele_freq=jnp.array(rng.random((n_ancestries, n_sites)).astype(np.float32)),
    )
    decode = DecodeResult(calls=calls, max_post=max_post_orig)
    result = AncestryResult(calls=calls, model=model, chrom="chr1", decode=decode)
    chrom_data = ChromData(
        geno=rng.integers(0, 2, size=(n_haps, n_sites)).astype(np.uint8),
        pos_bp=pos_bp,
        pos_cm=np.linspace(0, 10, n_sites),
        chrom="chr1",
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test.chr1.decode.parquet")
        write_decode_parquet(result, chrom_data, out_path, include_max_post=True)

        data = read_decode_parquet(out_path)
        np.testing.assert_array_equal(data["calls"], calls.astype(np.uint8))
        np.testing.assert_array_equal(data["pos_bp"], pos_bp)
        assert data["chrom"] == "chr1"
        assert data["T"] == n_sites
        assert data["K"] == n_ancestries
        # float16 precision: exact bit-pattern round-trip
        expected_f16 = max_post_orig.astype(np.float16)
        np.testing.assert_array_equal(data["max_post"], expected_f16)


def test_write_decode_parquet_no_max_post():
    """write_decode_parquet omits max_post when include_max_post=False."""
    import jax.numpy as jnp

    n_ancestries, n_haps, n_sites = 2, 4, 20
    rng = np.random.default_rng(66)
    calls = rng.integers(0, n_ancestries, size=(n_haps, n_sites)).astype(np.int8)

    model = AncestryModel(
        n_ancestries=n_ancestries,
        mu=jnp.array([0.5, 0.5]),
        gen_since_admix=20.0,
        allele_freq=jnp.array(rng.random((n_ancestries, n_sites)).astype(np.float32)),
    )
    result = AncestryResult(calls=calls, model=model, chrom="chr2")
    chrom_data = ChromData(
        geno=rng.integers(0, 2, size=(n_haps, n_sites)).astype(np.uint8),
        pos_bp=np.arange(n_sites, dtype=np.int64) * 1000,
        pos_cm=np.linspace(0, 5, n_sites),
        chrom="chr2",
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "test.chr2.decode.parquet")
        write_decode_parquet(result, chrom_data, out_path, include_max_post=False)

        data = read_decode_parquet(out_path)
        assert "max_post" not in data
        np.testing.assert_array_equal(data["calls"], calls.astype(np.uint8))


def test_read_decode_parquet_multichunk_max_post(tmp_path):
    """Regression: max_post must not be silently overwritten with chunks[0]
    when a row group has multiple chunks per column."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    H_per_chunk, T, n_chunks = 10, 4, 3
    total_H = H_per_chunk * n_chunks

    calls_arrays = []
    mp_arrays = []
    for i in range(n_chunks):
        # Distinct content per chunk so any cross-contamination is detectable
        c_bytes = bytes([i + 1] * (H_per_chunk * T))
        c_off = np.arange(0, H_per_chunk * T + 1, T, dtype=np.int64)
        calls_arrays.append(pa.Array.from_buffers(
            pa.large_binary(), H_per_chunk,
            [None, pa.py_buffer(c_off), pa.py_buffer(c_bytes)],
        ))

        mp_vals = np.full(H_per_chunk * T, i * 10.0, dtype=np.float16)
        mp_off = np.arange(0, H_per_chunk * T * 2 + 1, T * 2, dtype=np.int64)
        mp_arrays.append(pa.Array.from_buffers(
            pa.large_binary(), H_per_chunk,
            [None, pa.py_buffer(mp_off), pa.py_buffer(mp_vals.tobytes())],
        ))

    schema = pa.schema(
        [("calls", pa.large_binary()), ("max_post", pa.large_binary())],
        metadata={
            b"T": str(T).encode("ascii"),
            b"K": b"4",
            b"chrom": b"1",
            b"calls_dtype": b"uint8",
            b"max_post_dtype": b"float16",
            b"pos_bp": np.arange(T, dtype=np.int64).tobytes(),
        },
    )
    table = pa.table({
        "calls": pa.chunked_array(calls_arrays),
        "max_post": pa.chunked_array(mp_arrays),
    }, schema=schema)

    path = tmp_path / "multichunk.parquet"
    # Row group covers the full table so all chunks land in one row group
    pq.write_table(table, str(path), row_group_size=total_H * 2)

    # Verify the file actually has the multi-chunk structure we need
    pf = pq.ParquetFile(str(path))
    rg = pf.read_row_group(0)
    # If pyarrow fused chunks despite our setup, skip — test is moot
    if len(rg.column("calls").chunks) <= 1:
        pytest.skip("pyarrow fused chunks; cannot exercise multi-chunk path")

    data = read_decode_parquet(str(path))

    # Expected: chunk i's max_post is all-(i*10). If the bug is present,
    # all rows will be zero (chunks[0]'s value).
    for i in range(n_chunks):
        s, e = i * H_per_chunk, (i + 1) * H_per_chunk
        expected = np.full((H_per_chunk, T), i * 10.0, dtype=np.float16)
        np.testing.assert_array_equal(
            data["max_post"][s:e], expected,
            err_msg=f"max_post chunk {i} corrupted — likely chunks[0] bug",
        )


def test_bucket_parquet_merge_preserves_order(tmp_path):
    """Merging 3 per-bucket parquets whose haps are interleaved (NOT in
    bucket-contiguous order) produces a single output parquet whose rows
    are in monotone hap-id order."""
    T = 12
    K = 3
    pos_bp = np.arange(T, dtype=np.int64) * 1000

    bucket_hap_indices = [
        np.array([0, 3, 6, 9], dtype=np.int64),
        np.array([1, 4, 7, 10], dtype=np.int64),
        np.array([2, 5, 8, 11], dtype=np.int64),
    ]

    bucket_paths = []
    for b, hap_ids in enumerate(bucket_hap_indices):
        path = tmp_path / f"bucket{b}.parquet"
        bucket_paths.append(path)
        writer = DecodeParquetWriter(
            str(path), T=T, K=K, chrom="sim", pos_bp=pos_bp,
            include_max_post=True,
        )
        n = len(hap_ids)
        calls = np.full((n, T), -1, dtype=np.int8)
        max_post = np.zeros((n, T), dtype=np.float16)
        for i, hid in enumerate(hap_ids):
            calls[i, :] = (hid % K)
            max_post[i, :] = np.float16(hid * 0.01)
        writer.write_batch(calls, max_post)
        writer.close()

    out_path = tmp_path / "merged.parquet"
    _merge_bucket_parquets(
        bucket_paths, bucket_hap_indices, out_path,
        chrom="sim", pos_bp=pos_bp, T=T, K=K, include_max_post=True,
        out_row_group_size=5,
    )
    data = read_decode_parquet(str(out_path))
    assert data["calls"].shape == (12, T)
    for h in range(12):
        got = int(data["calls"][h, 0])
        assert got == h % K, f"row {h} calls should be {h % K}, got {got}"
        np.testing.assert_allclose(
            float(data["max_post"][h, 0]), float(np.float16(h * 0.01)), atol=1e-6
        )
