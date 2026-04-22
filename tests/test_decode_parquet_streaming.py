"""Tests for streaming decode parquet writer/reader.

Verifies that the chunked ParquetWriter produces bit-exact roundtrips
and that multiple row groups are created for large H.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from popout.datatypes import (
    AncestryModel,
    AncestryResult,
    ChromData,
    DecodeResult,
)
from popout.output import read_decode_parquet, write_decode_parquet


def _make_result(H=500, T=200, A=4, with_max_post=True, seed=0):
    rng = np.random.default_rng(seed)
    calls = rng.integers(0, A, size=(H, T), dtype=np.int8)
    max_post = (
        rng.uniform(0.3, 0.99, size=(H, T)).astype(np.float32)
        if with_max_post
        else None
    )
    decode = DecodeResult(calls=calls, max_post=max_post)
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.ones(A) / A,
        gen_since_admix=20.0,
        allele_freq=jnp.array(
            rng.random((A, T)).astype(np.float32),
        ),
    )
    result = AncestryResult(
        calls=calls, model=model, chrom="chr1", decode=decode,
    )
    chrom_data = ChromData(
        geno=np.zeros((H, T), dtype=np.uint8),
        pos_bp=np.arange(T, dtype=np.int64) * 1000,
        pos_cm=np.arange(T, dtype=np.float64) * 0.001,
        chrom="chr1",
    )
    return result, chrom_data


@pytest.mark.parametrize("H,T,with_mp", [
    (100, 50, True),
    (100, 50, False),
    (500, 200, True),
    (33, 17, True),        # non-uniform last chunk
    (75_000, 100, True),   # exceeds default hap_chunk boundary
])
def test_roundtrip_identical(tmp_path, H, T, with_mp):
    result, cdata = _make_result(H=H, T=T, with_max_post=with_mp)
    out = str(tmp_path / "decode.parquet")
    write_decode_parquet(result, cdata, out, include_max_post=with_mp)

    data = read_decode_parquet(out)
    np.testing.assert_array_equal(
        data["calls"], result.calls.view(np.uint8),
    )
    if with_mp:
        np.testing.assert_array_equal(
            data["max_post"], result.decode.max_post.astype(np.float16),
        )
    else:
        assert "max_post" not in data
    np.testing.assert_array_equal(data["pos_bp"], cdata.pos_bp)
    assert data["chrom"] == "chr1"
    assert data["T"] == T
    assert data["K"] == result.model.n_ancestries


def test_row_groups_multiple(tmp_path):
    """Writer must produce >1 row group when H exceeds hap_chunk."""
    import pyarrow.parquet as pq

    result, cdata = _make_result(H=120_000, T=50, with_max_post=False)
    out = str(tmp_path / "decode.parquet")
    write_decode_parquet(result, cdata, out, include_max_post=False)

    pf = pq.ParquetFile(out)
    assert pf.num_row_groups >= 2, (
        f"Expected multiple row groups for H=120000, got {pf.num_row_groups}"
    )


def test_small_hap_chunk(tmp_path):
    """Explicit small hap_chunk produces correct results."""
    result, cdata = _make_result(H=500, T=100, with_max_post=True)
    out = str(tmp_path / "decode.parquet")
    write_decode_parquet(result, cdata, out, include_max_post=True, hap_chunk=100)

    data = read_decode_parquet(out)
    np.testing.assert_array_equal(
        data["calls"], result.calls.view(np.uint8),
    )
    np.testing.assert_array_equal(
        data["max_post"], result.decode.max_post.astype(np.float16),
    )

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(out)
    assert pf.num_row_groups == 5  # 500 / 100
