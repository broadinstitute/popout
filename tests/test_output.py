"""Tests for output module."""

import tempfile
from pathlib import Path

import numpy as np

from popout.datatypes import AncestryModel, AncestryResult, ChromData
from popout.output import write_model


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
