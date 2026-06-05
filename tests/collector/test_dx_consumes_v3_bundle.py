"""Phase 7: popout DX consumes the v3 FLARE bundle without code changes.

Phase 6 of the retrofit changed the FLARE bundle from v2 (anonymous
``ancestry_0..K-1`` columns, postS-derived multi-component
``labels.json``) to v3 (named panel columns, by_name-derived 1-to-1
``labels.json``). This test proves that popout DX's FLARE consumers
work against **both** bundle shapes without modification, because they
already key off ``labels.json:popout_to_rf_label`` /
``rf_to_popout_components`` rather than column-name strings.

The cohort-bundle global.tsv reader (``popout.viz._loaders.read_global_tsv``)
discards the header row, so v2 anonymous vs v3 named columns are
indistinguishable at the data-shape layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from validation.popout_dx.scripts.dx_loaders import (
    RF_LABELS_CANONICAL,
    load_flare_global,
    project_to_rf_basis,
)


def _write_global(path: Path, header_cols: list[str],
                  sample_ids: list[str], q: np.ndarray) -> None:
    """Emit a ``global.tsv`` with the given header column names + data."""
    assert q.shape == (len(sample_ids), len(header_cols))
    with open(path, "w") as f:
        f.write("sample_id\t" + "\t".join(header_cols) + "\n")
        for sid, row in zip(sample_ids, q):
            f.write(sid + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")


@pytest.fixture
def flare_q():
    """A small FLARE-shaped proportion matrix: 4 samples × 5 panel components."""
    return np.array(
        [
            [0.95, 0.02, 0.01, 0.01, 0.01],  # sample_0 ~ afr
            [0.05, 0.85, 0.02, 0.05, 0.03],  # sample_1 ~ amr
            [0.02, 0.10, 0.80, 0.05, 0.03],  # sample_2 ~ eas
            [0.01, 0.04, 0.05, 0.85, 0.05],  # sample_3 ~ eur
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_ids():
    return [f"sample_{i}" for i in range(4)]


def _v3_labels_json() -> dict:
    """v3 FLARE labels.json: by_name → 1-to-1 mapping into SP6."""
    return {
        "tool": "FLARE",
        "rf_ref_labels": list(RF_LABELS_CANONICAL),
        "popout_to_rf_label": {"0": "afr", "1": "amr", "2": "eas",
                                "3": "eur", "4": "sas"},
        "rf_to_popout_components": {"afr": [0], "amr": [1], "eas": [2],
                                     "eur": [3], "sas": [4]},
    }


def _v2_labels_json() -> dict:
    """v2 FLARE labels.json: postS-derived with multi-component lumping.

    This mirrors the real bug: cluster's popout_to_rf_label folds two
    FLARE components into the same RF label (the fake-subancestry path
    we retired in Phase 6).
    """
    return {
        "tool": "FLARE",
        "rf_ref_labels": list(RF_LABELS_CANONICAL),
        "popout_to_rf_label": {"0": "afr", "1": "amr", "2": "amr",
                                "3": "eur", "4": "sas"},
        "rf_to_popout_components": {"afr": [0], "amr": [1, 2],
                                     "eur": [3], "sas": [4]},
    }


def test_load_flare_global_named_columns(tmp_path: Path, flare_q, sample_ids):
    """v3-style: named panel columns. ``load_flare_global`` returns the
    same array shape as v2-style; the header is discarded."""
    path = tmp_path / "v3.global.tsv"
    _write_global(path, ["afr", "amr", "eas", "eur", "sas"], sample_ids, flare_q)
    sids, q = load_flare_global(path)
    assert sids == sample_ids
    assert q.shape == flare_q.shape
    np.testing.assert_allclose(q, flare_q, atol=1e-6)


def test_load_flare_global_anonymous_columns(tmp_path: Path, flare_q, sample_ids):
    """v2-style: ancestry_0..K-1 columns. Same data, same return shape."""
    path = tmp_path / "v2.global.tsv"
    cols = [f"ancestry_{i}" for i in range(5)]
    _write_global(path, cols, sample_ids, flare_q)
    sids, q = load_flare_global(path)
    assert sids == sample_ids
    assert q.shape == flare_q.shape
    np.testing.assert_allclose(q, flare_q, atol=1e-6)


def test_project_to_rf_basis_v3_labels(flare_q):
    """v3 labels.json: 1-to-1. Projection is the identity on the panel cols."""
    rf = project_to_rf_basis(flare_q.astype(np.float64), source="flare",
                              labels=_v3_labels_json())
    assert rf.shape == (4, len(RF_LABELS_CANONICAL))
    # SP6 col order: afr, amr, eas, eur, mid, sas
    np.testing.assert_allclose(rf[:, 0], flare_q[:, 0], atol=1e-6)  # afr
    np.testing.assert_allclose(rf[:, 1], flare_q[:, 1], atol=1e-6)  # amr
    np.testing.assert_allclose(rf[:, 2], flare_q[:, 2], atol=1e-6)  # eas
    np.testing.assert_allclose(rf[:, 3], flare_q[:, 3], atol=1e-6)  # eur
    np.testing.assert_allclose(rf[:, 4], 0.0)                         # mid (none)
    np.testing.assert_allclose(rf[:, 5], flare_q[:, 4], atol=1e-6)  # sas


def test_project_to_rf_basis_v2_labels(flare_q):
    """v2 labels.json: postS-lumped (afr.1+afr.2 → afr column). Projection
    sums all components that map to the same RF label — the same mechanism
    works for any cardinality."""
    rf = project_to_rf_basis(flare_q.astype(np.float64), source="flare",
                              labels=_v2_labels_json())
    assert rf.shape == (4, len(RF_LABELS_CANONICAL))
    # SP6 col order: afr, amr, eas, eur, mid, sas
    np.testing.assert_allclose(rf[:, 0], flare_q[:, 0], atol=1e-6)  # afr (1 comp)
    np.testing.assert_allclose(rf[:, 1],
                                flare_q[:, 1] + flare_q[:, 2], atol=1e-6)  # amr (2 lumped)
    np.testing.assert_allclose(rf[:, 2], 0.0)                         # eas (none in v2)
    np.testing.assert_allclose(rf[:, 3], flare_q[:, 3], atol=1e-6)  # eur
    np.testing.assert_allclose(rf[:, 4], 0.0)                         # mid (none)
    np.testing.assert_allclose(rf[:, 5], flare_q[:, 4], atol=1e-6)  # sas


def test_project_to_rf_basis_v3_rows_sum_to_one(flare_q):
    """v3 labels.json has total coverage of SP5, so rows preserve mass."""
    rf = project_to_rf_basis(flare_q.astype(np.float64), source="flare",
                              labels=_v3_labels_json())
    np.testing.assert_allclose(rf.sum(axis=1), flare_q.sum(axis=1), atol=1e-6)
