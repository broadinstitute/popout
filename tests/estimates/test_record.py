"""Phase 1: Estimate dataclass + serdes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from popout.estimates import Estimate
from popout.labelspace import get
from popout.labelspace.registry import make_native_space


SP6 = get("SP6")


def _make_flare_estimate(n_samples: int = 3) -> Estimate:
    ls = make_native_space("flare", ("afr", "amr", "eas", "eur", "sas"))
    return Estimate(
        tool="flare",
        scope=("cluster_000", "chr1"),
        sample_ids=tuple(f"s{i:03d}" for i in range(n_samples)),
        label_space=ls,
        proportions=np.array([
            [0.1, 0.2, 0.0, 0.6, 0.1],
            [0.0, 0.0, 0.5, 0.4, 0.1],
            [0.9, 0.0, 0.0, 0.05, 0.05],
        ][:n_samples], dtype=np.float64),
        hard_calls=None,
        provenance={"source": "synthetic"},
    )


def test_construct_and_basic_access():
    e = _make_flare_estimate()
    assert e.tool == "flare"
    assert e.n_samples == 3
    assert e.members == ("afr", "amr", "eas", "eur", "sas")
    np.testing.assert_allclose(e.column("afr"), [0.1, 0.0, 0.9])


def test_reject_anonymous_column_names():
    with pytest.raises(ValueError, match="must be named"):
        Estimate(
            tool="flare", scope=("x",),
            sample_ids=("s0",),
            label_space=make_native_space("flare", ("ancestry_0", "ancestry_1")),
            proportions=np.array([[0.5, 0.5]]),
        )


def test_reject_shape_mismatch():
    with pytest.raises(ValueError, match="rows"):
        Estimate(
            tool="flare", scope=("x",),
            sample_ids=("s0", "s1"),       # 2 ids
            label_space=make_native_space("flare", ("a", "b")),
            proportions=np.array([[0.5, 0.5]]),    # 1 row
        )


def test_roundtrip_json(tmp_path: Path):
    e = _make_flare_estimate()
    out = tmp_path / "estimate.json"
    e.dump(out)
    back = Estimate.load(out)
    assert e == back


def test_roundtrip_named_tsv(tmp_path: Path):
    e = _make_flare_estimate()
    tsv = tmp_path / "global.tsv"
    e.to_named_tsv(tsv)
    text = tsv.read_text()
    assert text.startswith("sample_id\tafr\tamr\teas\teur\tsas\n")

    back = Estimate.from_named_tsv(tsv, tool="flare",
                                    scope=("cluster_000", "chr1"))
    assert back.members == e.members
    assert back.sample_ids == e.sample_ids
    np.testing.assert_allclose(back.proportions, e.proportions, atol=1e-6)


def test_from_named_tsv_refuses_anonymous_columns(tmp_path: Path):
    tsv = tmp_path / "anon.tsv"
    tsv.write_text("sample_id\tancestry_0\tancestry_1\ns0\t0.5\t0.5\n")
    with pytest.raises(ValueError, match="anonymous columns"):
        Estimate.from_named_tsv(tsv, tool="flare", scope=("x",))


def test_sp6_loaded_estimate_round_trips(tmp_path: Path):
    e = Estimate(
        tool="rf", scope=("cohort",),
        sample_ids=("s0", "s1"),
        label_space=SP6,
        proportions=np.array([
            [0.7, 0.1, 0.05, 0.1, 0.0, 0.05],
            [0.0, 0.0, 0.0, 0.95, 0.05, 0.0],
        ]),
        hard_calls=np.array(["afr", "eur"], dtype=object),
    )
    out = tmp_path / "rf.json"
    e.dump(out)
    back = Estimate.load(out)
    assert back.label_space is SP6
    assert back.hard_calls is not None
    np.testing.assert_array_equal(back.hard_calls, ["afr", "eur"])
