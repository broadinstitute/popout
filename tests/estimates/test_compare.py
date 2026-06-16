"""Phase 1: Estimate × Estimate comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest

from popout.estimates import Estimate, compare
from popout.labelspace import get
from popout.labelspace.registry import SP5, SP6, make_native_space


def _flare_estimate(proportions: list[list[float]],
                    sample_ids: list[str] | None = None) -> Estimate:
    sample_ids = sample_ids or [f"s{i:03d}" for i in range(len(proportions))]
    return Estimate(
        tool="flare", scope=("cohort",),
        sample_ids=tuple(sample_ids),
        label_space=make_native_space("flare", ("afr", "amr", "eas", "eur", "sas")),
        proportions=np.asarray(proportions, dtype=np.float64),
    )


def _rye_estimate(proportions: list[list[float]],
                  sample_ids: list[str] | None = None) -> Estimate:
    """Note: rye native order is (eur, eas, amr, afr, sas) — distinct from SP5."""
    sample_ids = sample_ids or [f"s{i:03d}" for i in range(len(proportions))]
    return Estimate(
        tool="rye", scope=("cohort",),
        sample_ids=tuple(sample_ids),
        label_space=make_native_space("rye", ("eur", "eas", "amr", "afr", "sas")),
        proportions=np.asarray(proportions, dtype=np.float64),
    )


def _rf_estimate(proportions: list[list[float]], hard_calls: list[str],
                 sample_ids: list[str] | None = None) -> Estimate:
    sample_ids = sample_ids or [f"s{i:03d}" for i in range(len(proportions))]
    return Estimate(
        tool="rf", scope=("cohort",),
        sample_ids=tuple(sample_ids),
        label_space=SP6,
        proportions=np.asarray(proportions, dtype=np.float64),
        hard_calls=np.asarray(hard_calls, dtype=object),
    )


# ── Basic concordance ───────────────────────────────────────────────────


def test_compare_identical_estimates_yields_ccc_one():
    e = _flare_estimate([[0.2, 0.1, 0.0, 0.6, 0.1],
                         [0.0, 0.0, 0.4, 0.5, 0.1],
                         [0.9, 0.05, 0.0, 0.05, 0.0]])
    # rye in its native order; we'll project to SP5.
    r = _rye_estimate([[0.6, 0.0, 0.1, 0.2, 0.1],
                       [0.5, 0.4, 0.0, 0.0, 0.1],
                       [0.05, 0.0, 0.05, 0.9, 0.0]])
    # Compare e with a permuted-to-rye-order copy of itself's proportions —
    # but cheaper: just compare e with a rye that holds the SAME values
    # under its native ordering. Easier: build the rye from e by permuting
    # columns.
    perm = [e.label_space.index(m) for m in ("eur", "eas", "amr", "afr", "sas")]
    r = _rye_estimate(e.proportions[:, perm].tolist())
    out = compare(e, r, target_space="SP5", mid_rule="drop")
    for lab in SP5.members:
        ccc = out.per_label[lab]["ccc"]
        if e.column(lab).std() == 0:
            assert math.isnan(ccc) or ccc == 0.0
        else:
            assert math.isclose(ccc, 1.0, abs_tol=1e-9), (lab, ccc)


def test_compare_target_space_auto_picks_sp5_when_no_mid():
    e = _flare_estimate([[0.5, 0.5, 0.0, 0.0, 0.0]])
    r = _rye_estimate([[0.0, 0.0, 0.5, 0.5, 0.0]])
    out = compare(e, r)
    assert out.target_space.tag == "SP5"


def test_compare_roster_intersection_default():
    e = _flare_estimate(
        [[0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
        sample_ids=["s0", "s1"],
    )
    r = _rye_estimate(
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]],
        sample_ids=["s0", "s2"],     # s1 missing on right
    )
    out = compare(e, r, mid_rule="drop")
    assert out.sample_ids == ("s0",)
    assert out.per_label["afr"]["n"] == 1


# ── MID-fold-to-eur (FLARE vs RF) ───────────────────────────────────────


def test_compare_flare_vs_rf_with_mid_fold():
    # Two samples. FLARE has no MID column (panel is SP5-native);
    # RF puts 0.3 mass on mid for s0 and zero for s1. After
    # fold_to_eur, the RF eur column becomes 0.3 + (RF's eur).
    e = _flare_estimate([[0.0, 0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0]])
    rf = _rf_estimate(
        proportions=[[0.0, 0.0, 0.0, 0.7, 0.3, 0.0],   # eur=0.7, mid=0.3
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],   # sas=1.0
        hard_calls=["mid", "sas"],
    )
    out = compare(e, rf, target_space="SP5", mid_rule="fold_to_eur")
    # FLARE's eur is [1.0, 0.0]; RF after fold: eur column = [1.0, 0.0].
    assert out.target_space.tag == "SP5"
    np.testing.assert_allclose(
        [out.per_label["eur"]["cluster_mu"]], [0.5],
    )
    # Confusion: s0 was rf=mid → eur after fold; rf=eur means RF column
    # eur counts s0. FLARE's hard_calls is None so no confusion is built
    # in this case.
    assert out.confusion is None
    assert "MID->eur" in out.tag


def test_compare_flare_vs_rf_drop_rule_zeroes_mid_mass():
    e = _flare_estimate([[0.0, 0.0, 0.0, 1.0, 0.0]])
    rf = _rf_estimate(
        proportions=[[0.0, 0.0, 0.0, 0.7, 0.3, 0.0]],
        hard_calls=["mid"],
    )
    out = compare(e, rf, target_space="SP5", mid_rule="drop")
    # RF eur stays 0.7 (mid dropped, not redistributed).
    assert math.isclose(out.per_label["eur"]["cluster_mu"], 1.0)  # anchor mean
    # right-side eur for this sample is 0.7, so per-sample error is 0.3
    np.testing.assert_allclose(
        [out.per_label["eur"]["mae_mean"]], [0.3], rtol=1e-6,
    )


# ── Hard-call metrics ──────────────────────────────────────────────────


def test_compare_hard_calls_yield_confusion_and_cluster_metrics():
    # Two RF Estimates so both have hard_calls.
    rf_a = _rf_estimate(
        proportions=[[0.9, 0.0, 0.0, 0.1, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        hard_calls=["afr", "eur"],
    )
    rf_b = _rf_estimate(
        proportions=[[0.85, 0.0, 0.0, 0.1, 0.05, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        hard_calls=["afr", "eur"],
    )
    out = compare(rf_a, rf_b, target_space="SP6", mid_rule=None)
    assert out.confusion is not None
    assert out.confusion.shape == (6, 6)
    # diagonal counts: 1 afr, 1 eur
    assert out.confusion[0, 0] == 1   # afr×afr
    assert out.confusion[3, 3] == 1   # eur×eur
    assert out.hard_metrics is not None
    assert math.isclose(out.hard_metrics["ari"], 1.0)
    assert math.isclose(out.hard_metrics["nmi"], 1.0)


# ── Tag format ─────────────────────────────────────────────────────────


def test_tag_lists_only_participating_tools():
    e = _flare_estimate([[1.0, 0.0, 0.0, 0.0, 0.0]])
    r = _rye_estimate([[0.0, 0.0, 0.0, 1.0, 0.0]])
    out = compare(e, r, mid_rule="drop")
    assert "popout" not in out.tag
    assert "rf" not in out.tag
    assert "flare=>name" in out.tag
    assert "rye=>name" in out.tag
    assert out.tag.startswith("L=SP5/MID-")
