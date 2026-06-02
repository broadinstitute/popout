"""Phase 3: projector invariants for proportions, tract codes, and collapse."""

from __future__ import annotations

import numpy as np
import pytest

from popout.labelspace import (
    Assignment,
    collapse,
    get,
    project_proportions,
    project_tracts,
)
from popout.benchmark.common import MISSING_LABEL, TractSet


SP6 = get("SP6")
SP5 = get("SP5")


# ── Reusable assignment builders ────────────────────────────────────────


def _assignment_two_into_afr() -> Assignment:
    """7 popout components mapping into SP6, with components 0 & 5 → afr."""
    p2rf = {0: "afr", 1: "amr", 2: "eas", 3: "eur",
            4: "mid", 5: "afr", 6: "sas"}
    return Assignment(
        target_space=SP6, source={"tool": "popout"},
        method="corrH", input_space="allele_freq",
        component_to_label=p2rf,
        label_to_components={"afr": [0, 5], "amr": [1], "eas": [2],
                             "eur": [3], "mid": [4], "sas": [6]},
        subcomponent_names={}, diagnostics={}, provenance={},
    )


def _assignment_with_unassigned() -> Assignment:
    return Assignment(
        target_space=SP6, source={"tool": "popout"},
        method="manual", input_space="hard_call",
        component_to_label={0: "afr", 1: "unassigned", 2: "eur"},
        label_to_components={"afr": [0], "eur": [2], "unassigned": [1]},
        subcomponent_names={}, diagnostics={}, provenance={},
    )


# ── project_proportions ─────────────────────────────────────────────────


def test_project_proportions_basic_sum():
    rng = np.random.default_rng(0)
    q = rng.dirichlet([1.0] * 7, size=5)
    a = _assignment_two_into_afr()
    out = project_proportions(q, a)
    assert out.shape == (5, 6)
    # afr column equals sum of columns 0 + 5.
    np.testing.assert_allclose(out[:, SP6.index("afr")], q[:, 0] + q[:, 5])
    # Total mass preserved (no unassigned).
    np.testing.assert_allclose(out.sum(axis=1), q.sum(axis=1))


def test_project_proportions_order_invariant_under_component_shuffle():
    """Swapping the order of [0, 5] in label_to_components["afr"] yields the same sum."""
    rng = np.random.default_rng(1)
    q = rng.dirichlet([1.0] * 7, size=8)
    a1 = _assignment_two_into_afr()
    a2 = Assignment(**{**a1.__dict__, "label_to_components": {
        **a1.label_to_components, "afr": [5, 0]
    }})
    np.testing.assert_array_equal(project_proportions(q, a1),
                                   project_proportions(q, a2))


def test_project_proportions_drops_unassigned_mass():
    rng = np.random.default_rng(2)
    q = rng.dirichlet([1.0] * 3, size=4)
    a = _assignment_with_unassigned()
    out = project_proportions(q, a)
    assert out.shape == (4, 6)
    # afr + eur should equal q[:,0] + q[:,2] (component 1 was unassigned).
    afr = out[:, SP6.index("afr")]
    eur = out[:, SP6.index("eur")]
    np.testing.assert_allclose(afr, q[:, 0])
    np.testing.assert_allclose(eur, q[:, 2])
    # Total is 1 - q[:,1].
    np.testing.assert_allclose(afr + eur, q[:, 0] + q[:, 2])


# ── project_tracts ──────────────────────────────────────────────────────


def _make_tractset(calls: np.ndarray, label_map: dict[int, str]) -> TractSet:
    n_haps, n_sites = calls.shape
    return TractSet(
        tool_name="popout", chrom="chr1",
        hap_ids=np.array([f"s{i//2}:{i%2}" for i in range(n_haps)], dtype=object),
        site_positions=np.arange(n_sites, dtype=np.int64) * 10000 + 1,
        calls=calls.astype(np.uint16),
        label_map=label_map,
    )


def test_project_tracts_remaps_via_lut():
    a = _assignment_two_into_afr()
    calls = np.array([[0, 5, 1, 2, 3, 4, 6, 0],
                      [5, 0, 6, 4, 3, 2, 1, 5]])
    ts = _make_tractset(calls, {i: f"src_{i}" for i in range(7)})
    out = project_tracts(ts, a)
    # 0, 5 → afr (target index 0); 1 → amr (1); 2 → eas (2); 3 → eur (3); 4 → mid (4); 6 → sas (5)
    expected = np.array([[0, 0, 1, 2, 3, 4, 5, 0],
                         [0, 0, 5, 4, 3, 2, 1, 0]], dtype=np.uint16)
    np.testing.assert_array_equal(out.calls, expected)
    assert out.label_map == {i: m for i, m in enumerate(SP6.members)}


def test_project_tracts_preserves_missing_label():
    a = _assignment_two_into_afr()
    calls = np.array([[0, MISSING_LABEL, 1, 2]])
    ts = _make_tractset(calls, {0: "x", 1: "y", 2: "z"})
    out = project_tracts(ts, a)
    assert out.calls[0, 1] == MISSING_LABEL


def test_project_tracts_rejects_unknown_source_code():
    a = _assignment_two_into_afr()
    calls = np.array([[0, 99, 1]])
    ts = _make_tractset(calls, {0: "x", 99: "y", 1: "z"})
    with pytest.raises(ValueError):
        project_tracts(ts, a)


# ── collapse SP6 → SP5 ──────────────────────────────────────────────────


def test_collapse_sp6_to_sp5_drop_mid():
    q = np.array([[0.1, 0.2, 0.0, 0.3, 0.4, 0.0],
                  [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]])
    out = collapse(q, SP6, SP5, rule="drop")
    assert out.shape == (2, 5)
    # MID column (index 4) is dropped; the rest preserved.
    np.testing.assert_array_equal(out[0], [0.1, 0.2, 0.0, 0.3, 0.0])
    np.testing.assert_array_equal(out[1], [0.0, 0.0, 0.5, 0.5, 0.0])


def test_collapse_rejects_unsupported_rule():
    q = np.zeros((1, 6))
    with pytest.raises(ValueError):
        collapse(q, SP6, SP5, rule="redistribute")


def test_collapse_rejects_mismatched_dst():
    q = np.zeros((1, 6))
    with pytest.raises(ValueError):
        collapse(q, SP6, SP6, rule="drop")


# ── Routing sanity: dx_loaders.project_to_rf_basis ──────────────────────


def test_dx_loaders_routes_through_labelspace():
    from validation.popout_dx.scripts.dx_loaders import project_to_rf_basis

    rng = np.random.default_rng(3)
    q = rng.dirichlet([1.0] * 7, size=4)
    labels = {
        "rf_ref_labels": list(SP6.members),
        "popout_to_rf_label": {"0": "afr", "1": "amr", "2": "eas",
                                "3": "eur", "4": "mid", "5": "afr", "6": "sas"},
        "rf_to_popout_components": {
            "afr": [0, 5], "amr": [1], "eas": [2],
            "eur": [3], "mid": [4], "sas": [6],
        },
    }
    out = project_to_rf_basis(q, source="popout", labels=labels)
    # Numerically equivalent to the direct labelspace.project_proportions call.
    a = _assignment_two_into_afr()
    np.testing.assert_array_equal(out, project_proportions(q, a))
