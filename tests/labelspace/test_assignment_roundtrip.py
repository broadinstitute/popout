"""Phase 1: Assignment round-trips and v1 → v2 upgrade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from popout.labelspace import Assignment, get
from popout.labelspace.assignment import SCHEMA_TAG


SP6 = get("SP6")


def _make_assignment() -> Assignment:
    return Assignment(
        target_space=SP6,
        source={"tool": "popout", "run": "fixture", "K": 7},
        method="corrH",
        input_space="allele_freq",
        component_to_label={0: "afr", 1: "amr", 2: "eas",
                            3: "eur", 4: "mid", 5: "afr", 6: "sas"},
        label_to_components={"afr": [0, 5], "amr": [1], "eas": [2],
                              "eur": [3], "mid": [4], "sas": [6]},
        subcomponent_names={0: "afr.1", 5: "afr.2"},
        diagnostics={"correlations": [[1.0] * 6] * 7,
                     "n_overlapping_units": 1234, "unit": "sites"},
        provenance={"reference": "1kg_superpop_GRCh38",
                    "params": {"slope_threshold": -0.05}},
    )


def test_dump_load_roundtrip(tmp_path: Path):
    a = _make_assignment()
    p = tmp_path / "labels.v2.json"
    a.dump(p)
    b = Assignment.load(p)
    assert a == b


def test_dump_carries_schema_tag(tmp_path: Path):
    a = _make_assignment()
    p = tmp_path / "labels.v2.json"
    a.dump(p)
    payload = json.loads(p.read_text())
    assert payload["schema"] == SCHEMA_TAG
    assert payload["target_space"] == "SP6"
    # keys are stringified for JSON safety
    assert "0" in payload["component_to_label"]


def test_from_dict_roundtrip():
    a = _make_assignment()
    b = Assignment.from_dict(a.to_dict())
    assert a == b


def test_v1_compare_to_rf_upgrade(tmp_path: Path):
    """A labels.json shaped like compare_to_rf.py output upgrades cleanly."""
    v1 = {
        "tool": "popout",
        "rf_ref_labels": ["afr", "amr", "eas", "eur", "mid", "sas"],
        "popout_to_rf_label": {"0": "afr", "1": "eur", "2": "amr"},
        "rf_to_popout_components": {"afr": [0], "eur": [1], "amr": [2]},
        "n_overlapping_sites": 12345,
        "correlations": [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.95, 0.0, 0.0],
                         [0.0, 0.0, 0.85, 0.0, 0.0, 0.0]],
        "slope_matrix": [[0.5, None, None, None, None, None],
                         [None, None, None, 0.6, None, None],
                         [None, None, 0.4, None, None, None]],
    }
    p = tmp_path / "labels.v1.json"
    p.write_text(json.dumps(v1))
    a = Assignment.load(p)
    assert a.target_space is SP6
    assert a.method == "postS"
    assert a.input_space == "posterior"
    assert a.component_to_label == {0: "afr", 1: "eur", 2: "amr"}
    assert a.diagnostics["slope_matrix"] is not None
    assert a.diagnostics["n_overlapping_units"] == 12345


def test_v1_popout_label_upgrade(tmp_path: Path):
    """A labels.json shaped like popout/label.py output upgrades to corrH."""
    v1 = {
        "rf_ref_labels": ["afr", "amr", "eas", "eur", "mid", "sas"],
        "popout_to_rf_label": {"0": "afr", "1": "eur"},
        "rf_to_popout_components": {"afr": [0], "eur": [1]},
        "n_overlapping_sites": 9999,
        "correlations": [[0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.95, 0.0, 0.0]],
    }
    p = tmp_path / "labels.v1.json"
    p.write_text(json.dumps(v1))
    a = Assignment.load(p)
    assert a.method == "corrH"
    assert a.input_space == "allele_freq"


def test_invalid_method_rejected():
    with pytest.raises(ValueError):
        Assignment(
            target_space=SP6, source={}, method="bogus",
            input_space="posterior",
            component_to_label={}, label_to_components={},
            subcomponent_names={}, diagnostics={}, provenance={},
        )


def test_component_to_label_validates_against_target():
    with pytest.raises(ValueError):
        Assignment(
            target_space=SP6, source={}, method="corrH",
            input_space="allele_freq",
            component_to_label={0: "not-a-real-label"},
            label_to_components={}, subcomponent_names={},
            diagnostics={}, provenance={},
        )


def test_unassigned_sentinel_allowed():
    a = Assignment(
        target_space=SP6, source={}, method="manual",
        input_space="hard_call",
        component_to_label={0: "unassigned"},
        label_to_components={}, subcomponent_names={},
        diagnostics={}, provenance={},
    )
    assert a.component_to_label[0] == "unassigned"
