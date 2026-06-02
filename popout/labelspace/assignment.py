"""``Assignment``: one map, one schema, with provenance.

Replaces the two divergent ``labels.json`` producers (``popout/label.py``
and ``validation/scripts/compare_to_rf.py``) — both schemas survive as
serialization formats but the in-memory representation is unified.

See ``my_notes/labels/LABEL_SPACE.md`` §5.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from .registry import LabelSpace, get, make_native_space


SCHEMA_TAG = "labelspace/2"
"""Version identifier on labels.json v2."""


# ── Method codes ─────────────────────────────────────────────────────────

METHOD_CODES = ("corrH", "postS", "confH", "name", "manual")
"""Matching strategies (LABEL_SPACE.md §3): corr-Hungarian, posterior-slope,
confusion-Hungarian, exact-name, analyst-supplied CSV. Phase 2 implements
the bodies; this list pins the legal values."""

INPUT_SPACES = ("allele_freq", "posterior", "hard_call")


# ── Assignment dataclass ────────────────────────────────────────────────


@dataclasses.dataclass
class Assignment:
    """The map and its provenance.

    Every field is required; ``diagnostics`` and ``provenance`` are
    open-ended dicts (specific keys depend on ``method``).
    """

    target_space: LabelSpace
    source: dict[str, Any]
    method: str
    input_space: str
    component_to_label: dict[int, str]
    label_to_components: dict[str, list[int]]
    subcomponent_names: dict[int, str]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.method not in METHOD_CODES:
            raise ValueError(
                f"method {self.method!r} not in {METHOD_CODES}"
            )
        if self.input_space not in INPUT_SPACES:
            raise ValueError(
                f"input_space {self.input_space!r} not in {INPUT_SPACES}"
            )
        # component_to_label values must be valid members of the target
        # space OR canonical "unassigned" sentinel.
        invalid = [
            (k, v) for k, v in self.component_to_label.items()
            if v != "unassigned" and v not in self.target_space.members
        ]
        if invalid:
            raise ValueError(
                f"component_to_label has values outside {self.target_space.tag}: {invalid}"
            )

    # ── Equivalence ───────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Assignment):
            return NotImplemented
        return (
            self.target_space.tag == other.target_space.tag
            and self.target_space.members == other.target_space.members
            and self.source == other.source
            and self.method == other.method
            and self.input_space == other.input_space
            and self.component_to_label == other.component_to_label
            and self.label_to_components == other.label_to_components
            and self.subcomponent_names == other.subcomponent_names
            and _canonical(self.diagnostics) == _canonical(other.diagnostics)
            and _canonical(self.provenance) == _canonical(other.provenance)
        )

    def __hash__(self) -> int:  # not strictly hashable but useful for sets-of-one
        return hash((self.target_space.tag, self.method,
                     tuple(sorted(self.component_to_label.items()))))

    # ── (de)serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Canonical v2 dict (sortable, JSON-safe)."""
        return {
            "schema": SCHEMA_TAG,
            "target_space": self.target_space.tag,
            "target_members": list(self.target_space.members),
            "source": _canonical(self.source),
            "method": self.method,
            "input_space": self.input_space,
            "component_to_label": {str(k): v for k, v in
                                   sorted(self.component_to_label.items())},
            "label_to_components": {k: list(v) for k, v in
                                    sorted(self.label_to_components.items())},
            "subcomponent_names": {str(k): v for k, v in
                                   sorted(self.subcomponent_names.items())},
            "diagnostics": _canonical(self.diagnostics),
            "provenance": _canonical(self.provenance),
        }

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2,
                                         sort_keys=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Assignment":
        """Read v1 or v2 from disk; always return an in-memory v2 ``Assignment``."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Assignment":
        schema = d.get("schema")
        if schema == SCHEMA_TAG:
            return _from_v2(d)
        # v1: identified by the legacy keys, no schema field.
        if "popout_to_rf_label" in d:
            return _from_v1(d)
        raise ValueError(
            f"unrecognized labels.json shape: schema={schema!r}, "
            f"keys={sorted(d.keys())[:8]}"
        )

    # ── Tag delegation (Phase 5) ─────────────────────────────────────

    @property
    def tag(self) -> str:
        return self.provenance.get("tag", "")

    def version_hash(self) -> str:
        # Defer to shorthand to avoid import cycle.
        from .shorthand import version_hash
        return version_hash(self)


# ── Helpers ──────────────────────────────────────────────────────────────


def _canonical(obj: Any) -> Any:
    """Recursively sort dict keys / convert numpy scalars so equality is order-free."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── v1 → in-memory v2 upgrade ───────────────────────────────────────────


def _from_v1(d: dict[str, Any]) -> Assignment:
    """Read the legacy schema produced by popout/label.py or compare_to_rf.py.

    Both producers shared these keys:
      ``popout_to_rf_label``     dict[str(int), str]
      ``rf_to_popout_components`` dict[str, list[int]]
      ``rf_ref_labels``          list[str]
      ``correlations``           K x |L*| nested list
    compare_to_rf.py also adds: ``tool``, ``slope_matrix``, ``max_cal_matrix``,
    ``merge_group_stats``, ``n_overlapping_sites`` (samples instead of sites).

    The v1→v2 upgrade is purely a re-organisation: no numbers are
    recomputed and no fields are dropped (they move under
    ``diagnostics`` / ``provenance``).
    """
    p2rf_raw = d["popout_to_rf_label"]
    rf2pop_raw = d["rf_to_popout_components"]
    rf_ref = tuple(d.get("rf_ref_labels", ("afr", "amr", "eas", "eur", "mid", "sas")))

    # Pick the target space — SP6 if exactly the canonical 6 labels, else
    # a native space tagged on the tool. Either way the rf_ref tuple is
    # the truth.
    if rf_ref == get("SP6").members:
        target = get("SP6")
    else:
        target = make_native_space(
            d.get("tool", "ref").lower(),
            rf_ref,
        )

    component_to_label = {int(k): v for k, v in p2rf_raw.items()}
    label_to_components = {k: [int(x) for x in v] for k, v in rf2pop_raw.items()}

    # v1 did not record subcomponent names; leave empty (Phase 3 fills
    # them via labelspace.naming).
    subcomponent_names: dict[int, str] = {}

    diagnostics: dict[str, Any] = {
        "correlations": d.get("correlations"),
    }
    if "n_overlapping_sites" in d:
        # compare_to_rf overloads the name with "n_overlapping_samples";
        # we keep the legacy key but also stash the unit semantics.
        diagnostics["n_overlapping_units"] = d["n_overlapping_sites"]
        diagnostics["unit"] = (
            "samples" if d.get("tool") in ("popout", "FLARE", "flare")
            else "sites"
        )
    if "slope_matrix" in d:
        diagnostics["slope_matrix"] = d["slope_matrix"]
    if "max_cal_matrix" in d:
        diagnostics["max_cal_matrix"] = d["max_cal_matrix"]
    if "merge_group_stats" in d:
        diagnostics["merge_group_stats"] = d["merge_group_stats"]

    # method/input_space inference is heuristic — slope_matrix signals
    # postS, anything else from compare_to_rf is corrH on freq.
    if "slope_matrix" in d:
        method = "postS"
        input_space = "posterior"
    else:
        method = "corrH"
        input_space = "allele_freq"

    provenance: dict[str, Any] = {
        "tool": d.get("tool", "popout"),
        "upgraded_from": "labels.json v1",
    }

    return Assignment(
        target_space=target,
        source={"tool": d.get("tool", "popout")},
        method=method,
        input_space=input_space,
        component_to_label=component_to_label,
        label_to_components=label_to_components,
        subcomponent_names=subcomponent_names,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _from_v2(d: dict[str, Any]) -> Assignment:
    members = tuple(d["target_members"])
    canonical = get("SP6").members if d["target_space"] == "SP6" else members
    if d["target_space"] == "SP6":
        target = get("SP6")
    elif d["target_space"] == "SP5":
        target = get("SP5")
    else:
        target = make_native_space(d["target_space"], members)
    return Assignment(
        target_space=target,
        source=d["source"],
        method=d["method"],
        input_space=d["input_space"],
        component_to_label={int(k): v for k, v in d["component_to_label"].items()},
        label_to_components={k: list(v) for k, v in d["label_to_components"].items()},
        subcomponent_names={int(k): v for k, v in d["subcomponent_names"].items()},
        diagnostics=d.get("diagnostics", {}),
        provenance=d.get("provenance", {}),
    )
