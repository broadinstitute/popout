"""Phase-0 characterization goldens for the label-space retrofit.

Captures byte-identical / numerically-identical snapshots of every label-
producing or label-consuming function that the migration plan
(`my_notes/labels/LABEL_SPACE_RETROFIT.md`) routes through the new
``popout.labelspace`` module. The goldens are the regression oracle for
Phases 2 and 4 (byte-identical) and the explicit-diff target for Phase 3
(stable subcomponent renumbering).

Set ``LABELSPACE_UPDATE_GOLDENS=1`` to regenerate the goldens on disk.
The CI default is to assert against committed values.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from popout.label import _assign_labels, _correlation_matrix  # noqa: E402
from popout.viz._style import ancestry_names                  # noqa: E402

from validation.popout_dx.scripts.dx_loaders import (         # noqa: E402
    project_to_rf_basis,
    RF_LABELS_CANONICAL,
)

from .conftest import (                                        # noqa: E402
    GOLDENS_DIR,
    SP6,
    synthetic_freq_inputs,
    synthetic_popout_rf_pair,
    synthetic_projection_inputs,
    synthetic_rf_q,
    synthetic_rye_q,
    synthetic_tractset_calls,
    write_compare_to_rf_fixture,
)


UPDATE = bool(os.environ.get("LABELSPACE_UPDATE_GOLDENS"))


# ── snapshot helpers ─────────────────────────────────────────────────────


def _canonical_json(obj) -> str:
    """Stable JSON: sorted keys, fixed precision for floats, no trailing nl."""
    def _convert(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    return json.dumps(obj, sort_keys=True, indent=2, default=_convert)


def _snapshot_json(name: str, obj) -> None:
    path = GOLDENS_DIR / f"{name}.json"
    payload = _canonical_json(obj)
    if UPDATE or not path.exists():
        path.write_text(payload + "\n")
        return
    expected = path.read_text().rstrip("\n")
    assert payload == expected, (
        f"{name}: golden drifted\n"
        f"  expected: {path}\n"
        f"  got: {payload[:200]}{'…' if len(payload) > 200 else ''}\n"
        f"Set LABELSPACE_UPDATE_GOLDENS=1 to regenerate if the change is intentional."
    )


def _snapshot_npy(name: str, arr: np.ndarray) -> None:
    path = GOLDENS_DIR / f"{name}.npy"
    arr = np.ascontiguousarray(arr)
    if UPDATE or not path.exists():
        np.save(path, arr)
        return
    expected = np.load(path)
    assert arr.shape == expected.shape, f"{name}: shape {arr.shape} != {expected.shape}"
    assert arr.dtype == expected.dtype, f"{name}: dtype {arr.dtype} != {expected.dtype}"
    if np.issubdtype(arr.dtype, np.floating):
        np.testing.assert_array_max_ulp(arr, expected, maxulp=0)
    else:
        np.testing.assert_array_equal(arr, expected)


# ── G1. popout/label.py: corr_hungarian on synthetic frequencies ─────────


def test_golden_corr_hungarian():
    inf, ref, ref_names = synthetic_freq_inputs()
    corr = _correlation_matrix(inf, ref)
    label_map, merge_map = _assign_labels(corr, ref_names)
    snapshot = {
        "ref_names": ref_names,
        "correlations": corr,
        "label_map": {str(k): v for k, v in label_map.items()},
        "merge_map": merge_map,
    }
    _snapshot_json("popout_label_corr_hungarian", snapshot)


# ── G2. compare_to_rf.py: posterior_slope via subprocess ─────────────────


def test_golden_posterior_slope(tmp_path: Path):
    global_tsv, rf_tsv, out_dir = write_compare_to_rf_fixture(tmp_path)
    script = REPO_ROOT / "validation" / "scripts" / "compare_to_rf.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--popout-global", str(global_tsv),
         "--rf-ancestry", str(rf_tsv),
         "--out-dir", str(out_dir)],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"compare_to_rf.py failed:\n{res.stdout}\n{res.stderr}"
    labels = json.loads((out_dir / "labels.json").read_text())
    # Trim noisy keys for snapshot stability across float-print quirks.
    snapshot = {
        "tool": labels["tool"],
        "rf_ref_labels": labels["rf_ref_labels"],
        "popout_to_rf_label": labels["popout_to_rf_label"],
        "rf_to_popout_components": labels["rf_to_popout_components"],
        "n_overlapping_sites": labels["n_overlapping_sites"],
        "correlations": np.array(labels["correlations"]).round(6).tolist(),
        # slope/max_cal are None-or-float; round only the floats.
        "slope_matrix": [[None if v is None else round(v, 6) for v in row]
                         for row in labels["slope_matrix"]],
    }
    _snapshot_json("compare_to_rf_posterior_slope", snapshot)


# ── G3. project_to_rf_basis: popout / rye / rf paths ─────────────────────


def test_golden_project_to_rf_basis_popout():
    q, labels = synthetic_projection_inputs()
    out = project_to_rf_basis(q, source="popout", labels=labels)
    _snapshot_npy("project_to_rf_basis_popout", out)


def test_golden_project_to_rf_basis_rye():
    q = synthetic_rye_q()
    out = project_to_rf_basis(q, source="rye")
    _snapshot_npy("project_to_rf_basis_rye", out)


def test_golden_project_to_rf_basis_rf():
    q = synthetic_rf_q()
    out = project_to_rf_basis(q, source="rf")
    _snapshot_npy("project_to_rf_basis_rf", out)


# ── G4. remap_to_rf_codes: integer LUT projection on a TractSet ──────────


def test_golden_remap_to_rf_codes():
    # Import locally so an import-time failure on a sibling dependency
    # doesn't poison the rest of the suite.
    from validation.popout_dx.scripts.dx_local_align_metrics import (
        remap_to_rf_codes,
    )
    from popout.benchmark.common import MISSING_LABEL, TractSet

    calls = synthetic_tractset_calls()
    n_haps, n_sites = calls.shape
    label_map = {
        0: "ancestry_0", 1: "ancestry_1", 2: "ancestry_2",
        3: "ancestry_3", 4: "ancestry_4", 5: "ancestry_5",
        6: "ancestry_6",
    }
    ts = TractSet(
        tool_name="popout",
        chrom="chr1",
        hap_ids=np.array([f"s{i//2}:{i%2}" for i in range(n_haps)], dtype=object),
        site_positions=np.arange(n_sites, dtype=np.int64) * 10000 + 1,
        calls=calls,
        label_map=label_map,
    )
    _, labels = synthetic_projection_inputs()
    out = remap_to_rf_codes(ts, labels)
    _snapshot_npy("remap_to_rf_codes_calls", out.calls)
    assert out.label_map == {i: name for i, name in enumerate(RF_LABELS_CANONICAL)}
    assert MISSING_LABEL == 65535   # pins the constant we baked into the fixture


# ── G5. subcomponent naming ───────────────────────────────────────────────
#
# Phase 3 of the label-space retrofit replaces the legacy global-index
# rule (``afr.0``, ``afr.5`` with index gaps) with a 1-based, dense rule
# (``afr.1``, ``afr.2`` ranked by descending correlation). Both the
# legacy and the new rule are exercised here so any future drift is
# caught.


def _legacy_global_index_names(p2rf: dict[int, str], n_anc: int) -> list[str]:
    """The pre-Phase-3 rule from compare_to_rf.py:291 — kept as the diff anchor."""
    counts = Counter(p2rf.values())
    return [
        f"{p2rf[i]}.{i}" if counts[p2rf[i]] > 1 else p2rf[i]
        for i in range(n_anc)
    ]


_NAMING_CASES = {
    "all_singletons": (
        {0: "afr", 1: "amr", 2: "eas", 3: "eur", 4: "mid", 5: "sas"}, 6,
    ),
    "afr_split_with_gap": (
        {0: "afr", 1: "amr", 2: "eas", 3: "eur", 4: "mid", 5: "afr", 6: "sas"}, 7,
    ),
    "eur_three_way": (
        {0: "eur", 1: "eur", 2: "afr", 3: "amr", 4: "eur", 5: "sas"}, 6,
    ),
}


def test_golden_subcomponent_names_legacy_v1():
    """Snapshot the pre-Phase-3 global-index rule for documentation."""
    snapshot = {
        case: _legacy_global_index_names(p2rf, n)
        for case, (p2rf, n) in _NAMING_CASES.items()
    }
    _snapshot_json("subcomponent_names_legacy_v1", snapshot)


def test_golden_subcomponent_names_stable_rank_v2():
    """Snapshot the post-Phase-3 stable-rank rule.

    Two ways the new rule is reached today:
      * popout/viz/_style.py::ancestry_names (consumer-facing)
      * popout.labelspace.naming.ordered_subcomponent_names (canonical)
    Both must produce the same list for the same input.
    """
    from popout.labelspace.naming import ordered_subcomponent_names

    snapshot = {}
    for case, (p2rf, n) in _NAMING_CASES.items():
        via_style = ancestry_names(
            n, labels={"popout_to_rf_label": {int(k): v for k, v in p2rf.items()}},
        )
        via_canonical = ordered_subcomponent_names(p2rf)
        assert via_style == via_canonical, (
            f"{case}: viz._style and labelspace.naming disagree:\n"
            f"  viz._style: {via_style}\n  canonical:  {via_canonical}"
        )
        snapshot[case] = via_style
    _snapshot_json("subcomponent_names_stable_rank_v2", snapshot)


def test_phase3_renumbering_diff_documented():
    """Pin the *expected* diff between v1 (legacy) and v2 (stable-rank)."""
    from popout.labelspace.naming import ordered_subcomponent_names

    legacy = {
        case: _legacy_global_index_names(p2rf, n)
        for case, (p2rf, n) in _NAMING_CASES.items()
    }
    new = {
        case: ordered_subcomponent_names(p2rf)
        for case, (p2rf, n) in _NAMING_CASES.items()
    }
    # all_singletons: identical (no splits → no suffix change)
    assert legacy["all_singletons"] == new["all_singletons"]
    # afr_split_with_gap: legacy yields afr.0/afr.5, new yields afr.1/afr.2
    assert legacy["afr_split_with_gap"][0] == "afr.0"
    assert legacy["afr_split_with_gap"][5] == "afr.5"
    assert new["afr_split_with_gap"][0] == "afr.1"
    assert new["afr_split_with_gap"][5] == "afr.2"
    # eur_three_way: legacy yields eur.0/eur.1/eur.4, new yields eur.1/eur.2/eur.3
    assert sorted(s for s in legacy["eur_three_way"] if s.startswith("eur.")) == ["eur.0", "eur.1", "eur.4"]
    assert sorted(s for s in new["eur_three_way"] if s.startswith("eur.")) == ["eur.1", "eur.2", "eur.3"]
