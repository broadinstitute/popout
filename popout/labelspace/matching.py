"""Named matching strategies that produce ``Assignment`` objects.

Each strategy moves verbatim from its previous home; numerics are
byte-identical to the Phase-0 goldens. See
``my_notes/labels/LABEL_SPACE.md`` §3 for the design.

Functions
---------
``corr_hungarian``    Pearson correlation + Hungarian (popout/label.py)
``posterior_slope``    correlation argmax + calibration-slope override
                       (validation/scripts/compare_to_rf.py)
``confusion_hungarian`` Hungarian on a hard-call confusion matrix
                       (popout/benchmark/align.py)
``by_name``            exact-name match
``manual``             analyst-supplied CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .assignment import Assignment
from .registry import LabelSpace


# ── Building blocks ──────────────────────────────────────────────────────


def _pearson_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows of ``a`` and rows of ``b``.

    Lifted verbatim from popout/label.py::_correlation_matrix.
    """
    combined = np.vstack([a, b])
    full_corr = np.corrcoef(combined)
    M = a.shape[0]
    return full_corr[:M, M:]


def _hungarian_or_merge(
    corr: np.ndarray, ref_names: Sequence[str],
) -> tuple[dict[int, str], dict[str, list[int]]]:
    """Assign labels via Hungarian when K_inf ≤ K_ref, else argmax-merge.

    Lifted verbatim from popout/label.py::_assign_labels (the body is
    structurally identical; we return name-keyed dicts).
    """
    K_inf, K_ref = corr.shape
    if K_inf <= K_ref:
        row_ind, col_ind = linear_sum_assignment(-corr)
        label_map = {int(r): ref_names[int(c)] for r, c in zip(row_ind, col_ind)}
    else:
        best_ref = np.argmax(corr, axis=1)
        label_map = {int(i): ref_names[int(best_ref[i])] for i in range(K_inf)}

    merge_map: dict[str, list[int]] = {}
    for idx, name in sorted(label_map.items()):
        merge_map.setdefault(name, []).append(idx)

    # Within each merge group: rank by correlation with the assigned label.
    # popout/label.py uses descending corr; Phase 2 preserves that exactly.
    for name, indices in merge_map.items():
        ref_col = ref_names.index(name)
        indices.sort(key=lambda i: -corr[i, ref_col])

    return label_map, merge_map


# ── corr_hungarian ──────────────────────────────────────────────────────


def corr_hungarian(
    freq_inf: np.ndarray,
    freq_ref: np.ndarray,
    target: LabelSpace,
    *,
    ref_names: Sequence[str] | None = None,
    n_overlapping_sites: int | None = None,
    source: dict | None = None,
    provenance: dict | None = None,
) -> Assignment:
    """Match inferred allele frequencies to reference frequencies.

    Pearson correlation matrix → Hungarian (when K_inf ≤ K_ref) or
    argmax (when K_inf > K_ref). Replaces popout/label.py.
    """
    if ref_names is None:
        ref_names = list(target.members)
    else:
        ref_names = list(ref_names)
    if list(target.members) != ref_names:
        raise ValueError(
            f"target.members {list(target.members)} != ref_names {ref_names}"
        )
    if freq_ref.shape[0] != len(ref_names):
        raise ValueError(
            f"freq_ref has {freq_ref.shape[0]} rows but ref_names has {len(ref_names)}"
        )
    corr = _pearson_corr_matrix(freq_inf, freq_ref)
    label_map, merge_map = _hungarian_or_merge(corr, ref_names)
    return Assignment(
        target_space=target,
        source=dict(source or {}),
        method="corrH",
        input_space="allele_freq",
        component_to_label=label_map,
        label_to_components=merge_map,
        subcomponent_names={},
        diagnostics={
            "correlations": corr.tolist(),
            "n_overlapping_units": (
                int(n_overlapping_sites) if n_overlapping_sites is not None
                else int(freq_inf.shape[1])
            ),
            "unit": "sites",
        },
        provenance=dict(provenance or {}),
    )


# ── posterior_slope ─────────────────────────────────────────────────────


def _posterior_correlations(popout_mat: np.ndarray, rf_prob: np.ndarray) -> np.ndarray:
    """K_popout × K_rf Pearson r over per-sample posterior proportions."""
    K_pop = popout_mat.shape[1]
    K_rf = rf_prob.shape[1]
    out = np.zeros((K_pop, K_rf), dtype=np.float64)
    for pa in range(K_pop):
        for ri in range(K_rf):
            out[pa, ri] = np.corrcoef(popout_mat[:, pa], rf_prob[:, ri])[0, 1]
    return out


def _calibration_slopes(
    popout_mat: np.ndarray,
    rf_prob: np.ndarray,
    *,
    n_bins: int = 20,
    min_bin_n: int = 100,
    min_populated_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per (popout_idx, rf_label) calibration slope + max binned mean.

    Lifted from compare_to_rf.py L235-259; default thresholds preserved.
    """
    K_pop = popout_mat.shape[1]
    K_rf = rf_prob.shape[1]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    slope = np.full((K_pop, K_rf), np.nan, dtype=np.float64)
    max_cal = np.full((K_pop, K_rf), np.nan, dtype=np.float64)
    for ri in range(K_rf):
        x = rf_prob[:, ri]
        bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        for pa in range(K_pop):
            y = popout_mat[:, pa]
            bx, by = [], []
            max_val = 0.0
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() < min_bin_n:
                    continue
                m = float(y[mask].mean())
                bx.append(0.5 * (edges[b] + edges[b + 1]))
                by.append(m)
                if m > max_val:
                    max_val = m
            max_cal[pa, ri] = max_val
            if len(bx) < min_populated_bins:
                continue
            slope[pa, ri], _ = np.polyfit(bx, by, 1)
    return slope, max_cal


def posterior_slope(
    popout_mat: np.ndarray,
    rf_prob: np.ndarray,
    target: LabelSpace,
    *,
    slope_threshold: float = -0.05,
    source: dict | None = None,
    provenance: dict | None = None,
) -> Assignment:
    """Match popout posteriors to RF probs via correlation + slope override.

    Lifted from validation/scripts/compare_to_rf.py L220-318. Default
    ``slope_threshold = -0.05`` matches the original script.
    """
    if rf_prob.shape[1] != len(target.members):
        raise ValueError(
            f"rf_prob has {rf_prob.shape[1]} columns but target {target.tag} "
            f"has {len(target.members)} members"
        )

    rf_ref = list(target.members)
    n_popout = popout_mat.shape[1]
    n_rf = rf_prob.shape[1]

    corr = _posterior_correlations(popout_mat, rf_prob)
    slope_mat, max_cal = _calibration_slopes(popout_mat, rf_prob)

    # Argmax with slope override (verbatim L262-282 of compare_to_rf.py).
    component_to_label: dict[int, str] = {}
    overrides: list[tuple[int, str, str, float, float]] = []
    for pa in range(n_popout):
        best_ri = int(np.argmax(corr[pa]))
        r_label = rf_ref[best_ri]
        r_slope = slope_mat[pa, best_ri]
        if not np.isnan(r_slope) and r_slope < slope_threshold:
            slope_best_ri = int(np.nanargmax(slope_mat[pa, :]))
            slope_best_val = slope_mat[pa, slope_best_ri]
            if not np.isnan(slope_best_val) and slope_best_val > 0:
                slope_label = rf_ref[slope_best_ri]
                overrides.append((pa, r_label, slope_label,
                                   float(r_slope), float(slope_best_val)))
                component_to_label[pa] = slope_label
                continue
        component_to_label[pa] = r_label

    # Sort merge groups by descending corr against the assigned label.
    label_to_components: dict[str, list[int]] = {}
    for pa, name in component_to_label.items():
        label_to_components.setdefault(name, []).append(pa)
    for name, indices in label_to_components.items():
        ref_col = rf_ref.index(name)
        indices.sort(key=lambda i: -corr[i, ref_col])

    diagnostics = {
        "correlations": corr.tolist(),
        "slope_matrix": np.where(np.isnan(slope_mat), None, slope_mat).tolist(),
        "max_cal_matrix": np.where(np.isnan(max_cal), None, max_cal).tolist(),
        "n_overlapping_units": int(popout_mat.shape[0]),
        "unit": "samples",
        "overrides": [
            {"component": pa, "from_label": old, "to_label": new,
             "from_slope": old_slope, "to_slope": new_slope}
            for pa, old, new, old_slope, new_slope in overrides
        ],
    }
    return Assignment(
        target_space=target,
        source=dict(source or {"tool": "popout"}),
        method="postS",
        input_space="posterior",
        component_to_label=component_to_label,
        label_to_components=label_to_components,
        subcomponent_names={},
        diagnostics=diagnostics,
        provenance=dict(provenance or {"params": {"slope_threshold": slope_threshold}}),
    )


# ── confusion_hungarian ─────────────────────────────────────────────────


def confusion_hungarian(
    calls_src: np.ndarray,
    calls_ref: np.ndarray,
    target: LabelSpace,
    *,
    src_labels: Sequence[int] | None = None,
    ref_labels: Sequence[int] | None = None,
    missing_label: int | None = None,
    source: dict | None = None,
    provenance: dict | None = None,
) -> Assignment:
    """Hungarian on a hard-call confusion matrix.

    Lifted from popout/benchmark/align.py::match_labels. Operates on
    flat integer call vectors / matrices. ``target`` provides the
    string names for the ref-side integer codes (assumed to be the
    ordinals 0..K-1 in ``target.members`` order, which matches today's
    benchmark convention).
    """
    if src_labels is None:
        src_set = set(np.unique(calls_src).tolist())
        if missing_label is not None:
            src_set.discard(missing_label)
        src_labels = sorted(src_set)
    if ref_labels is None:
        ref_set = set(np.unique(calls_ref).tolist())
        if missing_label is not None:
            ref_set.discard(missing_label)
        ref_labels = sorted(ref_set)

    K_src, K_ref = len(src_labels), len(ref_labels)
    K = max(K_src, K_ref)
    C = np.zeros((K, K), dtype=np.int64)
    for i, src_lab in enumerate(src_labels):
        src_mask = calls_src == src_lab
        for j, ref_lab in enumerate(ref_labels):
            ref_mask = calls_ref == ref_lab
            C[i, j] = int((src_mask & ref_mask).sum())
    row_ind, col_ind = linear_sum_assignment(-C)

    component_to_label: dict[int, str] = {}
    for r, c in zip(row_ind, col_ind):
        if r < K_src and c < K_ref:
            ref_int = int(ref_labels[c])
            if 0 <= ref_int < len(target.members):
                component_to_label[int(src_labels[r])] = target.members[ref_int]
            else:
                component_to_label[int(src_labels[r])] = f"unassigned"

    label_to_components: dict[str, list[int]] = {}
    for src, lab in component_to_label.items():
        label_to_components.setdefault(lab, []).append(src)

    return Assignment(
        target_space=target,
        source=dict(source or {}),
        method="confH",
        input_space="hard_call",
        component_to_label=component_to_label,
        label_to_components=label_to_components,
        subcomponent_names={},
        diagnostics={
            "confusion_matrix": C.tolist(),
            "src_labels": list(map(int, src_labels)),
            "ref_labels": list(map(int, ref_labels)),
            "n_overlapping_units": int((calls_src == calls_src).sum()),
            "unit": "sites",
        },
        provenance=dict(provenance or {}),
    )


# ── by_name ─────────────────────────────────────────────────────────────


def by_name(
    src_names: Sequence[str],
    target: LabelSpace,
    *,
    source: dict | None = None,
    case_insensitive: bool = True,
    provenance: dict | None = None,
) -> Assignment:
    """Exact-name match. ``src_names[i]`` is the name of component ``i``.

    Any name not in ``target.members`` produces ``"unassigned"`` (per the
    totality property of LABEL_SPACE.md §3.1).
    """
    target_lc = {m.lower(): m for m in target.members} if case_insensitive \
        else {m: m for m in target.members}
    component_to_label: dict[int, str] = {}
    for i, name in enumerate(src_names):
        key = name.lower() if case_insensitive else name
        component_to_label[i] = target_lc.get(key, "unassigned")
    label_to_components: dict[str, list[int]] = {}
    for i, lab in component_to_label.items():
        label_to_components.setdefault(lab, []).append(i)
    return Assignment(
        target_space=target,
        source=dict(source or {}),
        method="name",
        input_space="hard_call",
        component_to_label=component_to_label,
        label_to_components=label_to_components,
        subcomponent_names={},
        diagnostics={"n_overlapping_units": int(len(src_names)),
                     "unit": "components"},
        provenance=dict(provenance or {}),
    )


# ── manual ──────────────────────────────────────────────────────────────


def manual(
    csv_path: str | Path,
    target: LabelSpace,
    *,
    source: dict | None = None,
    provenance: dict | None = None,
) -> Assignment:
    """Read an analyst-supplied CSV ``component,label`` map.

    Two columns, no header tolerated; ``label`` must be a member of
    ``target`` or the literal ``"unassigned"``.
    """
    import csv as _csv
    component_to_label: dict[int, str] = {}
    with open(csv_path) as f:
        reader = _csv.reader(f)
        for n, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"{csv_path}:{n}: expected 2 columns, got {row}")
            try:
                idx = int(row[0])
            except ValueError:
                # tolerate a header row
                if n == 1:
                    continue
                raise ValueError(f"{csv_path}:{n}: component {row[0]!r} not int-coercible")
            lab = row[1].strip()
            if lab != "unassigned" and lab not in target.members:
                raise ValueError(
                    f"{csv_path}:{n}: label {lab!r} not in {target.tag} "
                    f"members {list(target.members)}"
                )
            component_to_label[idx] = lab
    label_to_components: dict[str, list[int]] = {}
    for i, lab in component_to_label.items():
        label_to_components.setdefault(lab, []).append(i)
    return Assignment(
        target_space=target,
        source=dict(source or {}),
        method="manual",
        input_space="hard_call",
        component_to_label=component_to_label,
        label_to_components=label_to_components,
        subcomponent_names={},
        diagnostics={"n_overlapping_units": int(len(component_to_label)),
                     "unit": "components"},
        provenance=dict(provenance or {"csv": str(csv_path)}),
    )
