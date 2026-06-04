"""Estimate × Estimate → ConcordanceResult.

Projects both Estimates into a target label space, computes per-label
concordance metrics (Pearson r, Lin's CCC, MAE, Jaccard@τ), an optional
confusion matrix (when both sides have hard calls), and the label-
permutation-invariant cluster metrics (ARI / NMI / V-measure) from
:mod:`popout.labelspace.metrics`. Returns one record with a canonical
figure tag.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np

from popout.labelspace import (
    Assignment,
    by_name,
    cluster_eval,
    collapse,
    project_proportions,
)
from popout.labelspace.registry import (
    SP5,
    SP6,
    LabelSpace,
    get,
)
from popout.labelspace.shorthand import format as format_tag

from .record import Estimate


# Defaults pinned from the FLARE-validate / popout-DX μ-gating contract.
JACCARD_THRESHOLDS: tuple[float, ...] = (0.10, 0.25, 0.50)
MU_GATE = 0.01
PEARSON_THRESHOLD = 0.95
CCC_THRESHOLD = 0.90


@dataclasses.dataclass(frozen=True)
class ConcordanceResult:
    """Result of comparing two Estimates in a common target space."""

    target_space: LabelSpace
    mid_rule: str | None
    pair: tuple[str, str]                              # (left.tool, right.tool)
    sample_ids: tuple[str, ...]
    per_label: dict[str, dict[str, float | bool | None]]
    confusion: np.ndarray | None
    confusion_row_labels: tuple[str, ...] | None
    confusion_col_labels: tuple[str, ...] | None
    hard_metrics: dict[str, float] | None              # ARI / NMI / V-measure
    tag: str
    provenance: dict


# ── Internals ───────────────────────────────────────────────────────────


def _lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float((2 * cov) / denom)


def _per_label_metrics(anchor: np.ndarray, other: np.ndarray) -> dict:
    cluster_mu = float(anchor.mean())
    if anchor.std() == 0 or other.std() == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(anchor, other)[0, 1])
    ccc = _lin_ccc(anchor, other)
    err = np.abs(anchor - other)
    out = {
        "cluster_mu": cluster_mu,
        "n": int(anchor.size),
        "pearson_r": pearson_r,
        "ccc": ccc,
        "mae_mean": float(err.mean()),
        "mae_median": float(np.median(err)),
        "mae_p95": float(np.percentile(err, 95)),
    }
    for tau in JACCARD_THRESHOLDS:
        a = anchor >= tau
        b = other >= tau
        inter = int((a & b).sum())
        union = int((a | b).sum())
        out[f"jaccard_{tau:.2f}"] = float(inter / union) if union > 0 else float("nan")
    if cluster_mu < MU_GATE:
        out["pass"] = None
    else:
        out["pass"] = bool(
            (not np.isnan(pearson_r) and pearson_r >= PEARSON_THRESHOLD)
            and (not np.isnan(ccc) and ccc >= CCC_THRESHOLD)
        )
    return out


def _project_to_target(
    estimate: Estimate,
    target: LabelSpace,
    mid_rule: str | None,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Project an Estimate's proportions (+ hard_calls) into ``target``.

    Returns ``(proportions_in_target, hard_calls_in_target, method_code)``.
    method_code is the matching method (``"name"`` / ``"corrH"`` /
    ``"postS"`` / ``"confH"`` / ``"manual"``) — pinned to ``"name"`` here
    since every Estimate's loader is required to produce named columns.
    """
    src = estimate.label_space

    # Step 1: source-space → SP6 (or stay in src if already in target's
    # ancestor). For named tools we just permute the columns; for popout
    # the loader already projected, so src == target up to MID.
    if src.tag == target.tag:
        proportions = estimate.proportions.copy()
    elif (src.has_mid == target.has_mid
          and set(src.members) == set(target.members)):
        # same membership, different order → permute
        perm = [src.index(m) for m in target.members]
        proportions = estimate.proportions[:, perm]
    elif target.tag == "SP5" and src.has_mid:
        # source has MID, target doesn't — collapse with rule
        # First, normalise src to SP6 ordering so collapse() sees a
        # canonical input.
        if src.tag != "SP6":
            sp6_perm = [src.index(m) if m in src.members else None
                        for m in SP6.members]
            if any(p is None for p in sp6_perm):
                raise ValueError(
                    f"compare: {src.tag} cannot be projected to SP6 "
                    f"(missing members {[m for m, p in zip(SP6.members, sp6_perm) if p is None]})"
                )
            in_sp6 = estimate.proportions[:, sp6_perm]
        else:
            in_sp6 = estimate.proportions
        proportions = collapse(
            in_sp6, SP6, SP5,
            rule=("fold_to_eur" if mid_rule == "fold_to_eur" else "drop"),
        )
    elif target.tag == "SP5" and not src.has_mid:
        # source already MID-less; permute to SP5 order
        missing = [m for m in target.members if m not in src.members]
        if missing:
            raise ValueError(
                f"compare: source {src.tag} missing target labels {missing}"
            )
        perm = [src.index(m) for m in target.members]
        proportions = estimate.proportions[:, perm]
    else:
        raise ValueError(
            f"compare: don't know how to project {src.tag} → {target.tag}"
        )

    # hard_calls go through the same name-based folding when present.
    hard = None
    if estimate.hard_calls is not None:
        if target.tag == "SP5" and src.has_mid and mid_rule == "fold_to_eur":
            hard = np.array(
                ["eur" if h == "mid" else h for h in estimate.hard_calls],
                dtype=object,
            )
        elif target.tag == "SP5" and src.has_mid and mid_rule == "drop":
            # samples whose hard call was MID become "unassigned" — they
            # are dropped from any hard-call metric the caller emits.
            hard = np.array(
                ["unassigned" if h == "mid" else h for h in estimate.hard_calls],
                dtype=object,
            )
        else:
            hard = estimate.hard_calls.copy()

    return proportions, hard, "name"


def _build_assignments_for_tag(
    left: Estimate, right: Estimate, target: LabelSpace,
) -> list[Assignment]:
    """One Assignment per side for the figure tag. Both use ``by_name``
    here since loaders deliver named columns; the tag method thus
    reads ``flare=>name | rye=>name``. (Popout's Estimate, when present,
    has already had its non-name matching applied at load time; the
    ``by_name`` here just confirms the labels alignment in the tag.)"""
    a_left = by_name(left.label_space.members, target,
                     source={"tool": left.tool})
    a_right = by_name(right.label_space.members, target,
                      source={"tool": right.tool})
    return [a_left, a_right]


# ── Public entry ────────────────────────────────────────────────────────


def compare(
    left: Estimate,
    right: Estimate,
    *,
    target_space: LabelSpace | str | None = None,
    mid_rule: str | None = None,
    sample_ids: Iterable[str] | None = None,
) -> ConcordanceResult:
    """Compare two Estimates in a target space.

    Parameters
    ----------
    left, right
        The two Estimates. ``left`` is the *anchor* — the μ-gate fires on
        its column means.
    target_space
        ``LabelSpace`` or its registry tag (``"SP6"``, ``"SP5"``). When
        ``None`` the function picks the smallest space both can reach:
        SP5 if either side lacks MID, SP6 otherwise.
    mid_rule
        How to handle MID when projecting from SP6 to SP5. ``"drop"``
        (default when both sides natively lack MID) or ``"fold_to_eur"``
        (recommended when one side carries MID mass — typically FLARE
        vs RF).
    sample_ids
        Optional roster. Defaults to the intersection of both Estimates'
        sample_ids (preserving ``left``'s order). Both sides are
        subset+aligned to this roster.
    """
    if isinstance(target_space, str):
        target = get(target_space)
    elif target_space is None:
        target = SP5 if (
            not left.label_space.has_mid or not right.label_space.has_mid
        ) else SP6
    else:
        target = target_space

    # Roster alignment.
    if sample_ids is None:
        left_idx = {s: i for i, s in enumerate(left.sample_ids)}
        roster = [s for s in left.sample_ids if s in set(right.sample_ids)]
    else:
        roster = list(sample_ids)
    left_idx = {s: i for i, s in enumerate(left.sample_ids)}
    right_idx = {s: i for i, s in enumerate(right.sample_ids)}
    missing_l = [s for s in roster if s not in left_idx]
    missing_r = [s for s in roster if s not in right_idx]
    if missing_l or missing_r:
        raise ValueError(
            f"compare: roster missing samples "
            f"(left={missing_l[:3]}{'...' if len(missing_l) > 3 else ''}; "
            f"right={missing_r[:3]}{'...' if len(missing_r) > 3 else ''})"
        )
    left_q = left.proportions[[left_idx[s] for s in roster]]
    right_q = right.proportions[[right_idx[s] for s in roster]]
    left_hc = (left.hard_calls[[left_idx[s] for s in roster]]
               if left.hard_calls is not None else None)
    right_hc = (right.hard_calls[[right_idx[s] for s in roster]]
                if right.hard_calls is not None else None)

    # Reconstruct minimal Estimates on the roster, then project both.
    left_roster = dataclasses.replace(
        left, sample_ids=tuple(roster),
        proportions=left_q, hard_calls=left_hc,
    )
    right_roster = dataclasses.replace(
        right, sample_ids=tuple(roster),
        proportions=right_q, hard_calls=right_hc,
    )
    left_p, left_hard, _ = _project_to_target(left_roster, target, mid_rule)
    right_p, right_hard, _ = _project_to_target(right_roster, target, mid_rule)

    # Per-label metrics (anchor = left).
    per_label: dict[str, dict] = {}
    for j, lab in enumerate(target.members):
        per_label[lab] = _per_label_metrics(left_p[:, j], right_p[:, j])

    # Confusion + hard-cluster metrics (only when both sides emit hard).
    confusion = None
    confusion_row_labels: tuple[str, ...] | None = None
    confusion_col_labels: tuple[str, ...] | None = None
    hard_metrics: dict | None = None
    if left_hard is not None and right_hard is not None:
        members = list(target.members) + ["unassigned"]
        col_labels = tuple(target.members)
        row_labels = tuple(target.members)
        cm = np.zeros((len(row_labels), len(col_labels)), dtype=np.int64)
        for l_call, r_call in zip(left_hard, right_hard):
            if l_call == "unassigned" or r_call == "unassigned":
                continue
            try:
                r_idx = row_labels.index(l_call)
                c_idx = col_labels.index(r_call)
            except ValueError:
                continue
            cm[r_idx, c_idx] += 1
        confusion = cm
        confusion_row_labels = row_labels
        confusion_col_labels = col_labels
        mask = np.array([
            (l != "unassigned" and r != "unassigned")
            for l, r in zip(left_hard, right_hard)
        ])
        if mask.any():
            hard_metrics = cluster_eval(left_hard[mask], right_hard[mask])

    assignments = _build_assignments_for_tag(left, right, target)
    tag = format_tag(target, assignments, mid_rule=mid_rule)

    provenance = {
        "left_tool": left.tool, "right_tool": right.tool,
        "left_scope": list(left.scope), "right_scope": list(right.scope),
        "target_space": target.tag,
        "mid_rule": mid_rule,
        "n_samples_compared": int(len(roster)),
        "pearson_threshold": PEARSON_THRESHOLD,
        "ccc_threshold": CCC_THRESHOLD,
        "mu_gate": MU_GATE,
    }
    return ConcordanceResult(
        target_space=target,
        mid_rule=mid_rule,
        pair=(left.tool, right.tool),
        sample_ids=tuple(roster),
        per_label=per_label,
        confusion=confusion,
        confusion_row_labels=confusion_row_labels,
        confusion_col_labels=confusion_col_labels,
        hard_metrics=hard_metrics,
        tag=tag,
        provenance=provenance,
    )
