"""Local-mode bp-weighted confusion (FLARE vs popout).

Reads ``cohort/bp_confusion_segments.tsv.gz`` (one row per segment)
and pools by ``(flare_rf_label, popout_rf_label)``, weighting each
cell by segment length in bp. Renders a row-normalised heatmap
(rows = FLARE label, columns = popout label).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .._helpers import fmt_pct, read_tsv


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "bp_confusion_segments.tsv.gz"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    flare_col = "flare_rf_label" if "flare_rf_label" in col else "flare_anc"
    pop_col = "popout_rf_label" if "popout_rf_label" in col else "popout_anc"
    start_col = "seg_start_bp" if "seg_start_bp" in col else "start_bp"
    end_col = "seg_end_bp" if "seg_end_bp" in col else "end_bp"
    if any(c not in col for c in (flare_col, pop_col, start_col, end_col)):
        return {"present": False}

    flare_labels: list[str] = []
    popout_labels: list[str] = []
    bp_grid: dict[tuple[str, str], int] = {}
    for r in rows:
        fa = r[col[flare_col]]
        pa = r[col[pop_col]]
        try:
            length = int(r[col[end_col]]) - int(r[col[start_col]])
        except ValueError:
            continue
        if length <= 0:
            continue
        bp_grid[(fa, pa)] = bp_grid.get((fa, pa), 0) + length
        if fa not in flare_labels:
            flare_labels.append(fa)
        if pa not in popout_labels:
            popout_labels.append(pa)

    flare_labels.sort()
    popout_labels.sort()
    if not flare_labels or not popout_labels:
        return {"present": False}
    cm = np.array(
        [[bp_grid.get((fa, pa), 0) for pa in popout_labels]
         for fa in flare_labels],
        dtype=float,
    )
    total = float(cm.sum())
    diag = 0.0
    for i, fa in enumerate(flare_labels):
        if fa in popout_labels:
            diag += cm[i, popout_labels.index(fa)]
    agree = diag / total if total > 0 else 0.0

    return {
        "present": True,
        "flare_labels": flare_labels,
        "popout_labels": popout_labels,
        "cm": cm,
        "agreement": agree,
        "agreement_pct": fmt_pct(agree),
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.4))
        ax.text(0.5, 0.5, "no bp-confusion data", ha="center", va="center")
        ax.axis("off")
        return fig

    cm = data["cm"]
    row_labels = data["flare_labels"]
    col_labels = data["popout_labels"]

    fig, ax = plt.subplots(
        figsize=(0.65 * len(col_labels) + 2.0, 0.65 * len(row_labels) + 1.6),
    )
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if row_sums[i, 0] <= 0:
                ax.text(j, i, "—", ha="center", va="center",
                        color="#888", fontsize=8)
            else:
                v = norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.55 else "#222",
                        fontsize=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("popout label ->")
    ax.set_ylabel("FLARE label ↓")
    ax.set_title(
        f"bp-weighted confusion — diag {data['agreement_pct']}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="row fraction (bp)")
    fig.tight_layout()
    return fig
