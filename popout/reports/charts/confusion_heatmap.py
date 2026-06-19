"""FLARE x {RF, Rye} cohort confusion matrix - heatmap with recall + precision margins.

Sums the per-cluster (other_call, FLARE call) counts cohort-wide and
renders the row-normalised recall heatmap with explicit recall and
precision margins. ``options.source`` (``rf`` or ``rye``) selects the
cohort file:

  - ``rf``  -> ``cohort/confusion_rf.tsv``  (rows = RF labels incl. MID)
  - ``rye`` -> ``cohort/confusion_rye.tsv`` (rows = FLARE calls, cols
              = Rye calls; symmetric, no MID)

**Reporting principle**: render the bundle's cohort table faithfully -
no MID folding, no label canonicalisation. The bundle is the source
of truth.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5, SP6

from .._helpers import read_tsv, topn


def compute(ctx, section=None) -> dict:
    """Render the cohort-summed confusion table for the chosen tool.

    ``options.source`` (``rf`` default for back-compat or ``rye``)
    selects which cohort file to read and which column names to expect.
    """
    opts = section.options if section is not None else {}
    source = opts.get("source", "rf")
    if source == "rf":
        path = ctx.bundle_dir / "cohort" / "confusion_rf.tsv"
        row_col, col_col = "rf_label", "flare_call"
        row_axis_name = "RF call (reference)"
        col_axis_name = "FLARE hard call"
        title_other = "RF"
    elif source == "rye":
        path = ctx.bundle_dir / "cohort" / "confusion_rye.tsv"
        row_col, col_col = "flare_call", "rye_call"
        row_axis_name = "FLARE call"
        col_axis_name = "Rye hard call"
        title_other = "Rye"
    else:
        raise ValueError(
            f"confusion_heatmap options.source must be 'rf' or 'rye'; "
            f"got {source!r}"
        )
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False, "source": source}
    col = {h: i for i, h in enumerate(header)}

    cell: dict[tuple[str, str], int] = {}
    row_labels: list[str] = []
    col_labels: list[str] = []
    for r in rows:
        try:
            rv = r[col[row_col]]
            cv = r[col[col_col]]
            n = int(r[col["n"]])
        except (IndexError, KeyError, ValueError):
            continue
        cell[(rv, cv)] = cell.get((rv, cv), 0) + n
        if rv not in row_labels:
            row_labels.append(rv)
        if cv not in col_labels:
            col_labels.append(cv)

    # Sort by SP6 order (canonical), then any extras at the back.
    row_labels.sort(
        key=lambda x: SP6.members.index(x) if x in SP6.members else 99
    )
    col_labels.sort(
        key=lambda x: SP6.members.index(x) if x in SP6.members else 99
    )

    M = np.zeros((len(row_labels), len(col_labels)), dtype=float)
    for i, rv in enumerate(row_labels):
        for j, cv in enumerate(col_labels):
            M[i, j] = cell.get((rv, cv), 0)

    row_sums = M.sum(axis=1, keepdims=True)
    col_sums = M.sum(axis=0, keepdims=True)
    recall = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
    precision = np.divide(M, col_sums, out=np.zeros_like(M), where=col_sums > 0)

    diag_recall = {
        row_labels[i]: recall[i, i]
        for i in range(min(M.shape))
        if i < len(row_labels)
    }
    worst_recall = topn(list(diag_recall.items()), n=3, reverse=False)
    off_pairs: list[tuple[str, float]] = []
    for i, rv in enumerate(row_labels):
        for j, cv in enumerate(col_labels):
            if rv == cv:
                continue
            if recall[i, j] > 0.02:
                off_pairs.append((
                    f"{row_col}=`{rv}` -> {col_col}=`{cv}` (n={int(M[i, j]):,})",
                    recall[i, j],
                ))
    top_conf = topn(off_pairs, n=3)

    return {
        "present": True,
        "source": source,
        "M": M,
        "recall": recall,
        "precision": precision,
        # Back-compat: existing templates use rf_labels / flare_calls.
        "rf_labels": row_labels,
        "flare_calls": col_labels,
        # Generic names for the renderer + new templates.
        "row_labels": row_labels,
        "col_labels": col_labels,
        "row_axis_name": row_axis_name,
        "col_axis_name": col_axis_name,
        "title_other": title_other,
        "diag_recall": diag_recall,
        "worst_recall": worst_recall,
        "top_conf": top_conf,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no confusion data", ha="center", va="center")
        ax.axis("off")
        return fig

    M = data["M"]
    recall = data["recall"]
    precision = data["precision"]
    rf_labels = data["row_labels"]
    flare_calls = data["col_labels"]
    row_axis_name = data.get("row_axis_name", "RF call (reference)")
    col_axis_name = data.get("col_axis_name", "FLARE hard call")
    title_other = data.get("title_other", "RF")

    fig = plt.figure(
        figsize=(max(7.5, 0.85 * len(flare_calls) + 4.5),
                 max(5.0, 0.75 * len(rf_labels) + 3.0)),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 0.18, 0.05],
        height_ratios=[1.0, 0.10],
        wspace=0.08, hspace=0.05,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_recall = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_prec = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_corner = fig.add_subplot(gs[1, 1])

    im = ax.imshow(recall, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(flare_calls)))
    ax.set_xticklabels(flare_calls, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rf_labels)))
    ax.set_yticklabels(rf_labels, fontsize=10)
    ax.set_ylabel(row_axis_name, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            n = int(M[i, j])
            txt = f"{recall[i, j]:.2f}\n({n:,})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if recall[i, j] > 0.55 else "black")
            if rf_labels[i] != flare_calls[j] and recall[i, j] > 0.05:
                rect = mpatches.Rectangle(
                    (j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                    edgecolor="#c62828", linewidth=1.6,
                )
                ax.add_patch(rect)

    # Right-hand recall column.
    ax_recall.set_xlim(0, 1)
    ax_recall.set_xticks([])
    ax_recall.tick_params(left=False, labelleft=False)
    ax_recall.set_title("recall", fontsize=9, pad=4)
    for spine in ("top", "right", "bottom", "left"):
        ax_recall.spines[spine].set_visible(False)
    for i in range(M.shape[0]):
        if i < recall.shape[1]:
            v = recall[i, i]
            ax_recall.text(0.5, i, f"{v:.2f}" if v == v else "—",
                           ha="center", va="center", fontsize=10, color="#222")
        else:
            ax_recall.text(0.5, i, "—", ha="center", va="center",
                           fontsize=10, color="#222")

    # Bottom precision row.
    ax_prec.set_ylim(0, 1)
    ax_prec.set_yticks([])
    ax_prec.tick_params(bottom=False, labelbottom=False)
    for spine in ("top", "right", "bottom", "left"):
        ax_prec.spines[spine].set_visible(False)
    ax_prec.set_ylabel("precision", fontsize=9, rotation=0,
                       ha="right", va="center", labelpad=12)
    for j in range(M.shape[1]):
        v = precision[j, j] if j < precision.shape[0] else float("nan")
        ax_prec.text(j, 0.5, f"{v:.2f}" if v == v else "—",
                     ha="center", va="center", fontsize=10, color="#222")
    ax_prec.set_xlim(ax.get_xlim())

    ax_corner.axis("off")
    fig.colorbar(im, cax=ax_cbar, label="row-normalized recall")
    title = f"Cohort-summed FLARE vs {title_other} confusion matrix"
    fig.suptitle(
        title + f"\n(rows = {row_axis_name}  ·  cols = {col_axis_name}  ·  "
                "diagonals = correct  ·  red-bordered cells = systematic confusion > 5%)",
        fontsize=11,
    )
    ax.set_xlabel(col_axis_name, fontsize=10)
    return fig
