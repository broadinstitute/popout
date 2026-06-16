"""FLARE vs RF cohort confusion matrix — heatmap with recall + precision margins.

Sums the per-cluster (RF call, FLARE call) counts cohort-wide and
renders the row-normalised recall heatmap, with explicit recall and
precision margins.

**Reporting principle**: this module renders the bundle's
``cohort/confusion_rf.tsv`` faithfully — no MID folding, no label
canonicalisation, no subancestry stripping. The bundle is the source
of truth. Any data-quality fixes (MID handling, retiring the
postS-introduced ``afr.N`` strings, …) belong in the stats collector
and are tracked in ``my_notes/validation/COLLECTOR_FIXES.md``.

Data: ``cohort/confusion_rf.tsv`` with columns (cluster_id, chrom,
rf_label, flare_call, n).
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5, SP6

from .._helpers import read_tsv, topn


def compute(ctx, section=None) -> dict:
    """Render the cohort-summed RF×FLARE confusion exactly as the bundle
    presents it. The report applies no transformations of its own — if
    the bundle's labels need to be canonicalised, that's the stats
    collector's job (tracked in my_notes/validation/COLLECTOR_FIXES.md).
    """
    path = ctx.bundle_dir / "cohort" / "confusion_rf.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}

    cell: dict[tuple[str, str], int] = {}
    rf_labels: list[str] = []
    flare_calls: list[str] = []
    for r in rows:
        try:
            rf = r[col["rf_label"]]
            fc = r[col["flare_call"]]
            n = int(r[col["n"]])
        except (IndexError, KeyError, ValueError):
            continue
        cell[(rf, fc)] = cell.get((rf, fc), 0) + n
        if rf not in rf_labels:
            rf_labels.append(rf)
        if fc not in flare_calls:
            flare_calls.append(fc)

    # Sort by SP6 order (canonical), then any extras alphabetically.
    rf_labels.sort(
        key=lambda x: SP6.members.index(x) if x in SP6.members else 99
    )
    flare_calls.sort(
        key=lambda x: SP6.members.index(x) if x in SP6.members else 99
    )

    M = np.zeros((len(rf_labels), len(flare_calls)), dtype=float)
    for i, rf in enumerate(rf_labels):
        for j, fc in enumerate(flare_calls):
            M[i, j] = cell.get((rf, fc), 0)

    row_sums = M.sum(axis=1, keepdims=True)
    col_sums = M.sum(axis=0, keepdims=True)
    recall = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
    precision = np.divide(M, col_sums, out=np.zeros_like(M), where=col_sums > 0)

    diag_recall = {
        rf_labels[i]: recall[i, i]
        for i in range(min(M.shape))
        if i < len(rf_labels)
    }
    worst_recall = topn(list(diag_recall.items()), n=3, reverse=False)
    off_pairs: list[tuple[str, float]] = []
    for i, rf in enumerate(rf_labels):
        for j, fc in enumerate(flare_calls):
            if rf == fc:
                continue
            if recall[i, j] > 0.02:
                off_pairs.append((
                    f"RF=`{rf}` -> FLARE=`{fc}` (n={int(M[i, j]):,})",
                    recall[i, j],
                ))
    top_conf = topn(off_pairs, n=3)

    return {
        "present": True,
        "M": M,
        "recall": recall,
        "precision": precision,
        "rf_labels": rf_labels,
        "flare_calls": flare_calls,
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
    rf_labels = data["rf_labels"]
    flare_calls = data["flare_calls"]

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
    ax.set_ylabel("RF call (reference)", fontsize=10)
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
    title = "Cohort-summed FLARE vs RF confusion matrix"
    fig.suptitle(
        title + "\n(rows = RF argmax superpop  ·  cols = FLARE argmax  ·  "
                "diagonals = correct  ·  red-bordered cells = systematic confusion > 5%)",
        fontsize=11,
    )
    ax.set_xlabel("FLARE hard call", fontsize=10)
    return fig
