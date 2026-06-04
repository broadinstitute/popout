"""Hard-call confusion heatmap — popout vs each comparison tool.

Reads ``cohort/popout_vs_<tool>.confusion.tsv`` for each tool the
bundle exposes. Pools across every (cluster, chrom) and emits one
row-normalised heatmap per tool, arranged in a single multi-panel
figure. Faithful: no in-report relabelling (no MID fold, no
canonicalisation) — render what the collector wrote.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP6

from .._helpers import fmt_pct, read_tsv, to_float


ANCHOR_TOOL = "popout"


def _pool_confusion(path) -> tuple[list[str], list[str], np.ndarray] | None:
    header, rows = read_tsv(path)
    if not rows:
        return None
    col = {h: i for i, h in enumerate(header)}
    if "popout_label" not in col:
        return None
    pop_col = "popout_label"
    col_labels = [c for c in header
                  if c not in ("cluster_id", "chrom", pop_col, "total")]
    row_label_order: list[str] = []
    pooled: dict[str, np.ndarray] = {}
    for r in rows:
        lab = r[col[pop_col]]
        if lab == "total":
            continue
        vec = np.zeros(len(col_labels), dtype=float)
        for j, c in enumerate(col_labels):
            vec[j] = to_float(r[col[c]]) or 0.0
        if lab not in pooled:
            pooled[lab] = vec.copy()
            row_label_order.append(lab)
        else:
            pooled[lab] += vec

    sp6 = list(SP6.members)
    row_labels = [lab for lab in sp6 if lab in pooled] + \
                 [lab for lab in row_label_order if lab not in sp6]
    cm = np.vstack([pooled[lab] for lab in row_labels])
    return row_labels, col_labels, cm


def compute(ctx, section=None) -> dict:
    tools = [t for t in (ctx.bundle.get("tools") or []) if t != ANCHOR_TOOL]
    panels: list[dict] = []
    for tool in tools:
        path = ctx.bundle_dir / "cohort" / f"popout_vs_{tool}.confusion.tsv"
        pooled = _pool_confusion(path)
        if pooled is None:
            continue
        row_labels, col_labels, cm = pooled
        total = float(cm.sum())
        diag = 0.0
        for i, lab in enumerate(row_labels):
            if lab in col_labels:
                diag += cm[i, col_labels.index(lab)]
        agree = diag / total if total > 0 else 0.0
        panels.append({
            "tool": tool,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "cm": cm,
            "agreement": agree,
            "agreement_pct": fmt_pct(agree),
        })
    if not panels:
        return {"present": False}
    return {"present": True, "panels": panels}


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.4))
        ax.text(0.5, 0.5, "no confusion matrices found",
                ha="center", va="center")
        ax.axis("off")
        return fig

    panels = data["panels"]
    n_panels = len(panels)
    panel_h = 0.55 * max(len(p["row_labels"]) for p in panels) + 1.6
    panel_w = 0.65 * max(len(p["col_labels"]) for p in panels) + 2.0
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(panel_w, panel_h * n_panels + 0.6),
    )
    if n_panels == 1:
        axes = [axes]

    for ax, p in zip(axes, panels):
        cm = p["cm"]
        row_labels = p["row_labels"]
        col_labels = p["col_labels"]
        tool = p["tool"]
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
        ax.set_xlabel(f"{tool} label →")
        ax.set_ylabel("popout label ↓")
        ax.set_title(f"popout vs {tool} — diag {p['agreement_pct']}",
                     fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7, label="row fraction")

    fig.tight_layout()
    return fig
