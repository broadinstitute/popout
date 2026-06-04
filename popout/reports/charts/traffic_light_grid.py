"""Traffic-light pass-rate grid — popout vs each comparison tool, per RF label.

One row per comparison tool (flare/rye/rf — whichever the bundle
exposes), one column per SP6 RF label. Each cell shows the fraction
of μ-evaluable clusters that passed the per-(cluster, RF-label) μ-gated
threshold for that pair. Cells where every cluster was μ-gated render
as the grey ``μ·∅`` marker.

Data source: ``cohort_summary.json:pairs[]`` (already aggregated by
the collector).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP6


ANCHOR_TOOL = "popout"

TL_GREEN = "#117733"
TL_YELLOW = "#DDCC77"
TL_RED = "#CC3311"
TL_GREY = "#888888"


def _tl_color(pair: dict) -> str:
    n_pass = int(pair.get("n_clusters_passing", 0) or 0)
    n_fail = int(pair.get("n_clusters_failing", 0) or 0)
    n_eval = n_pass + n_fail
    if n_eval == 0:
        return TL_GREY
    frac = n_pass / n_eval
    if frac >= 0.9:
        return TL_GREEN
    if frac >= 0.5:
        return TL_YELLOW
    return TL_RED


def compute(ctx, section=None) -> dict:
    pairs = ctx.bundle.get("pairs") or []
    if not pairs:
        return {"present": False}
    tools = [t for t in (ctx.bundle.get("tools") or []) if t != ANCHOR_TOOL]
    if not tools:
        return {"present": False}
    rf_labels = list(SP6.members)
    by_key: dict[tuple[str, str], dict] = {
        (p["tool"], p["rf_label"]): p for p in pairs
    }
    return {
        "present": True,
        "tools": tools,
        "rf_labels": rf_labels,
        "pairs_by_key": by_key,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no pair-summary data in bundle",
                ha="center", va="center")
        ax.axis("off")
        return fig

    tools = data["tools"]
    rf_labels = data["rf_labels"]
    by_key = data["pairs_by_key"]
    n_rows, n_cols = len(tools), len(rf_labels)
    fig, ax = plt.subplots(figsize=(0.95 * n_cols + 1.6, 0.7 * n_rows + 1.2))

    for i, tool in enumerate(tools):
        for j, lab in enumerate(rf_labels):
            p = by_key.get((tool, lab))
            color = _tl_color(p) if p else TL_GREY
            ax.add_patch(plt.Rectangle(
                (j, n_rows - 1 - i), 1, 1,
                facecolor=color, edgecolor="white", lw=1.0,
            ))
            text = "—"
            if p is not None:
                n_pass = int(p.get("n_clusters_passing", 0) or 0)
                n_fail = int(p.get("n_clusters_failing", 0) or 0)
                n_null = int(p.get("n_clusters_null", 0) or 0)
                n_eval = n_pass + n_fail
                if n_eval > 0:
                    text = f"{n_pass}/{n_eval}"
                elif n_null > 0:
                    text = "μ·∅"
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, text,
                    ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(rf_labels, fontsize=10)
    ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels([f"popout vs {t}" for t in tools], fontsize=10)
    ax.set_aspect("equal")
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Pass rate per (tool, RF label)\n"
                 "cell = #passing / #μ-evaluable clusters",
                 fontsize=10)
    fig.tight_layout()
    return fig
