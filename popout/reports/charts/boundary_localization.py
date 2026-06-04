"""Local-mode boundary localization — log-scale distance histogram +
flanking-label match bar.

Each row of ``cohort/boundary_localization.tsv`` represents one
FLARE switch and the nearest popout switch. We render two panels:

  - left: log-scale histogram of ``|distance_bp|``
  - right: matched / unmatched flanking-label bar
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .._helpers import fmt_pct, read_tsv, to_float


TL_GREEN = "#117733"
TL_RED = "#CC3311"


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "boundary_localization.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    dist_col = "distance_bp" if "distance_bp" in col else None
    flank_col = "flanking_label_match" if "flanking_label_match" in col else None
    if dist_col is None:
        return {"present": False}

    distances: list[float] = []
    matched: list[bool] = []
    for r in rows:
        d = to_float(r[col[dist_col]])
        if d is not None:
            distances.append(abs(d))
        if flank_col is not None:
            v = r[col[flank_col]]
            if v in ("true", "True", "1"):
                matched.append(True)
            elif v in ("false", "False", "0"):
                matched.append(False)

    n_match = sum(1 for m in matched if m)
    n_total = len(matched)
    return {
        "present": True,
        "distances": distances,
        "matched": matched,
        "n_match": n_match,
        "n_total": n_total,
        "match_pct": fmt_pct(n_match / n_total) if n_total > 0 else "—",
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.4))
        ax.text(0.5, 0.5, "no boundary-localization data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    distances = data["distances"]
    matched = data["matched"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    if distances:
        bins = np.logspace(2, 8, 30)
        axes[0].hist(distances, bins=bins, color=TL_GREEN, alpha=0.85)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("|distance_bp| to nearest popout switch")
        axes[0].set_ylabel("# FLARE switches")
        axes[0].set_title("Boundary localization (log-scale)", fontsize=10)
        for s in ("top", "right"):
            axes[0].spines[s].set_visible(False)
    else:
        axes[0].text(0.5, 0.5, "no distance data", ha="center", va="center")
        axes[0].axis("off")

    if matched:
        n_match = data["n_match"]
        n_total = data["n_total"]
        axes[1].bar(
            ["matched", "unmatched"],
            [n_match, n_total - n_match],
            color=[TL_GREEN, TL_RED],
        )
        axes[1].set_ylabel("# FLARE switches")
        axes[1].set_title(
            f"Flanking-label match ({data['match_pct']})",
            fontsize=10,
        )
        for s in ("top", "right"):
            axes[1].spines[s].set_visible(False)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return fig
