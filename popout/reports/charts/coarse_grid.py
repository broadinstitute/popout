"""Local-mode coarse-grid sweep — diagonal-fraction vs resolution.

Reads ``cohort/coarse_grid_summary.tsv`` (one row per
(cluster, sample, hap, chrom, resolution_mb)). Renders each
(cluster, sample, hap, chrom) series as a faint grey line + the
per-resolution median in bold.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .._helpers import read_tsv, to_float


TL_GREY = "#888888"
TL_GREEN = "#117733"


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "coarse_grid_summary.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    if "resolution_mb" not in col or "diagonal_fraction" not in col:
        return {"present": False}

    series: dict[tuple[str, str, str, str], list[tuple[float, float]]] = {}
    per_x: dict[float, list[float]] = {}
    for r in rows:
        res = to_float(r[col["resolution_mb"]])
        diag = to_float(r[col["diagonal_fraction"]])
        if res is None or diag is None:
            continue
        sample = r[col["sample"]] if "sample" in col else ""
        hap = r[col["hap"]] if "hap" in col else ""
        chrom = r[col["chrom"]] if "chrom" in col else ""
        cid = r[col["cluster_id"]] if "cluster_id" in col else ""
        key = (cid, chrom, sample, hap)
        series.setdefault(key, []).append((res, diag))
        per_x.setdefault(res, []).append(diag)

    if not series:
        return {"present": False}

    medians = sorted(per_x)
    median_ys = [float(np.median(per_x[x])) for x in medians]
    return {
        "present": True,
        "series": series,
        "median_xs": medians,
        "median_ys": median_ys,
        "n_series": len(series),
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.4))
        ax.text(0.5, 0.5, "no coarse-grid data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    series = data["series"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for pts in series.values():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=TL_GREY, alpha=0.15, lw=0.8)
    ax.plot(data["median_xs"], data["median_ys"],
            color=TL_GREEN, lw=2.2, label="median")
    ax.set_xscale("log")
    ax.set_xlabel("resolution (Mb)")
    ax.set_ylabel("diagonal fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coarse-grid sweep: agreement vs resolution", fontsize=11,
                 loc="left")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
