"""Per-cluster FLARE primary-ancestry decomposition.

One stacked horizontal bar per ``cluster_id``. Sample counts are taken
from the chr1 rows of ``cohort_global.tsv`` (chr1 is FLARE's training
chromosome and every sample appears exactly once at chr1); the primary
ancestry per sample is the argmax of its FLARE proportion vector.

Per-chromosome variation in the primary call is documented separately
in the per-chrom drift / panel-coverage section; this section is the
cluster-level snapshot.

Recovered from the pre-aa94436 ``cohort_composition`` chart, simplified
onto v6's named-column ``cohort_global.tsv`` (no merged_groups_rf
indirection) and collapsed to one row per cluster_id.

Data: ``cohort/cohort_global.tsv``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP6

from .._helpers import cluster_styles, read_tsv


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "cohort_global.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    required = {"cluster_id", "chrom", "sample_id"}
    if not required.issubset(col):
        return {"present": False}
    # Ancestry columns are everything in the header that names an SP6
    # member (FLARE has no MID, so this is SP5 in practice; the SP6
    # check keeps the chart honest if a future panel grows).
    anc_cols: list[tuple[str, int]] = [
        (h, i) for h, i in col.items() if h in SP6.members
    ]
    if not anc_cols:
        return {"present": False}

    # Collapse to one row per cluster_id. Use chr1 as the canonical
    # per-sample row (FLARE trains on chr1; each sample appears at chr1
    # exactly once). This avoids the 15 x 22 = 330-bar wall the (cluster,
    # chrom) stratification would produce.
    counts: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
        except (IndexError, KeyError):
            continue
        if chrom != "chr1":
            continue
        try:
            vals = [(name, float(r[i])) for name, i in anc_cols]
        except (IndexError, ValueError):
            continue
        if not vals:
            continue
        primary = max(vals, key=lambda kv: kv[1])[0]
        d = counts.setdefault(cid, {})
        d[primary] = d.get(primary, 0) + 1
        totals[cid] = totals.get(cid, 0) + 1

    if not counts:
        return {"present": False}

    cohort_counts: dict[str, int] = {}
    for d in counts.values():
        for rf, n in d.items():
            cohort_counts[rf] = cohort_counts.get(rf, 0) + n

    cc_keys = sorted(counts.keys(), key=lambda k: -totals.get(k, 0))
    rf_set = [a for a in SP6.members if a in cohort_counts] + sorted(
        rf for rf in cohort_counts if rf not in SP6.members
    )

    # Per-cluster top-1 + top-2 ancestries for the prose callout.
    cluster_summary: list[dict] = []
    for cc in cc_keys:
        cc_total = totals[cc]
        if cc_total == 0:
            continue
        sorted_ancs = sorted(
            counts[cc].items(), key=lambda kv: -kv[1],
        )
        top1 = sorted_ancs[0]
        top2 = sorted_ancs[1] if len(sorted_ancs) > 1 else (None, 0)
        cluster_summary.append({
            "cluster": cc,
            "n": cc_total,
            "top1": top1[0],
            "top1_frac": top1[1] / cc_total,
            "top2": top2[0],
            "top2_frac": (top2[1] / cc_total) if top2[0] else 0.0,
        })

    return {
        "present": True,
        "counts": counts,
        "totals": totals,
        "cc_keys": cc_keys,
        "rf_set": rf_set,
        "cluster_summary": cluster_summary,
        "cluster_styles": cluster_styles(counts.keys()),
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no per-cluster composition data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    counts = data["counts"]
    totals = data["totals"]
    cc_keys = data["cc_keys"]
    rf_set = data["rf_set"]
    cstyles: dict[str, dict[str, str]] = data.get("cluster_styles", {})

    n_clusters = len(cc_keys)
    fig = plt.figure(figsize=(9.0, max(3.5, 0.35 * n_clusters + 1.4)))
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[max(1.0, 0.35 * n_clusters), 0.55],
        hspace=0.30,
    )
    ax_clusters = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    def _color(rf: str) -> str:
        return palette.get(rf, "#888888")

    cc_labels = list(cc_keys)
    y = np.arange(n_clusters)
    left_arr = np.zeros(n_clusters)
    legend_handles: list[tuple[str, object]] = []
    for rf in rf_set:
        vals = np.array([counts[cc].get(rf, 0) for cc in cc_keys])
        if not vals.any():
            continue
        bars = ax_clusters.barh(
            y, vals, left=left_arr, color=_color(rf),
            edgecolor="white", linewidth=0.5, height=0.72,
        )
        legend_handles.append((rf, bars[0]))
        left_arr += vals
    ax_clusters.set_yticks(y)
    ax_clusters.set_yticklabels(cc_labels, fontsize=9)
    # Color each y-tick label by the cluster's identity color and stamp
    # the per-cluster marker shape just to the left of the bar, both
    # matching the (color, marker) the raindrop will carry in the
    # downstream raincloud charts.
    for label, cid in zip(ax_clusters.get_yticklabels(), cc_keys):
        style = cstyles.get(cid)
        if style:
            label.set_color(style["color"])
            label.set_fontweight("bold")
    ax_clusters.invert_yaxis()
    total_max = max(totals.values()) if totals else 1
    if cstyles:
        # Pin the marker at a fraction of total_max to the LEFT of the
        # bar's origin so it sits in the y-tick gutter.
        marker_x = -total_max * 0.012
        for i, cid in enumerate(cc_keys):
            style = cstyles.get(cid)
            if not style:
                continue
            ax_clusters.scatter(
                marker_x, i, marker=style["marker"], s=44,
                color=style["color"], edgecolor="white",
                linewidth=0.5, clip_on=False, zorder=4,
            )
    for i, cc in enumerate(cc_keys):
        ax_clusters.text(
            left_arr[i] + total_max * 0.01, i,
            f"  n={totals.get(cc, 0):,}",
            ha="left", va="center", fontsize=8, color="#222",
        )
    ax_clusters.set_xlim(0, total_max * 1.12)
    ax_clusters.set_xlabel("number of samples", fontsize=10)
    ax_clusters.set_title(
        "FLARE primary-ancestry decomposition per (cluster, chrom)",
        fontsize=12, loc="left",
    )
    for spine in ("top", "right"):
        ax_clusters.spines[spine].set_visible(False)

    ax_legend.legend(
        [h for _, h in legend_handles],
        [rf for rf, _ in legend_handles],
        title="ancestry", loc="center",
        ncol=min(len(legend_handles), 6),
        fontsize=9, title_fontsize=9, frameon=False,
    )
    return fig
