"""FLARE tract length per ancestry — raincloud in log space.

K2 raincloud + sina rain (audition_tract.T1) on a log10 x-axis. One
row per ancestry; cohort-pooled bar shows the n_tracts-weighted mean
Mb, the half-violin is a log-space KDE of per-(cluster, chrom) means,
and the open diamond marks the per-ancestry model expectation
``100 / (median model_T_gen × K=5)``.

Data: ``cohort/tract_length_stats.tsv`` with columns
``(cluster_id, chrom, ancestry_name, n_tracts, mean_Mb, implied_T_gen,
model_T_gen)``. Cluster is *not* a chart axis.

Reporting principle: rendered faithfully. The ``ancestry_name`` field
is whatever the collector wrote.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP6

from .._helpers import (
    cluster_styles,
    n_weighted_mean,
    raincloud_panel,
    read_tsv,
    to_float,
    topn,
)


K_PANEL = 5  # FLARE's panel size for the model expectation formula.


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "tract_length_stats.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}

    by_anc: dict[str, list[tuple[str, str, int, float, float | None, float | None]]] = {}
    cluster_ids_seen: set[str] = set()
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            cc = f"{cid}·{chrom}"
            name = r[col["ancestry_name"]]
            n_tracts = int(float(r[col["n_tracts"]]))
            mean_mb = to_float(r[col["mean_Mb"]])
            implied_t = to_float(r[col["implied_T_gen"]])
            model_t = to_float(r[col["model_T_gen"]])
        except (IndexError, KeyError, ValueError):
            continue
        if mean_mb is None or mean_mb <= 0:
            continue
        by_anc.setdefault(name, []).append(
            (cid, cc, n_tracts, mean_mb, implied_t, model_t)
        )
        cluster_ids_seen.add(cid)
    if not by_anc:
        return {"present": False}

    labels = sorted(
        by_anc.keys(),
        key=lambda a: SP6.members.index(a.split(".")[0])
        if a.split(".")[0] in SP6.members else 99,
    )

    pooled_mb: dict[str, float | None] = {}
    pooled_n: dict[str, int] = {}
    model_ref_mb: dict[str, float] = {}
    for anc in labels:
        items = by_anc[anc]
        pooled_mb[anc] = n_weighted_mean([(n, mb) for _, _, n, mb, _, _ in items])
        pooled_n[anc] = sum(n for _, _, n, _, _, _ in items)
        models = [t for _, _, _, _, _, t in items if t is not None and t > 0]
        if models:
            med_t = sorted(models)[len(models) // 2]
            model_ref_mb[anc] = 100.0 / (med_t * K_PANEL)

    devs: list[tuple[str, float]] = []
    for anc, v in pooled_mb.items():
        ref = model_ref_mb.get(anc)
        if v is None or ref is None:
            continue
        devs.append((
            f"`{anc}` (empirical {v:.2f} Mb vs model {ref:.2f} Mb)",
            abs(v - ref),
        ))
    top_dev = topn(devs, n=3)

    return {
        "present": True,
        "by_anc": by_anc,
        "labels": labels,
        "pooled_mb": pooled_mb,
        "pooled_n": pooled_n,
        "model_ref_mb": model_ref_mb,
        "top_dev": top_dev,
        "cluster_styles": cluster_styles(cluster_ids_seen),
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no tract-length data", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = data["labels"]
    by_anc = data["by_anc"]
    pooled_mb = data["pooled_mb"]
    model_ref_mb = data["model_ref_mb"]
    cstyles: dict[str, dict[str, str]] = data.get("cluster_styles", {})

    per_row = {anc: [mb for _, _, _, mb, _, _ in by_anc[anc]] for anc in labels}
    per_row_clusters = {
        anc: [cid for cid, _, _, _, _, _ in by_anc[anc]] for anc in labels
    }

    all_x = [v for vals in per_row.values() for v in vals if v > 0]
    all_x += [v for v in pooled_mb.values() if v is not None and v > 0]
    all_x += [v for v in model_ref_mb.values() if v > 0]
    if not all_x:
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no positive tract-length data",
                ha="center", va="center")
        ax.axis("off")
        return fig
    x_lo = 10 ** math.floor(math.log10(min(all_x)))
    x_hi = 10 ** math.ceil(math.log10(max(all_x)))

    n_anc = len(labels)
    n_clusters = len(cstyles)
    cluster_rows = (n_clusters + 7) // 8 if n_clusters else 0
    cluster_legend_h = 0.30 + 0.18 * cluster_rows if cstyles else 0.0
    chart_h = max(3.0, 0.95 * n_anc + 0.7)
    fig = plt.figure(figsize=(11.0, chart_h + 0.6 + cluster_legend_h))
    height_ratios = [chart_h, 0.45]
    if cstyles:
        height_ratios.append(cluster_legend_h)
    gs = fig.add_gridspec(
        nrows=len(height_ratios), ncols=1,
        height_ratios=height_ratios, hspace=0.4,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")
    ax_clusters = None
    if cstyles:
        ax_clusters = fig.add_subplot(gs[2, 0])
        ax_clusters.axis("off")

    raincloud_panel(
        ax, labels, pooled_mb, per_row,
        palette=palette, x_lo=x_lo, x_hi=x_hi,
        title=("Tract length per ancestry  ·  raincloud + sina rain (log scale)"),
        xlabel="mean tract length (Mb, log scale)",
        log=True,
        pooled_fmt="{:.2f} Mb",
        clusters_by_label=per_row_clusters,
        cluster_style_map=cstyles or None,
    )

    for i, anc in enumerate(labels):
        ref = model_ref_mb.get(anc)
        if ref is None or ref <= 0:
            continue
        ax.scatter(ref, i, marker="D", s=110, facecolor="white",
                   edgecolor="#222", linewidth=1.5, zorder=8)
        ax.text(ref * 1.05, i + 0.05, f"model {ref:.1f} Mb",
                fontsize=7.5, color="#444", va="bottom", zorder=8)

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    violin_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888", alpha=0.42)
    rain_proxy = plt.scatter([], [], s=20, color="#888",
                             edgecolor="white", linewidth=0.4)
    model_proxy = plt.scatter([], [], marker="D", s=80, facecolor="white",
                              edgecolor="#222", linewidth=1.4)
    ax_legend.legend(
        [bar_proxy, violin_proxy, rain_proxy, model_proxy],
        ["bar = cohort-pooled (n_tracts-weighted)",
         "half-violin = log-space KDE of per-(cluster, chrom) means",
         "raindrop = one (cluster, chrom) mean, by cluster (color + shape)",
         f"open diamond = model expectation = 100 / (median model_T_gen x K={K_PANEL})"],
        loc="center", ncol=4, fontsize=8.5, frameon=False,
    )
    if ax_clusters is not None:
        cluster_handles = [
            plt.scatter([], [], s=44, color=style["color"],
                        marker=style["marker"],
                        edgecolor="white", linewidth=0.4)
            for style in cstyles.values()
        ]
        ax_clusters.legend(
            cluster_handles, list(cstyles.keys()),
            loc="center", ncol=min(n_clusters, 8),
            fontsize=8, frameon=False, title="cluster",
            title_fontsize=9, columnspacing=1.0, handletextpad=0.4,
        )
    return fig
