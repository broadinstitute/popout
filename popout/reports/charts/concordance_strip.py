"""FLARE vs Rye concordance — per-ancestry Pearson r + Lin's CCC.

Two stacked **raincloud + sina-rain** panels (r on top, CCC below).
Each ancestry row is split into three strict, non-overlapping lanes:

    violin band  = [i-0.46, i-0.14]   half-violin (KDE) above the bar
    bar          = [i-0.08, i+0.08]   cohort-pooled n-weighted value
    rain band    = [i+0.14, i+0.46]   sina raindrops (jitter ∝ density)

Each raindrop is one ``(cluster_id, chrom)`` value, coloured by the row
ancestry. The dashed reference lines are the 0.95 / 0.90 pass
thresholds. Lanes are partitioned so nothing spills into a neighbouring
ancestry row.

Layout adapted from ``my_notes/graphs/scale_proto_K2.py`` — the chosen
audition.

Data: ``cohort/concordance_metrics.tsv`` (per ``(cluster, chrom,
ancestry)`` with ``cluster_mu / n_samples / pearson_r / ccc``). Rows
with ``cluster_mu < 0.01`` are μ-gated and excluded.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP5

from .._helpers import (
    cluster_styles,
    n_weighted_mean,
    raincloud_panel,
    read_tsv,
    to_float,
    topn,
)


PEARSON_REF = 0.95
CCC_REF = 0.90


# ── compute ────────────────────────────────────────────────────────────


def compute(ctx, section=None) -> dict:
    opts = section.options if section is not None else {}
    # Cohort filename of the per-(cluster, chrom, ancestry) concordance
    # table. Set per-section in the YAML at the section's top level
    # (NOT under a nested ``options:`` block — see config.SectionSpec):
    # ``source: rye`` -> ``concordance_metrics_rye.tsv``;
    # ``source: rf``  -> ``concordance_metrics_rf.tsv``.
    source = opts.get("source", "rye")
    if source not in ("rye", "rf"):
        raise ValueError(
            f"concordance_strip section 'source' must be 'rye' or 'rf'; "
            f"got {source!r}"
        )
    filename = f"concordance_metrics_{source}.tsv"
    path = ctx.bundle_dir / "cohort" / filename
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False, "source": source}
    col = {h: i for i, h in enumerate(header)}
    by_anc: dict[str, list[tuple[str, str, int, float, float, float]]] = {}
    cluster_ids_seen: set[str] = set()
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            cc = f"{cid}·{chrom}"
            anc = r[col["ancestry"]]
            mu = to_float(r[col["cluster_mu"]])
            n = int(float(r[col["n_samples"]]))
            pr = to_float(r[col["pearson_r"]])
            cccv = to_float(r[col["ccc"]])
        except (IndexError, KeyError, ValueError):
            continue
        if mu is None or mu < 0.01:
            continue
        if pr is None and cccv is None:
            continue
        by_anc.setdefault(anc, []).append((
            cid, cc, n, mu,
            pr if pr is not None else float("nan"),
            cccv if cccv is not None else float("nan"),
        ))
        cluster_ids_seen.add(cid)
    if not by_anc:
        return {"present": False, "all_gated": True, "source": source}

    sp5 = list(SP5.members)
    labels = [a for a in sp5 if a in by_anc] + sorted(
        a for a in by_anc if a not in sp5
    )

    pooled_r: dict[str, float | None] = {}
    pooled_ccc: dict[str, float | None] = {}
    for anc in labels:
        items = by_anc[anc]
        pooled_r[anc] = n_weighted_mean(
            [(n, pr) for _, _, n, _, pr, _ in items]
        )
        pooled_ccc[anc] = n_weighted_mean(
            [(n, cc) for _, _, n, _, _, cc in items]
        )

    gaps: list[tuple[str, float]] = []
    for anc, items in by_anc.items():
        for _cid, cc, _n, _mu, pr, cccv in items:
            if pr == pr and cccv == cccv:
                gaps.append((f"{cc}·{anc}", pr - cccv))
    top_gap = topn(gaps, n=3)

    cstyles = cluster_styles(cluster_ids_seen)

    return {
        "present": True,
        "source": source,
        "by_anc": by_anc,
        "labels": labels,
        "pooled_r": pooled_r,
        "pooled_ccc": pooled_ccc,
        "top_gap": top_gap,
        "pearson_ref": PEARSON_REF,
        "ccc_ref": CCC_REF,
        "cluster_styles": cstyles,
    }


# ── render: raincloud + sina rain (K2 layout) ─────────────────────────


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(
            0.5, 0.5,
            "no μ-evaluable concordance data"
            if data.get("all_gated") else "no concordance data",
            ha="center", va="center",
        )
        ax.axis("off")
        return fig

    labels = data["labels"]
    by_anc = data["by_anc"]
    pooled_r = data["pooled_r"]
    pooled_ccc = data["pooled_ccc"]
    cstyles: dict[str, dict[str, str]] = data.get("cluster_styles", {})
    n_anc = len(labels)

    # Per-row value lists for the two panels (Pearson r above, Lin's CCC
    # below). Aligned cluster_id lists let raincloud_panel color each
    # raindrop by cluster.
    per_row_r: dict[str, list[float]] = {}
    per_row_r_clusters: dict[str, list[str]] = {}
    per_row_ccc: dict[str, list[float]] = {}
    per_row_ccc_clusters: dict[str, list[str]] = {}
    for anc in labels:
        items = by_anc[anc]
        per_row_r[anc] = [pr for _cid, _cc, _n, _mu, pr, _ in items]
        per_row_r_clusters[anc] = [cid for cid, _cc, _n, _mu, _pr, _ in items]
        per_row_ccc[anc] = [ccc for _cid, _cc, _n, _mu, _pr, ccc in items]
        per_row_ccc_clusters[anc] = [cid for cid, _cc, _n, _mu, _pr, _ in items]

    all_r = [v for vals in per_row_r.values() for v in vals if v == v] + [
        v for v in pooled_r.values() if v is not None
    ]
    all_c = [v for vals in per_row_ccc.values() for v in vals if v == v] + [
        v for v in pooled_ccc.values() if v is not None
    ]
    x_lo = max(0.0, min([*all_r, *all_c, 0.90]) - 0.05)
    x_hi = 1.02

    n_clusters = len(cstyles)
    # Two rows of cluster swatches at <=8 per row; size the strip to fit.
    cluster_rows = (n_clusters + 7) // 8 if n_clusters else 0
    cluster_legend_h = 0.30 + 0.18 * cluster_rows if cstyles else 0.0
    chart_h = max(2.6, 0.95 * n_anc + 0.7)
    fig = plt.figure(
        figsize=(10.0, 2 * chart_h + 0.9 + cluster_legend_h),
    )
    height_ratios = [chart_h, chart_h, 0.45]
    if cstyles:
        height_ratios.append(cluster_legend_h)
    gs = fig.add_gridspec(
        nrows=len(height_ratios), ncols=1,
        height_ratios=height_ratios,
        hspace=0.35,
    )
    ax_r = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.axis("off")
    ax_clusters = None
    if cstyles:
        ax_clusters = fig.add_subplot(gs[3, 0])
        ax_clusters.axis("off")

    other = {"rye": "Rye", "rf": "RF"}.get(data.get("source", "rye"), "other tool")
    raincloud_panel(
        ax_r, labels, pooled_r, per_row_r,
        palette=palette, x_lo=x_lo, x_hi=x_hi,
        title=f"Pearson r vs {other} - rank linearity  ·  raincloud + sina rain",
        xlabel="Pearson r",
        threshold=PEARSON_REF,
        threshold_label=f"pass threshold: {PEARSON_REF:.2f}",
        clusters_by_label=per_row_r_clusters,
        cluster_style_map=cstyles or None,
    )
    raincloud_panel(
        ax_c, labels, pooled_ccc, per_row_ccc,
        palette=palette, x_lo=x_lo, x_hi=x_hi,
        title=f"Lin's CCC vs {other} - linearity + calibration  ·  raincloud + sina rain",
        xlabel="Lin's CCC",
        threshold=CCC_REF,
        threshold_label=f"pass threshold: {CCC_REF:.2f}",
        clusters_by_label=per_row_ccc_clusters,
        cluster_style_map=cstyles or None,
    )

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    violin_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888", alpha=0.42)
    rain_proxy = plt.scatter([], [], s=20, color="#888",
                             edgecolor="white", linewidth=0.4)
    ax_legend.legend(
        [bar_proxy, violin_proxy, rain_proxy],
        ["bar = cohort-pooled (n-weighted)",
         "half-violin = KDE of per-(cluster, chrom) values",
         "raindrop = one (cluster, chrom) value, by cluster (color + shape)"],
        loc="center", ncol=3, fontsize=9, frameon=False,
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
