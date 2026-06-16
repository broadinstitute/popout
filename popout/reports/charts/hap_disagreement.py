"""Hap disagreement per FLARE top-1 ancestry: raincloud + sina rain.

Per-sample bp-fraction where hap1 and hap2 disagree on ancestry call,
bucketed by **FLARE's per-sample top-1 ancestry** (the new schema in
v5.0.0 bundles). The metric is FLARE-internal end to end; RF never
appears, MID never appears.

Layout: K2 raincloud + sina-style rain (audition_hap.H1).

Data: ``cohort/hap_disagreement.tsv`` with columns
``(cluster_id, chrom, flare_top1, n, mean, median)``. Each raindrop is
one ``(cluster_id, chrom)`` row. Cluster is *not* a chart axis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP5

from .._helpers import (
    n_weighted_mean,
    raincloud_panel,
    read_tsv,
    topn,
)


BASELINE_LO = 0.10
BASELINE_HI = 0.30


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "hap_disagreement.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    if "flare_top1" not in col:
        raise RuntimeError(
            f"{path}: expected v5 schema column 'flare_top1'; "
            f"header was {header!r}. Regenerate the bundle under "
            f"schema v5.0.0."
        )

    by_top1: dict[str, list[tuple[str, int, float]]] = {}
    for r in rows:
        try:
            cc = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
            top1 = r[col["flare_top1"]]
            n = int(r[col["n"]])
            mean = float(r[col["mean"]])
        except (IndexError, KeyError, ValueError):
            continue
        by_top1.setdefault(top1, []).append((cc, n, mean))
    if not by_top1:
        return {"present": False}

    labels = sorted(
        by_top1.keys(),
        key=lambda a: SP5.members.index(a) if a in SP5.members else 99,
    )
    pooled: dict[str, float | None] = {}
    pooled_n: dict[str, int] = {}
    for top1 in labels:
        items = [(n, m) for _, n, m in by_top1[top1] if n >= 5]
        pooled[top1] = n_weighted_mean(items)
        pooled_n[top1] = sum(n for n, _ in items)

    pure_hits: list[tuple[str, float]] = []
    for top1 in SP5.members:
        if top1 not in by_top1:
            continue
        for cc, _n, mean in by_top1[top1]:
            pure_hits.append((f"{cc} · top1={top1}", mean))
    top_pure = topn(pure_hits, n=3)

    spreads: list[tuple[str, float]] = []
    for top1, items in by_top1.items():
        means = [m for _, _, m in items]
        if len(means) > 1:
            spreads.append((top1, max(means) - min(means)))
    top_spread = topn(spreads, n=2)

    return {
        "present": True,
        "by_top1": by_top1,
        "labels": labels,
        "pooled": pooled,
        "pooled_n": pooled_n,
        "top_pure": top_pure,
        "top_spread": top_spread,
        "baseline_lo": BASELINE_LO,
        "baseline_hi": BASELINE_HI,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no hap-disagreement data", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = data["labels"]
    by_top1 = data["by_top1"]
    pooled = data["pooled"]
    n_lab = len(labels)

    per_row = {top1: [m for _, _, m in by_top1[top1]] for top1 in labels}
    all_x = [m for vals in per_row.values() for m in vals]
    all_x += [v for v in pooled.values() if v is not None]
    x_hi = max(0.45, (max(all_x) if all_x else 0.4) * 1.10)

    chart_h = max(3.0, 0.95 * n_lab + 0.7)
    fig = plt.figure(figsize=(10.0, chart_h + 0.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[chart_h, 0.45], hspace=0.4,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    ax.axvspan(BASELINE_LO, BASELINE_HI, color="#cccccc", alpha=0.30, zorder=0)
    ax.text(
        0.5 * (BASELINE_LO + BASELINE_HI), -0.55,
        "expected baseline for admixed top-1 strata (0.10-0.30)",
        ha="center", va="bottom", fontsize=8, color="#666",
    )

    raincloud_panel(
        ax, labels, pooled, per_row,
        palette=palette, x_lo=0.0, x_hi=x_hi,
        title=("Hap disagreement per FLARE top-1 ancestry  ·  raincloud + sina rain"),
        xlabel="mean hap-disagreement fraction (bp-weighted)",
    )

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    violin_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888", alpha=0.42)
    rain_proxy = plt.scatter([], [], s=20, color="#888",
                             edgecolor="white", linewidth=0.4)
    band_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#cccccc", alpha=0.5)
    ax_legend.legend(
        [bar_proxy, violin_proxy, rain_proxy, band_proxy],
        ["bar = stratum mean (n-weighted)",
         "half-violin = KDE of per-(cluster, chrom) means",
         "raindrop = one (cluster, chrom) mean",
         "admixed-baseline band (0.10-0.30)"],
        loc="center", ncol=4, fontsize=9, frameon=False,
    )
    return fig
