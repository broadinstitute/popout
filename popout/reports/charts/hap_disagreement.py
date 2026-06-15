"""Hap disagreement per superpop label — raincloud + sina rain.

Per-sample bp-fraction where hap1 and hap2 disagree on ancestry call,
grouped by the RF superpop label (one row per label). Layout: K2
raincloud + sina-style rain (audition_hap.H1). The grey band marks
the expected baseline for genuinely admixed labels (0.10–0.30).

Data: ``cohort/hap_disagreement.tsv`` with columns
``(cluster_id, chrom, rf_label, n, mean)``. Each raindrop is one
``(cluster_id, chrom)`` row. Cluster is *not* a chart axis.

Reporting principle: rendered as the bundle delivers it. Any
canonicalisation of the ``rf_label`` column is the stats collector's
job (``my_notes/validation/COLLECTOR_FIXES.md``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP6

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

    by_rf: dict[str, list[tuple[str, int, float]]] = {}
    for r in rows:
        try:
            cc = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
            rf = r[col["rf_label"]]
            n = int(r[col["n"]])
            mean = float(r[col["mean"]])
        except (IndexError, KeyError, ValueError):
            continue
        by_rf.setdefault(rf, []).append((cc, n, mean))
    if not by_rf:
        return {"present": False}

    labels = sorted(
        by_rf.keys(),
        key=lambda a: SP6.members.index(a) if a in SP6.members else 99,
    )
    pooled: dict[str, float | None] = {}
    pooled_n: dict[str, int] = {}
    for rf in labels:
        items = [(n, m) for _, n, m in by_rf[rf] if n >= 5]
        pooled[rf] = n_weighted_mean(items)
        pooled_n[rf] = sum(n for n, _ in items)

    pure_hits: list[tuple[str, float]] = []
    for rf in SP6.members:
        if rf not in by_rf:
            continue
        for cc, _n, mean in by_rf[rf]:
            pure_hits.append((f"{cc} · RF={rf}", mean))
    top_pure = topn(pure_hits, n=3)

    spreads: list[tuple[str, float]] = []
    for rf, items in by_rf.items():
        means = [m for _, _, m in items]
        if len(means) > 1:
            spreads.append((rf, max(means) - min(means)))
    top_spread = topn(spreads, n=2)

    return {
        "present": True,
        "by_rf": by_rf,
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
    by_rf = data["by_rf"]
    pooled = data["pooled"]
    n_lab = len(labels)

    per_row = {rf: [m for _, _, m in by_rf[rf]] for rf in labels}
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
        "expected baseline for admixed labels (0.10–0.30)",
        ha="center", va="bottom", fontsize=8, color="#666",
    )

    raincloud_panel(
        ax, labels, pooled, per_row,
        palette=palette, x_lo=0.0, x_hi=x_hi,
        title=("Hap disagreement per superpop label  ·  raincloud + sina rain"),
        xlabel="mean hap-disagreement fraction (bp-weighted)",
    )

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    violin_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888", alpha=0.42)
    rain_proxy = plt.scatter([], [], s=20, color="#888",
                             edgecolor="white", linewidth=0.4)
    band_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#cccccc", alpha=0.5)
    ax_legend.legend(
        [bar_proxy, violin_proxy, rain_proxy, band_proxy],
        ["bar = cohort-pooled (n-weighted)",
         "half-violin = KDE of per-(cluster, chrom) means",
         "raindrop = one (cluster, chrom) mean",
         "admixed-baseline band (0.10–0.30)"],
        loc="center", ncol=4, fontsize=9, frameon=False,
    )
    return fig
