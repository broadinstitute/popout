"""Hap disagreement per superpop label — cohort-pooled bar + per-cluster ticks.

Per-sample bp-fraction where hap1 and hap2 disagree on ancestry call,
grouped by the RF superpop label (rows). Shaded baseline band marks
the expected level for genuinely admixed labels.

Data: ``cohort/hap_disagreement.tsv`` with columns (cluster_id, chrom,
rf_label, n, mean).

Reporting principle: rendered as the bundle delivers it. Any
canonicalisation of the ``rf_label`` column is the stats collector's
job (``my_notes/validation/COLLECTOR_FIXES.md``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP6

from .._helpers import (
    n_weighted_mean,
    overlay_ticks,
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

    # Highest disagreement on pure-ancestry labels (proxy for phasing noise).
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

    # Per-(rf_label, cluster) rows for the supporting table.
    table_rows: list[dict] = []
    for rf in labels:
        for cc, n, mean in sorted(by_rf[rf], key=lambda t: t[0]):
            table_rows.append({
                "rf_label": rf, "cluster_chrom": cc,
                "n": n, "mean": mean,
            })

    return {
        "present": True,
        "by_rf": by_rf,
        "labels": labels,
        "pooled": pooled,
        "pooled_n": pooled_n,
        "top_pure": top_pure,
        "top_spread": top_spread,
        "table_rows": table_rows,
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

    chart_h = max(2.6, 0.7 * n_lab + 0.6)
    fig = plt.figure(figsize=(9.0, chart_h + 0.8))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[chart_h, 0.5], hspace=0.3,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    def _color(rf: str) -> str:
        return palette.get(rf.split(".")[0], "#888888")

    all_x = [m for items in by_rf.values() for _, _, m in items]
    all_x += [v for v in pooled.values() if v is not None]
    x_hi = max(0.45, (max(all_x) if all_x else 0.4) * 1.15)

    ax.axvspan(BASELINE_LO, BASELINE_HI, color="#cccccc", alpha=0.30, zorder=0)

    for i, rf in enumerate(labels):
        v = pooled.get(rf)
        if v is not None:
            ax.barh(i, v, color=_color(rf), edgecolor="white",
                    linewidth=0.6, height=0.62, zorder=2)
            ax.text(v + x_hi * 0.005, i, f"{v:.3f}",
                    ha="left", va="center", fontsize=9, color="#222")
        per_cluster = [m for _, _, m in by_rf[rf]]
        overlay_ticks(ax, i, per_cluster, color="#111",
                      tick_height=0.46, lw=1.6, alpha=0.95)
    ax.set_yticks(range(n_lab))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, x_hi)
    ax.set_xlabel("mean hap-disagreement fraction (bp-weighted)", fontsize=10)
    ax.set_title(
        "Hap disagreement per superpop label  ·  bar = cohort-pooled "
        "(n-weighted)  ·  ticks = per cluster · chrom",
        fontsize=11, loc="left",
    )
    ax.text(
        0.5 * (BASELINE_LO + BASELINE_HI), n_lab - 0.4,
        "expected baseline for admixed labels (0.10–0.30)",
        ha="center", va="bottom", fontsize=8, color="#666",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    tick_proxy = plt.Line2D([0], [0], color="#111", linewidth=1.6)
    band_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#cccccc", alpha=0.5)
    ax_legend.legend(
        [bar_proxy, tick_proxy, band_proxy],
        ["cohort-pooled mean (n-weighted)",
         "per-cluster mean",
         "expected baseline for admixed labels (0.10–0.30)"],
        loc="center", ncol=3, fontsize=9, frameon=False,
    )
    return fig
