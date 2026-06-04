"""FLARE switch rate — forest layout, cohort + per-cluster.

Per-haplotype ancestry-switch counts. The bold top row is the
n_haplotypes-weighted cohort line; dim rows below are per (cluster,
chrom). Each row shows min → p99 as a horizontal line; mean as a
vertical tick; median as a hollow square; max as a red ✕.

Data: ``cohort/switch_rate_stats.tsv`` (cluster_id, chrom,
n_haplotypes, min, median, mean, p99, max).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .._helpers import n_weighted_mean, read_tsv, topn


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "switch_rate_stats.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}

    items: list[tuple[str, int, float, float, float, float, float]] = []
    for r in rows:
        try:
            lab = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
            n_hap = int(float(r[col["n_haplotypes"]]))
            mn = float(r[col["min"]])
            med = float(r[col["median"]])
            mean = float(r[col["mean"]])
            p99 = float(r[col["p99"]])
            mx = float(r[col["max"]])
        except (IndexError, KeyError, ValueError):
            continue
        items.append((lab, n_hap, mn, med, mean, p99, mx))
    if not items:
        return {"present": False}
    items.sort(key=lambda t: -t[4])

    cohort_mean = n_weighted_mean([(it[1], it[4]) for it in items]) or 0.0
    cohort_med = n_weighted_mean([(it[1], it[3]) for it in items]) or 0.0
    cohort_min = min(it[2] for it in items)
    cohort_p99 = max(it[5] for it in items)
    cohort_max = max(it[6] for it in items)
    cohort_n_hap = sum(it[1] for it in items)

    gaps = [(it[0], it[6] - it[5]) for it in items]
    top_gap = topn(gaps, n=1)

    return {
        "present": True,
        "cluster_rows": items,
        "cohort_mean": cohort_mean,
        "cohort_med": cohort_med,
        "cohort_min": cohort_min,
        "cohort_p99": cohort_p99,
        "cohort_max": cohort_max,
        "cohort_n_hap": cohort_n_hap,
        "top_gap": top_gap,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no switch-rate data", ha="center", va="center")
        ax.axis("off")
        return fig

    items = data["cluster_rows"]
    cohort_mean = data["cohort_mean"]
    cohort_med = data["cohort_med"]
    cohort_min = data["cohort_min"]
    cohort_p99 = data["cohort_p99"]
    cohort_max = data["cohort_max"]
    cohort_n_hap = data["cohort_n_hap"]

    n = len(items)
    rows_total = 1 + n
    chart_h = max(2.6, 0.4 * rows_total + 1.0)
    fig = plt.figure(figsize=(9.5, chart_h + 0.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[chart_h, 0.45], hspace=0.3,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    x_max = max(cohort_max, max(it[6] for it in items)) * 1.05

    def _draw_row(y, mn, med, mean, p99, mx, *, bold):
        line_color = "#222" if bold else "#bbbbbb"
        lw = 2.4 if bold else 1.2
        ax.hlines(y, mn, p99, color=line_color, linewidth=lw,
                  zorder=2 if bold else 1)
        ax.vlines(mean, y - 0.30, y + 0.30,
                  color="#3366A8" if bold else "#5a8acb",
                  linewidth=2.2 if bold else 1.4, zorder=5)
        s = 90 if bold else 50
        ax.scatter(med, y, marker="s", facecolor="white",
                   edgecolor=line_color,
                   linewidth=1.4 if bold else 0.9, s=s, zorder=4)
        ax.scatter(mx, y, marker="x",
                   color="#c62828" if bold else "#d97a7a",
                   s=110 if bold else 60,
                   linewidth=2.4 if bold else 1.4, zorder=4)

    _draw_row(0, cohort_min, cohort_med, cohort_mean,
              cohort_p99, cohort_max, bold=True)
    for i, (_lab, _n, mn, med, mean, p99, mx) in enumerate(items, start=1):
        _draw_row(i, mn, med, mean, p99, mx, bold=False)

    yticks = [0] + list(range(1, n + 1))
    yticklabels = [f"cohort\n(n_hap={cohort_n_hap:,})"] + [it[0] for it in items]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.set_xlabel("ancestry switches per haplotype", fontsize=10)
    ax.set_title(
        "Switch rate  ·  bold top row = cohort  ·  dim rows = per cluster · chrom",
        fontsize=11, loc="left",
    )
    ax.text(x_max * 0.998, 0,
            f"  mean {cohort_mean:.1f}  ·  median {cohort_med:.1f}  ·  "
            f"p99 {cohort_p99:.0f}  ·  max {cohort_max:.0f}",
            ha="right", va="center", fontsize=8.5, color="#222",
            fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    line_proxy = plt.Line2D([0], [0], color="#222", linewidth=2.4)
    mean_proxy = plt.Line2D([0], [0], color="#3366A8", linewidth=2.2)
    med_proxy = plt.scatter([], [], marker="s", facecolor="white",
                             edgecolor="#222", s=80)
    max_proxy = plt.scatter([], [], marker="x", color="#c62828",
                             s=100, linewidth=2.4)
    ax_legend.legend(
        [line_proxy, mean_proxy, med_proxy, max_proxy],
        ["min → p99", "mean (vertical tick)",
         "median (hollow square)", "max (✕)"],
        loc="center", ncol=4, fontsize=9, frameon=False,
    )
    return fig
