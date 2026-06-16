"""FLARE switch rate per FLARE top-1 ancestry: raincloud + sina rain.

Each haplotype's switch count is bucketed by **FLARE's own dominant
ancestry on that haplotype** (column ``dominant_anc`` in
``cohort/switch_rate_per_hap.tsv``). One row per stratum.

Per row:
  - cohort-pooled bar = stratum mean (n_haplotypes-weighted)
  - half-violin = KDE of per-(cluster, chrom) per-hap mean switch
    counts within the stratum
  - sina rain = one raindrop per (cluster, chrom, dominant_anc) cell

Data:
  - ``cohort/switch_rate_per_hap.tsv`` (v5+): one row per
    ``(cluster_id, chrom, sample_id, hap, dominant_anc, n_switches)``.
  - ``cohort/switch_rate_stats.tsv``: cohort-aggregate min/median/p99/max
    for the headline.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP5

from .._helpers import (
    n_weighted_mean,
    raincloud_panel,
    read_tsv,
)


def _read_cohort_aggregate(bundle_dir) -> dict:
    header, rows = read_tsv(bundle_dir / "cohort" / "switch_rate_stats.tsv")
    if not rows:
        return {}
    col = {h: i for i, h in enumerate(header)}
    cohort_min = None
    cohort_p99 = None
    cohort_max = None
    cohort_n_hap = 0
    means = []
    medians = []
    for r in rows:
        try:
            n_hap = int(float(r[col["n_haplotypes"]]))
            mn = float(r[col["min"]])
            med = float(r[col["median"]])
            mean = float(r[col["mean"]])
            p99 = float(r[col["p99"]])
            mx = float(r[col["max"]])
        except (IndexError, KeyError, ValueError):
            continue
        cohort_min = mn if cohort_min is None else min(cohort_min, mn)
        cohort_p99 = p99 if cohort_p99 is None else max(cohort_p99, p99)
        cohort_max = mx if cohort_max is None else max(cohort_max, mx)
        cohort_n_hap += n_hap
        means.append((n_hap, mean))
        medians.append((n_hap, med))
    cohort_mean = n_weighted_mean(means) or 0.0
    cohort_med = n_weighted_mean(medians) or 0.0
    return {
        "cohort_min": cohort_min or 0.0,
        "cohort_max": cohort_max or 0.0,
        "cohort_p99": cohort_p99 or 0.0,
        "cohort_mean": cohort_mean,
        "cohort_med": cohort_med,
        "cohort_n_hap": cohort_n_hap,
    }


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "switch_rate_per_hap.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    for required in ("dominant_anc", "n_switches",
                     "cluster_id", "chrom", "sample_id", "hap"):
        if required not in col:
            raise RuntimeError(
                f"{path}: expected v5 column {required!r}; "
                f"header was {header!r}. Regenerate the bundle under "
                f"schema v5.0.0."
            )

    # Per (cluster, chrom, dominant_anc): collect haplotype switch counts.
    # Each raindrop in the chart is one (cluster, chrom, dom_anc) cell's
    # mean switch count, with n_hap haplotypes contributing.
    per_cell: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            dom = r[col["dominant_anc"]]
            sw = int(r[col["n_switches"]])
        except (IndexError, KeyError, ValueError):
            continue
        per_cell[(cid, chrom, dom)].append(sw)

    members = list(SP5.members)
    strata_rows: list[dict] = []
    for lab in members:
        cell_means: list[tuple[int, float]] = []
        per_row_means: list[float] = []
        all_switches: list[int] = []
        for (cid, chrom, dom), switches in per_cell.items():
            if dom != lab or not switches:
                continue
            cell_mean = sum(switches) / len(switches)
            cell_means.append((len(switches), cell_mean))
            per_row_means.append(cell_mean)
            all_switches.extend(switches)
        if not all_switches:
            strata_rows.append({
                "label": lab, "n_hap": 0, "n_cells": 0,
                "min": None, "median": None, "mean": None,
                "p99": None, "max": None, "means": [],
            })
            continue
        all_switches.sort()
        n = len(all_switches)
        s_mean = n_weighted_mean(cell_means) or 0.0
        s_med = float(all_switches[n // 2])
        p99_idx = max(0, int(round(0.99 * (n - 1))))
        strata_rows.append({
            "label": lab,
            "n_hap": n,
            "n_cells": len(cell_means),
            "min": int(all_switches[0]),
            "median": s_med,
            "mean": s_mean,
            "p99": int(all_switches[p99_idx]),
            "max": int(all_switches[-1]),
            "means": per_row_means,
        })

    aggregate = _read_cohort_aggregate(ctx.bundle_dir)
    n_unstratified = sum(1 for r in rows
                         if r[col["dominant_anc"]] not in members)

    return {
        "present": True,
        "strata_rows": strata_rows,
        "n_haplotypes_total": sum(r["n_hap"] for r in strata_rows),
        "n_unstratified_haplotypes": n_unstratified,
        **aggregate,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no switch-rate data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    strata = [r for r in data["strata_rows"] if r["n_hap"] > 0]
    if not strata:
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no strata with haplotypes",
                ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [r["label"] for r in strata]
    pooled = {r["label"]: r["mean"] for r in strata}
    per_row = {r["label"]: r["means"] for r in strata}

    all_x = [m for vals in per_row.values() for m in vals]
    all_x += [v for v in pooled.values() if v is not None]
    x_hi = (max(all_x) if all_x else 1.0) * 1.05
    x_lo = 0.0

    n = len(strata)
    chart_h = max(3.0, 0.95 * n + 0.7)
    fig = plt.figure(figsize=(10.5, chart_h + 0.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[chart_h, 0.45], hspace=0.4,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    raincloud_panel(
        ax, labels, pooled, per_row,
        palette=palette, x_lo=x_lo, x_hi=x_hi,
        title=("Switch rate per FLARE top-1 ancestry stratum  ·  "
               "raincloud + sina rain"),
        xlabel="ancestry switches per haplotype",
        pooled_fmt="{:.1f}",
    )

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    violin_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888", alpha=0.42)
    rain_proxy = plt.scatter([], [], s=20, color="#888",
                             edgecolor="white", linewidth=0.4)
    ax_legend.legend(
        [bar_proxy, violin_proxy, rain_proxy],
        ["bar = stratum mean (n_haplotypes-weighted)",
         "half-violin = KDE of per-(cluster, chrom) cell means",
         "raindrop = one (cluster, chrom, dominant_anc) cell mean"],
        loc="center", ncol=3, fontsize=9, frameon=False,
    )
    return fig
