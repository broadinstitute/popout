"""FLARE switch rate — raincloud + sina rain, per FLARE top-1 ancestry.

Each ``(cluster_id, chrom)`` row in ``cohort/switch_rate_stats.tsv`` is
assigned to the *modal* FLARE top-1 ancestry of its samples (from
``cohort_global.tsv`` via ``load_cohort_cube``). One row per stratum.

Per row:
  - cohort-pooled bar = n_haplotypes-weighted mean of that stratum's
    per-(cluster, chrom) means
  - half-violin = KDE of per-(cluster, chrom) means in the stratum
  - sina rain  = one raindrop per (cluster, chrom) mean

The previous per-(cluster, chrom) forest was illegible on production
cohorts (clusters × chroms grows linearly); top-1 ancestry collapses
the y-axis to N_strata while keeping the per-(cluster, chrom) drift
visible as the rain shape.

Data: ``cohort/switch_rate_stats.tsv`` (cluster_id, chrom, n_haplotypes,
min, median, mean, p99, max) and ``cohort_global.tsv`` (for the
(cluster, chrom) → modal top-1 ancestry map).
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5

from .._helpers import (
    load_cohort_cube,
    n_weighted_mean,
    raincloud_panel,
    read_tsv,
)


def _cluster_chrom_to_top1(cube_data: dict) -> dict[tuple[str, str], str]:
    if not cube_data:
        return {}
    cube = cube_data["cube"]
    members = list(cube_data["label_space"].members)
    chroms = list(cube_data["chroms"])
    chrom_idx = {c: i for i, c in enumerate(chroms)}
    sample_ids = list(cube_data["sample_ids"])
    cluster_of = cube_data["cluster_of"]
    sid_idx = {sid: i for i, sid in enumerate(sample_ids)}

    counts: dict[tuple[str, str], Counter] = {}
    for (sid, chrom), cid in cluster_of.items():
        si = sid_idx.get(sid)
        ci = chrom_idx.get(chrom)
        if si is None or ci is None:
            continue
        top1 = members[int(np.argmax(cube[si, ci, :]))]
        counts.setdefault((cid, chrom), Counter())[top1] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "switch_rate_stats.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}

    items: list[tuple[str, str, int, float, float, float, float, float]] = []
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            n_hap = int(float(r[col["n_haplotypes"]]))
            mn = float(r[col["min"]])
            med = float(r[col["median"]])
            mean = float(r[col["mean"]])
            p99 = float(r[col["p99"]])
            mx = float(r[col["max"]])
        except (IndexError, KeyError, ValueError):
            continue
        items.append((cid, chrom, n_hap, mn, med, mean, p99, mx))
    if not items:
        return {"present": False}

    mid_rule = (section.mid_rule if section is not None else None) or "drop"
    cube_data = load_cohort_cube(
        ctx.bundle_dir, label_space=SP5, mid_rule=mid_rule)
    cc_to_anc = _cluster_chrom_to_top1(cube_data)

    cohort_mean = n_weighted_mean(
        [(it[2], it[5]) for it in items]) or 0.0
    cohort_med = n_weighted_mean(
        [(it[2], it[4]) for it in items]) or 0.0
    cohort_min = min(it[3] for it in items)
    cohort_p99 = max(it[6] for it in items)
    cohort_max = max(it[7] for it in items)
    cohort_n_hap = sum(it[2] for it in items)

    members = list(SP5.members)
    strata_rows: list[dict] = []
    for lab in members:
        sub = [it for it in items if cc_to_anc.get((it[0], it[1])) == lab]
        means = [it[5] for it in sub]
        if not sub:
            strata_rows.append({"label": lab, "n_hap": 0, "n_cc": 0,
                                "min": None, "median": None, "mean": None,
                                "p99": None, "max": None, "means": []})
            continue
        s_mean = n_weighted_mean([(it[2], it[5]) for it in sub]) or 0.0
        s_med = n_weighted_mean([(it[2], it[4]) for it in sub]) or 0.0
        strata_rows.append({
            "label": lab,
            "n_hap": sum(it[2] for it in sub),
            "n_cc": len(sub),
            "min": min(it[3] for it in sub),
            "median": s_med,
            "mean": s_mean,
            "p99": max(it[6] for it in sub),
            "max": max(it[7] for it in sub),
            "means": means,
        })
    n_unmapped = sum(1 for it in items if (it[0], it[1]) not in cc_to_anc)

    return {
        "present": True,
        "strata_rows": strata_rows,
        "cohort_mean": cohort_mean,
        "cohort_med": cohort_med,
        "cohort_min": cohort_min,
        "cohort_p99": cohort_p99,
        "cohort_max": cohort_max,
        "cohort_n_hap": cohort_n_hap,
        "n_cluster_chrom": len(items),
        "n_unmapped_cluster_chrom": n_unmapped,
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
        ax.text(0.5, 0.5, "no strata with samples",
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
         "half-violin = KDE of per-(cluster, chrom) means",
         "raindrop = one (cluster, chrom) mean"],
        loc="center", ncol=3, fontsize=9, frameon=False,
    )
    return fig
