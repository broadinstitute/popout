"""Cohort composition — FLARE primary ancestry, cohort-pooled + per-cluster.

The top row of the figure is one wide horizontal bar showing the cohort's
total samples decomposed by FLARE primary ancestry (argmax over each
sample's FLARE proportion vector). The bottom row is a stacked
per-(cluster, chrom) decomposition, sorted by sample count.

Data sources (legacy cohort bundle schema):
  - ``cohort/cohort_global.tsv``   — per-sample proportion vectors
  - ``cohort/merged_groups_rf.tsv`` — per-(cluster, chrom) maps FLARE
                                       component index → SP6 label
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP6

from .._helpers import read_tsv


def _read_merged_groups_rf(bundle_dir: Path) -> dict[tuple[str, str], dict[int, str]]:
    """(cluster_id, chrom) → {FLARE component index → SP6 label}."""
    path = bundle_dir / "cohort" / "merged_groups_rf.tsv"
    header, rows = read_tsv(path)
    out: dict[tuple[str, str], dict[int, str]] = {}
    if not rows:
        return out
    col = {h: i for i, h in enumerate(header)}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            rf = r[col["rf_label"]]
            idxs = r[col["component_indices"]]
        except (IndexError, KeyError):
            continue
        d = out.setdefault((cid, chrom), {})
        for token in idxs.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                d[int(token)] = rf
            except ValueError:
                continue
    return out


# ── compute ────────────────────────────────────────────────────────────


def compute(ctx, section=None) -> dict:
    """Return the data dict consumed by both the chart and the template."""
    path = ctx.bundle_dir / "cohort" / "cohort_global.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    if "sample_id" not in col or "cluster_id" not in col or "chrom" not in col:
        return {"present": False}
    n_meta = col["sample_id"] + 1
    flare_to_rf = _read_merged_groups_rf(ctx.bundle_dir)

    counts: dict[tuple[str, str], dict[str, int]] = {}
    totals: dict[tuple[str, str], int] = {}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
        except (IndexError, KeyError):
            continue
        if len(r) <= n_meta:
            continue
        try:
            vals = [float(x) for x in r[n_meta:]]
        except ValueError:
            continue
        if not vals:
            continue
        primary_idx = max(range(len(vals)), key=lambda i: vals[i])
        rf = flare_to_rf.get((cid, chrom), {}).get(primary_idx)
        if rf is None:
            continue
        d = counts.setdefault((cid, chrom), {})
        d[rf] = d.get(rf, 0) + 1
        totals[(cid, chrom)] = totals.get((cid, chrom), 0) + 1

    if not counts:
        return {"present": False}

    cohort_counts: dict[str, int] = {}
    for cc, d in counts.items():
        for rf, n in d.items():
            cohort_counts[rf] = cohort_counts.get(rf, 0) + n
    cohort_total = sum(cohort_counts.values()) or 1
    cohort_fracs = {rf: cohort_counts[rf] / cohort_total for rf in cohort_counts}

    cc_keys = sorted(counts.keys(), key=lambda k: -totals.get(k, 0))
    rf_set = [a for a in SP6.members if a in cohort_counts] + sorted(
        {rf for rf in cohort_counts if rf not in SP6.members}
    )

    # Callout: dominant cohort ancestry, second, and any cluster
    # whose composition diverges from the cohort by >2× on some ancestry.
    sorted_rf = sorted(cohort_fracs.items(), key=lambda kv: -kv[1])
    dom = sorted_rf[0] if sorted_rf else None
    snd = sorted_rf[1] if len(sorted_rf) > 1 else None
    divergent: list[dict] = []
    for cc in cc_keys:
        cc_total = totals[cc]
        if cc_total < 50:
            continue
        for rf, cf in cohort_fracs.items():
            if cf < 0.02:
                continue
            cluster_f = counts[cc].get(rf, 0) / cc_total
            if cluster_f >= 2.0 * cf or (cf > 0 and cluster_f <= 0.5 * cf):
                divergent.append({
                    "cluster": cc[0], "chrom": cc[1], "rf_label": rf,
                    "cluster_frac": cluster_f, "cohort_frac": cf,
                    "ratio": cluster_f / cf if cf > 0 else float("inf"),
                })

    return {
        "present": True,
        "counts": counts,
        "totals": totals,
        "cohort_counts": cohort_counts,
        "cohort_total": cohort_total,
        "cohort_fracs": cohort_fracs,
        "cc_keys": cc_keys,
        "rf_set": rf_set,
        "dom": dom,
        "snd": snd,
        "divergent": divergent,
    }


# ── render ────────────────────────────────────────────────────────────


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort-composition data", ha="center", va="center")
        ax.axis("off")
        return fig

    counts = data["counts"]
    totals = data["totals"]
    cohort_counts = data["cohort_counts"]
    cohort_total = data["cohort_total"]
    cc_keys = data["cc_keys"]
    rf_set = data["rf_set"]

    n_clusters = len(cc_keys)
    fig = plt.figure(figsize=(9.0, max(4.0, 1.4 + 0.45 * n_clusters + 1.6)))
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[1.2, max(1.0, 0.4 * n_clusters), 0.55],
        hspace=0.45,
    )
    ax_cohort = fig.add_subplot(gs[0, 0])
    ax_clusters = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.axis("off")

    def _color(rf: str) -> str:
        return palette.get(rf, "#888888")

    # ── Cohort bar ──────────────────────────────────────────────────────
    left = 0.0
    legend_handles = []
    for rf in rf_set:
        v = cohort_counts.get(rf, 0)
        if v <= 0:
            continue
        bar = ax_cohort.barh(0, v, left=left, color=_color(rf),
                             edgecolor="white", linewidth=0.6, height=0.7)
        legend_handles.append((rf, bar[0]))
        pct = 100.0 * v / cohort_total
        if pct >= 4.0:
            ax_cohort.text(left + v / 2, 0,
                           f"{rf}\n{int(v):,} ({pct:.1f}%)",
                           ha="center", va="center", fontsize=9,
                           color="white", fontweight="bold")
        left += v
    ax_cohort.set_xlim(0, cohort_total * 1.005)
    ax_cohort.set_ylim(-0.6, 0.6)
    ax_cohort.set_yticks([0])
    ax_cohort.set_yticklabels([f"cohort\n(n={cohort_total:,})"], fontsize=10,
                               fontweight="bold")
    ax_cohort.set_xticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax_cohort.spines[spine].set_visible(False)
    ax_cohort.set_title("Cohort sample count per FLARE primary ancestry",
                         fontsize=12, loc="left")

    # ── Per-cluster bars ───────────────────────────────────────────────
    cc_labels = [f"{cid}·{chrom}" for cid, chrom in cc_keys]
    y = np.arange(n_clusters)
    left_arr = np.zeros(n_clusters)
    for rf in rf_set:
        vals = np.array([counts[cc].get(rf, 0) for cc in cc_keys])
        if not vals.any():
            continue
        ax_clusters.barh(y, vals, left=left_arr, color=_color(rf),
                         edgecolor="white", linewidth=0.5, height=0.72)
        left_arr += vals
    ax_clusters.set_yticks(y)
    ax_clusters.set_yticklabels(cc_labels, fontsize=9)
    ax_clusters.invert_yaxis()
    total_max = max(totals.values()) if totals else 1
    for i, cc in enumerate(cc_keys):
        ax_clusters.text(left_arr[i] + total_max * 0.01, i,
                         f"  n={totals.get(cc, 0):,}",
                         ha="left", va="center", fontsize=8, color="#222")
    ax_clusters.set_xlim(0, total_max * 1.12)
    ax_clusters.set_xlabel("number of samples", fontsize=10)
    ax_clusters.set_title("Decomposition by cluster · chrom (supporting view)",
                          fontsize=10, loc="left", color="#444")
    for spine in ("top", "right"):
        ax_clusters.spines[spine].set_visible(False)

    # ── Legend strip ───────────────────────────────────────────────────
    ax_legend.legend(
        [h for _, h in legend_handles],
        [rf for rf, _ in legend_handles],
        title="ancestry", loc="center",
        ncol=min(len(legend_handles), 6),
        fontsize=9, title_fontsize=9, frameon=False,
    )

    return fig
