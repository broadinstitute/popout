"""FLARE component × RF-assigned label calibration heatmap.

For every (FLARE component, RF label) pair, two numbers:

- ``max_cal``: largest mean FLARE proportion across binned RF
  probability — the strongest response of the component to the RF
  tool's signal for that label.
- ``slope`` (m): linear-regression slope of FLARE proportion on RF
  probability across samples (only defined where both axes have
  enough variance).

A well-behaved FLARE component answers exactly one RF label on the
diagonal with m ≈ 1. Off-diagonal cells with a bright color are
component-to-label leakage candidates. Cells with ``max_cal > 1``
(hatched) usually mean the reference panel under-represents that
ancestry.

Data: ``cohort/calibration_slope.tsv`` plus ``cohort/concordance_metrics_rf.tsv``
(for the per-(cluster, ancestry) FLARE-side cluster_mu used to weight pooling).

Reporting principle: rendered faithfully. Label names propagate from
the bundle without canonicalisation.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP6

from .._helpers import (
    n_weighted_mean,
    read_tsv,
    to_float,
    topn,
)


def _cluster_mu_map(bundle_dir) -> dict[tuple[str, str, str], float]:
    """(cluster_id, chrom, ancestry) -> cluster_mu, from
    concordance_metrics_rf.tsv. cluster_mu is FLARE-side (mean of the
    FLARE column for that ancestry on that cluster x chrom); it is
    tool-independent so reading from the RF table or the Rye table is
    equivalent. v6 prefers RF because it is a required cohort artifact
    while Rye is gated on rye_q."""
    path = bundle_dir / "cohort" / "concordance_metrics_rf.tsv"
    out: dict[tuple[str, str, str], float] = {}
    if not path.exists():
        return out
    header, rows = read_tsv(path)
    if not rows:
        return out
    col = {h: i for i, h in enumerate(header)}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            anc = r[col["ancestry"]]
            mu = to_float(r[col["cluster_mu"]])
        except (IndexError, KeyError):
            continue
        if mu is not None:
            out[(cid, chrom, anc)] = mu
    return out


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "calibration_slope.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}

    mu_map = _cluster_mu_map(ctx.bundle_dir)

    cells: dict[tuple[str, str],
                 list[tuple[float, float | None, float | None]]] = {}
    anc_set: list[str] = []
    rf_order: list[str] = []
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            anc = r[col["ancestry_name"]]
            rf = r[col["rf_label"]]
            slope = to_float(r[col["slope"]])
            max_cal = to_float(r[col["max_cal"]])
        except (IndexError, KeyError):
            continue
        mu = mu_map.get((cid, chrom, anc.split(".")[0]), 1.0)
        if mu is None:
            mu = 1.0
        cells.setdefault((anc, rf), []).append((mu, slope, max_cal))
        if anc not in anc_set:
            anc_set.append(anc)
        if rf not in rf_order:
            rf_order.append(rf)
    if not cells:
        return {"present": False}

    def _anc_key(a: str) -> tuple[int, str]:
        base = a.split(".")[0]
        primary = SP6.members.index(base) if base in SP6.members else 99
        return (primary, a)

    anc_rows = sorted(anc_set, key=_anc_key)
    rf_order = [r for r in SP6.members if r in rf_order] + \
               sorted(r for r in rf_order if r not in SP6.members)

    pool_max: dict[tuple[str, str], float | None] = {}
    pool_slope: dict[tuple[str, str], float | None] = {}
    for key, recs in cells.items():
        pool_max[key] = n_weighted_mean(
            [(w, m) for w, _, m in recs if m is not None]
        )
        slope_vals = sorted(s for _, s, _ in recs if s is not None)
        pool_slope[key] = (slope_vals[len(slope_vals) // 2]
                            if slope_vals else None)

    mu_by_anc: dict[str, float | None] = {}
    for anc in anc_rows:
        base = anc.split(".")[0]
        per_cluster_mu = [mu_map[k] for k in mu_map
                          if k[2] == base and mu_map[k] is not None]
        mu_by_anc[anc] = (sum(per_cluster_mu) / len(per_cluster_mu)
                          if per_cluster_mu else None)

    over1_cells: list[tuple[int, int]] = []
    for i, anc in enumerate(anc_rows):
        for j, rf in enumerate(rf_order):
            v = pool_max.get((anc, rf))
            if v is not None and v > 1.0:
                over1_cells.append((i, j))

    off_diag: list[tuple[str, float]] = []
    for (anc, rf), v in pool_max.items():
        if v is None:
            continue
        base = anc.split(".")[0]
        if base == rf:
            continue
        off_diag.append((f"FLARE=`{anc}` -> RF=`{rf}`", v))
    top_off = topn(off_diag, n=3)
    n_over1 = len(over1_cells)

    return {
        "present": True,
        "anc_rows": anc_rows,
        "rf_order": rf_order,
        "pool_max": pool_max,
        "pool_slope": pool_slope,
        "mu_by_anc": mu_by_anc,
        "over1_cells": over1_cells,
        "top_off": top_off,
        "n_over1": n_over1,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no calibration data", ha="center", va="center")
        ax.axis("off")
        return fig

    anc_rows = data["anc_rows"]
    rf_order = data["rf_order"]
    pool_max = data["pool_max"]
    pool_slope = data["pool_slope"]
    mu_by_anc = data["mu_by_anc"]
    over1_cells = data["over1_cells"]

    n_anc = len(anc_rows)
    n_rf = len(rf_order)

    M = np.full((n_anc, n_rf), np.nan)
    for i, anc in enumerate(anc_rows):
        for j, rf in enumerate(rf_order):
            v = pool_max.get((anc, rf))
            if v is None:
                continue
            M[i, j] = min(v, 1.0)

    fig = plt.figure(
        figsize=(2.8 + 0.85 * n_rf + 1.0,
                 max(3.5, 0.55 * n_anc + 1.5)),
    )
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.2, max(n_rf * 0.85, 3.0), 0.5],
        wspace=0.05,
    )
    ax_mu = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    def _color(anc: str) -> str:
        return palette.get(anc.split(".")[0], "#888888")

    # μ strip
    bar_max_w = 1.0
    for i, anc in enumerate(anc_rows):
        mu = mu_by_anc.get(anc)
        if mu is None:
            ax_mu.text(0.5, i, "μ=—", ha="center", va="center",
                       fontsize=8, color="#666")
            continue
        w = max(0.02, bar_max_w * min(mu, 1.0))
        ax_mu.add_patch(mpatches.Rectangle(
            (0.0, i - 0.32), w, 0.64,
            facecolor=_color(anc), edgecolor="#444444", linewidth=0.4,
        ))
        ax_mu.text(bar_max_w + 0.06, i, f"μ={mu:.2f}",
                    ha="left", va="center", fontsize=8, color="#222")
    ax_mu.set_xlim(0, bar_max_w + 0.7)
    ax_mu.set_ylim(n_anc - 0.5, -0.5)
    ax_mu.set_yticks(range(n_anc))
    ax_mu.set_yticklabels(anc_rows, fontsize=9)
    ax_mu.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax_mu.spines[spine].set_visible(False)
    ax_mu.set_title("cohort μ", fontsize=9, loc="left", color="#444")

    # Heatmap
    im = ax_heat.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax_heat.set_xticks(range(n_rf))
    ax_heat.set_xticklabels(rf_order, fontsize=10)
    ax_heat.set_xlabel("RF-assigned superpop label", fontsize=10)
    ax_heat.set_yticks(range(n_anc))
    ax_heat.set_yticklabels([])
    ax_heat.set_title(
        "Cohort-pooled calibration  ·  color = μ-weighted mean max_cal  ·  "
        "text = max_cal (m = median slope)",
        fontsize=11, loc="left",
    )
    for i in range(n_anc):
        for j in range(n_rf):
            v = pool_max.get((anc_rows[i], rf_order[j]))
            if v is None:
                continue
            s = pool_slope.get((anc_rows[i], rf_order[j]))
            txt_color = "white" if min(v, 1.0) < 0.55 else "black"
            txt = f"{v:.2f}\nm={s:.2f}" if s is not None else f"{v:.2f}"
            ax_heat.text(j, i, txt, ha="center", va="center",
                          fontsize=8, color=txt_color, linespacing=0.95)
    for (i, j) in over1_cells:
        ax_heat.add_patch(mpatches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1, fill=False,
            hatch="///", edgecolor="#222", linewidth=0.6,
        ))

    fig.colorbar(im, cax=ax_cbar, label="max_cal (0–1)")
    return fig
