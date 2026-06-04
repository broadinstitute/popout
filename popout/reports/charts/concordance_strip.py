"""FLARE vs Rye concordance — per-ancestry Pearson r + Lin's CCC.

Two horizontal-bar panels (r on top, CCC below). Bar length is the
n-weighted cohort-pooled value; vertical ticks overlay each cluster's
individual value. The dashed reference lines are the 0.95 / 0.90
thresholds used in the pass criterion.

Data: ``cohort/concordance_metrics.tsv`` (per (cluster, chrom,
ancestry) with cluster_mu / n_samples / pearson_r / ccc). Rows with
``cluster_mu < 0.01`` are μ-gated and excluded from pooling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5, SP6

from .._helpers import (
    n_weighted_mean,
    overlay_ticks,
    read_tsv,
    to_float,
    topn,
)


PEARSON_REF = 0.95
CCC_REF = 0.90


# ── compute ────────────────────────────────────────────────────────────


def compute(ctx) -> dict:
    path = ctx.bundle_dir / "cohort" / "concordance_metrics.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    by_anc: dict[str, list[tuple[str, int, float, float, float]]] = {}
    for r in rows:
        try:
            cc = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
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
            cc, n, mu,
            pr if pr is not None else float("nan"),
            cccv if cccv is not None else float("nan"),
        ))
    if not by_anc:
        return {"present": False, "all_gated": True}

    # SP5 order first, then any extra labels alphabetically.
    sp5 = list(SP5.members)
    labels = [a for a in sp5 if a in by_anc] + sorted(
        a for a in by_anc if a not in sp5
    )

    pooled_r: dict[str, float | None] = {}
    pooled_ccc: dict[str, float | None] = {}
    for anc in labels:
        items = by_anc[anc]
        pooled_r[anc] = n_weighted_mean([(n, pr) for _, n, _, pr, _ in items])
        pooled_ccc[anc] = n_weighted_mean([(n, cc) for _, n, _, _, cc in items])

    # Calibration-drift candidates (per (cluster, ancestry) r − CCC).
    gaps: list[tuple[str, float]] = []
    for anc, items in by_anc.items():
        for cc, _n, _mu, pr, cccv in items:
            if pr == pr and cccv == cccv:
                gaps.append((f"{cc}·{anc}", pr - cccv))
    top_gap = topn(gaps, n=3)

    # Per-(ancestry, cluster) rows for the supporting longtable.
    table_rows: list[dict] = []
    for anc in labels:
        for cc, n, mu, pr, cccv in sorted(by_anc[anc], key=lambda t: t[0]):
            table_rows.append({
                "ancestry": anc, "cluster_chrom": cc,
                "cluster_mu": mu, "n_samples": n,
                "pearson_r": (None if pr != pr else pr),
                "ccc": (None if cccv != cccv else cccv),
            })

    return {
        "present": True,
        "by_anc": by_anc,
        "labels": labels,
        "pooled_r": pooled_r,
        "pooled_ccc": pooled_ccc,
        "top_gap": top_gap,
        "table_rows": table_rows,
        "pearson_ref": PEARSON_REF,
        "ccc_ref": CCC_REF,
    }


# ── render ────────────────────────────────────────────────────────────


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
    n_anc = len(labels)

    cluster_r = {anc: [pr for _, _, _, pr, _ in by_anc[anc] if pr == pr]
                 for anc in labels}
    cluster_ccc = {anc: [cc for _, _, _, _, cc in by_anc[anc] if cc == cc]
                   for anc in labels}

    all_r = [v for vals in cluster_r.values() for v in vals] + [
        v for v in pooled_r.values() if v is not None
    ]
    all_c = [v for vals in cluster_ccc.values() for v in vals] + [
        v for v in pooled_ccc.values() if v is not None
    ]
    x_lo = max(0.0, min([*all_r, *all_c, 0.90]) - 0.04)
    x_hi = 1.01

    chart_h = max(2.0, 0.55 * n_anc + 0.6)
    fig = plt.figure(figsize=(8.5, 2 * chart_h + 1.4))
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[chart_h, chart_h, 0.5],
        hspace=0.35,
    )
    ax_r = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.axis("off")

    def _color(rf: str) -> str:
        return palette.get(rf, "#888888")

    def _draw(ax, pooled, per_cluster, ref, title, xlabel):
        y = list(range(n_anc))
        for i, anc in enumerate(labels):
            v = pooled.get(anc)
            if v is not None:
                ax.barh(i, v, color=_color(anc),
                        edgecolor="white", linewidth=0.6,
                        height=0.62, zorder=2)
                ax.text(v + (x_hi - x_lo) * 0.005, i,
                        f"{v:.3f}", ha="left", va="center",
                        fontsize=9, color="#222")
            overlay_ticks(ax, i, per_cluster.get(anc, []),
                          color="#111", tick_height=0.46, lw=1.6, alpha=0.9)
        ax.axvline(ref, color="#666", linestyle="--", linewidth=0.9, zorder=1)
        ax.text(ref, n_anc - 0.4, f" pass threshold: {ref:.2f}",
                fontsize=8, color="#555", ha="left", va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11, loc="left")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    _draw(ax_r, pooled_r, cluster_r, PEARSON_REF,
          "Pearson r — rank linearity (cohort-pooled, n-weighted)",
          "Pearson r")
    _draw(ax_c, pooled_ccc, cluster_ccc, CCC_REF,
          "Lin's CCC — linearity + calibration (cohort-pooled, n-weighted)",
          "Lin's CCC")

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    tick_proxy = plt.Line2D([0], [0], color="#111", linewidth=1.6)
    ax_legend.legend(
        [bar_proxy, tick_proxy],
        ["bar length = cohort-pooled value (n-weighted across clusters)",
         "vertical tick = one cluster's value"],
        loc="center", ncol=2, fontsize=9, frameon=False,
    )
    return fig
