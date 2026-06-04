"""FLARE tract length per ancestry — cohort-pooled bar + per-cluster ticks
+ model expectation reference.

Data: ``cohort/tract_length_stats.tsv`` with columns (cluster_id,
chrom, ancestry_name, n_tracts, mean_Mb, implied_T_gen, model_T_gen).

**Reporting principle**: rendered faithfully from the bundle. The
``ancestry_name`` field is whatever the collector wrote, including any
postS-introduced subancestries — they appear as-is. See
``my_notes/validation/COLLECTOR_FIXES.md``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP6

from .._helpers import (
    n_weighted_mean,
    overlay_ticks,
    read_tsv,
    to_float,
    topn,
)


K_PANEL = 5  # FLARE's panel size for the model expectation formula.


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "tract_length_stats.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}

    by_anc: dict[str, list[tuple[str, int, float, float | None, float | None]]] = {}
    for r in rows:
        try:
            cc = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
            name = r[col["ancestry_name"]]
            n_tracts = int(float(r[col["n_tracts"]]))
            mean_mb = to_float(r[col["mean_Mb"]])
            implied_t = to_float(r[col["implied_T_gen"]])
            model_t = to_float(r[col["model_T_gen"]])
        except (IndexError, KeyError, ValueError):
            continue
        if mean_mb is None or mean_mb <= 0:
            continue
        by_anc.setdefault(name, []).append(
            (cc, n_tracts, mean_mb, implied_t, model_t)
        )
    if not by_anc:
        return {"present": False}

    labels = sorted(
        by_anc.keys(),
        key=lambda a: SP6.members.index(a.split(".")[0])
        if a.split(".")[0] in SP6.members else 99,
    )

    pooled_mb: dict[str, float | None] = {}
    pooled_n: dict[str, int] = {}
    model_ref_mb: dict[str, float] = {}
    for anc in labels:
        items = by_anc[anc]
        pooled_mb[anc] = n_weighted_mean([(n, mb) for _, n, mb, _, _ in items])
        pooled_n[anc] = sum(n for _, n, _, _, _ in items)
        models = [t for _, _, _, _, t in items if t is not None and t > 0]
        if models:
            med_t = sorted(models)[len(models) // 2]
            model_ref_mb[anc] = 100.0 / (med_t * K_PANEL)

    # Largest cohort-pooled deviation from the model expectation.
    devs: list[tuple[str, float]] = []
    for anc, v in pooled_mb.items():
        ref = model_ref_mb.get(anc)
        if v is None or ref is None:
            continue
        devs.append((
            f"`{anc}` (empirical {v:.2f} Mb vs model {ref:.2f} Mb)",
            abs(v - ref),
        ))
    top_dev = topn(devs, n=3)

    return {
        "present": True,
        "by_anc": by_anc,
        "labels": labels,
        "pooled_mb": pooled_mb,
        "pooled_n": pooled_n,
        "model_ref_mb": model_ref_mb,
        "top_dev": top_dev,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no tract-length data", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = data["labels"]
    by_anc = data["by_anc"]
    pooled_mb = data["pooled_mb"]
    model_ref_mb = data["model_ref_mb"]

    n_anc = len(labels)
    chart_h = max(2.6, 0.7 * n_anc + 0.6)
    fig = plt.figure(figsize=(9.0, chart_h + 0.8))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[chart_h, 0.5], hspace=0.3,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")
    ax.set_xscale("log")

    def _color(anc: str) -> str:
        return palette.get(anc.split(".")[0], "#888888")

    all_x = [v for _, _, mb, _, _ in
             (it for items in by_anc.values() for it in items)
             for v in [mb]]
    all_x += [v for v in pooled_mb.values() if v is not None]
    all_x += list(model_ref_mb.values())
    x_lo = max(0.1, min(all_x) * 0.7) if all_x else 0.1
    x_hi = (max(all_x) * 1.5) if all_x else 100.0

    for i, anc in enumerate(labels):
        v = pooled_mb.get(anc)
        if v is not None:
            ax.barh(i, v - x_lo, left=x_lo, color=_color(anc),
                    edgecolor="white", linewidth=0.6, height=0.62, zorder=2)
            ax.text(v * 1.04, i, f"{v:.2f} Mb",
                    ha="left", va="center", fontsize=9, color="#222")
        per_cluster = [mb for _, _, mb, _, _ in by_anc[anc]]
        overlay_ticks(ax, i, per_cluster, color="#111",
                      tick_height=0.48, lw=1.5, alpha=0.95)
        if anc in model_ref_mb:
            yref = model_ref_mb[anc]
            ax.vlines(yref, i - 0.34, i + 0.34, color="#666", linewidth=1.4,
                      linestyle="--", zorder=4)
    ax.set_yticks(range(n_anc))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("mean tract length (Mb, log scale)", fontsize=10)
    ax.set_title(
        "Tract length per ancestry  ·  bar = cohort-pooled (n_tracts-weighted)  ·  "
        "ticks = per-cluster empirical means  ·  dashed = model expectation",
        fontsize=11, loc="left",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    tick_proxy = plt.Line2D([0], [0], color="#111", linewidth=1.5)
    model_proxy = plt.Line2D([0], [0], color="#666", linewidth=1.4,
                              linestyle="--")
    ax_legend.legend(
        [bar_proxy, tick_proxy, model_proxy],
        ["cohort-pooled empirical mean (n_tracts-weighted)",
         "per-cluster empirical mean",
         f"model expectation = 100 / (median model_T_gen × K={K_PANEL})"],
        loc="center", ncol=3, fontsize=9, frameon=False,
    )
    return fig
