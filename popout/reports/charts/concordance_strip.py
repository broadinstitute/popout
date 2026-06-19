"""FLARE vs Rye concordance — per-ancestry Pearson r + Lin's CCC.

Two stacked **raincloud + sina-rain** panels (r on top, CCC below).
Each ancestry row is split into three strict, non-overlapping lanes:

    violin band  = [i-0.46, i-0.14]   half-violin (KDE) above the bar
    bar          = [i-0.08, i+0.08]   cohort-pooled n-weighted value
    rain band    = [i+0.14, i+0.46]   sina raindrops (jitter ∝ density)

Each raindrop is one ``(cluster_id, chrom)`` value, coloured by the row
ancestry. The dashed reference lines are the 0.95 / 0.90 pass
thresholds. Lanes are partitioned so nothing spills into a neighbouring
ancestry row.

Layout adapted from ``my_notes/graphs/scale_proto_K2.py`` — the chosen
audition.

Data: ``cohort/concordance_metrics.tsv`` (per ``(cluster, chrom,
ancestry)`` with ``cluster_mu / n_samples / pearson_r / ccc``). Rows
with ``cluster_mu < 0.01`` are μ-gated and excluded.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP5

from .._helpers import (
    n_weighted_mean,
    raincloud_panel,
    read_tsv,
    to_float,
    topn,
)


PEARSON_REF = 0.95
CCC_REF = 0.90


# ── compute ────────────────────────────────────────────────────────────


def compute(ctx, section=None) -> dict:
    opts = section.options if section is not None else {}
    # Cohort filename of the per-(cluster, chrom, ancestry) concordance
    # table. Set per-section in the YAML: ``options.source: rye`` reads
    # ``concordance_metrics_rye.tsv``; ``options.source: rf`` reads the
    # FLARE-vs-RF table written by compare_to_rf.py.
    source = opts.get("source", "rye")
    if source not in ("rye", "rf"):
        raise ValueError(
            f"concordance_strip options.source must be 'rye' or 'rf'; "
            f"got {source!r}"
        )
    filename = f"concordance_metrics_{source}.tsv"
    path = ctx.bundle_dir / "cohort" / filename
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False, "source": source}
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
        return {"present": False, "all_gated": True, "source": source}

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

    gaps: list[tuple[str, float]] = []
    for anc, items in by_anc.items():
        for cc, _n, _mu, pr, cccv in items:
            if pr == pr and cccv == cccv:
                gaps.append((f"{cc}·{anc}", pr - cccv))
    top_gap = topn(gaps, n=3)

    return {
        "present": True,
        "source": source,
        "by_anc": by_anc,
        "labels": labels,
        "pooled_r": pooled_r,
        "pooled_ccc": pooled_ccc,
        "top_gap": top_gap,
        "pearson_ref": PEARSON_REF,
        "ccc_ref": CCC_REF,
    }


# ── render: raincloud + sina rain (K2 layout) ─────────────────────────


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

    per_row_r = {anc: [pr for _, _, _, pr, _ in by_anc[anc] if pr == pr]
                 for anc in labels}
    per_row_ccc = {anc: [cc for _, _, _, _, cc in by_anc[anc] if cc == cc]
                   for anc in labels}

    all_r = [v for vals in per_row_r.values() for v in vals] + [
        v for v in pooled_r.values() if v is not None
    ]
    all_c = [v for vals in per_row_ccc.values() for v in vals] + [
        v for v in pooled_ccc.values() if v is not None
    ]
    x_lo = max(0.0, min([*all_r, *all_c, 0.90]) - 0.05)
    x_hi = 1.02

    chart_h = max(2.6, 0.95 * n_anc + 0.7)
    fig = plt.figure(figsize=(10.0, 2 * chart_h + 0.9))
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[chart_h, chart_h, 0.45],
        hspace=0.35,
    )
    ax_r = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.axis("off")

    other = {"rye": "Rye", "rf": "RF"}.get(data.get("source", "rye"), "other tool")
    raincloud_panel(
        ax_r, labels, pooled_r, per_row_r,
        palette=palette, x_lo=x_lo, x_hi=x_hi,
        title=f"Pearson r vs {other} - rank linearity  ·  raincloud + sina rain",
        xlabel="Pearson r",
        threshold=PEARSON_REF,
        threshold_label=f"pass threshold: {PEARSON_REF:.2f}",
    )
    raincloud_panel(
        ax_c, labels, pooled_ccc, per_row_ccc,
        palette=palette, x_lo=x_lo, x_hi=x_hi,
        title=f"Lin's CCC vs {other} - linearity + calibration  ·  raincloud + sina rain",
        xlabel="Lin's CCC",
        threshold=CCC_REF,
        threshold_label=f"pass threshold: {CCC_REF:.2f}",
    )

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    violin_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888", alpha=0.42)
    rain_proxy = plt.scatter([], [], s=20, color="#888",
                             edgecolor="white", linewidth=0.4)
    ax_legend.legend(
        [bar_proxy, violin_proxy, rain_proxy],
        ["bar = cohort-pooled (n-weighted)",
         "half-violin = KDE of per-(cluster, chrom) values",
         "raindrop = one (cluster, chrom) value (sina jitter)"],
        loc="center", ncol=3, fontsize=9, frameon=False,
    )
    return fig
