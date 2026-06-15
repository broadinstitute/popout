"""Three CCC metrics vs chr1 X-dominant threshold T.

Sweeps the chr1 X-proportion threshold T across a configurable set
(default {0.50, 0.70, 0.85, 0.95}). For each (T, ancestry) computes:

- ``drift``      = chr1 median − mean(chr2..22 medians)
- ``stdev_med``  = median per-sample std(X-prop across chr1..chr22)
- ``mad_mean``   = mean off-diagonal per-chrom-pair median absolute
                    difference per sample

Three side-by-side line plots, one panel per metric, one line per
ancestry. n-annotations at endpoints.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5

from .._helpers import load_cohort_cube


def compute(ctx, section=None) -> dict:
    section_opts = section.options if section is not None else {}
    thresholds = tuple(section_opts.get(
        "thresholds", (0.50, 0.70, 0.85, 0.95)))
    mid_rule = (section.mid_rule if section is not None else None) or "drop"

    cube_data = load_cohort_cube(
        ctx.bundle_dir, label_space=SP5, mid_rule=mid_rule)
    if not cube_data:
        return {"present": False}

    cube = cube_data["cube"]
    members = list(cube_data["label_space"].members)
    n_chroms = len(cube_data["chroms"])
    chrom_std = cube.std(axis=1, ddof=0)

    series: dict[str, dict] = {lab: {"T": [], "n": [], "drift": [],
                                      "stdev_med": [], "mad_mean": []}
                                for lab in members}
    for T in thresholds:
        for ai, lab in enumerate(members):
            mask = cube[:, 0, ai] >= T
            n_dom = int(mask.sum())
            series[lab]["T"].append(float(T))
            series[lab]["n"].append(n_dom)
            if n_dom < 20:
                for k in ("drift", "stdev_med", "mad_mean"):
                    series[lab][k].append(float("nan"))
                continue
            chr1_med = float(np.median(cube[mask, 0, ai]))
            per_med = [float(np.median(cube[mask, c, ai]))
                       for c in range(n_chroms)]
            drift = chr1_med - float(np.mean(per_med[1:]))
            stdev_med = float(np.median(chrom_std[mask, ai]))
            M = cube[mask, :, ai]
            mad = np.zeros((n_chroms, n_chroms))
            for j in range(n_chroms):
                mad[:, j] = np.median(np.abs(M - M[:, j:j+1]), axis=0)
            off = mad.copy(); np.fill_diagonal(off, np.nan)
            mad_mean = float(np.nanmean(off))
            series[lab]["drift"].append(drift)
            series[lab]["stdev_med"].append(stdev_med)
            series[lab]["mad_mean"].append(mad_mean)

    return {
        "present": True,
        "thresholds": list(thresholds),
        "series": series,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort_global.tsv data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    thresholds = data["thresholds"]
    series = data["series"]
    metrics = [
        ("drift",     "chr1 median  −  mean(chr2..22 medians)"),
        ("stdev_med", "median per-sample std(X-prop across chr1..chr22)"),
        ("mad_mean",  "mean off-diag median |X-prop(chr_i) − X-prop(chr_j)|"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (key, ylab) in zip(axes, metrics):
        ax.axhline(0, color="#aaa", linewidth=0.8, zorder=1)
        for lab, s in series.items():
            color = palette.get(lab, "#888888")
            Ts = s["T"]; ys = s[key]; ns = s["n"]
            ax.plot(Ts, ys, color=color, linewidth=2.6, marker="o",
                    markersize=10, markeredgecolor="white",
                    markeredgewidth=0.6, label=f"{lab}-dominant", zorder=3)
            for t, y, n in [(Ts[0], ys[0], ns[0]),
                            (Ts[-1], ys[-1], ns[-1])]:
                if y != y:
                    continue
                ha = "right" if t == Ts[0] else "left"
                dx = -6 if t == Ts[0] else 6
                ax.annotate(f"n={n:,}", (t, y),
                            textcoords="offset points", xytext=(dx, 6),
                            fontsize=8.5, color=color, ha=ha)
        ax.set_xlabel("chr1 X-dominant threshold  T")
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_xticks(thresholds)
        if thresholds:
            ax.set_xlim(min(thresholds) - 0.05, max(thresholds) + 0.05)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [plt.Line2D([0], [0], color=palette.get(lab, "#888"),
                          linewidth=2.6, marker="o", markersize=8,
                          label=f"{lab}-dominant") for lab in series]
    fig.legend(handles=handles, loc="lower center", frameon=False,
               ncol=len(series), bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.suptitle(
        "Three CCC metrics vs chr1 X-dominant threshold T",
        y=1.00, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig
