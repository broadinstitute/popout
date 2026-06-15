"""std(X-prop across chr1..chr22) per sample, within X-dominant strata.

For each ancestry X, restrict to samples with chr1 X-proportion ≥
``threshold`` (default 0.95). For each such sample compute the
standard deviation of X-proportion across the 22 chroms. One
histogram panel per ancestry; log y-axis. Median and p95 of the
distribution are marked.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5

from .._helpers import load_cohort_cube


def compute(ctx, section=None) -> dict:
    section_opts = section.options if section is not None else {}
    threshold = float(section_opts.get("threshold", 0.95))
    mid_rule = (section.mid_rule if section is not None else None) or "drop"

    cube_data = load_cohort_cube(
        ctx.bundle_dir, label_space=SP5, mid_rule=mid_rule)
    if not cube_data:
        return {"present": False}

    cube = cube_data["cube"]
    members = list(cube_data["label_space"].members)
    chrom_std = cube.std(axis=1, ddof=0)

    strata: list[dict] = []
    for ai, lab in enumerate(members):
        mask = cube[:, 0, ai] >= threshold
        n_dom = int(mask.sum())
        if n_dom < 20:
            strata.append({"label": lab, "n": n_dom,
                           "values": None, "median": None, "p95": None})
            continue
        vals = chrom_std[mask, ai]
        strata.append({
            "label": lab, "n": n_dom,
            "values": vals,
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
        })

    return {"present": True, "threshold": threshold, "strata": strata}


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort_global.tsv data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    strata = data["strata"]
    threshold = data["threshold"]

    fig, axes = plt.subplots(1, len(strata),
                             figsize=(3.6 * len(strata), 4.6),
                             sharey=True)
    if len(strata) == 1:
        axes = [axes]
    bin_edges = np.linspace(0, 0.20, 81)
    for ax, s in zip(axes, strata):
        color = palette.get(s["label"], "#888888")
        if s["values"] is None:
            ax.text(0.5, 0.5, f"{s['label']}-dominant\nn={s['n']} (too few)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{s['label']}-dominant   n={s['n']:,}",
                         color=color, fontweight="bold", fontsize=13)
            continue
        ax.hist(s["values"], bins=bin_edges, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axvline(s["median"], color="#222", linestyle="--", linewidth=1.0)
        ax.axvline(s["p95"], color="#666", linestyle=":", linewidth=1.0)
        ax.text(s["median"], ax.get_ylim()[1] * 0.92,
                f" median {s['median']:.3f}", color="#222", fontsize=9,
                ha="left", va="top")
        ax.text(s["p95"], ax.get_ylim()[1] * 0.78,
                f" p95 {s['p95']:.3f}", color="#666", fontsize=9,
                ha="left", va="top")
        ax.set_yscale("log")
        ax.set_title(f"{s['label']}-dominant   n={s['n']:,}",
                     color=color, fontweight="bold", fontsize=13)
        ax.set_xlabel("std(X-prop across chr1..chr22)")
        ax.set_xlim(0, 0.20)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("samples (log)")
    fig.suptitle(
        f"std(X-prop across chr1..chr22) per sample, within X-dominant strata "
        f"(chr1 X-prop ≥ {threshold:.2f})",
        y=1.02, fontsize=14)
    fig.tight_layout()
    return fig
