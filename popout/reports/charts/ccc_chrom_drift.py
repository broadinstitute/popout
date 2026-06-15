"""Per-chrom median X-proportion within X-dominant strata.

For each ancestry X, restrict to samples with chr1 X-proportion ≥
``threshold`` (default 0.95). Plot one colored line per stratum,
each line = per-chrom median X-proportion across the stratum's
samples. Dashed colored reference line marks each stratum's
across-chrom median of medians.
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
    chroms = list(cube_data["chroms"])

    strata: list[dict] = []
    for ai, lab in enumerate(members):
        mask = cube[:, 0, ai] >= threshold
        n_dom = int(mask.sum())
        if n_dom < 20:
            strata.append({"label": lab, "n": n_dom, "per_chrom_median": None,
                           "across_chrom_median": None,
                           "chr1_median": None, "others_mean": None,
                           "drift": None})
            continue
        per_med = [float(np.median(cube[mask, c, ai]))
                   for c in range(len(chroms))]
        chr1_med = per_med[0]
        others_mean = float(np.mean(per_med[1:])) if len(per_med) > 1 else None
        strata.append({
            "label": lab, "n": n_dom,
            "per_chrom_median": per_med,
            "across_chrom_median": float(np.nanmedian(per_med)),
            "chr1_median": chr1_med,
            "others_mean": others_mean,
            "drift": (chr1_med - others_mean) if others_mean is not None else None,
        })

    return {
        "present": True,
        "threshold": threshold,
        "chroms": chroms,
        "strata": strata,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort_global.tsv data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    chroms = data["chroms"]
    strata = [s for s in data["strata"] if s["per_chrom_median"] is not None]
    threshold = data["threshold"]

    fig, ax = plt.subplots(figsize=(14, 8.5))
    x = np.arange(len(chroms))
    levels: list[float] = []
    for s in strata:
        color = palette.get(s["label"], "#888888")
        ax.plot(x, s["per_chrom_median"], color=color, linewidth=2.6,
                marker="o", markersize=8.5, markeredgecolor="white",
                markeredgewidth=0.8,
                label=f"{s['label']}-dominant   (n={s['n']:,})", zorder=4)
        ax.axhline(s["across_chrom_median"], color=color, linestyle="--",
                   linewidth=0.9, alpha=0.55, zorder=1)
        levels.append(s["across_chrom_median"])

    ax.set_xticks(x)
    ax.set_xticklabels(chroms, rotation=45)
    ax.set_xlabel("chromosome")
    ax.set_ylabel("self-ancestry proportion in stratum")
    ax.set_title(
        "Per-chromosome self-ancestry proportion within X-dominant strata\n"
        f"(samples with chr1 X-proportion ≥ {threshold:.2f}; "
        "line = per-chrom median of the SAME ancestry's proportion)",
        pad=12, fontsize=13)
    if levels:
        lo = min(levels) - 0.04
        hi = max(s["per_chrom_median"][0] for s in strata) + 0.02
        ax.set_ylim(lo, hi)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
