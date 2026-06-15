"""median |X-prop(chr_i) − X-prop(chr_j)| per sample, within X-dominant strata.

For each ancestry X, restrict to samples with chr1 X-proportion ≥
``threshold`` (default 0.95). For each chrom pair (i, j) compute
the median across the stratum of the within-sample absolute
difference between X-proportion on chr_i and chr_j. One 22×22
heatmap panel per ancestry; shared color scale.
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
    n_chroms = len(chroms)

    strata: list[dict] = []
    for ai, lab in enumerate(members):
        mask = cube[:, 0, ai] >= threshold
        n_dom = int(mask.sum())
        if n_dom < 50:
            strata.append({"label": lab, "n": n_dom, "mad": None})
            continue
        M = cube[mask, :, ai]
        mad = np.zeros((n_chroms, n_chroms), dtype=float)
        for j in range(n_chroms):
            mad[:, j] = np.median(np.abs(M - M[:, j:j+1]), axis=0)
        strata.append({"label": lab, "n": n_dom, "mad": mad})

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

    strata = data["strata"]
    chroms = data["chroms"]
    threshold = data["threshold"]
    short = [c.removeprefix("chr") for c in chroms]

    vmax_global = max(
        (float(np.nanmax(s["mad"])) for s in strata if s["mad"] is not None),
        default=0.1)

    fig, axes = plt.subplots(1, len(strata),
                             figsize=(4.0 * len(strata), 5.0))
    if len(strata) == 1:
        axes = [axes]
    im = None
    for ax, s in zip(axes, strata):
        color = palette.get(s["label"], "#888888")
        if s["mad"] is None:
            ax.text(0.5, 0.5, f"{s['label']}-dominant\nn={s['n']} (too few)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{s['label']}-dominant  n={s['n']:,}",
                         color=color, fontweight="bold", fontsize=13)
            continue
        im = ax.imshow(s["mad"], cmap="magma_r", vmin=0.0, vmax=vmax_global,
                       aspect="equal")
        ax.set_xticks(range(len(chroms)))
        ax.set_xticklabels(short, fontsize=7)
        ax.set_yticks(range(len(chroms)))
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(f"{s['label']}-dominant  n={s['n']:,}",
                     color=color, fontweight="bold", fontsize=13)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02,
                     label="median |X-prop(chr_i) − X-prop(chr_j)| per sample")
    fig.suptitle(
        f"median |X-prop(chr_i) − X-prop(chr_j)| per sample, within "
        f"X-dominant strata (chr1 X-prop ≥ {threshold:.2f}; diag = 0)",
        y=1.02, fontsize=14)
    return fig
