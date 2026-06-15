"""Per-sample top-1 vs top-2 FLARE proportion, panelled by top-1.

For each chr1 sample compute its top-1 and top-2 ancestry
proportions. Plot one hexbin panel per top-1 ancestry: x =
top-1 proportion, y = top-2 proportion. Threshold reference lines
mark the dominant / leaning / admix cutoffs and the sum-to-1
ceiling for the remaining ancestries.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5

from .._helpers import load_cohort_cube


def compute(ctx, section=None) -> dict:
    opts = section.options if section is not None else {}
    thr_dom = float(opts.get("threshold_dominant", 0.95))
    thr_lean = float(opts.get("threshold_leaning", 0.85))
    thr_admix = float(opts.get("threshold_admix", 0.10))
    mid_rule = (section.mid_rule if section is not None else None) or "drop"

    cube_data = load_cohort_cube(
        ctx.bundle_dir, label_space=SP5, mid_rule=mid_rule)
    if not cube_data:
        return {"present": False}

    cube = cube_data["cube"]
    members = list(cube_data["label_space"].members)
    chr1 = cube[:, 0, :]
    order = np.argsort(chr1, axis=1)[:, ::-1]
    top1_val = np.take_along_axis(chr1, order[:, :1], axis=1).flatten()
    top2_val = np.take_along_axis(chr1, order[:, 1:2], axis=1).flatten()
    top1_idx = order[:, 0]

    return {
        "present": True,
        "members": members,
        "top1_val": top1_val,
        "top2_val": top2_val,
        "top1_idx": top1_idx,
        "n_samples": int(chr1.shape[0]),
        "threshold_dominant": thr_dom,
        "threshold_leaning": thr_lean,
        "threshold_admix": thr_admix,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort_global.tsv data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    members = data["members"]
    top1_val = data["top1_val"]
    top2_val = data["top2_val"]
    top1_idx = data["top1_idx"]
    n_samples = data["n_samples"]
    thr_dom = data["threshold_dominant"]
    thr_lean = data["threshold_leaning"]
    thr_admix = data["threshold_admix"]

    fig, axes = plt.subplots(1, len(members),
                             figsize=(3.6 * len(members), 5.2),
                             sharey=True, sharex=True)
    if len(members) == 1:
        axes = [axes]
    for ax, ai in zip(axes, range(len(members))):
        color = palette.get(members[ai], "#888")
        mask = top1_idx == ai
        n_a = int(mask.sum())
        if n_a == 0:
            ax.text(0.5, 0.5, "n=0", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.hexbin(top1_val[mask], top2_val[mask], gridsize=60, cmap="Greys",
                  bins="log", mincnt=1, extent=(0.2, 1.0, 0.0, 0.5))
        ax.axvline(thr_dom, color=color, linestyle="--", linewidth=1.4)
        ax.axvline(thr_lean, color=color, linestyle=":", linewidth=1.2)
        ax.axhline(thr_admix, color="#666", linestyle=":", linewidth=1.0)
        xs_b = np.linspace(0.5, 1.0, 50)
        ax.plot(xs_b, 1 - xs_b, color="#222", linewidth=0.7)
        ax.set_title(f"top-1 = {members[ai]}    n = {n_a:,}",
                     color=color, fontweight="bold", fontsize=13, pad=6)
        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(0.0, 0.50)
        ax.set_xlabel("top-1 proportion", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("top-2 proportion", fontsize=11)
    fig.suptitle(
        f"Per-sample top-1 vs top-2 FLARE proportion, panelled by top-1 "
        f"(N = {n_samples:,} chr1 samples)",
        y=1.00, fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig
