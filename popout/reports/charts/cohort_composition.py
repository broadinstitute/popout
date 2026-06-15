"""Cohort composition — FLARE top-1 ancestry, cohort + per-stratum summary.

The figure is a single wide horizontal bar: the cohort decomposed by each
sample's FLARE top-1 ancestry on chr1 (argmax over its proportion vector).
The data dict also carries a per-stratum summary that the template renders
as a small markdown table: sample count, percentage of cohort, and the
median across-chrom stability (stdev of the dominant-ancestry proportion
on chr2..22). The previous per-(cluster, chrom) supporting panel was
dropped: cluster is not the right stratification axis.

Data: ``cohort/cohort_global.tsv`` + ``cohort/merged_groups_rf.tsv`` via
``load_cohort_cube``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5

from .._helpers import load_cohort_cube, top1_strata


def compute(ctx, section=None) -> dict:
    mid_rule = (section.mid_rule if section is not None else None) or "drop"
    cube_data = load_cohort_cube(
        ctx.bundle_dir, label_space=SP5, mid_rule=mid_rule)
    if not cube_data:
        return {"present": False}

    cube = cube_data["cube"]
    members = list(cube_data["label_space"].members)
    n_samples = int(cube.shape[0])
    strata_masks = top1_strata(cube, members, chrom_idx=0)

    strata_summary: list[dict] = []
    cohort_counts: dict[str, int] = {}
    for ai, lab in enumerate(members):
        mask = strata_masks[lab]
        n = int(mask.sum())
        cohort_counts[lab] = n
        if n == 0:
            strata_summary.append({
                "label": lab, "n": 0, "frac": 0.0,
                "chrom_stdev_med": None,
            })
            continue
        # Stability proxy: per-sample stdev of dominant-ancestry proportion
        # across chr2..22; report the stratum's median.
        per_sample_std = np.std(cube[mask, 1:, ai], axis=1)
        strata_summary.append({
            "label": lab, "n": n, "frac": n / n_samples,
            "chrom_stdev_med": float(np.median(per_sample_std)),
        })

    cohort_fracs = {lab: c / n_samples for lab, c in cohort_counts.items()}
    sorted_fracs = sorted(cohort_fracs.items(), key=lambda kv: -kv[1])
    dom = sorted_fracs[0] if sorted_fracs else None
    snd = sorted_fracs[1] if len(sorted_fracs) > 1 else None

    return {
        "present": True,
        "n_samples": n_samples,
        "members": members,
        "cohort_counts": cohort_counts,
        "cohort_fracs": cohort_fracs,
        "strata_summary": strata_summary,
        "dom": dom,
        "snd": snd,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort-composition data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    members = data["members"]
    cohort_counts = data["cohort_counts"]
    n_samples = data["n_samples"]

    fig, ax = plt.subplots(figsize=(11.0, 2.2))
    left = 0.0
    handles = []
    for lab in members:
        v = cohort_counts.get(lab, 0)
        if v <= 0:
            continue
        color = palette.get(lab, "#888")
        bar = ax.barh(0, v, left=left, color=color,
                      edgecolor="white", linewidth=0.7, height=0.7)
        handles.append((lab, bar[0]))
        pct = 100.0 * v / n_samples
        if pct >= 3.0:
            ax.text(left + v / 2, 0,
                    f"{lab}\n{int(v):,} ({pct:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold")
        left += v

    ax.set_xlim(0, n_samples * 1.005)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([0])
    ax.set_yticklabels([f"cohort\n(n={n_samples:,})"],
                       fontsize=10, fontweight="bold")
    ax.set_xticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(
        "Cohort sample count per FLARE top-1 ancestry  (chr1 argmax)",
        fontsize=12, loc="left",
    )

    ax.legend(
        [h for _, h in handles], [lab for lab, _ in handles],
        loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=min(len(handles), 6), fontsize=9, frameon=False,
    )
    fig.tight_layout()
    return fig
