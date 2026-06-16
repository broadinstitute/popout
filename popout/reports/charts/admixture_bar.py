"""Per-sample admixture — two-panel summary, ancestry-stratified.

Panel A (cohort summary, fixed size): for each FLARE top-1 ancestry
stratum, a side-by-side pair of boxes showing the distribution of the
sample's dominant proportion and the sum of all non-dominant
proportions. Five strata × 2 boxes for SP5.

Panel B (admixed-only per-sample strip, capped): the classic
ADMIXTURE-style per-sample stacked bars but restricted to samples whose
top-1 proportion is < ``threshold_dominant`` (default 0.95), sorted by
top-1 ancestry -> descending top-1 proportion. Capped at
``max_admixed_samples`` (default 800) to keep the panel readable; the
template reports the cap.

Data: ``cohort/cohort_global.tsv`` + ``cohort/merged_groups_rf.tsv`` via
``load_cohort_cube``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5

from .._helpers import load_cohort_cube


def compute(ctx, section=None) -> dict:
    opts = section.options if section is not None else {}
    threshold_dominant = float(opts.get("threshold_dominant", 0.95))
    max_admixed_samples = int(opts.get("max_admixed_samples", 800))
    mid_rule = (section.mid_rule if section is not None else None) or "drop"

    cube_data = load_cohort_cube(
        ctx.bundle_dir, label_space=SP5, mid_rule=mid_rule)
    if not cube_data:
        return {"present": False}

    cube = cube_data["cube"]
    members = list(cube_data["label_space"].members)
    n_samples = int(cube.shape[0])
    chr1 = cube[:, 0, :]
    primary_idx = np.argmax(chr1, axis=1)
    primary_props = chr1[np.arange(n_samples), primary_idx]
    other_props = 1.0 - primary_props

    strata_summary: list[dict] = []
    for ai, lab in enumerate(members):
        mask = primary_idx == ai
        n = int(mask.sum())
        if n == 0:
            strata_summary.append({"label": lab, "n": 0,
                                   "dom": None, "other": None})
            continue
        strata_summary.append({
            "label": lab, "n": n,
            "dom": primary_props[mask].astype(float).tolist(),
            "other": other_props[mask].astype(float).tolist(),
        })

    admixed_mask = primary_props < threshold_dominant
    n_admixed = int(admixed_mask.sum())
    admix_idxs = np.where(admixed_mask)[0]
    # Sort by primary ancestry, then descending primary proportion.
    sort_key = np.lexsort((-primary_props[admix_idxs],
                           primary_idx[admix_idxs]))
    admix_idxs = admix_idxs[sort_key]
    capped = False
    if admix_idxs.size > max_admixed_samples:
        capped = True
        step = admix_idxs.size / max_admixed_samples
        pick = np.clip(np.round(np.arange(max_admixed_samples) * step).astype(int),
                       0, admix_idxs.size - 1)
        admix_idxs = admix_idxs[pick]

    return {
        "present": True,
        "n_samples": n_samples,
        "n_admixed": n_admixed,
        "n_admixed_shown": int(admix_idxs.size),
        "admixed_capped": capped,
        "threshold_dominant": threshold_dominant,
        "max_admixed_samples": max_admixed_samples,
        "members": members,
        "strata_summary": strata_summary,
        "admix_idxs": admix_idxs,
        "_cube_chr1": chr1,
        "_primary_idx": primary_idx,
    }


def _box(ax, x, values, color):
    if not values:
        return
    bp = ax.boxplot(
        values, positions=[x], widths=0.55,
        patch_artist=True, showfliers=False, whis=(5, 95),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#222")
        patch.set_linewidth(0.8)
    for med in bp["medians"]:
        med.set_color("#222")
        med.set_linewidth(1.4)
    for whisk in bp["whiskers"]:
        whisk.set_color("#444")
        whisk.set_linewidth(0.8)
    for cap in bp["caps"]:
        cap.set_color("#444")
        cap.set_linewidth(0.8)


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort_global.tsv data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    members = data["members"]
    strata = data["strata_summary"]
    thr = data["threshold_dominant"]
    n_admixed_shown = data["n_admixed_shown"]
    admix_idxs = data["admix_idxs"]
    chr1 = data["_cube_chr1"]
    present_strata = [s for s in strata if s["n"] > 0]

    fig = plt.figure(figsize=(11.0, 6.8))
    gs = fig.add_gridspec(
        nrows=3, ncols=1, height_ratios=[1.0, 1.0, 0.18], hspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.axis("off")

    # Panel A: per-stratum dom / other boxes.
    x_positions = []
    x_labels = []
    for i, s in enumerate(present_strata):
        color = palette.get(s["label"], "#888")
        x_dom = 2 * i + 1
        x_oth = 2 * i + 2
        _box(ax_a, x_dom, s["dom"], color)
        _box(ax_a, x_oth, s["other"], "#bbbbbb")
        x_positions.extend([x_dom, x_oth])
        x_labels.extend([f"{s['label']}\ndom", f"{s['label']}\nother"])

    if not present_strata:
        ax_a.text(0.5, 0.5, "no samples", ha="center", va="center",
                  transform=ax_a.transAxes)
    ax_a.axhline(thr, color="#c62828", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_a.text(0.995, thr, f"  dominant threshold {thr:.2f}",
              transform=ax_a.get_yaxis_transform(),
              ha="right", va="bottom", fontsize=8, color="#c62828")
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(x_labels, fontsize=9)
    ax_a.set_ylim(0, 1.02)
    ax_a.set_ylabel("proportion", fontsize=10)
    ax_a.set_title(
        "Panel A — dominant vs other proportion per top-1 ancestry stratum",
        fontsize=11, loc="left",
    )
    for spine in ("top", "right"):
        ax_a.spines[spine].set_visible(False)

    # Panel B: admixed-only per-sample stacked bars.
    n_display = admix_idxs.size
    if n_display == 0:
        ax_b.text(0.5, 0.5,
                  f"no admixed samples (top-1 < {thr:.2f})",
                  ha="center", va="center", transform=ax_b.transAxes)
        ax_b.axis("off")
    else:
        subset = chr1[admix_idxs]
        bottom = np.zeros(n_display, dtype=np.float32)
        x = np.arange(n_display)
        for j, lab in enumerate(members):
            h = subset[:, j].astype(np.float32)
            if not h.any():
                continue
            color = palette.get(lab, "#888")
            ax_b.bar(x, h, bottom=bottom, width=1.0,
                     color=color, edgecolor="none", linewidth=0)
            bottom += h
        ax_b.set_xlim(0, n_display)
        ax_b.set_ylim(0, 1.0)
        ax_b.set_xticks([])
        for spine in ("top", "right"):
            ax_b.spines[spine].set_visible(False)
        ax_b.set_ylabel("ancestry proportion", fontsize=10)
        ax_b.set_xlabel(
            f"admixed samples (top-1 < {thr:.2f}); "
            f"showing {n_display:,} of {data['n_admixed']:,}; "
            "sorted by top-1 ancestry then dominant proportion",
            fontsize=9,
        )
        ax_b.set_title("Panel B — admixed-only per-sample stack",
                       fontsize=11, loc="left")

    from matplotlib.patches import Rectangle
    handles = [Rectangle((0, 0), 1, 1, facecolor=palette.get(lab, "#888"))
               for lab in members]
    ax_legend.legend(handles, members, loc="center",
                     ncol=min(len(members), 6), fontsize=9, frameon=False)
    return fig
