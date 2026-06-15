"""Cohort decomposition by FLARE top-N ancestry label.

For each sample, derive a label mechanically from its chr1
5-ancestry FLARE proportion vector:

- ``X≥thr_dom`` if top-1 ≥ ``threshold_dominant`` (default 0.95)
- ``X-leaning`` if top-1 ≥ ``threshold_leaning`` (default 0.85)
                and top-2 < ``threshold_admix`` (default 0.10)
- ``X+Y+Z`` (alpha-sorted) if top-3 ≥ ``threshold_admix``
- ``X+Y`` (alpha-sorted) if top-2 ≥ ``threshold_admix``
- otherwise ``X-leaning``

Horizontal bar chart sorted by sample count.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5

from .._helpers import classify_sample_regime, load_cohort_cube


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
    n_samples = chr1.shape[0]

    labels = [
        classify_sample_regime(
            row, members,
            threshold_dominant=thr_dom,
            threshold_leaning=thr_lean,
            threshold_admix=thr_admix,
        )
        for row in chr1
    ]

    counts = Counter(labels)
    items = counts.most_common()
    thr_n = max(50, int(0.001 * n_samples))
    shown = [(lab, n) for lab, n in items if n >= thr_n]
    other_n = sum(n for lab, n in items if n < thr_n)
    if other_n:
        shown.append((f"other (<{thr_n})", other_n))

    return {
        "present": True,
        "n_samples": n_samples,
        "rows": shown,
        "threshold_dominant": thr_dom,
        "threshold_leaning": thr_lean,
        "threshold_admix": thr_admix,
    }


def _bar_color(lab: str, members: list[str], palette: dict[str, str]) -> str:
    if "≥" in lab:
        a = lab.split("≥")[0]
        return palette.get(a, "#888")
    if lab.endswith("-leaning"):
        a = lab.split("-")[0]
        return palette.get(a, "#888")
    if lab.startswith("other"):
        return "#888"
    parts = lab.split("+")
    if not parts:
        return "#888"
    cs = []
    for p in parts:
        c = palette.get(p)
        if c is None:
            continue
        cs.append(np.array(plt.matplotlib.colors.to_rgb(c)))
    if not cs:
        return "#888"
    avg = sum(cs) / len(cs)
    return tuple(avg)


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort_global.tsv data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    rows = list(data["rows"])
    rows.sort(key=lambda kv: kv[1])
    n_samples = data["n_samples"]
    members = list(SP5.members)
    thr_dom = data["threshold_dominant"]
    thr_lean = data["threshold_leaning"]
    thr_admix = data["threshold_admix"]

    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.32 * len(rows) + 1.5)))
    ys = np.arange(len(rows))
    fracs = np.array([n / n_samples for _, n in rows])
    colors = [_bar_color(lab, members, palette) for lab, _ in rows]
    ax.barh(ys, fracs, color=colors, edgecolor="white", linewidth=0.6)
    for y, (lab, n) in zip(ys, rows):
        ax.text(fracs[int(y)], y,
                f"  n={n:>7,}  ({n / n_samples * 100:5.2f}%)",
                va="center", ha="left", fontsize=10, family="monospace",
                color="#222")
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=10)
    ax.set_xlim(0, max(fracs, default=0.1) * 1.30)
    ax.set_xlabel(
        f"fraction of cohort  (N = {n_samples:,} chr1 samples)")
    ax.set_title(
        f"Cohort by FLARE top-N ancestry label "
        f"(X≥{thr_dom:.2f} ; X-leaning if top1≥{thr_lean:.2f} "
        f"& top2<{thr_admix:.2f} ; else alpha-sorted top-2 or top-3)",
        pad=10, fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
