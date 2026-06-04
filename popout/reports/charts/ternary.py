"""Ternary (3-way simplex) plot of per-sample proportions.

Only applicable when the cohort's effective label space resolves to
exactly 3 RF labels. Otherwise the chart returns an empty figure
with a "K != 3" note (i.e. it gates gracefully).

Reads ``cohort/cohort_global.tsv`` + ``cohort/merged_groups_rf.tsv``
and rebuckets per-sample FLARE proportions into SP6 RF labels (same
logic as ``admixture_bar.compute``).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from . import admixture_bar


def compute(ctx, section=None) -> dict:
    base = admixture_bar.compute(ctx)
    if not base.get("present"):
        return {"present": False, "reason": "no data"}
    labels = base["labels"]
    props = base["props"]
    nonempty = props.sum(axis=0) > 0
    used = [labels[j] for j in range(len(labels)) if nonempty[j]]
    if len(used) != 3:
        return {
            "present": False,
            "reason": f"effective K = {len(used)} (need exactly 3)",
            "used_labels": used,
        }
    # Project to the 3 nonempty columns.
    cols = [j for j in range(len(labels)) if nonempty[j]]
    triple = props[:, cols]
    triple = triple / triple.sum(axis=1, keepdims=True)
    return {
        "present": True,
        "labels": used,
        "triple": triple,
        "n_samples": triple.shape[0],
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.6))
        ax.text(
            0.5, 0.5,
            data.get("reason", "ternary plot not applicable"),
            ha="center", va="center", fontsize=10, color="#666",
        )
        ax.axis("off")
        return fig

    labels = data["labels"]
    triple = data["triple"]
    n_samples = triple.shape[0]

    # Subsample for rendering.
    MAX_POINTS = 10_000
    if n_samples > MAX_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, MAX_POINTS, replace=False)
        triple = triple[idx]
        n_samples = MAX_POINTS

    sqrt3_2 = math.sqrt(3) / 2
    x = triple[:, 1] + triple[:, 2] / 2
    y = triple[:, 2] * sqrt3_2

    # RGB-mix points by per-component palette color.
    from matplotlib.colors import to_rgb
    cs = [np.array(to_rgb(palette.get(lab, "#888888"))) for lab in labels]
    rgb = (
        triple[:, 0:1] * cs[0] +
        triple[:, 1:2] * cs[1] +
        triple[:, 2:3] * cs[2]
    )
    rgb = np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, sqrt3_2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=1)

    for frac in np.arange(0.1, 1.0, 0.1):
        ax.plot([frac / 2, 1 - frac / 2],
                [frac * sqrt3_2, frac * sqrt3_2],
                color="#DDDDDD", linewidth=0.5)
        ax.plot([frac, (1 + frac) / 2],
                [0, (1 - frac) * sqrt3_2],
                color="#DDDDDD", linewidth=0.5)
        ax.plot([1 - frac, (1 - frac) / 2 + frac * 0.5],
                [0, (1 - frac) * sqrt3_2 - frac * sqrt3_2 + frac * sqrt3_2],
                color="#DDDDDD", linewidth=0.5)

    ax.scatter(x, y, c=rgb, s=2.5, alpha=0.5, edgecolors="none")

    ax.text(0, -0.04, labels[0], ha="center", fontsize=11,
            fontweight="bold", color=palette.get(labels[0], "#222"))
    ax.text(1, -0.04, labels[1], ha="center", fontsize=11,
            fontweight="bold", color=palette.get(labels[1], "#222"))
    ax.text(0.5, sqrt3_2 + 0.03, labels[2], ha="center", fontsize=11,
            fontweight="bold", color=palette.get(labels[2], "#222"))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, sqrt3_2 + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(
        f"Three-way admixture (n={n_samples:,})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig
