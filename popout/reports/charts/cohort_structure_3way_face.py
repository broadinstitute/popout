"""(afr, eur, amr) face of the 5-simplex.

Subset chr1 samples where (afr + eur + amr) ≥ ``threshold`` (default
0.95). Renormalise to the 3-way simplex and plot as a ternary
hexbin. Composition-only marker labels at the corners and at the
half-mixes.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5

from .._helpers import load_cohort_cube


_TRI_KEYS = ("afr", "eur", "amr")


def compute(ctx, section=None) -> dict:
    opts = section.options if section is not None else {}
    threshold = float(opts.get("threshold", 0.95))
    keys = tuple(opts.get("ancestries", _TRI_KEYS))
    mid_rule = (section.mid_rule if section is not None else None) or "drop"

    cube_data = load_cohort_cube(
        ctx.bundle_dir, label_space=SP5, mid_rule=mid_rule)
    if not cube_data:
        return {"present": False}

    members = list(cube_data["label_space"].members)
    if not all(k in members for k in keys):
        return {"present": False}
    cube = cube_data["cube"]
    chr1 = cube[:, 0, :]
    idxs = [members.index(k) for k in keys]
    triple_sum = chr1[:, idxs].sum(axis=1)
    mask = triple_sum >= threshold
    n_total = int(chr1.shape[0])
    n_kept = int(mask.sum())
    sub = chr1[mask][:, idxs]
    if n_kept > 0:
        sub = sub / sub.sum(axis=1, keepdims=True)

    return {
        "present": True,
        "threshold": threshold,
        "keys": list(keys),
        "n_total": n_total,
        "n_kept": n_kept,
        "sub": sub if n_kept > 0 else None,
    }


def _tri_xy(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = p[:, 1] + 0.5 * p[:, 2]
    y = (math.sqrt(3) / 2) * p[:, 2]
    return x, y


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present") or data["sub"] is None:
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no 3-way subset found",
                ha="center", va="center")
        ax.axis("off")
        return fig

    sub = data["sub"]
    keys = data["keys"]
    threshold = data["threshold"]
    n_total = data["n_total"]
    n_kept = data["n_kept"]

    fig, ax = plt.subplots(figsize=(10.5, 9))
    xs, ys = _tri_xy(sub)
    hb = ax.hexbin(xs, ys, gridsize=70, cmap="magma_r", bins="log",
                   mincnt=1)
    tri = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#222", linewidth=1.0)

    # Corner labels (each is "<lab> = 1.00")
    corner_specs = [
        (keys[0], (0, 0),                       (-0.04, -0.04), "right"),
        (keys[1], (1, 0),                       (0.04, -0.04),  "left"),
        (keys[2], (0.5, math.sqrt(3) / 2),      (0, 0.03),      "center"),
    ]
    for nm, (x0, y0), (dx, dy), ha in corner_specs:
        c = palette.get(nm, "#888")
        ax.scatter(x0, y0, s=180, color=c, edgecolor="black",
                   linewidth=1.4, zorder=10)
        ax.annotate(f"{nm} = 1.00", (x0 + dx, y0 + dy), ha=ha,
                    va="center", fontsize=12, fontweight="bold", color=c)

    midpoints = [
        (f"½ {keys[0]} + ½ {keys[1]}", [0.5, 0.5, 0.0]),
        (f"½ {keys[1]} + ½ {keys[2]}", [0.0, 0.5, 0.5]),
        (f"½ {keys[0]} + ½ {keys[2]}", [0.5, 0.0, 0.5]),
        (f"⅓ {keys[0]} + ⅓ {keys[1]} + ⅓ {keys[2]}",
         [1 / 3, 1 / 3, 1 / 3]),
    ]
    for nm, p in midpoints:
        x_m, y_m = _tri_xy(np.array([p]))
        ax.scatter(x_m, y_m, s=70, marker="x", color="black",
                   linewidth=1.6, zorder=11)
        ax.annotate(nm, (x_m[0], y_m[0]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9.5, color="#222")

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.08, math.sqrt(3) / 2 + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"({keys[0]}, {keys[1]}, {keys[2]}) face of the 5-simplex  "
        f"(subset where the three sum to ≥ {threshold:.2f}; "
        f"n = {n_kept:,} of {n_total:,})",
        pad=12, fontsize=13)
    cb = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("samples (log)")
    fig.tight_layout()
    return fig
