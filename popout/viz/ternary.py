"""Ternary / triangle plot for K=3 admixture.

P4.2: Each sample as a point in equilateral triangle coordinates.
Only generated when n_ancestries == 3.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._style import ancestry_colors, popout_style
from ._loaders import read_global_tsv


def plot_ternary(
    prefix: str | Path,
    *,
    max_samples: int = 10000,
    point_size: float = 1.0,
    figsize: tuple[float, float] = (8, 7),
) -> "matplotlib.figure.Figure":
    """Ternary plot of three-way ancestry proportions.

    Parameters
    ----------
    prefix : path prefix or direct path to .global.tsv
    max_samples : max points to plot
    point_size : scatter point size
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tsv_path = (
        prefix if prefix.suffix == ".tsv"
        else prefix.with_name(prefix.name + ".global.tsv")
    )
    data = read_global_tsv(tsv_path)
    props = data.proportions  # (N, A)

    if data.n_ancestries != 3:
        raise ValueError(
            f"Ternary plot requires exactly 3 ancestries, got {data.n_ancestries}"
        )

    colors = ancestry_colors(3)

    # Subsample if needed
    if props.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(props.shape[0], max_samples, replace=False)
        props = props[idx]

    # Simplex → 2D projection (equilateral triangle)
    # Vertices: bottom-left (1,0,0), bottom-right (0,1,0), top (0,0,1)
    sqrt3_2 = math.sqrt(3) / 2
    x = props[:, 1] + props[:, 2] / 2
    y = props[:, 2] * sqrt3_2

    # Color each point by its dominant ancestry proportion using RGB mixing
    rgb = np.zeros((len(props), 3))
    from matplotlib.colors import to_rgb
    c0, c1, c2 = [np.array(to_rgb(colors[i])) for i in range(3)]
    for i in range(len(props)):
        rgb[i] = props[i, 0] * c0 + props[i, 1] * c1 + props[i, 2] * c2

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        # Draw triangle border
        tri_x = [0, 1, 0.5, 0]
        tri_y = [0, 0, sqrt3_2, 0]
        ax.plot(tri_x, tri_y, "k-", linewidth=1)

        # Grid lines (10% increments)
        for frac in np.arange(0.1, 1.0, 0.1):
            # Lines parallel to each edge
            # Bottom edge parallel
            x0 = frac / 2
            y0 = frac * sqrt3_2
            x1 = 1 - frac / 2
            y1 = frac * sqrt3_2
            ax.plot([x0, x1], [y0, y1], color="#DDDDDD", linewidth=0.5)
            # Left edge parallel
            x0 = frac
            y0 = 0
            x1 = (1 + frac) / 2
            y1 = (1 - frac) * sqrt3_2
            ax.plot([x0, x1], [y0, y1], color="#DDDDDD", linewidth=0.5)
            # Right edge parallel
            x0 = 0
            y0 = 0
            x1 = (1 - frac) / 2
            y1 = (1 - frac) * sqrt3_2
            ax.plot([x0 + frac * 0.5, x1 + frac * 0.5],
                    [frac * sqrt3_2 - y0 * sqrt3_2, y1],
                    color="#DDDDDD", linewidth=0.5)

        ax.scatter(x, y, c=rgb, s=point_size, alpha=0.5, edgecolors="none")

        # Vertex labels
        ax.text(0, -0.04, "Ancestry 0", ha="center", fontsize=10,
                fontweight="bold", color=colors[0])
        ax.text(1, -0.04, "Ancestry 1", ha="center", fontsize=10,
                fontweight="bold", color=colors[1])
        ax.text(0.5, sqrt3_2 + 0.03, "Ancestry 2", ha="center", fontsize=10,
                fontweight="bold", color=colors[2])

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, sqrt3_2 + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title("Three-Way Ancestry (Ternary Plot)", fontsize=13, fontweight="bold")
        fig.tight_layout()
    return fig
