"""Label correlation heatmap.

Shows how well each inferred ancestry correlates with each 1KG reference
population.  Highlights the assigned label mapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import ancestry_colors, ancestry_names, popout_style


def plot_label_correlation(
    prefix: str | Path,
    *,
    labels: dict | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> "matplotlib.figure.Figure":
    """Heatmap of Pearson correlation between inferred and reference ancestries.

    Parameters
    ----------
    prefix : unused (kept for API consistency), labels must be provided
    labels : labels dict from read_labels_json() — required
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    if labels is None or "correlations" not in labels:
        raise ValueError(
            "Label correlation plot requires --labels with a correlations matrix"
        )

    corr = labels["correlations"]  # (K_inf, K_ref) numpy array
    if isinstance(corr, list):
        corr = np.array(corr, dtype=np.float64)
    ref_names = labels.get("ref_names", [f"Ref {i}" for i in range(corr.shape[1])])
    n_inf = corr.shape[0]
    names = ancestry_names(n_inf, labels)
    n_overlap = labels.get("n_overlapping_sites", "?")

    # Identify assigned matches from label_map
    label_map = labels.get("label_map", {})
    assigned: set[tuple[int, int]] = set()
    for inf_idx, pop_name in label_map.items():
        inf_idx = int(inf_idx)
        if pop_name in ref_names:
            ref_idx = ref_names.index(pop_name)
            assigned.add((inf_idx, ref_idx))

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(ref_names)))
        ax.set_yticks(range(n_inf))
        ax.set_xticklabels(ref_names, fontsize=9, rotation=45, ha="right")
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Reference Population (1KG)")
        ax.set_ylabel("Inferred Ancestry")

        # Annotate cells with correlation values
        for i in range(n_inf):
            for j in range(len(ref_names)):
                val = corr[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                weight = "bold" if (i, j) in assigned else "normal"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=8,
                    color=color, fontweight=weight,
                )
                # Highlight assigned matches with border
                if (i, j) in assigned:
                    from matplotlib.patches import Rectangle
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=2,
                    ))

        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        ax.set_title(
            f"Label Correlation (n={n_overlap} overlapping sites)",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
    return fig
