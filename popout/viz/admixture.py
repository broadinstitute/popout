"""ADMIXTURE-style stacked bar plot and ancestry proportion density.

P1.1: Per-sample stacked bar (the classic population genetics overview).
P4.3: Per-ancestry proportion histogram/density.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import ancestry_colors, ancestry_names, popout_style
from ._loaders import read_global_tsv


def plot_admixture(
    prefix: str | Path,
    *,
    max_samples: int = 5000,
    title: str | None = None,
    labels: dict | None = None,
    figsize: tuple[float, float] = (16, 4),
) -> "matplotlib.figure.Figure":
    """ADMIXTURE-style stacked bar plot of per-sample ancestry proportions.

    Parameters
    ----------
    prefix : path prefix or direct path to .global.tsv
    max_samples : maximum samples to display (stratified subsample if exceeded)
    title : optional figure title
    labels : optional labels dict from read_labels_json()
    figsize : figure size in inches
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tsv_path = (
        prefix if prefix.suffix == ".tsv"
        else prefix.with_name(prefix.name + ".global.tsv")
    )
    data = read_global_tsv(tsv_path)
    props = data.proportions  # (N, A)
    n_samples, n_anc = props.shape
    colors = ancestry_colors(n_anc)
    names = ancestry_names(n_anc, labels)

    # Sort by dominant ancestry, then by proportion within group
    dominant = np.argmax(props, axis=1)
    sort_idx = np.lexsort((props[np.arange(n_samples), dominant], dominant))
    props = props[sort_idx]

    # Subsample if too many
    if n_samples > max_samples:
        step = n_samples / max_samples
        indices = np.round(np.arange(max_samples) * step).astype(int)
        indices = np.clip(indices, 0, n_samples - 1)
        props = props[indices]
        n_display = max_samples
    else:
        n_display = n_samples

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        # Build stacked image
        from matplotlib.colors import to_rgba
        rgba_colors = [to_rgba(c) for c in colors]

        n_vert = 200  # vertical resolution
        strip = np.zeros((n_vert, n_display, 4), dtype=np.float32)
        for si in range(n_display):
            cum = 0.0
            for a in range(n_anc):
                y_start = int(cum * n_vert)
                cum += props[si, a]
                y_end = int(cum * n_vert)
                for ch in range(4):
                    strip[y_start:y_end, si, ch] = rgba_colors[a][ch]

        ax.imshow(
            strip, aspect="auto", origin="lower",
            extent=[0, n_display, 0, 1],
            interpolation="nearest",
        )
        ax.set_xlim(0, n_display)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Samples (n={n_samples:,}, sorted by ancestry)")
        ax.set_ylabel("Ancestry Proportion")

        if title is None:
            name_str = ", ".join(names)
            title = f"Global Ancestry Proportions (K={n_anc}: {name_str})"
        ax.set_title(title)

        # Legend
        from matplotlib.patches import Rectangle
        legend_patches = [
            Rectangle((0, 0), 1, 1, facecolor=colors[a])
            for a in range(n_anc)
        ]
        ax.legend(
            legend_patches, names,
            loc="upper right", fontsize=7, ncol=min(n_anc, 4),
        )

        fig.tight_layout()
    return fig


def plot_ancestry_density(
    prefix: str | Path,
    *,
    n_bins: int = 50,
    labels: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> "matplotlib.figure.Figure":
    """Per-ancestry histogram of global ancestry proportions across samples.

    Parameters
    ----------
    prefix : path prefix or direct path to .global.tsv
    n_bins : number of histogram bins
    labels : optional labels dict from read_labels_json()
    figsize : figure size in inches (auto-scaled to n_ancestries if None)
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tsv_path = (
        prefix if prefix.suffix == ".tsv"
        else prefix.with_name(prefix.name + ".global.tsv")
    )
    data = read_global_tsv(tsv_path)
    props = data.proportions
    n_anc = data.n_ancestries
    colors = ancestry_colors(n_anc)
    names = ancestry_names(n_anc, labels)

    ncols = min(n_anc, 4)
    nrows = (n_anc + ncols - 1) // ncols
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    with popout_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for a in range(n_anc):
            ax = axes[a // ncols][a % ncols]
            ax.hist(
                props[:, a], bins=n_bins, color=colors[a],
                alpha=0.8, edgecolor="white", linewidth=0.3,
            )
            # Genome-wide mean proportion
            mean_prop = float(props[:, a].mean())
            ax.axvline(mean_prop, color="black", linestyle="--", linewidth=1,
                       alpha=0.6, label=f"mean={mean_prop:.2f}")
            ax.set_xlabel("Proportion")
            ax.set_ylabel("Count")
            ax.set_title(names[a], fontsize=11)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=7)

        # Hide unused axes
        for idx in range(n_anc, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("Ancestry Proportion Distributions", fontsize=13, fontweight="bold")
        fig.tight_layout()
    return fig
