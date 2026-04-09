"""Allele frequency divergence between ancestries.

P3.4: Pairwise F_ST-like divergence heatmap and per-site histogram.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import ancestry_colors, popout_style
from ._loaders import read_model_npz


def plot_freq_divergence(
    prefix: str | Path,
    *,
    figsize: tuple[float, float] = (12, 5),
) -> "matplotlib.figure.Figure":
    """Allele frequency divergence between ancestry pairs.

    Left panel: histogram of per-site mean pairwise |freq_a - freq_b|.
    Right panel: pairwise mean divergence matrix.

    Parameters
    ----------
    prefix : path prefix or direct path to .model.npz
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    npz_path = (
        prefix if prefix.suffix == ".npz"
        else prefix.with_name(prefix.name + ".model.npz")
    )

    model = read_model_npz(npz_path)
    freq = model["allele_freq"]  # (A, T)
    n_anc, n_sites = freq.shape
    colors = ancestry_colors(n_anc)

    # Compute pairwise divergence
    pairs = []
    pair_labels = []
    pair_means = np.zeros((n_anc, n_anc))
    for i in range(n_anc):
        for j in range(i + 1, n_anc):
            diff = np.abs(freq[i] - freq[j])
            pairs.append(diff)
            pair_labels.append(f"{i} vs {j}")
            m = float(diff.mean())
            pair_means[i, j] = m
            pair_means[j, i] = m

    with popout_style():
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left: histogram of per-site divergence for each pair
        ax = axes[0]
        for idx, (diff, label) in enumerate(zip(pairs, pair_labels)):
            ax.hist(diff, bins=50, alpha=0.5, label=label,
                    edgecolor="none", density=True)
        ax.set_xlabel("|freq_a - freq_b|")
        ax.set_ylabel("Density")
        ax.set_title("Per-Site Frequency Divergence")
        ax.legend(fontsize=6, ncol=2)

        # Right: pairwise mean divergence matrix
        ax = axes[1]
        im = ax.imshow(pair_means, cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(n_anc))
        ax.set_yticks(range(n_anc))
        ax.set_xticklabels([f"Anc {i}" for i in range(n_anc)], fontsize=7)
        ax.set_yticklabels([f"Anc {i}" for i in range(n_anc)], fontsize=7)
        ax.set_title("Mean Pairwise Divergence")

        # Annotate cells
        for i in range(n_anc):
            for j in range(n_anc):
                if i != j:
                    ax.text(j, i, f"{pair_means[i, j]:.3f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if pair_means[i, j] > pair_means.max() * 0.6 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Mean |Δfreq|")
        fig.suptitle("Allele Frequency Divergence Between Ancestries",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
    return fig
