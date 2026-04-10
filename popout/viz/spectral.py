"""PCA scatter plots colored by ancestry.

P3.1: PCA colored by final ancestry assignment.
P3.2: Side-by-side: GMM seed labels vs final ancestry.

Requires {prefix}.spectral.npz (saved during spectral initialization).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import ancestry_colors, ancestry_names, popout_style
from ._loaders import read_spectral_npz, read_global_tsv


def plot_pca_ancestry(
    prefix: str | Path,
    *,
    max_points: int = 50000,
    point_size: float = 0.5,
    labels: dict | None = None,
    figsize: tuple[float, float] = (8, 7),
) -> "matplotlib.figure.Figure":
    """PCA scatter colored by final ancestry assignment.

    Parameters
    ----------
    prefix : path prefix
    max_points : max haplotypes to plot
    point_size : scatter point size
    labels : optional labels dict from read_labels_json()
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    spec_path = prefix.with_name(prefix.name + ".spectral.npz")
    spec = read_spectral_npz(spec_path)
    if spec is None:
        raise FileNotFoundError(
            f"Spectral data not found at {spec_path}. "
            "Re-run popout to generate this file."
        )

    proj = spec["pca_proj"]  # (H, n_pc)
    gmm_labels = spec["gmm_labels"]  # (H,)

    # Try to load final ancestry from global.tsv for coloring
    global_path = prefix.with_name(prefix.name + ".global.tsv")
    if global_path.exists():
        data = read_global_tsv(global_path)
        # Map sample-level dominant ancestry to haplotype level
        dominant = np.argmax(data.proportions, axis=1)  # (n_samples,)
        # Duplicate for haplotypes
        final_labels = np.repeat(dominant, 2)
        if len(final_labels) != len(gmm_labels):
            final_labels = gmm_labels  # fallback
    else:
        final_labels = gmm_labels

    n_anc = int(final_labels.max()) + 1
    colors = ancestry_colors(n_anc)
    names = ancestry_names(n_anc, labels)

    # Subsample if needed
    if len(proj) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(proj), max_points, replace=False)
        proj = proj[idx]
        final_labels = final_labels[idx]

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        for a in range(n_anc):
            mask = final_labels == a
            if mask.any():
                ax.scatter(
                    proj[mask, 0], proj[mask, 1],
                    c=colors[a], s=point_size, alpha=0.3,
                    label=names[a], edgecolors="none",
                )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Colored by Final Ancestry")
        ax.legend(markerscale=5, fontsize=8)
        fig.tight_layout()
    return fig


def plot_seed_vs_final(
    prefix: str | Path,
    *,
    max_points: int = 50000,
    point_size: float = 0.5,
    labels: dict | None = None,
    figsize: tuple[float, float] = (16, 7),
) -> "matplotlib.figure.Figure":
    """Side-by-side PCA: GMM seed labels vs final EM ancestry.

    Parameters
    ----------
    prefix : path prefix
    max_points : max haplotypes per panel
    point_size : scatter point size
    labels : optional labels dict from read_labels_json()
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    spec_path = prefix.with_name(prefix.name + ".spectral.npz")
    spec = read_spectral_npz(spec_path)
    if spec is None:
        raise FileNotFoundError(
            f"Spectral data not found at {spec_path}. "
            "Re-run popout to generate this file."
        )

    proj = spec["pca_proj"]
    gmm_labels = spec["gmm_labels"]

    # Load final ancestry
    global_path = prefix.with_name(prefix.name + ".global.tsv")
    if global_path.exists():
        data = read_global_tsv(global_path)
        dominant = np.argmax(data.proportions, axis=1)
        final_labels = np.repeat(dominant, 2)
        if len(final_labels) != len(gmm_labels):
            final_labels = gmm_labels
    else:
        final_labels = gmm_labels

    n_anc = max(int(gmm_labels.max()), int(final_labels.max())) + 1
    colors = ancestry_colors(n_anc)
    names = ancestry_names(n_anc, labels)

    # Subsample
    if len(proj) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(proj), max_points, replace=False)
        proj = proj[idx]
        gmm_labels = gmm_labels[idx]
        final_labels = final_labels[idx]

    with popout_style():
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, labels_arr, title in [
            (axes[0], gmm_labels, "Spectral Seed (GMM)"),
            (axes[1], final_labels, "Final EM Ancestry"),
        ]:
            for a in range(n_anc):
                mask = labels_arr == a
                if mask.any():
                    ax.scatter(
                        proj[mask, 0], proj[mask, 1],
                        c=colors[a], s=point_size, alpha=0.3,
                        label=names[a], edgecolors="none",
                    )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(title)
            ax.legend(markerscale=5, fontsize=7)

        fig.suptitle("Spectral Initialization vs Final Ancestry",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
    return fig
