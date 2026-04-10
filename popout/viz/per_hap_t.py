"""Per-haplotype admixture time (T) distribution.

P3.3: Histogram of per-haplotype T estimates — unique to popout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import popout_style
from ._loaders import read_model_npz


def plot_per_hap_t(
    prefix: str | Path,
    *,
    n_bins: int = 60,
    labels: dict | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> "matplotlib.figure.Figure":
    """Histogram of per-haplotype estimated T (generations since admixture).

    Parameters
    ----------
    prefix : path prefix or direct path to .model.npz
    n_bins : number of histogram bins
    labels : optional labels dict (unused, kept for API consistency)
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    npz_path = (
        prefix if prefix.suffix == ".npz"
        else prefix.with_name(prefix.name + ".model.npz")
    )

    model = read_model_npz(npz_path)
    if "gen_per_hap" not in model:
        raise ValueError(
            "Per-haplotype T not found in model.npz. "
            "Re-run popout with --per-hap-T to generate this data."
        )

    t_per_hap = model["gen_per_hap"]
    global_t = float(model.get("gen_since_admix", 0))

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            t_per_hap, bins=n_bins, color="#4477AA",
            alpha=0.8, edgecolor="white", linewidth=0.3,
        )

        # Mark global T
        if global_t > 0:
            ax.axvline(global_t, color="#EE6677", linestyle="--", linewidth=2,
                       label=f"Global T = {global_t:.1f}")

        # Mark bucket boundaries if available
        if "bucket_centers" in model:
            centers = model["bucket_centers"]
            for c in centers:
                ax.axvline(c, color="#CCCCCC", linestyle=":", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Generations Since Admixture (T)")
        ax.set_ylabel("Number of Haplotypes")
        ax.set_title("Per-Haplotype Admixture Time Distribution")

        # Stats annotation
        median_t = float(np.median(t_per_hap))
        mean_t = float(np.mean(t_per_hap))
        ax.text(
            0.97, 0.95,
            f"Mean = {mean_t:.1f}\nMedian = {median_t:.1f}\n"
            f"n = {len(t_per_hap):,} haplotypes",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig
