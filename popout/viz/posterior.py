"""Posterior confidence distribution.

P2.1: Histogram of per-tract mean posterior confidence.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._style import popout_style
from ._loaders import read_tracts, read_summary


def plot_posterior_confidence(
    prefix: str | Path,
    *,
    n_bins: int = 50,
    figsize: tuple[float, float] = (8, 5),
) -> "matplotlib.figure.Figure":
    """Histogram of posterior confidence (max posterior per tract).

    Falls back to summary-level mean if per-tract posteriors unavailable.

    Parameters
    ----------
    prefix : path prefix
    n_bins : number of histogram bins
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")

    # Try to read per-tract posteriors
    posteriors = []
    if tracts_path.exists():
        for t in read_tracts(tracts_path):
            if not math.isnan(t.mean_posterior):
                posteriors.append(t.mean_posterior)

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        if posteriors:
            arr = np.array(posteriors)
            ax.hist(
                arr, bins=n_bins, range=(0, 1),
                color="#4477AA", alpha=0.8, edgecolor="white", linewidth=0.3,
            )
            mean_conf = arr.mean()
            ax.axvline(mean_conf, color="#EE6677", linestyle="--", linewidth=1.5,
                       label=f"Mean = {mean_conf:.3f}")
            ax.set_xlabel("Mean Posterior (per tract)")
            ax.set_ylabel("Count")
            ax.set_title("Posterior Confidence Distribution")
            ax.legend()
        else:
            # Fall back to summary-level stat
            summary_path = prefix.with_name(prefix.name + ".summary.json")
            if summary_path.exists():
                summary = read_summary(summary_path)
                mean_conf = summary.get("output", {}).get("mean_posterior_confidence")
                if mean_conf is not None:
                    ax.text(
                        0.5, 0.5,
                        f"Mean posterior confidence: {mean_conf:.4f}\n"
                        f"(Per-tract posteriors not available;\n"
                        f"re-run with --probs for detailed histogram)",
                        ha="center", va="center", fontsize=12,
                        transform=ax.transAxes,
                    )
                    ax.set_title("Posterior Confidence")
                else:
                    ax.text(
                        0.5, 0.5, "No posterior data available",
                        ha="center", va="center", fontsize=12,
                        transform=ax.transAxes,
                    )
            else:
                ax.text(
                    0.5, 0.5, "No posterior data available",
                    ha="center", va="center", fontsize=12,
                    transform=ax.transAxes,
                )

        fig.tight_layout()
    return fig
