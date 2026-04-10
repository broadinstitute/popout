"""Posterior confidence distribution.

P2.1: Histogram of per-tract mean posterior confidence, optionally broken
down by ancestry, with cumulative distribution overlay.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._style import ancestry_colors, ancestry_names, popout_style
from ._loaders import read_tracts, read_summary


def plot_posterior_confidence(
    prefix: str | Path,
    *,
    n_bins: int = 50,
    labels: dict | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> "matplotlib.figure.Figure":
    """Histogram of posterior confidence (max posterior per tract).

    When per-tract posteriors are available, shows per-ancestry breakdown
    as overlapping step histograms with a cumulative distribution overlay.
    Falls back to summary-level mean if per-tract posteriors unavailable.

    Parameters
    ----------
    prefix : path prefix
    n_bins : number of histogram bins
    labels : optional labels dict from read_labels_json()
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")

    # Try to read per-tract posteriors grouped by ancestry
    posteriors_by_anc: dict[int, list[float]] = {}
    if tracts_path.exists():
        for t in read_tracts(tracts_path):
            if not math.isnan(t.mean_posterior):
                posteriors_by_anc.setdefault(t.ancestry, []).append(t.mean_posterior)

    has_posteriors = any(len(v) > 0 for v in posteriors_by_anc.values())

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        if has_posteriors:
            all_posteriors = np.concatenate([
                np.array(v) for v in posteriors_by_anc.values()
            ])
            n_anc = max(posteriors_by_anc.keys()) + 1
            colors = ancestry_colors(n_anc)
            names = ancestry_names(n_anc, labels)
            bins = np.linspace(0, 1, n_bins + 1)

            # Per-ancestry step histograms
            for a in sorted(posteriors_by_anc.keys()):
                arr = np.array(posteriors_by_anc[a])
                ax.hist(
                    arr, bins=bins, histtype="step", linewidth=1.5,
                    color=colors[a], label=names[a], density=True,
                )

            # Overall mean
            mean_conf = float(all_posteriors.mean())
            ax.axvline(mean_conf, color="black", linestyle="--", linewidth=1.5,
                       label=f"Mean = {mean_conf:.3f}")

            # Cumulative distribution on secondary axis
            ax2 = ax.twinx()
            sorted_post = np.sort(all_posteriors)
            cdf = np.arange(1, len(sorted_post) + 1) / len(sorted_post)
            ax2.plot(sorted_post, cdf, color="gray", linewidth=1.5,
                     alpha=0.5, label="CDF")
            ax2.set_ylabel("Cumulative Fraction", color="gray")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="y", labelcolor="gray")

            ax.set_xlabel("Mean Posterior (per tract)")
            ax.set_ylabel("Density")
            ax.set_title("Posterior Confidence Distribution")
            ax.legend(fontsize=7, loc="upper left")
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
