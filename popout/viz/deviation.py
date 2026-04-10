"""Ancestry deviation (selection scan) plot.

Shows z-score of local ancestry frequency deviation from genome-wide mean
along the genome.  Outlier windows are candidates for ancestry-specific
selection or adaptive introgression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import (
    CHROM_ORDER, ancestry_colors, ancestry_names, chrom_sort_key,
    normalize_chrom, popout_style,
)
from ._loaders import read_tracts


def plot_ancestry_deviation(
    prefix: str | Path,
    *,
    window_mb: float = 1.0,
    labels: dict | None = None,
    figsize: tuple[float, float] = (18, 8),
) -> "matplotlib.figure.Figure":
    """Ancestry deviation z-score along the genome.

    For each genomic window, computes z = (local_prop - genome_mean) / genome_sd
    and plots Manhattan-style per ancestry.

    Parameters
    ----------
    prefix : path prefix
    window_mb : window size in Mb
    labels : optional labels dict from read_labels_json()
    figsize : figure size
    """
    import matplotlib.pyplot as plt
    from .genome import _compute_windowed_ancestry

    prefix = Path(prefix)
    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")

    coverage, total_bp = _compute_windowed_ancestry(tracts_path, window_mb)

    if not coverage:
        raise ValueError("No tracts found")

    # Determine ancestries
    all_anc = set()
    for chrom_data in coverage.values():
        for bin_data in chrom_data.values():
            all_anc.update(bin_data.keys())
    n_anc = max(all_anc) + 1
    colors = ancestry_colors(n_anc)
    names = ancestry_names(n_anc, labels)

    chroms = sorted(
        [c for c in coverage.keys() if c in CHROM_ORDER],
        key=chrom_sort_key,
    )

    # Compute per-window proportions and genome-wide stats per ancestry
    props_by_anc: dict[int, list[float]] = {a: [] for a in range(n_anc)}
    window_info: list[tuple[str, int, dict[int, float]]] = []  # (chrom, bin, {anc: prop})

    for chrom in chroms:
        for b in sorted(coverage[chrom].keys()):
            tot = total_bp[chrom].get(b, 1)
            prop_dict = {}
            for a in range(n_anc):
                p = coverage[chrom][b].get(a, 0) / tot
                prop_dict[a] = p
                props_by_anc[a].append(p)
            window_info.append((chrom, b, prop_dict))

    # Genome-wide mean and sd per ancestry
    stats = {}
    for a in range(n_anc):
        arr = np.array(props_by_anc[a])
        stats[a] = (float(arr.mean()), float(arr.std()))

    with popout_style():
        nrows = n_anc
        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
        if nrows == 1:
            axes = [axes]

        for ai, a in enumerate(range(n_anc)):
            ax = axes[ai]
            mean_a, sd_a = stats[a]

            offset = 0.0
            tick_positions = []
            tick_labels_list = []
            prev_chrom = None

            for chrom, b, prop_dict in window_info:
                if prev_chrom is not None and chrom != prev_chrom:
                    # Chromosome boundary — compute offset
                    max_prev = max(
                        bb for cc, bb, _ in window_info if cc == prev_chrom
                    )
                    sep_x = offset + (max_prev + 1) * window_mb
                    ax.axvline(sep_x, color="#EEEEEE", linewidth=0.3)
                    offset = sep_x + window_mb

                if prev_chrom != chrom:
                    # Record tick for this chromosome
                    chrom_bins = [
                        bb for cc, bb, _ in window_info if cc == chrom
                    ]
                    mid = offset + (max(chrom_bins) * window_mb) / 2
                    tick_positions.append(mid)
                    tick_labels_list.append(chrom.replace("chr", ""))

                prev_chrom = chrom

            # Now plot points
            offset = 0.0
            prev_chrom = None
            xs = []
            zs = []

            for chrom, b, prop_dict in window_info:
                if prev_chrom is not None and chrom != prev_chrom:
                    max_prev = max(
                        bb for cc, bb, _ in window_info if cc == prev_chrom
                    )
                    offset = offset + (max_prev + 1) * window_mb + window_mb

                x = offset + b * window_mb
                z = (prop_dict[a] - mean_a) / sd_a if sd_a > 0 else 0
                xs.append(x)
                zs.append(z)
                prev_chrom = chrom

            # Alternating chromosome colors
            ax.scatter(xs, zs, s=1, c=colors[a], alpha=0.5, rasterized=True)

            # Significance thresholds
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
            ax.axhline(2, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.axhline(-2, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.axhline(3, color="red", linestyle=":", linewidth=0.5, alpha=0.4)
            ax.axhline(-3, color="red", linestyle=":", linewidth=0.5, alpha=0.4)

            ax.set_ylabel(f"{names[a]}\nz-score", fontsize=8)
            ax.set_ylim(-5, 5)

            if ai == 0:
                ax.set_title("Ancestry Deviation Along Genome (Selection Scan)")

        # X-axis on bottom panel
        axes[-1].set_xticks(tick_positions)
        axes[-1].set_xticklabels(tick_labels_list, fontsize=7)
        axes[-1].set_xlabel("Chromosome")

        fig.tight_layout()
    return fig
