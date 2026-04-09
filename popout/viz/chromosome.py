"""Per-chromosome ancestry boxplots.

P4.1: Boxplot of per-sample ancestry proportion grouped by chromosome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import ancestry_colors, chrom_sort_key, normalize_chrom, popout_style
from ._loaders import read_tracts


def plot_chromosome_boxplots(
    prefix: str | Path,
    *,
    ancestry: int = 0,
    max_samples: int = 5000,
    figsize: tuple[float, float] = (14, 5),
) -> "matplotlib.figure.Figure":
    """Per-chromosome boxplot of one ancestry's proportion across samples.

    Parameters
    ----------
    prefix : path prefix
    ancestry : which ancestry to plot (default: 0)
    max_samples : subsample individuals if cohort is larger
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")

    # Accumulate per-sample per-chrom ancestry coverage
    # coverage[chrom][sample] = {ancestry: total_bp}
    coverage: dict[str, dict[str, dict[int, int]]] = {}
    total_bp: dict[str, dict[str, int]] = {}

    for t in read_tracts(tracts_path):
        c = normalize_chrom(t.chrom)
        bp = t.end_bp - t.start_bp
        coverage.setdefault(c, {}).setdefault(t.sample, {})
        coverage[c][t.sample][t.ancestry] = coverage[c][t.sample].get(t.ancestry, 0) + bp
        total_bp.setdefault(c, {})
        total_bp[c][t.sample] = total_bp[c].get(t.sample, 0) + bp

    if not coverage:
        raise ValueError("No tracts found")

    chroms = sorted(
        [c for c in coverage.keys() if c.startswith("chr")],
        key=chrom_sort_key,
    )

    # Build per-chrom proportion arrays
    box_data = []
    labels = []
    for chrom in chroms:
        samples = list(coverage[chrom].keys())
        if len(samples) > max_samples:
            samples = samples[:max_samples]
        props = []
        for s in samples:
            tot = total_bp[chrom].get(s, 1)
            anc_bp = coverage[chrom][s].get(ancestry, 0)
            props.append(anc_bp / tot)
        if props:
            box_data.append(props)
            labels.append(chrom.replace("chr", ""))

    colors = ancestry_colors(max(a for c in coverage.values()
                                  for s in c.values()
                                  for a in s.keys()) + 1)

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        bp = ax.boxplot(
            box_data, tick_labels=labels, patch_artist=True,
            showfliers=False, widths=0.6,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[ancestry])
            patch.set_alpha(0.7)

        ax.set_xlabel("Chromosome")
        ax.set_ylabel(f"Ancestry {ancestry} Proportion")
        ax.set_title(f"Per-Chromosome Ancestry {ancestry} Proportion Distribution")
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
    return fig
