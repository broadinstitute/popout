"""Tract length distributions and switch rate along genome.

P1.3: Full tract length histogram with theoretical exponential overlay.
P2.4: Switch rate along genomic position (QC diagnostic).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import (
    CHROM_ORDER, ancestry_colors, chrom_sort_key, normalize_chrom, popout_style,
)
from ._loaders import (
    collect_tract_lengths_by_ancestry, read_model_text, read_tracts,
)


def plot_tract_lengths(
    prefix: str | Path,
    *,
    n_bins: int = 80,
    log_scale: bool = True,
    show_theoretical: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> "matplotlib.figure.Figure":
    """Tract length distribution per ancestry with optional exponential overlay.

    Parameters
    ----------
    prefix : path prefix
    n_bins : number of histogram bins
    log_scale : use log scale on y-axis
    show_theoretical : overlay theoretical exponential from fitted T and mu
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")
    lengths = collect_tract_lengths_by_ancestry(tracts_path)

    if not lengths:
        raise ValueError("No tracts found")

    n_anc = max(lengths.keys()) + 1
    colors = ancestry_colors(n_anc)

    # Load model for theoretical overlay
    model = None
    if show_theoretical:
        model_path = prefix.with_name(prefix.name + ".model")
        if model_path.exists():
            model = read_model_text(model_path)

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        all_lengths = np.concatenate([np.array(v) for v in lengths.values()])
        max_len = np.percentile(all_lengths, 99.5)
        bins = np.linspace(0, max_len, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        for a in sorted(lengths.keys()):
            arr = np.array(lengths[a])
            counts, _ = np.histogram(arr, bins=bins)
            density = counts / (len(arr) * bin_width) if len(arr) > 0 else counts
            ax.bar(
                bin_centers, density, width=bin_width * 0.9,
                color=colors[a], alpha=0.6, label=f"Ancestry {a} (n={len(arr):,})",
            )

        # Theoretical exponential overlay
        if model is not None and show_theoretical:
            T = model.get("gen_since_admix", 20)
            mu = model.get("mu", [])
            if mu:
                x = np.linspace(0.01, max_len, 300)
                for a in sorted(lengths.keys()):
                    if a < len(mu):
                        # Rate = T * (1 - mu[a]) per Morgan ≈ T * (1-mu[a]) / 100 per Mb
                        # Tracts are in Mb; genetic distance ≈ 1 cM/Mb ≈ 0.01 M/Mb
                        rate = T * (1 - mu[a]) * 0.01  # per Mb
                        if rate > 0:
                            y = rate * np.exp(-rate * x)
                            ax.plot(x, y, "--", color=colors[a], linewidth=2,
                                    alpha=0.8)

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-5)
        ax.set_xlabel("Tract Length (Mb)")
        ax.set_ylabel("Density")
        ax.set_title("Ancestry Tract Length Distribution")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return fig


def plot_switch_rate(
    prefix: str | Path,
    *,
    window_mb: float = 5.0,
    figsize: tuple[float, float] = (16, 4),
) -> "matplotlib.figure.Figure":
    """Switch rate (ancestry transitions per Mb) along the genome.

    Parameters
    ----------
    prefix : path prefix
    window_mb : window size in Mb for binning switches
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")

    # Count tract starts (= switches) in windows along the genome
    switch_counts: dict[str, dict[int, int]] = {}  # chrom -> {bin -> count}
    n_haps_per_chrom: dict[str, set] = {}
    for t in read_tracts(tracts_path):
        c = normalize_chrom(t.chrom)
        bin_idx = int(t.start_bp / (window_mb * 1e6))
        switch_counts.setdefault(c, {})
        switch_counts[c][bin_idx] = switch_counts[c].get(bin_idx, 0) + 1
        n_haps_per_chrom.setdefault(c, set()).add((t.sample, t.haplotype))

    if not switch_counts:
        raise ValueError("No tracts found")

    chroms = sorted(switch_counts.keys(), key=chrom_sort_key)
    # Filter to standard autosomes
    chroms = [c for c in chroms if c in CHROM_ORDER]

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        offset = 0
        tick_positions = []
        tick_labels = []

        for ci, chrom in enumerate(chroms):
            counts = switch_counts[chrom]
            n_haps = len(n_haps_per_chrom.get(chrom, set())) or 1
            max_bin = max(counts.keys()) if counts else 0

            xs = []
            ys = []
            for b in range(max_bin + 1):
                xs.append(offset + b * window_mb)
                # Rate = switches per haplotype per Mb
                ys.append(counts.get(b, 0) / (n_haps * window_mb))

            color = "#4477AA" if ci % 2 == 0 else "#AA3377"
            ax.fill_between(xs, ys, alpha=0.5, color=color, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=0.5, alpha=0.8)

            mid = offset + (max_bin * window_mb) / 2
            tick_positions.append(mid)
            tick_labels.append(chrom.replace("chr", ""))
            offset += (max_bin + 2) * window_mb

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Chromosome")
        ax.set_ylabel("Switches / haplotype / Mb")
        ax.set_title("Ancestry Switch Rate Along Genome")
        fig.tight_layout()
    return fig
