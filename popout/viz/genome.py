"""Ancestry along the genome: Manhattan-style and multi-individual painting.

P2.2: Mean ancestry proportion vs genomic position (Manhattan-style).
P2.3: Multi-individual chromosome painting for one chromosome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import (
    CHROM_ORDER, ancestry_colors, ancestry_names, chrom_sort_key,
    normalize_chrom, popout_style,
)
from ._loaders import read_tracts


def _compute_windowed_ancestry(
    tracts_path: Path,
    window_mb: float = 1.0,
) -> tuple[
    dict[str, dict[int, dict[int, float]]],
    dict[str, dict[int, float]],
]:
    """Shared helper: compute per-window ancestry coverage from tracts.

    Returns (coverage, total_bp) dicts keyed by chrom -> bin -> ancestry -> bp.
    """
    window_bp = int(window_mb * 1e6)
    coverage: dict[str, dict[int, dict[int, float]]] = {}
    total_bp: dict[str, dict[int, float]] = {}

    for t in read_tracts(tracts_path):
        c = normalize_chrom(t.chrom)
        bin_start = t.start_bp // window_bp
        bin_end = t.end_bp // window_bp
        for b in range(bin_start, bin_end + 1):
            w_start = max(t.start_bp, b * window_bp)
            w_end = min(t.end_bp, (b + 1) * window_bp)
            overlap = w_end - w_start
            if overlap <= 0:
                continue
            coverage.setdefault(c, {}).setdefault(b, {})
            coverage[c][b][t.ancestry] = coverage[c][b].get(t.ancestry, 0) + overlap
            total_bp.setdefault(c, {})
            total_bp[c][b] = total_bp[c].get(b, 0) + overlap

    return coverage, total_bp


def plot_ancestry_along_genome(
    prefix: str | Path,
    *,
    window_mb: float = 1.0,
    labels: dict | None = None,
    figsize: tuple[float, float] = (18, 5),
) -> "matplotlib.figure.Figure":
    """Mean ancestry proportion along the genome (Manhattan-style).

    Parameters
    ----------
    prefix : path prefix
    window_mb : window size in Mb for binning
    labels : optional labels dict from read_labels_json()
    figsize : figure size
    """
    import matplotlib.pyplot as plt

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

    # Compute genome-wide mean proportion per ancestry
    total_cov = {a: 0.0 for a in range(n_anc)}
    grand_total = 0.0
    for chrom in chroms:
        for b in coverage[chrom]:
            tot = total_bp[chrom].get(b, 1)
            grand_total += tot
            for a in range(n_anc):
                total_cov[a] += coverage[chrom][b].get(a, 0)
    genome_means = {a: total_cov[a] / grand_total if grand_total > 0 else 0
                    for a in range(n_anc)}

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        offset = 0.0
        tick_positions = []
        tick_labels = []

        for ci, chrom in enumerate(chroms):
            bins = sorted(coverage[chrom].keys())
            if not bins:
                continue
            max_bin = bins[-1]

            for a in range(n_anc):
                xs = []
                ys = []
                for b in bins:
                    xs.append(offset + b * window_mb)
                    tot = total_bp[chrom].get(b, 1)
                    ys.append(coverage[chrom][b].get(a, 0) / tot)
                ax.plot(xs, ys, color=colors[a], linewidth=1.5, alpha=0.9)

            mid = offset + (max_bin * window_mb) / 2
            tick_positions.append(mid)
            tick_labels.append(chrom.replace("chr", ""))

            # Separator
            sep_x = offset + (max_bin + 1) * window_mb
            ax.axvline(sep_x, color="#CCCCCC", linewidth=0.5)
            offset = sep_x + window_mb

        # Genome-wide mean horizontal lines
        for a in range(n_anc):
            ax.axhline(genome_means[a], color=colors[a], linestyle="--",
                       linewidth=0.8, alpha=0.4)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Chromosome")
        ax.set_ylabel("Ancestry Proportion")
        ax.set_ylim(0, 1)
        ax.set_title("Mean Ancestry Proportion Along Genome")

        # Legend
        from matplotlib.lines import Line2D
        legend_lines = [
            Line2D([0], [0], color=colors[a], linewidth=2)
            for a in range(n_anc)
        ]
        ax.legend(
            legend_lines, names,
            loc="upper right", fontsize=7, ncol=min(n_anc, 4),
        )

        fig.tight_layout()
    return fig


def plot_multi_individual(
    prefix: str | Path,
    chrom: str = "chr1",
    *,
    max_individuals: int = 500,
    window_mb: float = 1.0,
    labels: dict | None = None,
    figsize: tuple[float, float] = (16, 10),
) -> "matplotlib.figure.Figure":
    """Multi-individual chromosome painting for one chromosome.

    Rows = individuals (sorted by dominant ancestry), columns = genomic
    position. Colored by dominant ancestry at each window.

    Parameters
    ----------
    prefix : path prefix
    chrom : chromosome to plot
    max_individuals : maximum individuals to display
    window_mb : window size in Mb
    labels : optional labels dict from read_labels_json()
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")
    window_bp = int(window_mb * 1e6)

    # Collect per-sample per-window dominant ancestry
    sample_data: dict[str, dict[int, dict[int, dict[int, float]]]] = {}
    max_anc = 0

    for t in read_tracts(tracts_path, chrom=chrom):
        sample_data.setdefault(t.sample, {}).setdefault(t.haplotype, {})
        bin_start = t.start_bp // window_bp
        bin_end = t.end_bp // window_bp
        for b in range(bin_start, bin_end + 1):
            w_start = max(t.start_bp, b * window_bp)
            w_end = min(t.end_bp, (b + 1) * window_bp)
            overlap = w_end - w_start
            if overlap > 0:
                sample_data[t.sample][t.haplotype].setdefault(b, {})
                anc_dict = sample_data[t.sample][t.haplotype][b]
                anc_dict[t.ancestry] = anc_dict.get(t.ancestry, 0) + overlap
                if t.ancestry > max_anc:
                    max_anc = t.ancestry

    if not sample_data:
        raise ValueError(f"No tracts found for chromosome {chrom}")

    n_anc = max_anc + 1
    colors = ancestry_colors(n_anc)
    names = ancestry_names(n_anc, labels)

    samples = sorted(sample_data.keys())
    # Determine global bin range
    all_bins = set()
    for s in samples:
        for hap_data in sample_data[s].values():
            all_bins.update(hap_data.keys())
    n_bins = max(all_bins) + 1 if all_bins else 0

    # Build haplotype-level matrix
    rows = []
    row_dominant_ancestry = []
    for s in samples:
        for hap in [0, 1]:
            hap_data = sample_data[s].get(hap, {})
            row = np.full(n_bins, -1, dtype=np.int8)
            anc_sum = np.zeros(n_anc)
            for b in range(n_bins):
                if b in hap_data:
                    best_a = max(hap_data[b], key=hap_data[b].get)
                    row[b] = best_a
                    anc_sum[best_a] += 1
            rows.append(row)
            row_dominant_ancestry.append(np.argmax(anc_sum) if anc_sum.sum() > 0 else 0)

    # Subsample if needed
    if len(rows) > max_individuals * 2:
        step = len(rows) / (max_individuals * 2)
        indices = np.round(np.arange(max_individuals * 2) * step).astype(int)
        indices = np.clip(indices, 0, len(rows) - 1)
        rows = [rows[i] for i in indices]
        row_dominant_ancestry = [row_dominant_ancestry[i] for i in indices]

    # Sort by dominant ancestry
    sort_idx = np.argsort(row_dominant_ancestry)
    matrix = np.array([rows[i] for i in sort_idx])

    # Build RGB image
    from matplotlib.colors import to_rgb
    rgb_colors = [to_rgb(c) for c in colors]
    bg_color = (0.94, 0.94, 0.94)

    img = np.full((matrix.shape[0], matrix.shape[1], 3), bg_color, dtype=np.float32)
    for a in range(n_anc):
        mask = matrix == a
        for ch in range(3):
            img[:, :, ch][mask] = rgb_colors[a][ch]

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            img, aspect="auto", interpolation="nearest",
            extent=[0, n_bins * window_mb, 0, matrix.shape[0]],
        )
        ax.set_xlabel(f"Position on {chrom} (Mb)")
        ax.set_ylabel(f"Haplotypes (n={matrix.shape[0]:,})")
        ax.set_title(f"Multi-Individual Ancestry Painting — {chrom}")

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
