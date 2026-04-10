"""Per-chromosome ancestry boxplots.

P4.1: Boxplot of per-sample ancestry proportion grouped by chromosome.
Supports all-ancestry grouped view or single-ancestry view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import (
    ancestry_colors, ancestry_names, chrom_sort_key, normalize_chrom, popout_style,
)
from ._loaders import read_tracts


def plot_chromosome_boxplots(
    prefix: str | Path,
    *,
    ancestry: int | None = None,
    max_samples: int = 5000,
    labels: dict | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> "matplotlib.figure.Figure":
    """Per-chromosome boxplot of ancestry proportions across samples.

    When *ancestry* is None (default), shows all ancestries as grouped
    boxplots per chromosome.  When set to an integer, shows only that
    ancestry.

    Parameters
    ----------
    prefix : path prefix
    ancestry : which ancestry to plot (default: all)
    max_samples : subsample individuals if cohort is larger
    labels : optional labels dict from read_labels_json()
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

    # Determine all ancestries present
    all_anc = set()
    for c in coverage.values():
        for s in c.values():
            all_anc.update(s.keys())
    n_anc = max(all_anc) + 1

    colors = ancestry_colors(n_anc)
    names = ancestry_names(n_anc, labels)

    # Determine which ancestries to plot
    anc_list = [ancestry] if ancestry is not None else list(range(n_anc))

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        if len(anc_list) == 1:
            # Single-ancestry mode
            a = anc_list[0]
            box_data = []
            chrom_labels = []
            for chrom in chroms:
                samples = list(coverage[chrom].keys())
                if len(samples) > max_samples:
                    samples = samples[:max_samples]
                props = []
                for s in samples:
                    tot = total_bp[chrom].get(s, 1)
                    anc_bp = coverage[chrom][s].get(a, 0)
                    props.append(anc_bp / tot)
                if props:
                    box_data.append(props)
                    chrom_labels.append(chrom.replace("chr", ""))

            bp = ax.boxplot(
                box_data, tick_labels=chrom_labels, patch_artist=True,
                showfliers=False, widths=0.6,
                medianprops=dict(color="black", linewidth=1.5),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[a])
                patch.set_alpha(0.7)

            # Genome-wide mean
            all_props = [p for bd in box_data for p in bd]
            if all_props:
                ax.axhline(np.mean(all_props), color=colors[a], linestyle="--",
                           linewidth=1, alpha=0.5, label=f"mean={np.mean(all_props):.3f}")
                ax.legend(fontsize=7)

            ax.set_ylabel(f"{names[a]} Proportion")
            ax.set_title(f"Per-Chromosome {names[a]} Proportion Distribution")

        else:
            # All-ancestry grouped mode
            n_chroms = len(chroms)
            group_width = 0.8
            bar_width = group_width / n_anc
            positions_by_anc: dict[int, list[float]] = {a: [] for a in anc_list}
            box_data_by_anc: dict[int, list[list[float]]] = {a: [] for a in anc_list}
            chrom_centers = []

            for ci, chrom in enumerate(chroms):
                center = ci * (n_anc + 1)
                chrom_centers.append(center + (n_anc - 1) / 2 * bar_width)
                samples = list(coverage[chrom].keys())
                if len(samples) > max_samples:
                    samples = samples[:max_samples]

                for ai, a in enumerate(anc_list):
                    pos = center + ai * bar_width
                    positions_by_anc[a].append(pos)
                    props = []
                    for s in samples:
                        tot = total_bp[chrom].get(s, 1)
                        anc_bp = coverage[chrom][s].get(a, 0)
                        props.append(anc_bp / tot)
                    box_data_by_anc[a].append(props if props else [0])

            for a in anc_list:
                bp = ax.boxplot(
                    box_data_by_anc[a],
                    positions=positions_by_anc[a],
                    widths=bar_width * 0.8,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=1),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(colors[a])
                    patch.set_alpha(0.7)

            ax.set_xticks(chrom_centers)
            ax.set_xticklabels([c.replace("chr", "") for c in chroms], fontsize=7)
            ax.set_ylabel("Ancestry Proportion")
            ax.set_title("Per-Chromosome Ancestry Proportion Distribution")

            # Legend
            from matplotlib.patches import Rectangle
            legend_patches = [
                Rectangle((0, 0), 1, 1, facecolor=colors[a], alpha=0.7)
                for a in anc_list
            ]
            ax.legend(legend_patches, [names[a] for a in anc_list],
                      fontsize=7, ncol=min(n_anc, 4))

        ax.set_xlabel("Chromosome")
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
    return fig
