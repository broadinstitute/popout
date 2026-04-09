"""Karyogram / chromosome painting for a single individual.

Shows all 22 autosomes, two horizontal bars per chromosome (one per
haplotype), colored by ancestry. The signature LAI figure.
"""

from __future__ import annotations

from pathlib import Path

from ._style import (
    CHROM_ORDER, ancestry_colors, chrom_length, chrom_sort_key,
    normalize_chrom, popout_style,
)
from ._loaders import read_tracts, Tract


def plot_karyogram(
    prefix: str | Path,
    sample: str,
    *,
    title: str | None = None,
    n_ancestries: int | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> "matplotlib.figure.Figure":
    """Draw a karyogram for one individual.

    Parameters
    ----------
    prefix : path prefix (or direct path to tracts.tsv.gz)
    sample : sample name to plot
    title : optional figure title
    n_ancestries : if known, fixes the color palette size
    figsize : figure size in inches
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    prefix = Path(prefix)
    tracts_path = (
        prefix if prefix.suffix == ".gz"
        else prefix.with_name(prefix.name + ".tracts.tsv.gz")
    )

    # Collect tracts for this sample
    tracts_by_chrom: dict[str, list[Tract]] = {}
    max_ancestry = 0
    for t in read_tracts(tracts_path, sample=sample):
        c = normalize_chrom(t.chrom)
        tracts_by_chrom.setdefault(c, []).append(t)
        if t.ancestry > max_ancestry:
            max_ancestry = t.ancestry

    if not tracts_by_chrom:
        raise ValueError(f"No tracts found for sample {sample!r}")

    if n_ancestries is None:
        n_ancestries = max_ancestry + 1
    colors = ancestry_colors(n_ancestries)

    # Sort chromosomes
    chroms = sorted(tracts_by_chrom.keys(), key=chrom_sort_key)

    with popout_style():
        fig, ax = plt.subplots(figsize=figsize)

        bar_height = 0.35
        gap = 0.1
        row_height = 2 * bar_height + gap + 0.4
        max_bp = max(chrom_length(c) or max(t.end_bp for t in tracts_by_chrom[c])
                     for c in chroms)

        for row, chrom in enumerate(chroms):
            y_base = (len(chroms) - 1 - row) * row_height
            tracts = tracts_by_chrom[chrom]
            c_len = chrom_length(chrom) or max(t.end_bp for t in tracts)

            # Background chromosome outline
            for hap_offset in [0, bar_height + gap]:
                ax.add_patch(Rectangle(
                    (0, y_base + hap_offset), c_len, bar_height,
                    facecolor="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.5,
                ))

            # Ancestry tracts
            for t in tracts:
                hap_offset = 0 if t.haplotype == 0 else bar_height + gap
                width = t.end_bp - t.start_bp
                ax.add_patch(Rectangle(
                    (t.start_bp, y_base + hap_offset), width, bar_height,
                    facecolor=colors[t.ancestry % len(colors)],
                    edgecolor="none",
                ))

            # Chromosome label
            ax.text(
                -max_bp * 0.02,
                y_base + bar_height + gap / 2,
                chrom.replace("chr", ""),
                ha="right", va="center", fontsize=8, fontweight="bold",
            )

        ax.set_xlim(-max_bp * 0.06, max_bp * 1.02)
        ax.set_ylim(-0.3, len(chroms) * row_height)
        ax.set_xlabel("Genomic Position (bp)")
        ax.set_yticks([])

        if title is None:
            title = f"Ancestry Karyogram — {sample}"
        ax.set_title(title, fontsize=13, fontweight="bold")

        # Legend
        legend_patches = [
            Rectangle((0, 0), 1, 1, facecolor=colors[a])
            for a in range(n_ancestries)
        ]
        ax.legend(
            legend_patches,
            [f"Ancestry {a}" for a in range(n_ancestries)],
            loc="upper right", frameon=True, fontsize=8,
        )

        fig.tight_layout()
    return fig
