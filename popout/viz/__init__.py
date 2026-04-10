"""Popout visualization suite.

Generate publication-quality and diagnostic plots from popout LAI results.

Usage (programmatic)::

    from popout.viz import plot_karyogram, plot_admixture, plot_tract_lengths
    fig = plot_karyogram("results/cohort", sample="NA12878")
    fig = plot_admixture("results/cohort")

Usage (CLI)::

    popout viz --prefix results/cohort --out figures/
"""

from __future__ import annotations

from .admixture import plot_admixture, plot_ancestry_density
from .convergence import plot_convergence
from .gallery import generate_gallery, viz_main
from .karyogram import plot_karyogram
from .posterior import plot_posterior_confidence
from .tracts import plot_tract_lengths, plot_switch_rate
from .genome import plot_ancestry_along_genome, plot_multi_individual
from .chromosome import plot_chromosome_boxplots
from .freq_divergence import plot_freq_divergence
from .per_hap_t import plot_per_hap_t
from .ternary import plot_ternary
from .deviation import plot_ancestry_deviation
from .label_correlation import plot_label_correlation

__all__ = [
    "plot_admixture",
    "plot_ancestry_density",
    "plot_ancestry_along_genome",
    "plot_ancestry_deviation",
    "plot_chromosome_boxplots",
    "plot_convergence",
    "plot_freq_divergence",
    "plot_karyogram",
    "plot_label_correlation",
    "plot_multi_individual",
    "plot_per_hap_t",
    "plot_posterior_confidence",
    "plot_switch_rate",
    "plot_ternary",
    "plot_tract_lengths",
    "generate_gallery",
    "viz_main",
]
