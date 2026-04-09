"""Gallery orchestrator and CLI entry point for popout viz.

Discovers available output files and generates all applicable plots.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ._loaders import discover_files, read_model_text, read_global_tsv

log = logging.getLogger(__name__)

# Map of plot name -> (required files, generator function name, module)
PLOT_REGISTRY = {
    "admixture": (["global_tsv"], "plot_admixture", "admixture"),
    "ancestry_density": (["global_tsv"], "plot_ancestry_density", "admixture"),
    "karyogram": (["tracts"], "plot_karyogram", "karyogram"),
    "tract_lengths": (["tracts"], "plot_tract_lengths", "tracts"),
    "switch_rate": (["tracts"], "plot_switch_rate", "tracts"),
    "ancestry_along_genome": (["tracts"], "plot_ancestry_along_genome", "genome"),
    "multi_individual": (["tracts"], "plot_multi_individual", "genome"),
    "convergence": ([], "plot_convergence", "convergence"),  # needs stats or summary
    "posterior": ([], "plot_posterior_confidence", "posterior"),  # gracefully degrades
    "chromosome_boxplots": (["tracts"], "plot_chromosome_boxplots", "chromosome"),
    "freq_divergence": (["model_npz"], "plot_freq_divergence", "freq_divergence"),
    "per_hap_t": (["model_npz"], "plot_per_hap_t", "per_hap_t"),
    "ternary": (["global_tsv"], "plot_ternary", "ternary"),
    "pca_ancestry": (["spectral_npz"], "plot_pca_ancestry", "spectral"),
    "seed_vs_final": (["spectral_npz", "global_tsv"], "plot_seed_vs_final", "spectral"),
}


def generate_gallery(
    prefix: str | Path,
    out_dir: str | Path,
    *,
    fmt: str = "png",
    dpi: int = 300,
    plots: list[str] | None = None,
    sample: str | None = None,
) -> list[Path]:
    """Generate all applicable plots for a popout run.

    Parameters
    ----------
    prefix : output prefix from the popout run
    out_dir : directory to write plots into
    fmt : image format (png, pdf, svg)
    dpi : resolution
    plots : optional list of plot names to generate (default: all applicable)
    sample : sample name for karyogram (required for karyogram, optional for others)

    Returns
    -------
    List of paths to generated plot files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = discover_files(prefix)
    log.info("Discovered files: %s", ", ".join(available.keys()))

    # Determine which plots to generate
    if plots is not None:
        requested = plots
    else:
        requested = list(PLOT_REGISTRY.keys())

    # Check if ternary is applicable (K=3 only)
    n_anc = None
    if "global_tsv" in available:
        try:
            data = read_global_tsv(available["global_tsv"])
            n_anc = data.n_ancestries
        except Exception:
            pass

    generated: list[Path] = []

    for name in requested:
        if name not in PLOT_REGISTRY:
            log.warning("Unknown plot: %s", name)
            continue

        required_files, func_name, module_name = PLOT_REGISTRY[name]

        # Check file requirements
        missing = [f for f in required_files if f not in available]
        if missing:
            log.info("Skipping %s: missing %s", name, ", ".join(missing))
            continue

        # Skip ternary if K != 3
        if name == "ternary" and n_anc is not None and n_anc != 3:
            log.info("Skipping ternary: n_ancestries=%d (need 3)", n_anc)
            continue

        # Skip karyogram if no sample specified
        if name == "karyogram" and sample is None:
            log.info("Skipping karyogram: no --sample specified")
            continue

        try:
            # Dynamic import
            mod = __import__(f"popout.viz.{module_name}", fromlist=[func_name])
            func = getattr(mod, func_name)

            # Call with appropriate args
            if name == "karyogram":
                fig = func(prefix, sample)
            else:
                fig = func(prefix)

            out_path = out_dir / f"{name}.{fmt}"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            generated.append(out_path)
            log.info("Generated: %s", out_path)
        except Exception as e:
            log.warning("Failed to generate %s: %s", name, e)

    log.info("Generated %d plots in %s/", len(generated), out_dir)
    return generated


def viz_main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``popout viz``."""
    parser = argparse.ArgumentParser(
        description="Generate visualization gallery from popout results",
    )
    parser.add_argument("--prefix", required=True,
                        help="Output prefix from popout run")
    parser.add_argument("--out", required=True,
                        help="Output directory for plots")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png",
                        help="Plot format (default: png)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Plot DPI (default: 300)")
    parser.add_argument("--sample", default=None,
                        help="Sample name for karyogram")
    parser.add_argument("--plots", default=None,
                        help="Comma-separated list of plots to generate "
                             "(default: all applicable)")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    plot_list = args.plots.split(",") if args.plots else None

    generate_gallery(
        args.prefix,
        args.out,
        fmt=args.format,
        dpi=args.dpi,
        plots=plot_list,
        sample=args.sample,
    )
