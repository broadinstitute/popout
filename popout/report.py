"""Generate QC reports from popout summary statistics.

Usage:
    popout report --stats cohort.summary.json --out report/
    popout report --stats cohort.summary.json --out report/ --format pdf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def report_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate QC report from popout summary statistics",
    )
    parser.add_argument("--stats", required=True,
                        help="Path to .summary.json file")
    parser.add_argument("--out", required=True,
                        help="Output directory for plots")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png",
                        help="Plot format (default: png)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Plot DPI (default: 150)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib is required for report generation. "
                  "Install with: pip install matplotlib")
        raise SystemExit(1)

    with open(args.stats) as f:
        summary = json.load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format
    dpi = args.dpi

    plots_written = 0

    if summary.get("em_convergence"):
        plot_convergence(summary, out_dir, fmt, dpi)
        plots_written += 1

    if summary.get("spectral", {}).get("singular_values"):
        plot_eigenvalue_spectrum(summary, out_dir, fmt, dpi)
        plots_written += 1

    if summary.get("output", {}).get("genome_wide_ancestry_proportions"):
        plot_ancestry_proportions(summary, out_dir, fmt, dpi)
        plots_written += 1

    if summary.get("output", {}).get("tract_stats_by_ancestry"):
        plot_tract_lengths(summary, out_dir, fmt, dpi)
        plots_written += 1

    if summary.get("timing"):
        plot_timing_breakdown(summary, out_dir, fmt, dpi)
        plots_written += 1

    if summary.get("site_filter_funnel"):
        plot_site_filter_funnel(summary, out_dir, fmt, dpi)
        plots_written += 1

    log.info("Wrote %d plots to %s/", plots_written, out_dir)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_convergence(summary: dict, out_dir: Path, fmt: str, dpi: int) -> None:
    """EM convergence: max/mean delta_freq vs iteration."""
    import matplotlib.pyplot as plt

    records = summary["em_convergence"]
    if not records:
        return

    iterations = [r["iteration"] for r in records if "max_delta_freq" in r]
    max_deltas = [r["max_delta_freq"] for r in records if "max_delta_freq" in r]
    mean_deltas = [r["mean_delta_freq"] for r in records if "mean_delta_freq" in r]

    if not iterations:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(iterations, max_deltas, "o-", label="max Δ(freq)", color="#2196F3")
    if mean_deltas:
        ax.semilogy(iterations[:len(mean_deltas)], mean_deltas, "s--",
                     label="mean Δ(freq)", color="#FF9800")
    ax.axhline(1e-4, color="gray", linestyle=":", alpha=0.5, label="convergence threshold")
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Allele Frequency Change")
    ax.set_title("EM Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"convergence.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_eigenvalue_spectrum(summary: dict, out_dir: Path, fmt: str, dpi: int) -> None:
    """Singular value spectrum with annotated n_ancestries."""
    import matplotlib.pyplot as plt

    spec = summary.get("spectral", {})
    sv = spec.get("singular_values", [])
    n_anc = spec.get("n_ancestries")

    if not sv:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: singular values
    ax = axes[0]
    ax.bar(range(len(sv)), sv, color="#4CAF50", alpha=0.8)
    if n_anc is not None:
        ax.axvline(n_anc - 1.5, color="red", linestyle="--", label=f"A = {n_anc}")
        ax.legend()
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular Value")
    ax.set_title("PCA Eigenvalue Spectrum")

    # Right: gap ratios
    ratios = spec.get("gap_ratios", [])
    if ratios:
        ax = axes[1]
        ax.bar(range(len(ratios)), ratios, color="#FF5722", alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        if n_anc is not None:
            ax.axvline(n_anc - 1.5, color="red", linestyle="--", label=f"A = {n_anc}")
            ax.legend()
        ax.set_xlabel("Gap Index (i → i+1)")
        ax.set_ylabel("Ratio S[i] / S[i+1]")
        ax.set_title("Eigenvalue Gap Ratios")

    fig.tight_layout()
    fig.savefig(out_dir / f"eigenvalues.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_ancestry_proportions(summary: dict, out_dir: Path, fmt: str, dpi: int) -> None:
    """Genome-wide ancestry proportions stacked bar."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    props = summary.get("output", {}).get("genome_wide_ancestry_proportions", [])
    if not props:
        return

    n_anc = len(props)
    colors = cm.Set2(range(n_anc))

    fig, ax = plt.subplots(figsize=(6, 5))
    bottom = 0.0
    for a in range(n_anc):
        ax.bar(0, props[a], bottom=bottom, color=colors[a],
               label=f"Ancestry {a} ({props[a]:.1%})")
        bottom += props[a]
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title("Genome-Wide Ancestry Proportions")
    ax.set_xticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / f"ancestry_proportions.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_tract_lengths(summary: dict, out_dir: Path, fmt: str, dpi: int) -> None:
    """Tract length summary per ancestry."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    tract_stats = summary.get("output", {}).get("tract_stats_by_ancestry", {})
    if not tract_stats:
        return

    ancestries = sorted(tract_stats.keys(), key=lambda x: int(x) if x.isdigit() else 99)
    n_anc = len(ancestries)
    colors = cm.Set2(range(n_anc))

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = range(n_anc)
    medians = [tract_stats[a].get("median_sites", 0) for a in ancestries]
    means = [tract_stats[a].get("mean_sites", 0) for a in ancestries]
    p5s = [tract_stats[a].get("p5_sites", 0) for a in ancestries]
    p95s = [tract_stats[a].get("p95_sites", 0) for a in ancestries]

    # Error bars from p5 to p95
    lower_err = [m - p for m, p in zip(medians, p5s)]
    upper_err = [p - m for m, p in zip(medians, p95s)]

    ax.bar(x_pos, medians, color=colors[:n_anc], alpha=0.8,
           yerr=[lower_err, upper_err], capsize=5, ecolor="gray")
    ax.scatter(x_pos, means, color="black", marker="D", zorder=5, s=30, label="mean")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Ancestry {a}" for a in ancestries])
    ax.set_ylabel("Tract Length (sites)")
    ax.set_title("Ancestry Tract Length Distribution")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"tract_lengths.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_timing_breakdown(summary: dict, out_dir: Path, fmt: str, dpi: int) -> None:
    """Timing breakdown by pipeline stage."""
    import matplotlib.pyplot as plt

    timing = summary.get("timing", {})
    if not timing:
        return

    # Aggregate by category
    categories: dict[str, float] = {}
    for name, secs in timing.items():
        if name.startswith("io/"):
            categories["I/O"] = categories.get("I/O", 0) + secs
        elif name.startswith("spectral"):
            categories["Spectral Init"] = categories.get("Spectral Init", 0) + secs
        elif name.startswith("e_step"):
            categories["E-step (HMM)"] = categories.get("E-step (HMM)", 0) + secs
        elif name.startswith("m_step"):
            categories["M-step"] = categories.get("M-step", 0) + secs
        elif name.startswith("chrom/"):
            pass  # skip per-chrom totals (would double-count)
        else:
            categories["Other"] = categories.get("Other", 0) + secs

    if not categories:
        return

    names = list(categories.keys())
    values = list(categories.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Seconds")
    ax.set_title("Pipeline Timing Breakdown")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"timing.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_site_filter_funnel(summary: dict, out_dir: Path, fmt: str, dpi: int) -> None:
    """Site filter funnel showing survival at each stage."""
    import matplotlib.pyplot as plt

    funnel = summary.get("site_filter_funnel", {})
    if not funnel:
        return

    # Aggregate across chromosomes
    stages = ["sites_biallelic", "sites_after_thinning", "sites_after_maf_mac", "sites_final"]
    stage_labels = ["Biallelic SNPs", "After Thinning", "After MAF/MAC", "Final Sites"]

    totals = {}
    for chrom, data in funnel.items():
        for stage in stages:
            if stage in data:
                totals[stage] = totals.get(stage, 0) + data[stage]

    # Only plot stages we have
    present = [(label, totals[stage]) for stage, label in zip(stages, stage_labels)
               if stage in totals]
    if not present:
        return

    labels, values = zip(*present)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4CAF50", "#8BC34A", "#FFC107", "#2196F3"]
    bars = ax.barh(range(len(labels)), values,
                   color=colors[:len(labels)], alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Sites (genome-wide)")
    ax.set_title("Site Filter Funnel")
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"site_funnel.{fmt}", dpi=dpi)
    plt.close(fig)
