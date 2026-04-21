"""Build a markdown benchmark report with plots."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from popout.benchmark.align import align_haps, align_sites, apply_label_map, match_labels
from popout.benchmark.common import TractSet
from popout.benchmark.metrics import (
    compute_all_metrics,
    per_ancestry_precision_recall,
    tract_length_stats,
)


def build_report(
    tracts: dict[str, TractSet],
    truth: Optional[TractSet] = None,
    output_dir: str | Path = "benchmark_report",
    site_strategy: str = "intersect",
    label_overrides: Optional[dict[str, dict[int, int]]] = None,
) -> Path:
    """Build the full benchmark report.

    Parameters
    ----------
    tracts : dict of tool_name -> TractSet
    truth : optional ground-truth TractSet
    output_dir : directory to write report and plots to
    site_strategy : "intersect" or "project_a_onto_b"
    label_overrides : optional per-tool label mappings {tool: {src: ref}}

    Returns path to the markdown file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    label_overrides = label_overrides or {}

    # Determine reference for label matching
    ref_ts = truth
    if ref_ts is None:
        # Use the first reference-based tool as reference
        for name, ts in tracts.items():
            if name != "popout":
                ref_ts = ts
                break

    # Align and compute metrics for each tool pair
    all_metrics: dict[str, dict] = {}
    aligned_tracts: dict[str, TractSet] = {}

    for name, ts in tracts.items():
        if truth is not None:
            a, b = align_sites(ts, truth, strategy=site_strategy)
            a, b = align_haps(a, b)

            # Label matching for reference-free tools
            if name == "popout" or all(v.isdigit() for v in ts.label_map.values()):
                if name in label_overrides:
                    mapping = label_overrides[name]
                else:
                    mapping = match_labels(a, b)
                a = apply_label_map(a, mapping)

            metrics = compute_all_metrics(a, b, b_is_truth=True)
            all_metrics[name] = metrics
            aligned_tracts[name] = a
        else:
            aligned_tracts[name] = ts

    # Tool-to-tool comparisons (no truth)
    tool_names = list(tracts.keys())
    pairwise_metrics: dict[str, dict] = {}
    for i, name_a in enumerate(tool_names):
        for name_b in tool_names[i + 1:]:
            ts_a, ts_b = align_sites(tracts[name_a], tracts[name_b], strategy=site_strategy)
            ts_a, ts_b = align_haps(ts_a, ts_b)
            pair_key = f"{name_a}_vs_{name_b}"
            pairwise_metrics[pair_key] = compute_all_metrics(ts_a, ts_b, b_is_truth=False)

    # Build markdown
    lines = []
    lines.append("# LAI Benchmark Report\n")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Tools compared: {', '.join(tracts.keys())}")
    chrom = next(iter(tracts.values())).chrom
    lines.append(f"Chromosome: {chrom}")
    lines.append(f"Ground truth: {'yes' if truth is not None else 'no'}")
    lines.append("")

    # Summary table
    if truth is not None and all_metrics:
        lines.append("## Summary\n")
        lines.append("| Tool | Per-site acc vs truth | Mean r\u00b2 vs truth | Wall time | Peak RSS |")
        lines.append("|------|----------------------|------------------|-----------|----------|")
        for name, m in all_metrics.items():
            wt = tracts[name].metadata.get("wall_time", "-")
            mem = tracts[name].metadata.get("peak_rss", "-")
            lines.append(
                f"| {name} | {m['per_site_accuracy']:.3f} | "
                f"{m['mean_r2']:.3f} | {wt} | {mem} |"
            )
        lines.append("")

        # Per-ancestry r2
        lines.append("## Per-ancestry r\u00b2 against truth\n")
        anc_keys = list(next(iter(all_metrics.values()))["per_ancestry_r2"].keys())
        ref_label_map = truth.label_map
        header = "| Ancestry | " + " | ".join(all_metrics.keys()) + " |"
        sep = "|----------|" + "|".join(["-------"] * len(all_metrics)) + "|"
        lines.append(header)
        lines.append(sep)
        for k in anc_keys:
            name = ref_label_map.get(k, str(k))
            vals = " | ".join(
                f"{m['per_ancestry_r2'].get(k, float('nan')):.3f}"
                for m in all_metrics.values()
            )
            lines.append(f"| {name} | {vals} |")
        lines.append("")

        # Per-ancestry precision/recall
        lines.append("## Per-ancestry precision / recall against truth\n")
        for name, m in all_metrics.items():
            lines.append(f"### {name}\n")
            lines.append("| Ancestry | Precision | Recall |")
            lines.append("|----------|-----------|--------|")
            pr = m["per_ancestry_precision_recall"]
            for k in sorted(pr.keys()):
                anc_name = ref_label_map.get(k, str(k))
                lines.append(
                    f"| {anc_name} | {pr[k]['precision']:.3f} | {pr[k]['recall']:.3f} |"
                )
            lines.append("")

    # Tract length statistics
    lines.append("## Tract length statistics\n")
    for name, ts in tracts.items():
        stats = tract_length_stats(ts)
        lines.append(f"### {name}\n")
        if stats["count"] == 0:
            lines.append("No tracts.\n")
            continue
        lines.append(f"Total tracts: {stats['count']}\n")
        lines.append("| Metric | Sites | Base pairs |")
        lines.append("|--------|-------|------------|")
        for metric in ["min", "max", "mean", "median", "q25", "q75"]:
            lines.append(
                f"| {metric} | {stats['sites'][metric]:.0f} | "
                f"{stats['bp'][metric]:,.0f} |"
            )
        lines.append("")

    # Tool-to-tool agreement
    if pairwise_metrics:
        lines.append("## Tool-to-tool agreement\n")
        for pair_key, m in pairwise_metrics.items():
            lines.append(f"### {pair_key}\n")
            lines.append(f"- Per-site accuracy: {m['per_site_accuracy']:.3f}")
            lines.append(f"- Mean r\u00b2: {m['mean_r2']:.3f}")
            lines.append(f"- Mean global fraction error: {m['global_fraction_error_mean']:.4f}")
            lines.append("")

    # Plots
    try:
        _plot_ancestry_r2(all_metrics, truth, plots_dir)
        lines.append("## Plots\n")
        lines.append("![per-ancestry r\u00b2](plots/ancestry_r2.png)")
    except Exception:
        pass

    try:
        _plot_tract_lengths(tracts, plots_dir)
        lines.append("![tract length CDF](plots/tract_lengths.png)")
    except Exception:
        pass

    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _plot_ancestry_r2(
    all_metrics: dict[str, dict],
    truth: Optional[TractSet],
    plots_dir: Path,
) -> None:
    """Grouped bar chart of per-ancestry r²."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not all_metrics:
        return

    tool_names = list(all_metrics.keys())
    first = next(iter(all_metrics.values()))
    anc_keys = list(first["per_ancestry_r2"].keys())
    anc_names = [
        truth.label_map.get(k, str(k)) if truth else str(k) for k in anc_keys
    ]

    x = np.arange(len(anc_keys))
    width = 0.8 / len(tool_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, tool in enumerate(tool_names):
        vals = [all_metrics[tool]["per_ancestry_r2"].get(k, 0) for k in anc_keys]
        ax.bar(x + i * width, vals, width, label=tool)

    ax.set_xlabel("Ancestry")
    ax.set_ylabel("r\u00b2")
    ax.set_title("Per-ancestry r\u00b2 against truth")
    ax.set_xticks(x + width * (len(tool_names) - 1) / 2)
    ax.set_xticklabels(anc_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "ancestry_r2.png", dpi=150)
    plt.close(fig)


def _plot_tract_lengths(tracts: dict[str, TractSet], plots_dir: Path) -> None:
    """CDF of tract lengths per tool."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ts in tracts.items():
        tract_list = ts.to_tracts()
        if not tract_list:
            continue
        bp_lengths = []
        for _, start_idx, end_idx, _ in tract_list:
            bp_start = ts.site_positions[start_idx]
            bp_end = ts.site_positions[min(end_idx - 1, ts.n_sites - 1)]
            bp_lengths.append(bp_end - bp_start)
        bp_lengths = np.sort(bp_lengths)
        cdf = np.arange(1, len(bp_lengths) + 1) / len(bp_lengths)
        ax.plot(bp_lengths / 1e6, cdf, label=name)

    ax.set_xlabel("Tract length (Mb)")
    ax.set_ylabel("CDF")
    ax.set_title("Tract length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "tract_lengths.png", dpi=150)
    plt.close(fig)
