#!/usr/bin/env python3
"""Render the five PNGs the artifact schema demands under
``structural/``, ``hap_disagreement/``, and ``regional/``.

This is a thin downstream step from ``validate_per_site_metrics.py`` —
it reads the summary JSONs/TSVs that step writes and renders matplotlib
figures over already-aggregated arrays. No VCF access, no tract
streaming.

Outputs (paths under ``--out-root``):

  structural/tract_length_distribution.png
  structural/switch_rate_distribution.png
  structural/switch_rate_distribution_log.png
  hap_disagreement/by_rf_label.png
  regional/regional_qc_<chrom>.png   (one per chrom present in windows.tsv.gz)

Usage:
    python render_collector_pngs.py \\
        --out-root work/cluster_007/chr1/ \\
        [--region-mask-bed centromere.bed ...]
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_region_masks(beds: list[Path]) -> list[tuple[str, int, int, str]]:
    out: list[tuple[str, int, int, str]] = []
    for bed in beds:
        default_name = bed.stem
        with open(bed) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("track"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                chrom, s, e = parts[0], int(parts[1]), int(parts[2])
                name = parts[3] if len(parts) >= 4 else default_name
                out.append((chrom, s, e, name))
    return out


def render_tract_length(structural_dir: Path) -> None:
    """Per-ancestry mean tract length (Mb) bar chart with model_T_gen
    annotated. Loses the per-tract log-histogram detail of the old
    popout viz, but the summary JSON has the salient numbers."""
    summary_path = structural_dir / "tract_length_summary.json"
    if not summary_path.exists():
        return
    data = json.loads(summary_path.read_text())
    per_anc = data.get("per_ancestry", [])
    if not per_anc:
        return
    names = [a["name"] for a in per_anc]
    means = [a["mean_Mb"] for a in per_anc]
    medians = [a["median_Mb"] for a in per_anc]
    n_tracts = [a["n_tracts"] for a in per_anc]
    model_T = per_anc[0].get("model_T_gen")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x, means, color="#4477AA", alpha=0.8, label="mean")
    ax.scatter(x, medians, marker="_", color="black", s=120, linewidths=2, label="median")
    for i, n in enumerate(n_tracts):
        ax.text(x[i], means[i], f"n={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Tract length (Mb)")
    title = "Tract length distribution per ancestry"
    if model_T:
        title += f"  ·  FLARE model T_gen={model_T:.2f}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path = structural_dir / "tract_length_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_switch_rate(structural_dir: Path) -> None:
    summary_path = structural_dir / "switch_rate_summary.json"
    if not summary_path.exists():
        return
    data = json.loads(summary_path.read_text())
    hist = data.get("histogram", [])
    if not hist:
        return
    labels = [
        (f"{b['bin_lo']}-{b['bin_hi']-1}" if b['bin_hi'] - b['bin_lo'] > 1 else str(b['bin_lo']))
        for b in hist
    ]
    counts = [b["count"] for b in hist]
    n_haps = data.get("n_haplotypes", sum(counts))
    median = data.get("median", 0.0)

    for log_scale, suffix, ylabel in [
        (False, "switch_rate_distribution.png", "Count"),
        (True, "switch_rate_distribution_log.png", "Count (log)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        ax.bar(x, counts, color="#4477AA", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Ancestry switches per haplotype (bins)")
        ax.set_ylabel(ylabel)
        suffix_title = " — log scale" if log_scale else ""
        ax.set_title(f"Switch rate distribution{suffix_title} "
                     f"(n={n_haps:,} haplotypes, median={median:.0f})")
        fig.tight_layout()
        out_path = structural_dir / suffix
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path}")


def render_hap_disagreement(hap_dir: Path) -> None:
    """Per-RF-label violin from per_sample.tsv (which has the raw
    per-sample disagree_frac — no information loss vs the old plot)."""
    per_sample = hap_dir / "per_sample.tsv"
    if not per_sample.exists():
        return
    by_label: dict[str, list[float]] = defaultdict(list)
    with open(per_sample) as f:
        header = f.readline().rstrip("\n").split("\t")
        lab_i = header.index("rf_hard_label")
        dis_i = header.index("disagreement_bp_frac")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            by_label[parts[lab_i]].append(float(parts[dis_i]))
    if not by_label:
        return

    label_order = sorted(by_label.keys(),
                         key=lambda k: (k == "mixed", k == "unjoined", k))
    data = [by_label[k] for k in label_order]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(range(1, len(label_order) + 1))
    ax.set_xticklabels([f"{k}\n(n={c})" for k, c in zip(label_order, counts)])
    ax.set_ylabel("Hap-disagreement bp fraction")
    ax.set_title("Hap-disagreement rate, by RF hard label")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_path = hap_dir / "by_rf_label.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_regional(regional_dir: Path, masks: list[tuple[str, int, int, str]],
                    fdr_q: float) -> None:
    """One manhattan-style figure per chrom in windows.tsv.gz."""
    windows_tsv = regional_dir / "windows.tsv.gz"
    if not windows_tsv.exists():
        return
    # Group rows by (chrom, ancestry_name); each row carries everything we need.
    by_chrom: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(windows_tsv, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        ci = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            row = {
                "chrom": parts[ci["chrom"]],
                "start": int(parts[ci["start"]]),
                "end": int(parts[ci["end"]]),
                "ancestry_name": parts[ci["ancestry_name"]],
                "z": float(parts[ci["z"]]),
                "p": float(parts[ci["p"]]),
                "q": float(parts[ci["q"]]),
                "mask_region": parts[ci["mask_region"]] if len(parts) > ci["mask_region"] else "",
            }
            by_chrom[row["chrom"]].append(row)
    for chrom, rows in by_chrom.items():
        ancestries = sorted({r["ancestry_name"] for r in rows})
        K = len(ancestries)
        fig, axs = plt.subplots(K, 1, figsize=(11, 1.5 * K + 0.5),
                                sharex=True, squeeze=False)
        for ai, an in enumerate(ancestries):
            ax = axs[ai, 0]
            ar = [r for r in rows if r["ancestry_name"] == an]
            xs = np.array([(r["start"] + r["end"]) / 2 / 1e6 for r in ar])
            ys = np.array([-np.log10(max(r["p"], 1e-300)) for r in ar])
            sig_mask = np.array([r["q"] < fdr_q for r in ar])
            ax.scatter(xs[~sig_mask], ys[~sig_mask], s=8, color="C0", alpha=0.6,
                       label="n.s.")
            ax.scatter(xs[sig_mask], ys[sig_mask], s=12, color="red",
                       label=f"q<{fdr_q}")
            for m_chrom, m_s, m_e, _ in masks:
                if m_chrom != chrom:
                    continue
                ax.axvspan(m_s / 1e6, m_e / 1e6, color="gray", alpha=0.15)
            ax.set_ylabel(f"{an}\n-log10 p")
            if ai == 0:
                ax.legend(fontsize=7, loc="upper right")
            if ai == K - 1:
                ax.set_xlabel(f"{chrom} position (Mb)")
        fig.suptitle(
            f"Regional QC — {chrom}\n"
            f"per-window mean ancestry proportion z-tested vs chrom mean; "
            f"red = FDR-significant (q<{fdr_q}).",
            y=0.995, fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = regional_dir / f"regional_qc_{chrom}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-root", type=Path, required=True,
                   help="Artifact work root; expects "
                        "{structural,hap_disagreement,regional}/ populated")
    p.add_argument("--region-mask-bed", type=Path, action="append", default=[])
    p.add_argument("--fdr-q", type=float, default=0.05)
    args = p.parse_args()

    structural_dir = args.out_root / "structural"
    hap_dir = args.out_root / "hap_disagreement"
    regional_dir = args.out_root / "regional"

    render_tract_length(structural_dir)
    render_switch_rate(structural_dir)
    render_hap_disagreement(hap_dir)
    masks = _load_region_masks(args.region_mask_bed)
    render_regional(regional_dir, masks, args.fdr_q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
