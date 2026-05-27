#!/usr/bin/env python3
"""§8.1 hap-disagreement rate (project_summary.md §5.3).

For each sample, compute the bp-weighted fraction of the genome where
hap1 and hap2 are called as different ancestries. Stratify by RF hard
label (see diagnostics/GLOSSARY.md) and optionally by named genomic
region from a BED.

Usage:
    python validate_hap_disagreement.py \\
        --tracts PATH/<prefix>.tracts.tsv.gz \\
        --rf-ancestry PATH/foxtrot_v4.ancestry_preds.tsv \\
        --region-bed PATH/regions.bed \\
        --out-dir PATH/diagnostics
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "popout"))
from popout.viz._loaders import read_tracts


def load_rf_hard_labels(path: Path, *, mixed_threshold: float = 0.8) -> dict[str, str]:
    """Return sample_id -> RF hard label ('afr'/'eur'/.../'mixed')."""
    out: dict[str, str] = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        rid_col = header.index("research_id")
        pred_col = header.index("ancestry_pred")
        prob_col = header.index("probabilities")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rid = parts[rid_col]
            pred = parts[pred_col]
            probs = ast.literal_eval(parts[prob_col])
            max_p = max(probs)
            out[rid] = pred if max_p >= mixed_threshold else "mixed"
    return out


def load_region_bed(path: Path) -> dict[str, list[tuple[int, int, str]]]:
    """Return chrom -> sorted list of (start, end, name) intervals."""
    out: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                raise ValueError(
                    f"region BED must have 4 columns (chrom start end name); got: {line!r}"
                )
            chrom, start, end, name = parts[0], int(parts[1]), int(parts[2]), parts[3]
            out[chrom].append((start, end, name))
    for chrom in out:
        out[chrom].sort()
    return dict(out)


def merge_walk(
    h1_tracts: list[tuple[str, int, int, int]],
    h2_tracts: list[tuple[str, int, int, int]],
) -> list[tuple[str, int, int, int, int]]:
    """Yield (chrom, seg_start, seg_end, anc_h1, anc_h2) segments via merge walk.

    Tracts on each haplotype are assumed sorted by (chrom, start_bp) and
    contiguous within a chromosome. Tract intervals are inclusive-inclusive
    on bp (FLARE-style: start_bp and end_bp are both site positions).
    """
    segments: list[tuple[str, int, int, int, int]] = []
    chroms = sorted(set(t[0] for t in h1_tracts) | set(t[0] for t in h2_tracts))
    for chrom in chroms:
        h1c = [t for t in h1_tracts if t[0] == chrom]
        h2c = [t for t in h2_tracts if t[0] == chrom]
        if not h1c or not h2c:
            raise RuntimeError(
                f"chrom {chrom} present on only one haplotype "
                f"(h1={len(h1c)} h2={len(h2c)}); refusing to compare"
            )
        i = j = 0
        # Walk to find shared start.
        cur = max(h1c[0][1], h2c[0][1])
        while i < len(h1c) and j < len(h2c):
            a_chrom, a_start, a_end, a_anc = h1c[i]
            b_chrom, b_start, b_end, b_anc = h2c[j]
            seg_end = min(a_end, b_end)
            if cur <= seg_end:
                segments.append((chrom, cur, seg_end, a_anc, b_anc))
            cur = seg_end + 1
            if a_end <= b_end:
                i += 1
            if b_end <= a_end:
                j += 1
    return segments


def segment_bp_agreement(
    segments: list[tuple[str, int, int, int, int]],
    *,
    region: tuple[int, int] | None = None,
    region_chrom: str | None = None,
) -> tuple[int, int]:
    """Return (agree_bp, total_bp), optionally restricted to one region."""
    agree = total = 0
    for chrom, s, e, a, b in segments:
        if region is not None:
            if region_chrom is not None and chrom != region_chrom:
                continue
            s_use = max(s, region[0])
            e_use = min(e, region[1])
            if s_use > e_use:
                continue
        else:
            s_use, e_use = s, e
        length = e_use - s_use + 1
        total += length
        if a == b:
            agree += length
    return agree, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tracts", type=Path, required=True)
    p.add_argument("--rf-ancestry", type=Path, required=True,
                   help="RF ancestry predictions (foxtrot_v4.ancestry_preds.tsv: research_id, ancestry_pred, probabilities)")
    p.add_argument("--region-bed", type=Path, default=None,
                   help="Optional 4-col BED (chrom start end name) for per-region stratification")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.tracts.exists():
        raise FileNotFoundError(args.tracts)
    if not args.rf_ancestry.exists():
        raise FileNotFoundError(args.rf_ancestry)

    print(f"Loading RF hard labels from {args.rf_ancestry}")
    rf_hard_labels = load_rf_hard_labels(args.rf_ancestry)
    print(f"  {len(rf_hard_labels):,} samples with RF labels")

    regions: dict[str, list[tuple[int, int, str]]] = {}
    if args.region_bed is not None:
        print(f"Loading region BED from {args.region_bed}")
        regions = load_region_bed(args.region_bed)
        n_intervals = sum(len(v) for v in regions.values())
        print(f"  {n_intervals} intervals across {len(regions)} chromosomes")

    print(f"Streaming tracts from {args.tracts}")
    per_sample: dict[str, dict[int, list[tuple[str, int, int, int]]]] = defaultdict(
        lambda: {1: [], 2: []}
    )
    for t in read_tracts(args.tracts):
        per_sample[t.sample][t.haplotype].append((t.chrom, t.start_bp, t.end_bp, t.ancestry))
    print(f"  {len(per_sample):,} samples")

    # Per-sample whole-genome metrics.
    rows: list[dict] = []
    per_region_rows: list[dict] = []
    n_missing_label = 0
    for sid, haps in per_sample.items():
        h1 = sorted(haps[1])
        h2 = sorted(haps[2])
        if not h1 or not h2:
            raise RuntimeError(f"sample {sid} missing tracts on hap1 or hap2")
        segs = merge_walk(h1, h2)
        agree, total = segment_bp_agreement(segs)
        if total == 0:
            raise RuntimeError(f"sample {sid} has zero overlap bp; tracts malformed")
        # Dominant ancestry per hap (bp-weighted).
        bp_h1: dict[int, int] = defaultdict(int)
        bp_h2: dict[int, int] = defaultdict(int)
        for c, s, e, a in h1:
            bp_h1[a] += e - s + 1
        for c, s, e, a in h2:
            bp_h2[a] += e - s + 1
        dom_h1 = max(bp_h1, key=bp_h1.get)
        dom_h2 = max(bp_h2, key=bp_h2.get)
        label = rf_hard_labels.get(sid)
        if label is None:
            n_missing_label += 1
            label = "unjoined"
        rows.append({
            "sample_id": sid,
            "agreement_bp_frac": agree / total,
            "disagreement_bp_frac": 1.0 - agree / total,
            "total_bp": total,
            "dominant_anc_h1": dom_h1,
            "dominant_anc_h2": dom_h2,
            "rf_hard_label": label,
        })
        for chrom, ivs in regions.items():
            for rs, re_, rname in ivs:
                r_agree, r_total = segment_bp_agreement(
                    segs, region=(rs, re_), region_chrom=chrom
                )
                if r_total == 0:
                    continue
                per_region_rows.append({
                    "sample_id": sid,
                    "rf_hard_label": label,
                    "region": rname,
                    "chrom": chrom,
                    "start": rs,
                    "end": re_,
                    "agreement_bp_frac": r_agree / r_total,
                    "disagreement_bp_frac": 1.0 - r_agree / r_total,
                    "total_bp": r_total,
                })

    if n_missing_label:
        print(f"  WARN: {n_missing_label} samples without RF labels; bucketed as 'unjoined'")

    # Write per-sample TSV.
    out_tsv = args.out_dir / "hap_disagreement.per_sample.tsv"
    with open(out_tsv, "w") as f:
        cols = ["sample_id", "rf_hard_label", "agreement_bp_frac",
                "disagreement_bp_frac", "total_bp", "dominant_anc_h1", "dominant_anc_h2"]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {out_tsv}")

    # Per-RF-label violin plot.
    by_label: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_label[r["rf_hard_label"]].append(r["disagreement_bp_frac"])
    label_order = sorted(by_label.keys(), key=lambda k: (k == "mixed", k == "unjoined", k))
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [by_label[k] for k in label_order]
    counts = [len(d) for d in data]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(range(1, len(label_order) + 1))
    ax.set_xticklabels([f"{k}\n(n={c})" for k, c in zip(label_order, counts)])
    ax.set_ylabel("Hap-disagreement bp fraction")
    ax.set_title("§8.1 hap-disagreement rate, by RF hard label")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_png = args.out_dir / "hap_disagreement_by_rf_label.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}")
    cohort_mean = float(np.mean([r["disagreement_bp_frac"] for r in rows]))
    print(f"  cohort mean hap-disagreement: {cohort_mean:.4f}")
    per_rf_label = []
    for k in label_order:
        vals = by_label[k]
        if vals:
            print(f"    {k:>10}: n={len(vals):4d}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")
            per_rf_label.append({
                "rf_label": k,
                "n": int(len(vals)),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
            })

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "cohort_mean_disagreement": cohort_mean,
        "n_samples": int(len(rows)),
        "n_samples_unjoined": int(n_missing_label),
        "per_rf_label": per_rf_label,
    }, indent=2))
    print(f"  wrote {summary_path}")

    # Per-region plot (only if region BED supplied).
    if per_region_rows:
        out_region_tsv = args.out_dir / "hap_disagreement.per_region.tsv"
        with open(out_region_tsv, "w") as f:
            cols = ["sample_id", "rf_hard_label", "region", "chrom", "start", "end",
                    "agreement_bp_frac", "disagreement_bp_frac", "total_bp"]
            f.write("\t".join(cols) + "\n")
            for r in per_region_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"  wrote {out_region_tsv}")

        by_region: dict[str, list[float]] = defaultdict(list)
        for r in per_region_rows:
            by_region[r["region"]].append(r["disagreement_bp_frac"])
        region_order = sorted(by_region.keys())
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(region_order)), 5))
        region_means = [float(np.mean(by_region[k])) for k in region_order]
        region_counts = [len(by_region[k]) for k in region_order]
        ax.bar(region_order, region_means)
        ax.axhline(cohort_mean, color="red", linestyle="--", linewidth=1,
                   label=f"cohort mean ({cohort_mean:.4f})")
        for i, c in enumerate(region_counts):
            ax.text(i, region_means[i], f"n={c}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Mean hap-disagreement bp fraction")
        ax.set_title("§8.1 hap-disagreement rate, by named genomic region")
        ax.legend()
        ax.set_ylim(bottom=0)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        out_png = args.out_dir / "hap_disagreement_by_region.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
