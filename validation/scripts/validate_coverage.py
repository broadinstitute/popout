#!/usr/bin/env python3
"""§8.1 output coverage + per-chromosome consistency checks.

Coverage:
  - Output sample set == input sample set (no silent sample drops).
  - Output site count == FLARE log `markers` (no silent site drops at write).

Per-chromosome consistency:
  - Global ancestry proportions stable across chroms.
  - On chr1-only data this is a degenerate single-row table; the script
    still lands the artifact so multi-chrom runs slot in unchanged.

Usage:
    python validate_coverage.py \\
        --global-tsv PATH/<prefix>.global.tsv \\
        --input-samples PATH/<input.vcf.gz | sample_list.txt> \\
        --qc-tsv PATH/<prefix>.qc.tsv \\
        --flare-log PATH/<prefix>.log \\
        --out-dir PATH/diagnostics
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "popout"))
from popout.viz._loaders import read_global_tsv


def read_qc_tsv(path: Path) -> dict[str, int]:
    """FLARE QC TSV is key<TAB>int per line."""
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split("\t")
            out[k] = int(v)
    return out


def load_input_samples(path: Path) -> list[str]:
    """Load input sample IDs from either a sample-list txt or a VCF header."""
    if path.suffix == ".gz" or path.suffix == ".vcf" or path.name.endswith(".vcf.gz"):
        import pysam
        vcf = pysam.VariantFile(str(path))
        return list(vcf.header.samples)
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def parse_log_markers(log_path: Path) -> int:
    """Pull the `markers` count from a FLARE log."""
    text = log_path.read_text()
    m = re.search(r"^\s*markers\s*:\s*(\d+)", text, re.MULTILINE)
    if m is None:
        raise RuntimeError(f"Could not find `markers` in FLARE log {log_path}")
    return int(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--global-tsv", type=Path, required=True)
    p.add_argument("--input-samples", type=Path, required=True,
                   help="Path to input VCF (.vcf.gz) or sample list (.txt, one per line)")
    # ★ v1.1: qc-tsv is optional; pre-pipeline test fixtures don't have one.
    # When omitted, qc-dependent checks emit SKIP rows instead of FAIL.
    p.add_argument("--qc-tsv", type=Path, default=None,
                   help="FLARE <prefix>.qc.tsv (★ v1.1 optional; SKIP when missing)")
    p.add_argument("--flare-log", type=Path, required=True,
                   help="FLARE <prefix>.log")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.global_tsv, args.input_samples, args.flare_log):
        if not path.exists():
            raise FileNotFoundError(path)
    have_qc = args.qc_tsv is not None and args.qc_tsv.exists()
    if args.qc_tsv is not None and not have_qc:
        raise FileNotFoundError(args.qc_tsv)

    print("Loading inputs...")
    ga = read_global_tsv(args.global_tsv)
    global_samples = set(ga.sample_names)
    input_samples = load_input_samples(args.input_samples)
    input_set = set(input_samples)
    qc: dict[str, int] = read_qc_tsv(args.qc_tsv) if have_qc else {}
    log_markers = parse_log_markers(args.flare_log)
    print(f"  global.tsv samples: {len(global_samples)}")
    print(f"  input samples:      {len(input_set)}")
    if have_qc:
        print(f"  qc gt_samples:      {qc.get('gt_samples')}")
        print(f"  qc out_samples:     {qc.get('out_samples')}")
        print(f"  qc out_records:     {qc.get('out_records')}")
    else:
        print(f"  qc.tsv:             NOT PROVIDED (qc-dependent checks will SKIP)")
    print(f"  log markers:        {log_markers}")

    # ── Coverage checks ────────────────────────────────────────────────
    checks: list[tuple[str, str, str]] = []  # (check, status, detail)

    # 1. sample-set equality
    only_in_input = input_set - global_samples
    only_in_global = global_samples - input_set
    if only_in_input or only_in_global:
        detail = f"input_only={len(only_in_input)} global_only={len(only_in_global)}"
        checks.append(("input_set_equals_output_set", "FAIL", detail))
        # Show up to 5 IDs from each side.
        if only_in_input:
            print(f"  FAIL: samples in input but not in global.tsv ({len(only_in_input)}):")
            for sid in list(only_in_input)[:5]:
                print(f"    {sid}")
        if only_in_global:
            print(f"  FAIL: samples in global.tsv but not in input ({len(only_in_global)}):")
            for sid in list(only_in_global)[:5]:
                print(f"    {sid}")
    else:
        checks.append(("input_set_equals_output_set", "PASS", f"n={len(global_samples)}"))

    # 2-4. qc.tsv-dependent checks. ★ v1.1: when qc.tsv isn't provided
    # (pre-pipeline test fixtures), emit SKIP rows instead of running the
    # check against missing data.
    if have_qc:
        # 2. qc.tsv sample-count consistency
        if qc.get("gt_samples") == qc.get("out_samples") == len(global_samples):
            checks.append(("qc_sample_count_consistent", "PASS",
                           f"gt={qc['gt_samples']} out={qc['out_samples']} global={len(global_samples)}"))
        else:
            checks.append(("qc_sample_count_consistent", "FAIL",
                           f"gt={qc.get('gt_samples')} out={qc.get('out_samples')} global={len(global_samples)}"))

        # 3. qc out_records vs log markers — no silent site drop at write.
        if qc.get("out_records") == log_markers:
            checks.append(("output_site_count_matches_log", "PASS",
                           f"out_records={qc['out_records']} log_markers={log_markers}"))
        else:
            checks.append(("output_site_count_matches_log", "FAIL",
                           f"out_records={qc.get('out_records')} log_markers={log_markers}"))

        # 4. Site-count plausibility (≥95% of *intersection*).
        ratio = (qc.get("out_records") or 0) / max(log_markers, 1)
        if ratio >= 0.95:
            checks.append(("site_coverage_ge_95pct_of_intersection", "PASS",
                           f"ratio={ratio:.4f}"))
        else:
            checks.append(("site_coverage_ge_95pct_of_intersection", "FAIL",
                           f"ratio={ratio:.4f}"))
    else:
        skip_detail = "qc.tsv not provided (pre-pipeline fixture)"
        checks.append(("qc_sample_count_consistent", "SKIP", skip_detail))
        checks.append(("output_site_count_matches_log", "SKIP", skip_detail))
        checks.append(("site_coverage_ge_95pct_of_intersection", "SKIP", skip_detail))

    out_check = args.out_dir / "coverage_check.tsv"
    with open(out_check, "w") as f:
        f.write("check\tstatus\tdetail\n")
        for c, s, d in checks:
            f.write(f"{c}\t{s}\t{d}\n")
    print(f"  wrote {out_check}")
    for c, s, d in checks:
        print(f"    [{s}] {c}  ({d})")

    n_fail = sum(1 for _, s, _ in checks if s == "FAIL")
    n_skip = sum(1 for _, s, _ in checks if s == "SKIP")

    # ── Per-chromosome consistency ────────────────────────────────────
    # v1_nc is chr1-only. The global.tsv schema has no chrom column
    # (one row per sample, ancestry proportions averaged over the genome).
    # On a single-chrom run, there is exactly one chrom-worth of evidence
    # and the consistency check is trivially a no-op. The artifact is
    # still written so multi-chrom merges of this file work unchanged.
    per_chrom_tsv = args.out_dir / "per_chrom_consistency.tsv"
    chroms = sorted(qc.keys() - {"gt_samples", "out_samples", "gt_records", "out_records"})
    # Pull out per-chrom out_records from qc.tsv.
    rows = []
    for k in chroms:
        if k.startswith("out_records."):
            chrom = k[len("out_records."):]
            n_out = qc[k]
            n_gt = qc.get(f"gt_records.{chrom}", 0)
            rows.append({"chrom": chrom, "gt_records": n_gt, "out_records": n_out})
    rows.sort(key=lambda r: r["chrom"])
    with open(per_chrom_tsv, "w") as f:
        f.write("chrom\tgt_records\tout_records\n")
        for r in rows:
            f.write(f"{r['chrom']}\t{r['gt_records']}\t{r['out_records']}\n")
    print(f"  wrote {per_chrom_tsv}  ({len(rows)} chromosomes)")

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(rows) == 0:
        ax.text(0.5, 0.5, "no per-chrom records in qc.tsv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    elif len(rows) == 1:
        ax.text(0.5, 0.5, f"single-chromosome run ({rows[0]['chrom']})\n"
                          f"out_records={rows[0]['out_records']:,}\n"
                          "across-chrom consistency deferred until multi-chrom run",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        # bp-level proportions are *not* in global.tsv (it averages over
        # the genome). For multi-chrom we'd want per-chrom averages from
        # per-chrom global files. For now, just plot out_records per chrom.
        chrom_names = [r["chrom"] for r in rows]
        out_vals = [r["out_records"] for r in rows]
        ax.bar(chrom_names, out_vals)
        ax.set_ylabel("out_records")
        ax.set_title("Per-chromosome output record count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out_png = args.out_dir / "per_chrom_consistency.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}")

    if n_fail:
        print(f"\nOVERALL: {n_fail} coverage check(s) FAILED ({n_skip} SKIP)")
        sys.exit(1)
    if n_skip:
        print(f"\nOVERALL: all non-skipped coverage checks PASS ({n_skip} SKIP)")
    else:
        print("\nOVERALL: all coverage checks PASS")


if __name__ == "__main__":
    main()
