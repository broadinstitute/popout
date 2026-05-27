#!/usr/bin/env python3
"""§R6: ref/target site-concordance audit.

For one chromosome, count exact `(chrom, pos, ref, alt)` overlap between
the FLARE reference VCF (the one fed to `flare ref=`) and the per-cluster
input gt= VCF. Classify reference variants missing from the target into
three buckets: `absent_in_target`, `position_match_but_alleles_differ`,
`exact_match_found_on_reinspection`. Emit a single-row summary TSV plus
a JSON with the pass flag (threshold ≥ 94.5% per PLAN2.md §2.2).

Lifted verbatim from `my_notes/lk_notebooks/lk_compare_ref_alt_records.ipynb`
cells 13 (`count_exact_variant_overlap_safe`), 16 (`inspect_target_at_positions`),
19 (`classify_exact_misses`), 21 (`summarize_overlap`).

Multiallelic note (from audit): pysam does NOT auto-split multiallelics.
The FLARE ref VCF is biallelic-SNPs-only; the per-cluster gt VCF may have
multiallelics. We count exact `(chrom, pos, ref, alt)` matches as-is —
multiallelic split-state mismatches surface as
`position_match_but_alleles_differ`, which is the intended classification.

Usage:
    python validate_ref_target_concordance.py \\
        --ref-vcf PATH/chr1.gnomad_lai_90.vcf.bgz \\
        --input-vcf PATH/cluster_007.chr1.gt.vcf.gz \\
        --chrom chr1 \\
        --out-dir PATH/diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pysam


# ─── PLAN2.md §2.2 acceptance threshold ───────────────────────────────────

PASS_THRESHOLD_PCT = 94.5


def ensure_tbi(vcf_path: Path) -> None:
    """pysam.VariantFile.fetch() requires a .tbi sibling; create one if missing.

    Cromwell localizes File inputs without automatically pulling sibling
    .tbi files. tabix is cheap (10-30s for a chr1-sized .vcf.gz) so we
    generate the index in-place rather than failing and asking the
    operator to re-prep inputs.
    """
    tbi = vcf_path.with_name(vcf_path.name + ".tbi")
    if tbi.exists():
        return
    print(f"  no index at {tbi}; running pysam.tabix_index() in place...")
    pysam.tabix_index(str(vcf_path), preset="vcf", force=False)
    if not tbi.exists():
        raise RuntimeError(f"tabix_index returned but {tbi} still missing")


# ─── Lift: lk_compare_ref_alt_records.ipynb cell 13 ───────────────────────


def count_exact_variant_overlap_safe(reference_vcf_path, target_vcf_path, chromosome):
    """Memory-bounded overlap count between ref and target on one chrom.

    Original lift (cell 13) built BOTH ref_set and target_set in memory,
    then took intersection / difference. For chr1 production phased VCFs
    target_set can hold tens of millions of tuples (~2 GB) and the task
    gets OOM-killed.

    Equivalent algorithm with bounded memory:
      - Build ref_set once (~600k tuples on chr1 → ~50 MB).
      - Stream target VCF; for each (chrom, pos, ref, alt), increment a
        target_total counter and, if the tuple is in ref_set, add it to
        a matched set. Never materialize target_set.
      - missing = ref_set - matched.
    """
    ref_vcf = pysam.VariantFile(reference_vcf_path)
    tgt_vcf = pysam.VariantFile(target_vcf_path)

    ref_contigs = set(ref_vcf.header.contigs)
    tgt_contigs = set(tgt_vcf.header.contigs)

    if chromosome not in ref_contigs:
        raise ValueError(f"{chromosome} not found in reference VCF contigs.")
    if chromosome not in tgt_contigs:
        raise ValueError(f"{chromosome} not found in target VCF contigs.")

    reference_set: set[tuple[str, int, str, str]] = {
        (rec.chrom, rec.pos, rec.ref, alt)
        for rec in ref_vcf.fetch(chromosome)
        for alt in (rec.alts or [])
    }

    matched: set[tuple[str, int, str, str]] = set()
    target_total = 0
    for rec in tgt_vcf.fetch(chromosome):
        for alt in (rec.alts or []):
            target_total += 1
            t = (rec.chrom, rec.pos, rec.ref, alt)
            if t in reference_set:
                matched.add(t)

    missing = reference_set - matched

    summary = pd.DataFrame([{
        "chromosome": chromosome,
        "reference_total_records": len(reference_set),
        "target_total_records_on_chromosome": target_total,
        "overlap_count": len(matched),
        "percent_reference_found_in_target": (
            100 * len(matched) / len(reference_set) if len(reference_set) > 0 else 0
        ),
    }])

    missing_df = pd.DataFrame(sorted(missing), columns=["chrom", "pos", "ref", "alt"])
    matched_df = pd.DataFrame(sorted(matched), columns=["chrom", "pos", "ref", "alt"])

    return summary, matched_df, missing_df


# ─── Lift: lk_compare_ref_alt_records.ipynb cell 16 ───────────────────────


def inspect_target_at_positions(target_vcf_path, positions_df):
    """For each (chrom, pos) in positions_df, pull records from target VCF
    and report REF/ALT alleles present at that position."""
    vcf = pysam.VariantFile(target_vcf_path)
    results = []

    for _, row in positions_df.iterrows():
        chrom = row["chrom"]
        pos = int(row["pos"])
        found_any = False
        try:
            for rec in vcf.fetch(chrom, pos - 1, pos):  # pysam is 0-based half-open
                if rec.pos == pos:
                    found_any = True
                    for alt in rec.alts or []:
                        results.append({
                            "chrom": chrom,
                            "pos": pos,
                            "target_ref": rec.ref,
                            "target_alt": alt,
                        })
            if not found_any:
                results.append({
                    "chrom": chrom, "pos": pos,
                    "target_ref": None, "target_alt": None,
                })
        except ValueError:
            results.append({
                "chrom": chrom, "pos": pos,
                "target_ref": "CHROM_NOT_FOUND", "target_alt": None,
            })

    return pd.DataFrame(results)


# ─── Lift: lk_compare_ref_alt_records.ipynb cell 19 ───────────────────────


def classify_exact_misses(missing_df, inspection_df):
    out = []
    for _, row in missing_df.iterrows():
        chrom, pos, ref, alt = row["chrom"], row["pos"], row["ref"], row["alt"]
        subset = inspection_df[
            (inspection_df["chrom"] == chrom) & (inspection_df["pos"] == pos)
        ]
        if subset.empty or subset["target_ref"].isna().all():
            status = "absent_in_target"
        else:
            exact_match = (
                (subset["target_ref"] == ref) & (subset["target_alt"] == alt)
            ).any()
            status = "exact_match_found_on_reinspection" if exact_match else "position_match_but_alleles_differ"
        out.append({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt, "status": status})
    return pd.DataFrame(out)


# ─── Lift: lk_compare_ref_alt_records.ipynb cell 21 ───────────────────────


def summarize_overlap(reference_total, exact_overlap_count, classified_df):
    absent = int((classified_df["status"] == "absent_in_target").sum())
    pos_only = int((classified_df["status"] == "position_match_but_alleles_differ").sum())
    recovered_exact = int((classified_df["status"] == "exact_match_found_on_reinspection").sum())

    return pd.DataFrame([{
        "reference_total": reference_total,
        "exact_overlap": exact_overlap_count,
        "exact_overlap_pct": 100 * exact_overlap_count / reference_total,
        "position_match_but_alleles_differ": pos_only,
        "position_match_but_alleles_differ_pct": 100 * pos_only / reference_total,
        "absent_in_target": absent,
        "absent_in_target_pct": 100 * absent / reference_total,
        "reinspection_exact_match_found": recovered_exact,
    }])


# ─── Main: orchestration + schema-file emit ───────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref-vcf", type=Path, required=True,
                   help="FLARE reference VCF (e.g. final_filtered_reference.chr1.vcf.gz)")
    p.add_argument("--input-vcf", type=Path, required=True,
                   help="Per-cluster gt= VCF that fed FLARE")
    p.add_argument("--chrom", required=True,
                   help="Chromosome to audit (e.g. chr1)")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.ref_vcf, args.input_vcf):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"R6 ref/target concordance audit | chrom={args.chrom}")
    print(f"  ref:    {args.ref_vcf}")
    print(f"  target: {args.input_vcf}")

    # pysam.fetch needs .tbi siblings; regenerate in place if Cromwell didn't
    # localize them (which it never does for unspecified File companions).
    ensure_tbi(args.ref_vcf)
    ensure_tbi(args.input_vcf)

    overlap_summary, _matched_df, missing_df = count_exact_variant_overlap_safe(
        str(args.ref_vcf), str(args.input_vcf), args.chrom,
    )
    ref_total = int(overlap_summary["reference_total_records"].iloc[0])
    exact_overlap = int(overlap_summary["overlap_count"].iloc[0])
    print(f"  reference variants: {ref_total:,}")
    print(f"  exact overlap:      {exact_overlap:,} "
          f"({overlap_summary['percent_reference_found_in_target'].iloc[0]:.2f}%)")
    print(f"  missing:            {len(missing_df):,}")

    # Reinspect missing positions for split-multiallelic / allele-mismatch cases.
    if len(missing_df):
        inspection_df = inspect_target_at_positions(str(args.input_vcf), missing_df)
        classified_df = classify_exact_misses(missing_df, inspection_df)
    else:
        classified_df = pd.DataFrame(columns=["chrom", "pos", "ref", "alt", "status"])

    summary = summarize_overlap(ref_total, exact_overlap, classified_df)

    # ── Emit wide-form summary TSV (schema §1.13 provenance entry). ──
    out_tsv = args.out_dir / "ref_target_concordance.tsv"
    summary.insert(0, "chrom", args.chrom)
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"  wrote {out_tsv}")

    # ── Emit summary JSON with pass flag. ──
    pass_flag = bool(float(summary["exact_overlap_pct"].iloc[0]) >= PASS_THRESHOLD_PCT)
    summary_json = {
        "chrom": args.chrom,
        "reference_total": ref_total,
        "exact_overlap": exact_overlap,
        "exact_overlap_pct": float(summary["exact_overlap_pct"].iloc[0]),
        "position_match_but_alleles_differ_pct": float(
            summary["position_match_but_alleles_differ_pct"].iloc[0]
        ),
        "absent_in_target_pct": float(summary["absent_in_target_pct"].iloc[0]),
        "reinspection_exact_match_found": int(
            summary["reinspection_exact_match_found"].iloc[0]
        ),
        "pass_threshold_pct": PASS_THRESHOLD_PCT,
        "pass": pass_flag,
    }
    out_json = args.out_dir / "ref_target_concordance_summary.json"
    out_json.write_text(json.dumps(summary_json, indent=2))
    print(f"  wrote {out_json}")
    print(f"  PASS={pass_flag}  (threshold={PASS_THRESHOLD_PCT}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
