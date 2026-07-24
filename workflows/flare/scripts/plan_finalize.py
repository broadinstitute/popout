#!/usr/bin/env python3
"""
plan_finalize.py — preflight for flare_finalize.wdl

Parse a FLARE-run manifest TSV (same schema as
validation/make_flare_validate_config.py:read_manifest_tsv) and bucket the
per-(cluster, chrom) rows by chromosome, so the WDL can scatter over chroms
and merge across clusters within each chrom.

Emits, under --out-dir:
  - chroms.txt              chromosome names, one per line, sorted
                            lexicographically (chr1, chr10, chr11, ...) so
                            downstream deliverable ordering is deterministic
  - anc_lists/<chrom>.txt   one anc_vcf URI per line, cluster_id-sorted
  - global_lists/<chrom>.txt one global_anc URI per line, cluster_id-sorted
  - stats.json              summary counters

The lists are ordered by cluster_id (sorted) so the row order of the
per-chrom global TSV concat is stable across reruns.

Hard-fails if:
  - the manifest is missing required columns
  - any (cluster, chrom) pair appears more than once (would double-count in
    the merge)
  - a chromosome has zero rows (empty scatter would be silent no-op)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_COLS = (
    "cluster_id", "chrom", "anc_vcf", "global_anc",
)


def natural_chrom_key(chrom: str) -> tuple:
    """Sort chr1, chr2, ..., chr9, chr10, ..., chr22, chrX, chrY, chrM."""
    m = re.fullmatch(r"chr(\d+|X|Y|M|MT)", chrom)
    if not m:
        return (99, chrom)
    tok = m.group(1)
    if tok.isdigit():
        return (0, int(tok))
    return ({"X": 1, "Y": 2, "M": 3, "MT": 3}.get(tok, 99), tok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir",  required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "anc_lists").mkdir(exist_ok=True)
    (args.out_dir / "global_lists").mkdir(exist_ok=True)

    with open(args.manifest) as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    fieldnames = list(reader.fieldnames or [])
    missing = [c for c in REQUIRED_COLS if c not in fieldnames]
    if missing:
        raise SystemExit(
            f"{args.manifest}: missing required columns: {missing}\n"
            f"got: {fieldnames}"
        )

    # chrom -> list of (cluster_id, anc_vcf, global_anc)
    by_chrom: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for row in reader:
        cid  = (row.get("cluster_id") or "").strip()
        chrm = (row.get("chrom")      or "").strip()
        anc  = (row.get("anc_vcf")    or "").strip()
        glo  = (row.get("global_anc") or "").strip()
        if not cid or not chrm:
            continue
        if not anc or not glo:
            raise SystemExit(
                f"{args.manifest}: (cluster={cid!r}, chrom={chrm!r}) has empty "
                f"anc_vcf or global_anc; both are required for the merge"
            )
        key = (cid, chrm)
        if key in seen_pairs:
            raise SystemExit(
                f"{args.manifest}: duplicate row for cluster={cid!r} chrom={chrm!r}"
            )
        seen_pairs.add(key)
        by_chrom[chrm].append((cid, anc, glo))

    if not by_chrom:
        raise SystemExit(f"{args.manifest}: no valid rows found")

    chroms = sorted(by_chrom.keys(), key=natural_chrom_key)
    with open(args.out_dir / "chroms.txt", "w") as fh:
        for c in chroms:
            fh.write(c + "\n")

    # Emit per-chrom URI lists with a zero-padded numeric prefix so a
    # lexicographic glob() at the WDL level returns them in the same order
    # as chroms.txt (which uses natural chr1, chr2, ..., chr10 ordering).
    # WDL 1.0 has no `suffix()` builtin and no map(), so aligning arrays via
    # sorted-glob is the cleanest fixed-point pattern.
    for idx, c in enumerate(chroms):
        rows = sorted(by_chrom[c], key=lambda r: r[0])
        anc_path = args.out_dir / "anc_lists"    / f"{idx:04d}__{c}.txt"
        glo_path = args.out_dir / "global_lists" / f"{idx:04d}__{c}.txt"
        with open(anc_path, "w") as fh:
            for _, anc, _ in rows:
                fh.write(anc + "\n")
        with open(glo_path, "w") as fh:
            for _, _, glo in rows:
                fh.write(glo + "\n")

    stats = {
        "num_chroms":    len(chroms),
        "num_rows":      sum(len(v) for v in by_chrom.values()),
        "rows_per_chrom": {c: len(by_chrom[c]) for c in chroms},
    }
    with open(args.out_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"plan_finalize: {json.dumps({k: v for k, v in stats.items() if k != 'rows_per_chrom'}, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
