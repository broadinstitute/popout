#!/usr/bin/env python3
"""
emit_cleaned_manifest.py

Rewrite a FLARE-run manifest TSV so that rows whose cluster_id was affected
by the sample-removal cleanup point at the cleaned artifacts.

For each row whose cluster_id is in --affected-clusters, replace anc_vcf
and global_anc with the GCS URIs formed by joining --cleaned-prefix with
the original basename. Unaffected rows pass through byte-verbatim. Header
and all other columns are preserved.

The cleanup delocalization contract (encoded in flare_cleanup.wdl) is:

    <cleaned_prefix>/<cluster_id>/<original_basename>

so the rewrite is a mechanical substitution of the GCS path prefix. This
script does not touch GCS; it just rewrites strings.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import PurePosixPath


REQUIRED_COLS = (
    "cluster_id", "chrom", "anc_vcf", "global_anc", "flare_model",
    "flare_log", "input_vcf",
)


def rewrite_uri(original: str, cleaned_prefix: str, cluster_id: str) -> str:
    """
    Replace the parent-directory portion of `original` with
    <cleaned_prefix>/<cluster_id>/, keeping the original basename.
    """
    if not original:
        return original
    basename = PurePosixPath(original).name
    prefix = cleaned_prefix.rstrip("/")
    return f"{prefix}/{cluster_id}/{basename}"


def read_affected(path) -> set[str]:
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-in",       required=True)
    ap.add_argument("--manifest-out",      required=True)
    ap.add_argument("--affected-clusters", required=True,
                    help="file with one cluster_id per line (empty file OK)")
    ap.add_argument("--cleaned-prefix",    required=True,
                    help="e.g. gs://.../flare_cleanup_v1")
    args = ap.parse_args()

    affected = read_affected(args.affected_clusters)

    with open(args.manifest_in) as fin:
        lines = [ln for ln in fin if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    fieldnames = list(reader.fieldnames or [])
    missing = [c for c in REQUIRED_COLS if c not in fieldnames]
    if missing:
        raise SystemExit(
            f"{args.manifest_in}: missing required columns: {missing}\n"
            f"got: {fieldnames}"
        )

    rewritten_rows = 0
    total_rows = 0
    with open(args.manifest_out, "w") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames,
                           delimiter="\t", lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for row in reader:
            if not (row.get("cluster_id") or "").strip():
                continue
            total_rows += 1
            cid = row["cluster_id"].strip()
            out = {k: (row.get(k) or "") for k in fieldnames}
            if cid in affected:
                out["anc_vcf"]    = rewrite_uri(out["anc_vcf"],    args.cleaned_prefix, cid)
                out["global_anc"] = rewrite_uri(out["global_anc"], args.cleaned_prefix, cid)
                rewritten_rows += 1
            w.writerow(out)

    print(
        f"emit_cleaned_manifest: {args.manifest_out}: "
        f"rewrote {rewritten_rows}/{total_rows} rows "
        f"(affected clusters: {len(affected)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
