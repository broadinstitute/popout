#!/usr/bin/env python3
"""
plan_cleanup.py — bookkeeping preflight for flare_cleanup.wdl

Given:
  - a manifest TSV in the schema
    validation/make_flare_validate_config.py:read_manifest_tsv already consumes
    (required cols: cluster_id, chrom, anc_vcf, global_anc, flare_model,
    flare_log, input_vcf; optional: flare_qc_tsv)
  - one per-cluster sample-list file per unique cluster_id in the manifest
    (basename minus extension must equal the cluster_id; format is one IID
    per line, blank and '#' lines ignored — same convention plink2 uses)
  - a drop-samples file (one research ID per line, '#' comments allowed)

Emit:
  - cleanup_audit.tsv        cluster_id, sample_id, in_cluster (Y/N)
  - affected_clusters.tsv    cluster_ids with >=1 overlap (one per line)
  - drops_<cluster_id>.txt   per affected cluster, only samples present in
                             that cluster (bcftools view --samples-file input)
  - manifest_affected_rows.tsv  the subset of manifest rows (header preserved)
                                whose cluster_id is affected — used by the WDL
                                scatter
  - stats.json               summary counters for W&B / manifest audit

Hard-fails:
  - drop-list sample present in ZERO clusters (drift between drop list and
    the cohort partitioning; a caller silently ignoring it would corrupt the
    delivery)
  - manifest references a cluster_id for which no sample-list was supplied
  - a supplied sample-list has no matching cluster_id in the manifest (would
    be dead weight; likely operator error)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_COLS = (
    "cluster_id", "chrom", "anc_vcf", "global_anc", "flare_model",
    "flare_log", "input_vcf",
)
OPTIONAL_COLS = ("flare_qc_tsv",)


def read_id_list(path: Path) -> list[str]:
    """One ID per line, strip blanks and '#' comments. Preserves input order."""
    ids: list[str] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return (fieldnames, rows). Fieldnames preserved verbatim from the header."""
    with open(path) as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    fieldnames = list(reader.fieldnames or [])
    missing = [c for c in REQUIRED_COLS if c not in fieldnames]
    if missing:
        raise SystemExit(
            f"{path}: manifest missing required columns: {missing}\n"
            f"got: {fieldnames}"
        )
    rows = []
    for row in reader:
        if not row.get("cluster_id", "").strip():
            continue
        rows.append({k: (row.get(k) or "").strip() for k in fieldnames})
    return fieldnames, rows


def cluster_id_from_sample_list(path: Path) -> str:
    """Basename minus extension, matching flare_pipeline.wdl:46-48."""
    return path.name.split(".", 1)[0] if "." in path.name else path.stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",  required=True, type=Path)
    ap.add_argument("--drop-ids",  required=True, type=Path)
    ap.add_argument("--cluster-sample-list", dest="sample_lists",
                    action="append", required=True, type=Path,
                    help="repeat once per cluster; basename before '.' must "
                         "equal the cluster_id in the manifest")
    ap.add_argument("--out-dir",   required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_manifest(args.manifest)
    manifest_cluster_ids = sorted({r["cluster_id"] for r in rows})

    drops = read_id_list(args.drop_ids)
    if not drops:
        raise SystemExit(f"{args.drop_ids}: no drop IDs parsed; refusing to run")
    drop_set = set(drops)

    # cluster_id -> set of sample IDs in that cluster
    per_cluster: dict[str, set[str]] = {}
    for sl in args.sample_lists:
        cid = cluster_id_from_sample_list(sl)
        if cid in per_cluster:
            raise SystemExit(
                f"--cluster-sample-list gave duplicate cluster_id {cid!r} "
                f"(from {sl})"
            )
        per_cluster[cid] = set(read_id_list(sl))

    supplied = set(per_cluster.keys())
    manifest_set = set(manifest_cluster_ids)

    missing_lists = manifest_set - supplied
    if missing_lists:
        raise SystemExit(
            f"manifest references clusters with no sample-list supplied: "
            f"{sorted(missing_lists)}"
        )
    unused_lists = supplied - manifest_set
    if unused_lists:
        raise SystemExit(
            f"sample-lists supplied for clusters not in manifest "
            f"(likely operator error): {sorted(unused_lists)}"
        )

    # cluster_id -> [dropped sample IDs present in that cluster]
    per_cluster_drops: dict[str, list[str]] = defaultdict(list)
    # sample_id -> set of clusters it was found in
    sample_locations: dict[str, set[str]] = defaultdict(set)
    for cid, members in per_cluster.items():
        hits = drop_set & members
        for sid in sorted(hits):
            per_cluster_drops[cid].append(sid)
            sample_locations[sid].add(cid)

    # Preserve drop-list input order in the audit.
    audit_path = args.out_dir / "cleanup_audit.tsv"
    orphans: list[str] = []
    with open(audit_path, "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["sample_id", "in_cluster", "cluster_ids"])
        for sid in drops:
            locs = sorted(sample_locations.get(sid, set()))
            w.writerow([sid, "Y" if locs else "N", ",".join(locs)])
            if not locs:
                orphans.append(sid)

    if orphans:
        raise SystemExit(
            f"drop-list contains {len(orphans)} sample(s) not present in any "
            f"cluster: {orphans[:10]}{'...' if len(orphans) > 10 else ''}\n"
            "either the drop list is out of sync with the cohort partitioning "
            "or a sample-list is missing; refusing to proceed"
        )

    affected = sorted(per_cluster_drops.keys())
    with open(args.out_dir / "affected_clusters.tsv", "w") as fh:
        for cid in affected:
            fh.write(cid + "\n")

    for cid in affected:
        with open(args.out_dir / f"drops_{cid}.txt", "w") as fh:
            for sid in per_cluster_drops[cid]:
                fh.write(sid + "\n")

    # Manifest subset for the WDL scatter. Header preserved verbatim.
    affected_manifest = args.out_dir / "manifest_affected_rows.tsv"
    affected_set = set(affected)
    with open(affected_manifest, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames,
                           delimiter="\t", lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for row in rows:
            if row["cluster_id"] in affected_set:
                w.writerow(row)

    stats = {
        "num_clusters_total":     len(per_cluster),
        "num_clusters_affected":  len(affected),
        "num_drop_ids":           len(drops),
        "num_drops_matched":      len(drops) - len(orphans),
        "num_manifest_rows":      len(rows),
        "num_affected_rows":      sum(1 for r in rows if r["cluster_id"] in affected_set),
    }
    with open(args.out_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"plan_cleanup: {json.dumps(stats, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
