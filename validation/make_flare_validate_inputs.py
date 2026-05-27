#!/usr/bin/env python3
"""
make_flare_validate_inputs.py

Emit `flare_validate_inputs.json` for the `flare_validate` Terra workflow.

Two input modes:

  --manifest-tsv  Explicit (cluster_id, chrom, anc_vcf, global_anc, flare_model,
                  flare_log, flare_qc_tsv, input_vcf) rows. Most reliable;
                  produce the TSV from `gcloud storage ls` against your
                  flare_pipeline run dir.

  --cromwell-dir + --input-vcf-dir
                  Walk a Cromwell output tree. Less reliable across pipeline
                  versions; use --manifest-tsv when in doubt.

Singletons (rf_ancestry, chrom_sizes, etc.) live at well-known repo paths
and can be overridden via flags.

Usage:
    ./make_flare_validate_inputs.py \\
        --manifest-tsv cluster_runs.tsv \\
        --run-name flare_v9_chr1 \\
        --out flare_validate_inputs.json

The manifest TSV must have a header row with these REQUIRED columns (order arbitrary):
    cluster_id   chrom   anc_vcf   global_anc   flare_model
    flare_log    input_vcf   ref_vcf
Plus one OPTIONAL column:
    flare_qc_tsv     (★ v1.1: empty/missing exercises the SKIP-coverage path)
Additional columns are ignored. Lines starting with '#' are skipped.

When --validate is set, every gs:// path is checked via `gcloud storage ls`
in parallel.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


WORKFLOW_NAME = "flare_validate"

# NO bucket paths are baked into this tool. All cohort-specific paths
# (rye_q, rf_ancestry, the per-(cluster, chrom) FLARE outputs, etc.) MUST
# be supplied at invocation time and live in inputs.json — never in
# checked-in source. See my_notes/validation/ for the canonical Terra
# paths the operator pastes into their local manifest TSV.

# Default for the chrom sizes (small static reference file, safe to ship
# in the repo). Lives alongside this script under validation/data/.
VALIDATION_DIR = Path(__file__).resolve().parent
DEFAULT_CHROM_SIZES = VALIDATION_DIR / "data" / "grch38.chrom.sizes"

# ★ v1.1: manifest gains `ref_vcf` column (R6 audit) and `flare_qc_tsv` becomes
# optional (pre-pipeline fixtures don't have one). Required cols are those
# every (cluster, chrom) row must populate.
REQUIRED_COLS = (
    "cluster_id", "chrom", "anc_vcf", "global_anc", "flare_model",
    "flare_log", "input_vcf", "ref_vcf",
)
OPTIONAL_COLS = ("flare_qc_tsv",)
GCLOUD_CONFIG = "pmi-ops"


def read_manifest_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path) as f:
        # Skip leading comment lines.
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    fieldnames = reader.fieldnames or []
    missing = [c for c in REQUIRED_COLS if c not in fieldnames]
    if missing:
        raise RuntimeError(
            f"{path}: manifest missing required columns: {missing}\n"
            f"got: {fieldnames}"
        )
    for row in reader:
        if not row["cluster_id"].strip() or not row["chrom"].strip():
            continue
        out = {k: row[k].strip() for k in REQUIRED_COLS}
        # Optional cols pass through when populated; empty string when absent.
        for k in OPTIONAL_COLS:
            out[k] = (row[k].strip() if k in fieldnames and row.get(k) else "")
        rows.append(out)
    return rows


def walk_cromwell_outputs(
    cromwell_dir: str, input_vcf_dir: str,
) -> list[dict[str, str]]:
    """Discover (cluster, chrom) tuples by listing the Cromwell output tree.

    Best-effort: walks `<cromwell_dir>/**/call-flare_task/<cluster>.<chrom>.*`
    and pairs with `<input_vcf_dir>/<cluster>.<chrom>.*.vcf.gz`. If the layout
    differs, use --manifest-tsv instead.
    """
    raise NotImplementedError(
        "Cromwell directory walking not implemented in V1. "
        "Produce a manifest TSV with `gcloud storage ls` and use --manifest-tsv."
    )


def build_inputs(rows: list[dict[str, str]], args: argparse.Namespace) -> dict:
    """Assemble the JSON payload for flare_validate.wdl."""
    cluster_runs = []
    for r in rows:
        cr: dict[str, str] = {
            "cluster_id":  r["cluster_id"],
            "chrom":       r["chrom"],
            "anc_vcf":     r["anc_vcf"],
            "global_anc":  r["global_anc"],
            "flare_model": r["flare_model"],
            "flare_log":   r["flare_log"],
            "input_vcf":   r["input_vcf"],
            "ref_vcf":     r["ref_vcf"],   # ★ v1.1 required
        }
        # ★ v1.1: flare_qc_tsv is optional — omit when blank so the WDL sees
        # `File?` = unset and the orchestrator falls through to SKIP-mode.
        if r.get("flare_qc_tsv"):
            cr["flare_qc_tsv"] = r["flare_qc_tsv"]
        cluster_runs.append(cr)

    payload: dict = {
        f"{WORKFLOW_NAME}.cluster_runs":    cluster_runs,
        f"{WORKFLOW_NAME}.rf_ancestry":     str(args.rf_ancestry),
        f"{WORKFLOW_NAME}.chrom_sizes":     str(args.chrom_sizes),
        f"{WORKFLOW_NAME}.run_name":        args.run_name,
        f"{WORKFLOW_NAME}.schema_version":  args.schema_version,
    }
    # ★ v1.1: Rye is the canonical concordance source. Always emit unless the
    # user explicitly overrides with --rye-q "" (empty string skips).
    if args.rye_q is not None and str(args.rye_q):
        payload[f"{WORKFLOW_NAME}.rye_q"] = str(args.rye_q)
    if args.panel_id:
        payload[f"{WORKFLOW_NAME}.panel_id"] = args.panel_id
    if args.ref_panel:
        payload[f"{WORKFLOW_NAME}.ref_panel"] = str(args.ref_panel)
    if args.self_id:
        payload[f"{WORKFLOW_NAME}.self_id"] = str(args.self_id)
    if args.popout_secondary_global:
        payload[f"{WORKFLOW_NAME}.popout_secondary_global"] = str(args.popout_secondary_global)
    if args.popout_secondary_labels:
        payload[f"{WORKFLOW_NAME}.popout_secondary_labels"] = str(args.popout_secondary_labels)
    if args.collation_config:
        payload[f"{WORKFLOW_NAME}.collation_config"] = str(args.collation_config)
    if args.previous_cohort_bundle:
        payload[f"{WORKFLOW_NAME}.previous_cohort_bundle"] = str(args.previous_cohort_bundle)
    if args.docker_image:
        payload[f"{WORKFLOW_NAME}.docker_image"] = args.docker_image
    if args.wandb_api_key_env:
        payload[f"{WORKFLOW_NAME}.wandb_api_key"] = args.wandb_api_key_env
    return payload


# ─── Validation ───────────────────────────────────────────────────────────


def gcs_check(path: str) -> tuple[str, bool, str]:
    try:
        r = subprocess.run(
            ["gcloud", "storage", "ls", f"--configuration={GCLOUD_CONFIG}", path],
            capture_output=True, text=True, timeout=30,
        )
        return path, r.returncode == 0, (r.stderr.strip() or "ok")
    except FileNotFoundError:
        return path, False, "gcloud not on PATH"
    except subprocess.TimeoutExpired:
        return path, False, "timeout"


def validate_paths(paths: list[str], workers: int = 16) -> int:
    if not shutil.which("gcloud"):
        sys.exit("error: --validate requires gcloud on PATH")
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(gcs_check, p): p for p in paths}
        for fut in as_completed(futures):
            path, ok, msg = fut.result()
            status = "ok     " if ok else "MISSING"
            print(f"{status}  {path}  ({msg})", file=sys.stderr)
            if not ok:
                failures += 1
    return failures


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest-tsv", type=Path,
                    help="TSV with one row per (cluster_id, chrom). Required cols: "
                         "cluster_id, chrom, anc_vcf, global_anc, flare_model, flare_log, "
                         "input_vcf, ref_vcf. Optional col: flare_qc_tsv (★ v1.1: omit to "
                         "exercise SKIP-on-missing-qc for pre-pipeline fixtures).")
    ap.add_argument("--cromwell-dir", type=str,
                    help="(experimental) walk a Cromwell output tree")
    ap.add_argument("--input-vcf-dir", type=str,
                    help="(experimental) dir of per-cluster gt= VCFs, for --cromwell-dir")
    ap.add_argument("--run-name", required=True,
                    help="Magicwand run name; also tags the cohort bundle filename")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path")
    ap.add_argument("--rf-ancestry", required=True,
                    help="Path to RF ancestry predictions TSV (e.g. foxtrot_v4.ancestry_preds.tsv). "
                         "Supply at invocation time; no default — bucket paths must NOT be baked in.")
    ap.add_argument("--chrom-sizes", type=Path, default=DEFAULT_CHROM_SIZES)
    ap.add_argument("--panel-id", default="")
    ap.add_argument("--ref-panel", type=Path, default=None,
                    help="Ref-panel TSV; only used for sha256 in per-cluster manifest")
    # ★ v1.1: --rye-q replaces --admixture-q / --admixture-fam. Optional —
    # when omitted (or empty), the orchestrator skips compare_to_rye and the
    # concordance/rye_* file family is absent from the artifact.
    ap.add_argument("--rye-q", default="",
                    help="Path to Rye Q TSV. No default — supply at invocation time.")
    ap.add_argument("--self-id", type=Path, default=None)
    ap.add_argument("--popout-secondary-global", type=Path, default=None)
    ap.add_argument("--popout-secondary-labels", type=Path, default=None)
    ap.add_argument("--collation-config", type=Path, default=None)
    ap.add_argument("--previous-cohort-bundle", type=Path, default=None)
    ap.add_argument("--docker-image", default=None,
                    help="Override the WDL's default docker image (lai-tools:latest)")
    ap.add_argument("--wandb-api-key-env", default=None,
                    help="Pass an api key string (NOT recommended; prefer a Terra workspace secret)")
    ap.add_argument("--schema-version", default="1.0.0")
    ap.add_argument("--validate", action="store_true",
                    help="gcloud-storage-ls every gs:// path before writing")
    ap.add_argument("--skip-bucket-prefix", action="append", default=[],
                    help="When --validate is set, skip paths starting with this prefix "
                         "(workstation auth may not cover all buckets the WDL will read). "
                         "Repeatable.")
    args = ap.parse_args()

    if args.manifest_tsv:
        rows = read_manifest_tsv(args.manifest_tsv)
    elif args.cromwell_dir:
        rows = walk_cromwell_outputs(args.cromwell_dir, args.input_vcf_dir or "")
    else:
        ap.error("either --manifest-tsv or --cromwell-dir is required")
    if not rows:
        sys.exit("error: no (cluster_id, chrom) rows discovered")

    n_clusters = len({r["cluster_id"] for r in rows})
    n_chroms = len({r["chrom"] for r in rows})
    print(f"discovered {len(rows)} cluster_run(s) "
          f"({n_clusters} clusters × {n_chroms} chroms)", file=sys.stderr)

    payload = build_inputs(rows, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)

    # Validation: only gs:// paths (local paths are checked at WDL localization).
    if args.validate:
        all_paths: set[str] = set()
        for r in rows:
            for col in ("anc_vcf", "global_anc", "flare_model", "flare_log",
                        "input_vcf", "ref_vcf"):
                if r[col].startswith("gs://"):
                    all_paths.add(r[col])
            qc = r.get("flare_qc_tsv", "")
            if qc and qc.startswith("gs://"):
                all_paths.add(qc)
        for single in (args.rf_ancestry, args.chrom_sizes, args.ref_panel,
                       args.rye_q, args.self_id,
                       args.popout_secondary_global, args.popout_secondary_labels,
                       args.collation_config, args.previous_cohort_bundle):
            if single and str(single).startswith("gs://"):
                all_paths.add(str(single))
        # Workstation auth may not cover every bucket the WDL will access
        # at run time (some require Cromwell-side credentials). The caller
        # can pass --skip-bucket-prefix to whitelist patterns whose presence
        # cannot be verified locally.
        skipped = {p for p in all_paths
                   if any(p.startswith(pref) for pref in args.skip_bucket_prefix)}
        all_paths -= skipped
        print(f"\nvalidating {len(all_paths)} gs:// paths "
              f"({len(skipped)} skipped via --skip-bucket-prefix)",
              file=sys.stderr)
        nfail = validate_paths(sorted(all_paths))
        if nfail:
            print(f"\n{nfail} path(s) missing", file=sys.stderr)
            return 2
        print("\nall gs:// paths ok", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
