#!/usr/bin/env python3
"""
make_flare_validate_config.py

Emit the v4.0.0 inputs pair for the `flare_validate` Terra workflow:

  <out>.config.json      cohort manifest the WDL's `discover_runs` task opens
  <out>.inputs.json      the 3-key inputs JSON Cromwell consumes

The WDL takes a single `File config_file` (Cromwell localizes it from gs://);
all per-(cluster, chrom) URIs and all cohort-singleton URIs live inside the
config. See validation/scripts/discover_runs.py for the schema.

Usage:
    ./make_flare_validate_config.py \\
        --manifest-tsv cluster_runs.tsv \\
        --run-name flare_v9_chr1 \\
        --rf-ancestry gs://.../foxtrot_v4.ancestry_preds.tsv \\
        --rye-q       gs://.../aou_admixture_estimates_rye_pruned_v9.Q \\
        --chrom-sizes gs://.../grch38.chrom.sizes \\
        --config-out  gs-staged/flare_validate_config.flare_v9_chr1.json \\
        --inputs-out  scripts/flare_validate_inputs.flare_v9_chr1.json

The manifest TSV must have a header row with these REQUIRED columns (order
arbitrary):
    cluster_id   chrom   anc_vcf   global_anc   flare_model
    flare_log    input_vcf
Plus one OPTIONAL column:
    flare_qc_tsv     (empty/missing exercises the SKIP-coverage path)
Additional columns are ignored. Lines starting with '#' are skipped.

When --validate is set, every gs:// path is checked via `gcloud storage ls`
in parallel before either output file is written.
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
SCHEMA_VERSION = "5.0.0"

VALIDATION_DIR = Path(__file__).resolve().parent
DEFAULT_CHROM_SIZES = VALIDATION_DIR / "data" / "grch38.chrom.sizes"

REQUIRED_COLS = (
    "cluster_id", "chrom", "anc_vcf", "global_anc", "flare_model",
    "flare_log", "input_vcf",
)
OPTIONAL_COLS = ("flare_qc_tsv",)
GCLOUD_CONFIG = "pmi-ops"


def read_manifest_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path) as f:
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
        for k in OPTIONAL_COLS:
            out[k] = (row[k].strip() if k in fieldnames and row.get(k) else "")
        rows.append(out)
    return rows


def build_config(rows: list[dict[str, str]], args: argparse.Namespace) -> dict:
    """Assemble the v4.0.0 config JSON."""
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
        }
        if r.get("flare_qc_tsv"):
            cr["flare_qc_tsv"] = r["flare_qc_tsv"]
        cluster_runs.append(cr)

    shared: dict[str, str] = {
        "rf_ancestry": str(args.rf_ancestry),
        "chrom_sizes": str(args.chrom_sizes),
    }
    for k, v in (
        ("rye_q",                   args.rye_q),
        ("self_id",                 args.self_id),
        ("popout_secondary_global", args.popout_secondary_global),
        ("popout_secondary_labels", args.popout_secondary_labels),
        ("ref_panel",               args.ref_panel),
        ("collation_config",        args.collation_config),
        ("previous_cohort_bundle",  args.previous_cohort_bundle),
    ):
        shared[k] = str(v) if v else ""

    cfg: dict = {
        "schema_version": SCHEMA_VERSION,
        "run_name":       args.run_name,
        "panel_id":       args.panel_id or "",
        "mid_rule":       args.mid_rule,
        "discovery": {
            "mode":         "manifest",
            "cluster_runs": cluster_runs,
        },
        "clusters": args.cluster_globs,
        "chroms":   args.chrom_globs,
        "shared":   shared,
    }
    return cfg


def build_inputs(config_uri: str, docker_image: str, wandb_api_key: str) -> dict:
    """3-key WDL inputs JSON."""
    payload: dict = {
        f"{WORKFLOW_NAME}.config_file": config_uri,
    }
    if docker_image:
        payload[f"{WORKFLOW_NAME}.docker_image"] = docker_image
    if wandb_api_key:
        payload[f"{WORKFLOW_NAME}.wandb_api_key"] = wandb_api_key
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
    ap.add_argument("--manifest-tsv", type=Path, required=True,
                    help="TSV with one row per (cluster_id, chrom). Required cols: "
                         "cluster_id, chrom, anc_vcf, global_anc, flare_model, flare_log, "
                         "input_vcf. Optional col: flare_qc_tsv (omit to exercise SKIP).")
    ap.add_argument("--run-name", required=True,
                    help="Magicwand run name; also tags the cohort bundle filename")
    ap.add_argument("--config-out", type=Path, required=True,
                    help="Output path for the v4.0.0 config JSON the WDL discover_runs task reads")
    ap.add_argument("--inputs-out", type=Path, required=True,
                    help="Output path for the 3-key Cromwell inputs JSON pointing at --config-out")
    ap.add_argument("--config-uri", default=None,
                    help="gs:// URI the inputs JSON should reference for config_file. "
                         "Defaults to the local --config-out path; override when the file "
                         "will be uploaded to a different GCS location.")
    ap.add_argument("--rf-ancestry", required=True,
                    help="Path/URI to the RF ancestry predictions TSV "
                         "(e.g. foxtrot_v4.ancestry_preds.tsv). No default.")
    ap.add_argument("--chrom-sizes", type=Path, default=DEFAULT_CHROM_SIZES)
    ap.add_argument("--panel-id", default="")
    ap.add_argument("--mid-rule", default="none",
                    choices=("none", "drop", "fold_to_eur"),
                    help="Cohort-side MID-handling rule for confusion_rf.tsv "
                         "(propagated to collate_cohort via discover_runs).")
    ap.add_argument("--ref-panel", type=Path, default=None,
                    help="Ref-panel VCF; only used for sha256 in per-cluster manifest")
    ap.add_argument("--rye-q", default="",
                    help="Path/URI to Rye Q TSV. No default — supply at invocation time.")
    ap.add_argument("--self-id", type=Path, default=None)
    ap.add_argument("--popout-secondary-global", type=Path, default=None)
    ap.add_argument("--popout-secondary-labels", type=Path, default=None)
    ap.add_argument("--collation-config", type=Path, default=None)
    ap.add_argument("--previous-cohort-bundle", type=Path, default=None)
    ap.add_argument("--cluster-globs", nargs="+",
                    default=["cluster_*", "null_cluster_*"],
                    help="Glob patterns applied to cluster_id at discover time")
    ap.add_argument("--chrom-globs", nargs="+", default=["chr*"],
                    help="Glob patterns applied to chrom at discover time")
    ap.add_argument("--docker-image", default=None,
                    help="Override the WDL's default docker image (lai-tools:latest)")
    ap.add_argument("--wandb-api-key-env", default=None,
                    help="Pass an api key string (NOT recommended; prefer a Terra workspace secret)")
    ap.add_argument("--validate", action="store_true",
                    help="gcloud-storage-ls every gs:// path before writing either output")
    ap.add_argument("--skip-bucket-prefix", action="append", default=[],
                    help="When --validate is set, skip paths starting with this prefix. Repeatable.")
    args = ap.parse_args()

    rows = read_manifest_tsv(args.manifest_tsv)
    if not rows:
        sys.exit("error: no (cluster_id, chrom) rows discovered")

    n_clusters = len({r["cluster_id"] for r in rows})
    n_chroms = len({r["chrom"] for r in rows})
    print(f"discovered {len(rows)} cluster_run(s) "
          f"({n_clusters} clusters x {n_chroms} chroms)", file=sys.stderr)

    if args.validate:
        all_paths: set[str] = set()
        for r in rows:
            for col in ("anc_vcf", "global_anc", "flare_model", "flare_log",
                        "input_vcf"):
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
        print("all gs:// paths ok\n", file=sys.stderr)

    config = build_config(rows, args)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {args.config_out}", file=sys.stderr)

    config_uri = args.config_uri or str(args.config_out)
    inputs = build_inputs(config_uri, args.docker_image or "",
                          args.wandb_api_key_env or "")
    args.inputs_out.parent.mkdir(parents=True, exist_ok=True)
    args.inputs_out.write_text(json.dumps(inputs, indent=2) + "\n")
    print(f"wrote {args.inputs_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
