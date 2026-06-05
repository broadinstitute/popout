#!/usr/bin/env python3
"""Build the flare_validate.wdl v4.0.0 inputs pair from a completed
flare_pipeline miniwdl run on the synthetic_flare dataset.

Two-step flow:
  1. Run flare_pipeline against the synthetic dataset (see
     workflows/flare/tests/README.md). This produces per-(cluster, chrom)
     FLARE outputs in miniwdl's call tree.
  2. Run this script on the resulting run dir. It walks the call tree,
     pairs each (cluster_id, chrom) with its FLARE outputs + Stage A
     input VCF, and emits a v4.0.0 config JSON plus the 3-key inputs JSON
     the WDL consumes.

The validate WDL is then driven with:

    miniwdl run workflows/flare/wdl/flare_validate.wdl \\
        -i workflows/flare/tests/synthetic_flare_validate_inputs.json

The script keeps no state and is safe to re-run.

Usage:
    python workflows/flare/tests/make_synthetic_flare_validate_dataset.py \\
        --pipeline-run-dir /tmp/miniwdl_flare_smoke/<run-id> \\
        --rf-ancestry data/synthetic_flare/rf_ancestry.tsv \\
        --chrom-sizes validation/data/grch38.chrom.sizes \\
        --config-out workflows/flare/tests/synthetic_flare_validate_config.json \\
        --inputs-out workflows/flare/tests/synthetic_flare_validate_inputs.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# Filename pattern FLARE uses: <cluster>.<chrom>.<suffix>.
# `output_prefix = cluster_ids[c] + "." + chromosomes[ci]` in flare_pipeline.wdl.
FLARE_FILE_PATTERN = re.compile(
    r"^(?P<cluster>[A-Za-z0-9_]+)\.(?P<chrom>chr[0-9XYM]+)\."
    r"(?P<suffix>anc\.vcf\.gz|global\.anc\.gz|model|log|qc\.tsv)$"
)

# Stage A's per-cluster gt VCF lands at
# <run-dir>/call-export_cluster_vcfs/shard-*/.../call-plink2_export_clusters_task/...
# with filenames like <cluster_id>_sample_list.<chrom>.aou.v9.phased.vcf.gz
# in production, or <cluster_id>.<chrom>.*.vcf.gz in synthetic.
GT_VCF_PATTERN = re.compile(
    r"^(?P<cluster>[A-Za-z0-9_]+?)[._](sample_list\.)?(?P<chrom>chr[0-9XYM]+)\..*\.vcf\.gz$"
)


def walk_for_flare_outputs(pipeline_run_dir: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """Scan the miniwdl run dir for FLARE outputs.

    Returns {(cluster_id, chrom): {suffix: path}}.
    """
    by_pair: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    suffix_to_key = {
        "anc.vcf.gz":   "anc_vcf",
        "global.anc.gz": "global_anc",
        "model":        "flare_model",
        "log":          "flare_log",
        "qc.tsv":       "flare_qc_tsv",
    }
    for path in pipeline_run_dir.rglob("*"):
        if not path.is_file():
            continue
        m = FLARE_FILE_PATTERN.match(path.name)
        if not m:
            continue
        key = suffix_to_key.get(m.group("suffix"))
        if key is None:
            continue
        pair = (m.group("cluster"), m.group("chrom"))
        # Multiple call directories may have the same output (cached copies,
        # re-runs); keep the first hit. Miniwdl puts canonical outputs under
        # `out/` but FLARE-stage artifacts also live under call-<task>/
        # execution dirs.
        if key not in by_pair[pair]:
            by_pair[pair][key] = path
    return dict(by_pair)


def walk_for_input_vcfs(pipeline_run_dir: Path) -> dict[tuple[str, str], Path]:
    """Locate per-(cluster, chrom) gt VCFs from Stage A's export task."""
    by_pair: dict[tuple[str, str], Path] = {}
    # Search anywhere under call-export_cluster_vcfs.
    candidates: list[Path] = []
    for stage_a in pipeline_run_dir.rglob("call-export_cluster_vcfs"):
        candidates.extend(stage_a.rglob("*.vcf.gz"))
    if not candidates:
        # Synthetic-flare layout: Stage A outputs land in any call dir; widen.
        candidates = list(pipeline_run_dir.rglob("*.gt.vcf.gz"))
    for path in candidates:
        if path.name.endswith(".tbi"):
            continue
        m = GT_VCF_PATTERN.match(path.name)
        if not m:
            continue
        pair = (m.group("cluster"), m.group("chrom"))
        if pair not in by_pair:
            by_pair[pair] = path
    return by_pair


def pair_inputs(
    flare_outputs: dict[tuple[str, str], dict[str, Path]],
    gt_vcfs: dict[tuple[str, str], Path],
) -> tuple[list[dict[str, str]], list[str]]:
    """Pair FLARE outputs with the matching input VCF. Returns
    (cluster_runs, warnings).
    """
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    required = ("anc_vcf", "global_anc", "flare_model", "flare_log", "flare_qc_tsv")
    for pair, files in sorted(flare_outputs.items()):
        missing = [k for k in required if k not in files]
        if missing:
            warnings.append(f"{pair[0]}.{pair[1]}: missing FLARE outputs {missing}; skipping")
            continue
        gt_vcf = gt_vcfs.get(pair)
        if gt_vcf is None:
            warnings.append(f"{pair[0]}.{pair[1]}: no Stage A gt VCF found; skipping")
            continue
        rows.append({
            "cluster_id":   pair[0],
            "chrom":        pair[1],
            "anc_vcf":      str(files["anc_vcf"].resolve()),
            "global_anc":   str(files["global_anc"].resolve()),
            "flare_model":  str(files["flare_model"].resolve()),
            "flare_log":    str(files["flare_log"].resolve()),
            "flare_qc_tsv": str(files["flare_qc_tsv"].resolve()),
            "input_vcf":    str(gt_vcf.resolve()),
        })
    return rows, warnings


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pipeline-run-dir", type=Path, required=True,
                    help="Miniwdl run dir for the flare_pipeline test (e.g. /tmp/miniwdl_flare_smoke/<id>)")
    ap.add_argument("--rf-ancestry", type=Path, required=True,
                    help="RF ancestry predictions TSV (synthetic or real)")
    ap.add_argument("--chrom-sizes", type=Path, required=True)
    ap.add_argument("--config-out", type=Path, required=True,
                    help="Output path for the v4.0.0 config JSON the WDL discover_runs reads")
    ap.add_argument("--inputs-out", type=Path, required=True,
                    help="Output path for the 3-key Cromwell inputs JSON pointing at --config-out")
    ap.add_argument("--run-name", default="synthetic_flare_validate")
    ap.add_argument("--docker-image", default=None,
                    help="Optional override; otherwise the WDL's default lai-tools:latest is used")
    args = ap.parse_args()

    if not args.pipeline_run_dir.is_dir():
        sys.exit(f"error: not a directory: {args.pipeline_run_dir}")
    for p in (args.rf_ancestry, args.chrom_sizes):
        if not p.exists():
            sys.exit(f"error: {p} not found")

    print(f"Scanning {args.pipeline_run_dir} for FLARE outputs...", file=sys.stderr)
    flare_outputs = walk_for_flare_outputs(args.pipeline_run_dir)
    print(f"  found {len(flare_outputs)} (cluster, chrom) pair(s) with at least one FLARE output", file=sys.stderr)

    gt_vcfs = walk_for_input_vcfs(args.pipeline_run_dir)
    print(f"  found {len(gt_vcfs)} per-cluster gt VCF(s)", file=sys.stderr)

    rows, warnings = pair_inputs(flare_outputs, gt_vcfs)
    for w in warnings:
        print(f"  WARN: {w}", file=sys.stderr)
    if not rows:
        sys.exit("error: no complete (cluster, chrom) tuples discovered; check --pipeline-run-dir")

    cluster_runs = []
    for r in rows:
        cluster_runs.append({
            "cluster_id":   r["cluster_id"],
            "chrom":        r["chrom"],
            "anc_vcf":      r["anc_vcf"],
            "global_anc":   r["global_anc"],
            "flare_model":  r["flare_model"],
            "flare_log":    r["flare_log"],
            "flare_qc_tsv": r["flare_qc_tsv"],
            "input_vcf":    r["input_vcf"],
        })

    config: dict = {
        "schema_version": "4.0.0",
        "run_name":       args.run_name,
        "panel_id":       "",
        "mid_rule":       "none",
        "discovery": {
            "mode":         "manifest",
            "cluster_runs": cluster_runs,
        },
        "clusters": ["cluster_*", "null_cluster_*"],
        "chroms":   ["chr*"],
        "shared": {
            "rf_ancestry":             str(args.rf_ancestry.resolve()),
            "chrom_sizes":             str(args.chrom_sizes.resolve()),
            "rye_q":                   "",
            "self_id":                 "",
            "popout_secondary_global": "",
            "popout_secondary_labels": "",
            "ref_panel":               "",
            "collation_config":        "",
            "previous_cohort_bundle":  "",
        },
    }

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nwrote {args.config_out} ({len(cluster_runs)} cluster_run(s))", file=sys.stderr)

    inputs: dict = {
        "flare_validate.config_file": str(args.config_out.resolve()),
    }
    if args.docker_image:
        inputs["flare_validate.docker_image"] = args.docker_image

    args.inputs_out.parent.mkdir(parents=True, exist_ok=True)
    args.inputs_out.write_text(json.dumps(inputs, indent=2) + "\n")
    print(f"wrote {args.inputs_out}", file=sys.stderr)
    print(f"\nRun: miniwdl run workflows/flare/wdl/flare_validate.wdl -i {args.inputs_out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
