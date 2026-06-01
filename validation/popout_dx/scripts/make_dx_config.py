#!/usr/bin/env python3
"""Write a popout DX YAML config from CLI flags.

Local-side helper. Lets you build the config without hand-editing YAML
and validates the tool subset before writing.

Example
-------
::

    python make_dx_config.py \\
      --run-name popout_dx_aou_v9_chr1 \\
      --tools popout,flare,rye,rf \\
      --popout-run-dir gs://.../popout-runs/aou_v9/2026-05-15/ \\
      --flare-cohort-bundle gs://.../cohort_bundle.flare_validate_chr1.v3.0.0.tar.gz \\
      --rye-q gs://.../aou_admixture_estimates_rye_pruned_v9.Q \\
      --rf-ancestry gs://.../foxtrot_v4.ancestry_preds.tsv \\
      --clusters 'cluster_*' \\
      --chroms 'chr1' \\
      --out scripts/popout_dx_config.chr1_all.yaml

For local-mode runs add ``--mode global_local --flare-anc-vcf-root gs://...``
and (optionally) ``--local-per-bucket-n``, ``--local-threshold``,
``--local-chroms``, ``--local-coarse-grids-mb``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Validate against the same vocabulary discover_runs enforces.
from validation.popout_dx.scripts.discover_runs import ALL_TOOLS, ANCHOR_TOOL


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"make_dx_config: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _yaml_quote(s: str) -> str:
    """Single-quote a YAML scalar so embedded ``:`` / ``$`` are safe."""
    if "'" in s:
        return '"' + s.replace('"', r"\"") + '"'
    return f"'{s}'"


def _emit_glob_list(name: str, vals: list[str]) -> str:
    items = ", ".join(_yaml_quote(v) for v in vals)
    return f"{name}: [{items}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tools", required=True,
                    help="comma-separated subset of popout,flare,rye,rf (popout required)")
    ap.add_argument("--out", required=True, type=Path)

    ap.add_argument("--flare-cohort-bundle",
                    help="path to a FLARE-validate cohort_bundle tarball (required when 'flare' in --tools)")
    ap.add_argument("--flare-anc-vcf-root",
                    help="GCS prefix where FLARE pipeline emits per-cluster .anc.vcf.gz "
                         "(required when --mode global_local)")

    ap.add_argument("--rye-q",
                    help="path to Rye Q TSV (required when 'rye' in --tools)")
    ap.add_argument("--rf-ancestry",
                    help="path to foxtrot RF ancestry TSV (required when 'rf' in --tools)")

    ap.add_argument("--clusters", nargs="+", default=["cluster_*"],
                    help="glob(s) matched against the cohort bundle's per_cluster/ tree")
    ap.add_argument("--chroms", nargs="+", default=["chr*"],
                    help="glob(s) matched against the cohort bundle's per_cluster/<cid>/")

    ap.add_argument("--mode", choices=("global", "global_local"), default="global")
    ap.add_argument("--local-per-bucket-n", type=int, default=25)
    ap.add_argument("--local-threshold", type=float, default=0.80)
    ap.add_argument("--local-rng-seed", type=int, default=42)
    ap.add_argument("--local-chroms", nargs="+", default=["chr1"])
    ap.add_argument("--local-coarse-grids-mb", type=int, nargs="+",
                    default=[1, 2, 5, 10, 20])

    ap.add_argument("--schema-version", default="1.0.0")
    args = ap.parse_args()

    # Tool validation matches discover_runs.validate_config.
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    if ANCHOR_TOOL not in tools:
        die(f"--tools must include {ANCHOR_TOOL!r}")
    extras = [t for t in tools if t != ANCHOR_TOOL]
    if not extras:
        die("--tools must include at least one comparison tool besides popout")
    for t in tools:
        if t not in ALL_TOOLS:
            die(f"unknown tool {t!r}; allowed: {ALL_TOOLS}")

    # Per-tool path requirements.
    if "flare" in tools and not args.flare_cohort_bundle:
        die("--flare-cohort-bundle is required when 'flare' is in --tools")
    if "rye" in tools and not args.rye_q:
        die("--rye-q is required when 'rye' is in --tools")
    if "rf" in tools and not args.rf_ancestry:
        die("--rf-ancestry is required when 'rf' is in --tools")
    if args.mode == "global_local" and not args.flare_anc_vcf_root:
        die("--flare-anc-vcf-root is required when --mode global_local")

    lines: list[str] = []
    lines.append(f"run_name: {_yaml_quote(args.run_name)}")
    lines.append(f"schema_version: {_yaml_quote(args.schema_version)}")
    lines.append(f"tools: [{', '.join(tools)}]")

    if "flare" in tools:
        lines.append("")
        lines.append("flare:")
        lines.append(f"  cohort_bundle: {_yaml_quote(args.flare_cohort_bundle)}")
        if args.flare_anc_vcf_root:
            lines.append(f"  anc_vcf_root: {_yaml_quote(args.flare_anc_vcf_root)}")
    if "rye" in tools:
        lines.append("")
        lines.append("rye:")
        lines.append(f"  q_path: {_yaml_quote(args.rye_q)}")
    if "rf" in tools:
        lines.append("")
        lines.append("rf:")
        lines.append(f"  ancestry_path: {_yaml_quote(args.rf_ancestry)}")

    lines.append("")
    lines.append(_emit_glob_list("clusters", args.clusters))
    lines.append(_emit_glob_list("chroms", args.chroms))

    if args.mode == "global_local":
        lines.append("")
        lines.append("local_sampling:")
        lines.append(f"  per_bucket_n: {args.local_per_bucket_n}")
        lines.append(f"  threshold:    {args.local_threshold}")
        lines.append(f"  rng_seed:     {args.local_rng_seed}")
        lines.append(f"  chroms: [{', '.join(_yaml_quote(c) for c in args.local_chroms)}]")
        lines.append(f"  coarse_grid_resolutions_mb: [{', '.join(str(x) for x in args.local_coarse_grids_mb)}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(
        f"make_dx_config: wrote {args.out} (tools={tools}, mode={args.mode}, "
        f"clusters={args.clusters}, chroms={args.chroms})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
