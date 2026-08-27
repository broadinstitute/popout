#!/usr/bin/env python3
"""Discover FLARE-validate scatter inputs from a JSON config.

The v4.0.0 contract for ``flare_validate.wdl`` is a single
``File config_file`` (gs://, Cromwell-localized) — not the
``Array[FlareClusterRun]`` struct array that v1.0.0..v3.0.0 took. This
script opens the localized config and emits a headerless TSV that the
workflow consumes via ``read_tsv()``; each row becomes one scatter shard.

Outputs (all into ``--out-dir``):
  ``runs_manifest.tsv``                no header; column order == TSV_COLUMNS;
                                        one row per (cluster_id, chrom) shard.
                                        Cohort singletons (rf_ancestry, rye_q,
                                        etc.) repeat in every row so the
                                        WDL's scatter-time String -> File
                                        coercion auto-localizes them per shard.
  ``runs_manifest.tsv.with_header.tsv`` same data with a header row for humans
  ``runs_manifest.json``                rich audit JSON
  ``run_name.txt``                      single line; the cohort run name
  ``schema_version.txt``                must equal EXPECTED_SCHEMA_VERSION
  ``mid_rule.txt``                      collate-side MID rule
  ``panel_id.txt``                      may be empty
  ``collation_config_uri.txt``          may be empty
  ``previous_cohort_bundle_uri.txt``    may be empty

Fails fast — no fallbacks, no silent drops. Missing required keys, wrong
schema version, unsupported discovery modes, partial cluster_runs entries,
and empty-after-globbing all raise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import sys
from pathlib import Path
from typing import Iterable, NoReturn


EXPECTED_SCHEMA_VERSION = "7.0.0"

# Headerless TSV column order. Indexes are referenced verbatim from the WDL
# scatter body — keep them in sync with workflows/flare/wdl/flare_validate.wdl.
#
# v7 note: the trailing "required-non-empty" invariant (see history) does
# not apply to the gutted extractor pipeline, which only reads row[0..2]
# (cluster_id, chrom, anc_vcf) — all three are always non-empty, so row
# access never runs off the end of the array even if Cromwell strips the
# empty tail. The full 15-column TSV is still emitted for anything that
# still consumes the wider schema.
TSV_COLUMNS: tuple[str, ...] = (
    "cluster_id",               # 0
    "chrom",                    # 1
    "anc_vcf",                  # 2
    "global_anc",               # 3
    "flare_model",              # 4
    "flare_log",                # 5
    "flare_qc_tsv",             # 6  optional
    "input_vcf",                # 7
    "rye_q",                    # 8  optional
    "self_id",                  # 9  optional
    "popout_secondary_global",  # 10 optional
    "popout_secondary_labels",  # 11 optional
    "ref_panel",                # 12 optional
    "rf_ancestry",              # 13 cohort singleton — required non-empty
    "chrom_sizes",              # 14 cohort singleton — required non-empty (anchors the tail)
)

# Per-cluster_run fields the manifest-mode config must carry. v7 gutted
# the validation pipeline down to a single tract-events extractor, which
# only needs cluster_id / chrom / anc_vcf. The other fields (global_anc,
# flare_model, flare_log, input_vcf, flare_qc_tsv) remain accepted for
# forward-compat with existing configs but are no longer required, and if
# absent from the config they land as empty strings in the manifest TSV.
REQUIRED_CLUSTER_RUN_FIELDS: tuple[str, ...] = (
    "cluster_id", "chrom", "anc_vcf",
)
OPTIONAL_CLUSTER_RUN_FIELDS: tuple[str, ...] = (
    "global_anc", "flare_model", "flare_log", "input_vcf", "flare_qc_tsv",
)

# v7 collapses the shared block to fully optional. The old v6 pipeline
# required rf_ancestry + chrom_sizes for the compare_to_rf / regional
# steps, both of which have been retired.
SHARED_REQUIRED_NONEMPTY: tuple[str, ...] = ()
SHARED_OPTIONAL: tuple[str, ...] = (
    "rf_ancestry", "chrom_sizes",
    "rye_q", "self_id",
    "popout_secondary_global", "popout_secondary_labels",
    "ref_panel",
    "collation_config", "previous_cohort_bundle",
)

SUPPORTED_DISCOVERY_MODES = ("manifest",)


def die(msg: str) -> NoReturn:
    print(f"discover_runs: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── Config validation ────────────────────────────────────────────────────


def validate_config(cfg: dict) -> None:
    """Hard-fail on any malformed config. No fallbacks."""
    if not isinstance(cfg, dict):
        die("config root must be a mapping")

    for key in ("schema_version", "run_name", "discovery", "shared"):
        if key not in cfg:
            die(f"config missing required top-level key {key!r}")

    sv = cfg["schema_version"]
    if sv != EXPECTED_SCHEMA_VERSION:
        die(
            f"config.schema_version {sv!r} != expected {EXPECTED_SCHEMA_VERSION!r}; "
            f"regenerate the config with the v{EXPECTED_SCHEMA_VERSION} "
            f"writer (validation/make_flare_validate_config.py)"
        )

    if not isinstance(cfg["run_name"], str) or not cfg["run_name"]:
        die("config.run_name must be a non-empty string")

    disc = cfg["discovery"]
    if not isinstance(disc, dict):
        die("config.discovery must be a mapping")
    mode = disc.get("mode")
    if mode not in SUPPORTED_DISCOVERY_MODES:
        die(
            f"config.discovery.mode {mode!r} not in {SUPPORTED_DISCOVERY_MODES}; "
            f"walk-mode is not implemented in v{EXPECTED_SCHEMA_VERSION}"
        )

    cluster_runs = disc.get("cluster_runs")
    if not isinstance(cluster_runs, list) or not cluster_runs:
        die("config.discovery.cluster_runs must be a non-empty list (mode=manifest)")
    for i, cr in enumerate(cluster_runs):
        if not isinstance(cr, dict):
            die(f"config.discovery.cluster_runs[{i}] must be a mapping")
        for f in REQUIRED_CLUSTER_RUN_FIELDS:
            v = cr.get(f)
            if not isinstance(v, str) or not v:
                die(
                    f"config.discovery.cluster_runs[{i}] missing required field "
                    f"{f!r} (cluster_id={cr.get('cluster_id')!r}, "
                    f"chrom={cr.get('chrom')!r})"
                )
        for f in OPTIONAL_CLUSTER_RUN_FIELDS:
            if f in cr and not isinstance(cr[f], str):
                die(
                    f"config.discovery.cluster_runs[{i}] field {f!r} must be a "
                    f"string when present"
                )

    shared = cfg["shared"]
    if not isinstance(shared, dict):
        die("config.shared must be a mapping")
    for k in SHARED_REQUIRED_NONEMPTY:
        v = shared.get(k)
        if not isinstance(v, str) or not v:
            die(f"config.shared.{k} must be a non-empty string URI")
    for k in SHARED_OPTIONAL:
        if k in shared and not isinstance(shared[k], str):
            die(f"config.shared.{k} must be a string when present (empty string = absent)")

    for fld in ("clusters", "chroms"):
        if fld in cfg:
            v = cfg[fld]
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                die(f"config.{fld} must be a list of glob strings")

    if "panel_id" in cfg and not isinstance(cfg["panel_id"], str):
        die("config.panel_id must be a string when present")
    if "mid_rule" in cfg and not isinstance(cfg["mid_rule"], str):
        die("config.mid_rule must be a string when present")


# ── Glob filtering ───────────────────────────────────────────────────────


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


def apply_glob_filters(
    cluster_runs: list[dict],
    cluster_globs: list[str],
    chrom_globs: list[str],
) -> list[dict]:
    selected = [
        cr for cr in cluster_runs
        if _matches_any(cr["cluster_id"], cluster_globs)
        and _matches_any(cr["chrom"], chrom_globs)
    ]
    if not selected:
        die(
            f"clusters={cluster_globs} x chroms={chrom_globs} matched no entries "
            f"in config.discovery.cluster_runs ({len(cluster_runs)} candidates; "
            f"first few cluster_ids: "
            f"{sorted({c['cluster_id'] for c in cluster_runs})[:5]})"
        )
    return selected


# ── Manifest assembly ────────────────────────────────────────────────────


def discover_from_manifest(cfg: dict) -> list[dict]:
    """Project the config's cluster_runs into TSV row dicts, attaching the
    cohort-singleton URIs from config.shared.
    """
    shared = cfg["shared"]
    rows: list[dict] = []
    for cr in cfg["discovery"]["cluster_runs"]:
        row = {
            "cluster_id":              cr["cluster_id"],
            "chrom":                   cr["chrom"],
            "anc_vcf":                 cr["anc_vcf"],
            "global_anc":              cr.get("global_anc", "") or "",
            "flare_model":             cr.get("flare_model", "") or "",
            "flare_log":               cr.get("flare_log", "") or "",
            "flare_qc_tsv":            cr.get("flare_qc_tsv", "") or "",
            "input_vcf":               cr.get("input_vcf", "") or "",
            "rf_ancestry":             shared.get("rf_ancestry", "") or "",
            "chrom_sizes":             shared.get("chrom_sizes", "") or "",
            "rye_q":                   shared.get("rye_q", "") or "",
            "self_id":                 shared.get("self_id", "") or "",
            "popout_secondary_global": shared.get("popout_secondary_global", "") or "",
            "popout_secondary_labels": shared.get("popout_secondary_labels", "") or "",
            "ref_panel":               shared.get("ref_panel", "") or "",
        }
        rows.append(row)
    return rows


def build_manifest(cfg: dict) -> tuple[dict, list[dict]]:
    mode = cfg["discovery"]["mode"]
    cluster_globs: list[str] = cfg.get("clusters", ["cluster_*", "null_cluster_*"])
    chrom_globs: list[str] = cfg.get("chroms", ["chr*"])

    if mode == "manifest":
        all_rows = discover_from_manifest(cfg)
    else:
        die(f"unhandled discovery mode {mode!r} (validate_config should have caught this)")

    rows = apply_glob_filters(all_rows, cluster_globs, chrom_globs)
    rejected = len(all_rows) - len(rows)
    if rejected:
        print(
            f"discover_runs: glob filter selected {len(rows)}/{len(all_rows)} "
            f"(cluster, chrom) pairs ({rejected} rejected)",
            file=sys.stderr,
        )

    rows.sort(key=lambda r: (r["cluster_id"], r["chrom"]))

    manifest = {
        "schema_version": cfg["schema_version"],
        "run_name":       cfg["run_name"],
        "panel_id":       cfg.get("panel_id", ""),
        "mid_rule":       cfg.get("mid_rule", "none"),
        "discovery_mode": mode,
        "cluster_globs":  cluster_globs,
        "chrom_globs":    chrom_globs,
        "generated_at":   dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_runs":         len(rows),
        "cluster_ids":    sorted({r["cluster_id"] for r in rows}),
        "chroms":         sorted({r["chrom"] for r in rows}),
        "shared":         cfg["shared"],
        "runs":           rows,
    }
    return manifest, rows


# ── TSV emission ─────────────────────────────────────────────────────────


def write_tsv(rows: Iterable[dict], path: Path) -> None:
    """Write the scatter-row TSV.

    Two files are produced:
      ``<path>``                   no header, machine-consumable by WDL
                                   ``read_tsv()`` (each row -> one scatter shard)
      ``<path>.with_header.tsv``   same data with a header line for humans
    """
    rows = list(rows)
    with open(path, "w") as f:
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in TSV_COLUMNS) + "\n")
    with open(str(path) + ".with_header.tsv", "w") as f:
        f.write("\t".join(TSV_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in TSV_COLUMNS) + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="JSON config (Cromwell-localizes from gs:// at task entry)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output dir; all sidecar files land here")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = json.load(f)

    validate_config(cfg)
    manifest, rows = build_manifest(cfg)

    json_path = args.out_dir / "runs_manifest.json"
    tsv_path = args.out_dir / "runs_manifest.tsv"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    write_tsv(rows, tsv_path)

    shared = cfg["shared"]
    (args.out_dir / "run_name.txt").write_text(cfg["run_name"] + "\n")
    (args.out_dir / "schema_version.txt").write_text(cfg["schema_version"] + "\n")
    (args.out_dir / "mid_rule.txt").write_text((cfg.get("mid_rule") or "none") + "\n")
    (args.out_dir / "panel_id.txt").write_text((cfg.get("panel_id") or "") + "\n")
    (args.out_dir / "collation_config_uri.txt").write_text(
        (shared.get("collation_config") or "") + "\n")
    (args.out_dir / "previous_cohort_bundle_uri.txt").write_text(
        (shared.get("previous_cohort_bundle") or "") + "\n")

    print(
        f"discover_runs: wrote {json_path} and {tsv_path} ({len(rows)} runs, "
        f"mode={manifest['discovery_mode']}, run_name={manifest['run_name']!r})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
