#!/usr/bin/env python3
"""popout DX cohort collator.

Gather step. Takes the per-cluster DX tarballs produced by every scatter
shard, untars each into a staging dir, and concatenates schema-shaped
TSVs into long-form cohort tables under ``cohort/`` with ``cluster_id``
and ``chrom`` columns prepended.

Output: ``cohort_dx.<run_name>.v<schema_version>.tar.gz`` containing the
``cohort_dx/`` tree spec'd in ``SCHEMA.md`` §2.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import sys
import tarfile
from pathlib import Path

from validation.popout_dx import schema as dx_schema


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"collate_dx: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── Per-cluster artifact extraction ─────────────────────────────────────


def extract_all(tarballs: list[Path], stage_root: Path) -> list[tuple[str, str, Path]]:
    """Untar every per-cluster artifact into ``stage_root``. Returns
    ``[(cluster_id, chrom, artifact_root), ...]``.
    """
    out: list[tuple[str, str, Path]] = []
    for tb in tarballs:
        dest = stage_root / tb.stem
        dest.mkdir(parents=True, exist_ok=True)
        artifact_root = dx_schema.read_cluster_artifact(tb, dest)
        cluster_id = artifact_root.parent.name
        chrom = artifact_root.name
        out.append((cluster_id, chrom, artifact_root))
    return sorted(out)


# ── Schema validation per artifact ──────────────────────────────────────


def validate_per_cluster(extracted: list[tuple[str, str, Path]]) -> None:
    n_err_total = 0
    for cluster_id, chrom, root in extracted:
        issues = dx_schema.validate_cluster_artifact(root)
        n_err = sum(1 for i in issues if i.severity == "error")
        if n_err:
            print(f"collate_dx: {cluster_id}/{chrom}: {n_err} schema error(s)",
                  file=sys.stderr)
            for iss in issues:
                if iss.severity == "error":
                    print(f"  {iss}", file=sys.stderr)
            n_err_total += n_err
    if n_err_total:
        die(f"per-cluster artifact validation failed ({n_err_total} total errors)")


# ── Long-form TSV concatenation ─────────────────────────────────────────


def _concat_tsv_with_prefix(
    sources: list[tuple[str, str, Path]],
    rel_path: str,
    out_path: Path,
    *,
    prefix_cols: tuple[str, ...] = ("cluster_id", "chrom"),
    optional: bool = False,
) -> int:
    """Concatenate the same per-cluster TSV across all clusters into a
    long-form output with ``prefix_cols`` (cluster_id, chrom) prepended.

    If a source is missing the TSV:
      * optional=True  → silently skip that source
      * optional=False → die

    Returns the total number of data rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    rows_out = 0
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
        for cluster_id, chrom, root in sources:
            src = root / rel_path
            if not src.exists():
                if optional:
                    continue
                die(f"{cluster_id}/{chrom}: missing required {rel_path}")
            with open(src) as fin:
                reader = csv.reader(fin, delimiter="\t")
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                if not header_written:
                    writer.writerow(list(prefix_cols) + header)
                    header_written = True
                for row in reader:
                    if not row:
                        continue
                    writer.writerow([cluster_id, chrom] + row)
                    rows_out += 1
    return rows_out


def _concat_tsv_gz_with_prefix(
    sources: list[tuple[str, str, Path]],
    rel_path: str,
    out_path: Path,
    *,
    optional: bool = False,
) -> int:
    """Same as ``_concat_tsv_with_prefix`` but for gzipped sources/output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    rows_out = 0
    with gzip.open(out_path, "wt", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
        for cluster_id, chrom, root in sources:
            src = root / rel_path
            if not src.exists():
                if optional:
                    continue
                die(f"{cluster_id}/{chrom}: missing required {rel_path}")
            opener = gzip.open if str(src).endswith(".gz") else open
            with opener(src, "rt") as fin:
                reader = csv.reader(fin, delimiter="\t")
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                if not header_written:
                    writer.writerow(["cluster_id", "chrom"] + header)
                    header_written = True
                for row in reader:
                    if not row:
                        continue
                    writer.writerow([cluster_id, chrom] + row)
                    rows_out += 1
    return rows_out


# ── Manifest unpivot ────────────────────────────────────────────────────


def write_manifest_tsv(
    sources: list[tuple[str, str, Path]], out_path: Path,
) -> None:
    """Per-shard manifest summary: one row per (cluster_id, chrom) with the
    headline fields from each manifest.json.
    """
    cols = ["cluster_id", "chrom", "mode", "tools",
            "n_samples", "n_ancestries_popout",
            "total_wallclock_seconds", "peak_rss_gb"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for cluster_id, chrom, root in sources:
            m = json.loads((root / "manifest.json").read_text())
            f.write("\t".join([
                cluster_id, chrom,
                str(m.get("mode", "")),
                ",".join(m.get("tools", [])),
                str(m.get("n_samples", "")),
                str(m.get("n_ancestries_popout", "")),
                str(m.get("total_wallclock_seconds", "")),
                str(m.get("peak_rss_gb", "")),
            ]) + "\n")


def write_tier1_long(
    sources: list[tuple[str, str, Path]], out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("cluster_id\tchrom\tkey\tvalue\n")
        for cluster_id, chrom, root in sources:
            t1 = root / "tier1_metrics.tsv"
            if not t1.exists():
                die(f"{cluster_id}/{chrom}: missing tier1_metrics.tsv")
            for line in t1.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                f.write(f"{cluster_id}\t{chrom}\t{parts[0]}\t{parts[1]}\n")


def write_pairwise_soft_summary_long(
    sources: list[tuple[str, str, Path]], out_path: Path,
) -> None:
    """Unpivot every per-cluster summary.json:pairs list into one long-form
    TSV: cluster_id, chrom, tool, rf_label, popout_mu, pearson_r, ccc,
    mae_mean, pass.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cluster_id", "chrom", "tool", "rf_label",
            "popout_mu", "pearson_r", "ccc", "mae_mean", "pass"]

    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    with open(out_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for cluster_id, chrom, root in sources:
            sj = root / "global" / "pairwise_soft" / "summary.json"
            if not sj.exists():
                continue
            summary = json.loads(sj.read_text())
            for p in summary.get("pairs", []):
                f.write("\t".join([
                    cluster_id, chrom,
                    str(p.get("tool", "")),
                    str(p.get("rf_label", "")),
                    _fmt(p.get("popout_mu")),
                    _fmt(p.get("pearson_r")),
                    _fmt(p.get("ccc")),
                    _fmt(p.get("mae_mean")),
                    _fmt(p.get("pass")),
                ]) + "\n")


# ── Cohort summary computation ──────────────────────────────────────────


def compute_cohort_summary(
    sources: list[tuple[str, str, Path]],
    tools: list[str],
    mode: str,
) -> dict:
    """Aggregate per-cluster pairwise_soft summaries into cohort means."""
    # Per (tool, rf_label): list of ccc values across clusters; pass counts.
    per_pair: dict[tuple[str, str], dict] = {}
    for cluster_id, chrom, root in sources:
        sj = root / "global" / "pairwise_soft" / "summary.json"
        if not sj.exists():
            continue
        summary = json.loads(sj.read_text())
        for p in summary.get("pairs", []):
            key = (p.get("tool", ""), p.get("rf_label", ""))
            acc = per_pair.setdefault(key, {"ccc": [], "pearson_r": [],
                                            "passing": 0, "failing": 0, "null": 0})
            ccc = p.get("ccc")
            if ccc is not None:
                acc["ccc"].append(ccc)
            r = p.get("pearson_r")
            if r is not None:
                acc["pearson_r"].append(r)
            verdict = p.get("pass")
            if verdict is True:
                acc["passing"] += 1
            elif verdict is False:
                acc["failing"] += 1
            else:
                acc["null"] += 1

    import math
    def _mean(xs: list) -> float | None:
        ys = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(sum(ys) / len(ys)) if ys else None

    pairs_summary: list[dict] = []
    for (tool, label), acc in sorted(per_pair.items()):
        denom = acc["passing"] + acc["failing"]
        pairs_summary.append({
            "tool": tool,
            "rf_label": label,
            "mean_ccc_across_clusters": _mean(acc["ccc"]),
            "mean_pearson_r_across_clusters": _mean(acc["pearson_r"]),
            "n_clusters_passing": acc["passing"],
            "n_clusters_failing": acc["failing"],
            "n_clusters_null": acc["null"],
            "fraction_clusters_passing": (acc["passing"] / denom) if denom > 0 else None,
        })

    out = {
        "n_clusters": len({cid for cid, _, _ in sources}),
        "n_chroms": len({chrom for _, chrom, _ in sources}),
        "n_artifacts": len(sources),
        "tools": tools,
        "mode": mode,
        "pairs": pairs_summary,
    }

    if mode == "global_local":
        bp_agree: list[float] = []
        cal_drift: list[float] = []
        for _, _, root in sources:
            sj = root / "local" / "local_summary.json"
            if not sj.exists():
                continue
            d = json.loads(sj.read_text())
            if d.get("bp_agreement") is not None:
                bp_agree.append(d["bp_agreement"])
            if d.get("calibration_drift_fraction") is not None:
                cal_drift.append(d["calibration_drift_fraction"])
        out["mean_bp_agreement"] = _mean(bp_agree)
        out["mean_calibration_drift_fraction"] = _mean(cal_drift)

    return out


# ── Cohort manifest + bundle tarball ────────────────────────────────────


def write_cohort_manifest(
    sources: list[tuple[str, str, Path]],
    out_path: Path,
    *,
    run_name: str,
    mode: str,
    tools: list[str],
    per_artifact_sha256: dict[str, str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": dx_schema.SCHEMA_VERSION,
        "run_name": run_name,
        "mode": mode,
        "tools": tools,
        "n_clusters": len({cid for cid, _, _ in sources}),
        "n_chroms": len({chrom for _, chrom, _ in sources}),
        "n_artifacts": len(sources),
        "cluster_ids": sorted({cid for cid, _, _ in sources}),
        "chroms": sorted({chrom for _, chrom, _ in sources}),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256_per_artifact": per_artifact_sha256,
    }
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")


def write_bundle_tarball(
    bundle_root: Path,
    out_tarball: Path,
    *,
    include_per_cluster: bool = False,
    per_cluster_sources: list[tuple[str, str, Path]] | None = None,
) -> None:
    """Tar ``bundle_root`` into ``out_tarball`` under prefix ``cohort_dx/``.
    Optionally include the unpacked per-cluster artifact trees.
    """
    out_tarball.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tarball, "w:gz") as tar:
        tar.add(bundle_root, arcname="cohort_dx")
        if include_per_cluster and per_cluster_sources:
            for cluster_id, chrom, root in per_cluster_sources:
                tar.add(root, arcname=f"cohort_dx/per_cluster/{cluster_id}/{chrom}")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tarballs", required=True, type=Path, nargs="+",
                    help="per-cluster DX tarballs to collate")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--mode", required=True, choices=("global", "global_local"))
    ap.add_argument("--tools", required=True,
                    help="comma-separated subset of popout,flare,rye,rf")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="staging dir for bundle assembly")
    ap.add_argument("--out-tarball", required=True, type=Path)
    ap.add_argument("--include-per-cluster", action="store_true",
                    help="embed unpacked per-cluster trees in the cohort bundle")
    args = ap.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    if not tools:
        die("--tools cannot be empty")

    stage = args.out_dir / "stage"
    bundle = args.out_dir / "cohort_dx"
    bundle.mkdir(parents=True, exist_ok=True)

    print(f"collate_dx: extracting {len(args.tarballs)} per-cluster tarball(s)",
          file=sys.stderr)
    sources = extract_all(args.tarballs, stage)
    validate_per_cluster(sources)

    cohort = bundle / "cohort"
    cohort.mkdir(parents=True, exist_ok=True)

    # Always-present tables.
    write_manifest_tsv(sources, cohort / "manifest.tsv")
    write_tier1_long(sources, cohort / "tier1_metrics.tsv")
    write_pairwise_soft_summary_long(sources, cohort / "pairwise_soft_summary.tsv")
    n_pm = _concat_tsv_with_prefix(
        sources, "global/pairwise_soft/per_sample_mae.tsv", cohort / "per_sample_mae.tsv",
    )
    print(f"collate_dx: per_sample_mae.tsv = {n_pm} rows", file=sys.stderr)

    # Tool-gated tables.
    if "flare" in tools:
        _concat_tsv_with_prefix(
            sources, "global/pairwise_hard/popout_vs_flare.confusion.tsv",
            cohort / "popout_vs_flare.confusion.tsv", optional=True,
        )
        _concat_tsv_with_prefix(
            sources, "global/pairwise_soft/popout_vs_flare.metrics.tsv",
            cohort / "popout_vs_flare.metrics.tsv", optional=True,
        )
    if "rye" in tools:
        _concat_tsv_with_prefix(
            sources, "global/pairwise_hard/popout_vs_rye.confusion.tsv",
            cohort / "popout_vs_rye.confusion.tsv", optional=True,
        )
        _concat_tsv_with_prefix(
            sources, "global/pairwise_soft/popout_vs_rye.metrics.tsv",
            cohort / "popout_vs_rye.metrics.tsv", optional=True,
        )
    if "rf" in tools:
        _concat_tsv_with_prefix(
            sources, "global/pairwise_hard/popout_vs_rf.confusion.tsv",
            cohort / "popout_vs_rf.confusion.tsv", optional=True,
        )
        _concat_tsv_with_prefix(
            sources, "global/pairwise_soft/popout_vs_rf.metrics.tsv",
            cohort / "popout_vs_rf.metrics.tsv", optional=True,
        )

    # Local-mode tables.
    if args.mode == "global_local":
        _concat_tsv_with_prefix(
            sources, "local/local_per_sample.tsv",
            cohort / "local_per_sample.tsv", optional=True,
        )
        _concat_tsv_gz_with_prefix(
            sources, "local/views/bp_confusion_segments.tsv.gz",
            cohort / "bp_confusion_segments.tsv.gz", optional=True,
        )
        _concat_tsv_with_prefix(
            sources, "local/views/boundary_localization.tsv",
            cohort / "boundary_localization.tsv", optional=True,
        )
        _concat_tsv_with_prefix(
            sources, "local/views/coarse_grid_summary.tsv",
            cohort / "coarse_grid_summary.tsv", optional=True,
        )

    # Cohort summary + manifest.
    summary = compute_cohort_summary(sources, tools, args.mode)
    (bundle / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    per_artifact_sha = {
        f"{cluster_id}/{chrom}": dx_schema.sha256_file(
            _resolve_artifact_tarball(args.tarballs, cluster_id, chrom)
        )
        for cluster_id, chrom, _ in sources
    }
    write_cohort_manifest(
        sources, bundle / "cohort_manifest.json",
        run_name=args.run_name, mode=args.mode, tools=tools,
        per_artifact_sha256=per_artifact_sha,
    )

    # Validate the assembled cohort tree against the schema.
    issues = dx_schema.validate_cohort_bundle(bundle)
    n_err = dx_schema.report_issues(issues, label=f"cohort_dx.{args.run_name}")
    if n_err:
        die(f"cohort bundle failed schema validation ({n_err} errors)")

    write_bundle_tarball(
        bundle, args.out_tarball,
        include_per_cluster=args.include_per_cluster,
        per_cluster_sources=sources if args.include_per_cluster else None,
    )
    print(
        f"collate_dx: wrote {args.out_tarball} ({len(sources)} per-cluster artifacts)",
        file=sys.stderr,
    )
    return 0


def _resolve_artifact_tarball(tarballs: list[Path], cluster_id: str, chrom: str) -> Path:
    """Find the tarball whose basename starts with ``<cluster_id>.<chrom>``."""
    needle = f"{cluster_id}.{chrom}.popout_dx."
    for tb in tarballs:
        if tb.name.startswith(needle):
            return tb
    die(f"could not resolve tarball for {cluster_id}/{chrom}")


if __name__ == "__main__":
    sys.exit(main())
