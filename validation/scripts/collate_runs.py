#!/usr/bin/env python3
"""FLARE validation Stage 2 — collate per-cluster artifacts into a cohort bundle.

See `validation/SCHEMA.md` §2 for the cohort bundle layout and
`my_notes/validation/PLAN.md` §3 for the design.

Usage:
    python collate_runs.py \\
        --cluster-artifacts a.tar.gz b.tar.gz ... \\
        --collation-config config.json \\
        --schema-version 1.1.0 \\
        --run-name flare_v9_chr1 \\
        --out-bundle cohort_bundle.tar.gz \\
        --out-summary cohort_summary.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import json
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

VALIDATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATION_DIR.parent))
from validation.schema import (
    SCHEMA_VERSION,
    read_cluster_artifact,
    report_issues,
    sha256_file,
    validate_cluster_artifact,
    validate_cohort_bundle,
)


# ── Logging ───────────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def _phase(msg: str) -> None:
    print(f"===== flare_collate: {msg} =====", file=sys.stderr, flush=True)


# ── Per-cluster artifact view ─────────────────────────────────────────────


@dataclasses.dataclass
class ClusterArtifact:
    cluster_id: str
    chrom: str
    artifact_dir: Path                  # the <cluster_id>/<chrom>/ subdir
    manifest: dict                      # parsed manifest.json
    sha256: str                         # of the source tarball


def _load_artifact(tarball: Path, staging: Path) -> ClusterArtifact:
    artifact_dir = read_cluster_artifact(tarball, staging)
    issues = validate_cluster_artifact(artifact_dir)
    n_err = sum(1 for i in issues if i.severity == "error")
    if n_err:
        report_issues(issues, label=f"artifact {tarball.name}")
        raise RuntimeError(f"{tarball.name}: {n_err} schema error(s); aborting")
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    return ClusterArtifact(
        cluster_id=manifest["cluster_id"],
        chrom=manifest["chrom"],
        artifact_dir=artifact_dir,
        manifest=manifest,
        sha256=sha256_file(tarball),
    )


# ── Long-form table writers ───────────────────────────────────────────────


def _write_header_once(path: Path, header: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(header + "\n")


def _append_lines(path: Path, lines: list[str]) -> None:
    with open(path, "a") as f:
        for line in lines:
            f.write(line + ("\n" if not line.endswith("\n") else ""))


def collate_cohort_global(arts: list[ClusterArtifact], out_path: Path) -> None:
    _write_header_once(out_path, "cluster_id\tchrom\tsample_id\tancestries")
    # Replace placeholder header with a more honest one: column count varies
    # per cluster, so the consumer must read the cluster's manifest to know K.
    out_path.write_text("cluster_id\tchrom\tsample_id\tancestry_props_tab_separated\n")
    with open(out_path, "a") as out:
        for art in arts:
            global_tsv = art.artifact_dir / "global.tsv"
            with open(global_tsv) as f:
                next(f)  # discard the popout-format header
                for line in f:
                    out.write(f"{art.cluster_id}\t{art.chrom}\t{line}")


def collate_coverage(arts: list[ClusterArtifact], out_path: Path) -> None:
    _write_header_once(out_path, "cluster_id\tchrom\tcheck\tstatus\tdetail")
    for art in arts:
        with open(art.artifact_dir / "coverage" / "coverage_check.tsv") as f:
            next(f)
            lines = [f"{art.cluster_id}\t{art.chrom}\t{line.rstrip()}" for line in f if line.strip()]
        _append_lines(out_path, lines)


def collate_manifest(arts: list[ClusterArtifact], out_path: Path) -> None:
    cols = ["cluster_id", "chrom", "n_samples", "n_markers", "n_ancestries",
            "coverage_passed", "total_wallclock_seconds", "peak_rss_gb",
            "cpu_wall_ratio", "flare_version", "panel_id", "generated_at"]
    _write_header_once(out_path, "\t".join(cols))
    rows = []
    for art in arts:
        m = art.manifest
        rows.append("\t".join(str(m.get(c, "") if c not in ("cluster_id", "chrom")
                                  else (art.cluster_id if c == "cluster_id" else art.chrom))
                              for c in cols))
    _append_lines(out_path, rows)


def collate_tier1(arts: list[ClusterArtifact], out_path: Path) -> None:
    _write_header_once(out_path, "cluster_id\tchrom\tkey\tvalue")
    for art in arts:
        with open(art.artifact_dir / "tier1_metrics.tsv") as f:
            lines = [f"{art.cluster_id}\t{art.chrom}\t{line.rstrip()}"
                     for line in f if line.strip()]
        _append_lines(out_path, lines)


def collate_soft_correlation_rf(arts: list[ClusterArtifact], out_path: Path) -> None:
    """Unpivot the wide rf_soft_correlation.tsv per cluster into long form."""
    _write_header_once(out_path, "cluster_id\tchrom\tflare_ancestry\trf_label\tr")
    for art in arts:
        src = art.artifact_dir / "soft_correlation" / "rf_soft_correlation.tsv"
        with open(src) as f:
            header = f.readline().rstrip("\n").split("\t")
            rf_labels = header[1:]
            rows = []
            for line in f:
                parts = line.rstrip("\n").split("\t")
                ancestry = parts[0]
                for rf_idx, rf_lab in enumerate(rf_labels):
                    if rf_idx + 1 >= len(parts):
                        continue
                    rows.append(f"{art.cluster_id}\t{art.chrom}\t{ancestry}\t{rf_lab}\t{parts[rf_idx+1]}")
        _append_lines(out_path, rows)


def collate_merged_groups_rf(arts: list[ClusterArtifact], out_path: Path) -> None:
    _write_header_once(
        out_path,
        "cluster_id\tchrom\trf_label\tmerged_r\tsummed_mu\tcomponent_indices\tcomponent_names",
    )
    for art in arts:
        with open(art.artifact_dir / "soft_correlation" / "rf_merged_groups.tsv") as f:
            next(f)
            rows = [f"{art.cluster_id}\t{art.chrom}\t{line.rstrip()}"
                    for line in f if line.strip()]
        _append_lines(out_path, rows)


def collate_concordance_metrics(arts: list[ClusterArtifact], out_path: Path) -> bool:
    """★ v1.1 (optional, gated on rye_q): concat per-cluster concordance_metrics.tsv."""
    cols = ["cluster_id", "chrom", "ancestry", "cluster_mu", "n_samples",
            "pearson_r", "ccc", "cosine_mean", "mae_mean", "mae_median", "mae_p95",
            "jaccard_at_0.10", "jaccard_at_0.25", "jaccard_at_0.50", "pass"]
    _write_header_once(out_path, "\t".join(cols))
    any_present = False
    for art in arts:
        src = art.artifact_dir / "concordance" / "concordance_metrics.tsv"
        if not src.exists():
            continue
        any_present = True
        with open(src) as f:
            next(f)
            rows = [f"{art.cluster_id}\t{art.chrom}\t{line.rstrip()}"
                    for line in f if line.strip()]
        _append_lines(out_path, rows)
    if not any_present:
        out_path.unlink(missing_ok=True)
    return any_present


def collate_confusion_rf(arts: list[ClusterArtifact], out_path: Path) -> None:
    """Unpivot the wide rf_confusion_matrix.tsv into long (cluster, rf_label, flare_call, n)."""
    _write_header_once(out_path, "cluster_id\tchrom\trf_label\tflare_call\tn")
    for art in arts:
        src = art.artifact_dir / "confusion" / "rf_confusion_matrix.tsv"
        with open(src) as f:
            header = f.readline().rstrip("\n").split("\t")
            # header[0] = "rf_label", header[1..-1] = popout ancestry names, header[-1] = "total"
            popout_names = header[1:-1]
            rows = []
            for line in f:
                parts = line.rstrip("\n").split("\t")
                rf_label = parts[0]
                if rf_label == "total":
                    continue
                for idx, name in enumerate(popout_names):
                    n = parts[1 + idx]
                    rows.append(f"{art.cluster_id}\t{art.chrom}\t{rf_label}\t{name}\t{n}")
        _append_lines(out_path, rows)


def collate_calibration_slope(arts: list[ClusterArtifact], out_path: Path) -> None:
    """Unpivot the wide slope_matrix.tsv into long form."""
    _write_header_once(out_path, "cluster_id\tchrom\tancestry_name\trf_label\tslope\tmax_cal")
    for art in arts:
        src = art.artifact_dir / "calibration" / "slope_matrix.tsv"
        with open(src) as f:
            header = f.readline().rstrip("\n").split("\t")
            slope_cols = [(i, h[:-len("_slope")]) for i, h in enumerate(header) if h.endswith("_slope")]
            max_cols = {h[:-len("_max")]: i for i, h in enumerate(header) if h.endswith("_max")}
            rows = []
            for line in f:
                parts = line.rstrip("\n").split("\t")
                ancestry = parts[0]
                for ci, rf_lab in slope_cols:
                    slope = parts[ci] if ci < len(parts) else "NA"
                    max_ci = max_cols.get(rf_lab)
                    max_v = parts[max_ci] if max_ci is not None and max_ci < len(parts) else "NA"
                    rows.append(f"{art.cluster_id}\t{art.chrom}\t{ancestry}\t{rf_lab}\t{slope}\t{max_v}")
        _append_lines(out_path, rows)


def collate_tract_length_stats(arts: list[ClusterArtifact], out_path: Path) -> None:
    cols = ["cluster_id", "chrom", "ancestry", "ancestry_name", "n_tracts",
            "mean_Mb", "median_Mb", "exp_fit_rate", "implied_T_gen", "model_T_gen"]
    _write_header_once(out_path, "\t".join(cols))
    for art in arts:
        src = art.artifact_dir / "structural" / "tract_length_summary.json"
        data = json.loads(src.read_text())
        rows = []
        for entry in data.get("per_ancestry", []):
            rows.append("\t".join([
                art.cluster_id, art.chrom,
                str(entry["ancestry"]), entry["name"], str(entry["n_tracts"]),
                f"{entry['mean_Mb']:.6f}", f"{entry['median_Mb']:.6f}",
                "NA" if entry.get("exp_fit_rate") is None else f"{entry['exp_fit_rate']:.6f}",
                "NA" if entry.get("implied_T_gen") is None else f"{entry['implied_T_gen']:.4f}",
                "NA" if entry.get("model_T_gen") is None else f"{entry['model_T_gen']:.4f}",
            ]))
        _append_lines(out_path, rows)


def collate_switch_rate_stats(arts: list[ClusterArtifact], out_path: Path) -> None:
    cols = ["cluster_id", "chrom", "n_haplotypes", "mean", "median", "p99", "min", "max"]
    _write_header_once(out_path, "\t".join(cols))
    rows = []
    for art in arts:
        d = json.loads((art.artifact_dir / "structural" / "switch_rate_summary.json").read_text())
        rows.append("\t".join([
            art.cluster_id, art.chrom,
            str(d["n_haplotypes"]), f"{d['mean']:.4f}", f"{d['median']:.2f}",
            f"{d['p99']:.2f}", str(d["min"]), str(d["max"]),
        ]))
    _append_lines(out_path, rows)


def collate_hap_disagreement(arts: list[ClusterArtifact], out_path: Path) -> None:
    cols = ["cluster_id", "chrom", "rf_label", "n", "mean", "median"]
    _write_header_once(out_path, "\t".join(cols))
    rows = []
    for art in arts:
        d = json.loads((art.artifact_dir / "hap_disagreement" / "summary.json").read_text())
        for entry in d.get("per_rf_label", []):
            rows.append("\t".join([
                art.cluster_id, art.chrom, entry["rf_label"],
                str(entry["n"]), f"{entry['mean']:.6f}", f"{entry['median']:.6f}",
            ]))
    _append_lines(out_path, rows)


def collate_regional_windows(arts: list[ClusterArtifact], out_path: Path) -> None:
    """Concatenate per-cluster regional/windows.tsv.gz, prepending cluster_id."""
    with gzip.open(out_path, "wt") as out:
        wrote_header = False
        for art in arts:
            src = art.artifact_dir / "regional" / "windows.tsv.gz"
            with gzip.open(src, "rt") as f:
                header = f.readline().rstrip("\n")
                if not wrote_header:
                    out.write("cluster_id\t" + header + "\n")
                    wrote_header = True
                for line in f:
                    out.write(art.cluster_id + "\t" + line)


def collate_self_id(arts: list[ClusterArtifact], out_path: Path) -> bool:
    cols = ["cluster_id", "chrom", "self_id", "n", "ancestry", "name", "mean_mu"]
    _write_header_once(out_path, "\t".join(cols))
    any_present = False
    for art in arts:
        src = art.artifact_dir / "self_id" / "check.tsv"
        if not src.exists():
            continue
        any_present = True
        with open(src) as f:
            next(f)
            rows = [f"{art.cluster_id}\t{art.chrom}\t{line.rstrip()}"
                    for line in f if line.strip()]
        _append_lines(out_path, rows)
    if not any_present:
        out_path.unlink(missing_ok=True)
    return any_present


# ── Cross-cluster meta-analysis on regional windows ───────────────────────


def regional_meta_analysis(out_windows_gz: Path, out_meta: Path) -> None:
    """For each (window, ancestry_name), count clusters flagged + Stouffer combine z."""
    # Window key: (chrom, start, end, ancestry_name).
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    clusters_seen: set[str] = set()
    with gzip.open(out_windows_gz, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            cluster = parts[idx["cluster_id"]]
            clusters_seen.add(cluster)
            chrom = parts[idx["chrom"]]
            start = int(parts[idx["start"]])
            end = int(parts[idx["end"]])
            anc = parts[idx["ancestry_name"]]
            try:
                z = float(parts[idx["z"]])
                p = float(parts[idx["p"]])
                q = float(parts[idx["q"]])
            except (ValueError, KeyError):
                continue
            mask = parts[idx["mask_region"]] if "mask_region" in idx else ""
            by_key[(chrom, start, end, anc)].append({
                "cluster": cluster, "z": z, "p": p, "q": q, "mask_region": mask,
            })

    n_clusters_total = len(clusters_seen)
    with open(out_meta, "w") as out:
        out.write("chrom\tstart\tend\tancestry_name\tn_clusters_flagged\t"
                  "n_clusters_total\tstouffer_z\tstouffer_p\tstouffer_q\tmask_region\n")
        rows = []
        for (chrom, start, end, anc), entries in sorted(by_key.items()):
            zs = np.array([e["z"] for e in entries])
            # Stouffer's combined z. Assumes one-sided sign-aware combination.
            stouffer_z = float(zs.sum() / np.sqrt(len(zs))) if len(zs) else 0.0
            stouffer_p = float(2.0 * norm.sf(abs(stouffer_z)))
            n_flagged = sum(1 for e in entries if e["q"] < 0.05)
            mask = next((e["mask_region"] for e in entries if e["mask_region"]), "")
            rows.append((chrom, start, end, anc, n_flagged, len(entries),
                         stouffer_z, stouffer_p, mask))
        # BH-FDR on stouffer p-values.
        if rows:
            ps = np.array([r[7] for r in rows])
            order = np.argsort(ps)
            n = len(ps)
            qs = np.empty(n)
            cummin = 1.0
            for rank, idx_ in enumerate(reversed(order)):
                k = n - rank
                q_raw = ps[idx_] * n / k
                cummin = min(cummin, q_raw)
                qs[idx_] = cummin
        else:
            qs = np.array([])
        for (chrom, start, end, anc, n_flagged, n_total, sz, sp, mask), q in zip(rows, qs):
            out.write(f"{chrom}\t{start}\t{end}\t{anc}\t{n_flagged}\t{n_total}\t"
                      f"{sz:+.4f}\t{sp:.4e}\t{q:.4e}\t{mask}\n")


# ── Summary + dashboard ───────────────────────────────────────────────────


def build_cohort_summary(
    arts: list[ClusterArtifact], cohort_dir: Path, *, run_name: str,
) -> dict:
    # Distinct cluster ids (a cluster may span multiple chroms).
    cluster_ids = sorted({a.cluster_id for a in arts})
    chroms = sorted({a.chrom for a in arts})

    # n_clusters_pass_coverage: a cluster passes if all its (cluster, chrom)
    # artifacts have coverage_passed: true.
    by_cluster: dict[str, list[ClusterArtifact]] = defaultdict(list)
    for a in arts:
        by_cluster[a.cluster_id].append(a)
    n_pass = sum(
        1 for cid, group in by_cluster.items()
        if all(g.manifest.get("coverage_passed") for g in group)
    )

    # μ-weighted mean merged-r per RF label, cohort-wide. Skip rows where
    # the RF label has no FLARE components mapped to it (summed_mu == 0)
    # — those are degenerate non-tests and a plain np.mean would drag the
    # cohort-wide signal down to zero for any label that's absent from one
    # cluster (e.g., a 3-cluster cohort where 2 clusters have no AFR gets a
    # misleading mean_merged_r_afr ≈ 0.33).
    mu_by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)  # [(merged_r, summed_mu)]
    mg_path = cohort_dir / "merged_groups_rf.tsv"
    if mg_path.exists():
        with open(mg_path) as f:
            header = f.readline().rstrip("\n").split("\t")
            i_lab = header.index("rf_label")
            i_r = header.index("merged_r")
            i_mu = header.index("summed_mu")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                try:
                    mu = float(parts[i_mu])
                    if mu <= 0:
                        continue
                    mu_by_label[parts[i_lab]].append((float(parts[i_r]), mu))
                except (ValueError, IndexError):
                    continue
    mean_merged_r: dict[str, float] = {}
    for lab, rs in mu_by_label.items():
        total_mu = sum(mu for _, mu in rs)
        if total_mu > 0:
            mean_merged_r[lab] = sum(r * mu for r, mu in rs) / total_mu

    # HLA / regional outlier counts via per-cluster regional summary.json.
    n_with_hla = 0
    n_outliers_outside = 0
    for a in arts:
        rg = a.artifact_dir / "regional" / "summary.json"
        if not rg.exists():
            continue
        d = json.loads(rg.read_text())
        if d.get("hla_overlap_n", 0) > 0:
            n_with_hla += 1
        n_outliers_outside += int(d.get("outside_mask_n", 0))

    # Traffic light per cluster (simple V1: red if any coverage failed,
    # yellow if any merged_r < 0.7, else green).
    traffic = {}
    for cid, group in by_cluster.items():
        if not all(g.manifest.get("coverage_passed") for g in group):
            traffic[cid] = "red"
            continue
        cid_rs: list[float] = []
        for g in group:
            mg = g.artifact_dir / "soft_correlation" / "rf_merged_groups.tsv"
            if not mg.exists():
                continue
            with open(mg) as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    try:
                        cid_rs.append(float(parts[1]))
                    except (ValueError, IndexError):
                        pass
        if cid_rs and min(cid_rs) < 0.7:
            traffic[cid] = "yellow"
        else:
            traffic[cid] = "green"

    total_wall_h = sum(a.manifest.get("total_wallclock_seconds", 0.0) for a in arts) / 3600.0
    total_peak = max((a.manifest.get("peak_rss_gb", 0.0) for a in arts), default=0.0)

    # ★ v1.1: mean of per-cluster global_ccc (cohort-wide R10 signal).
    gcccs: list[float] = []
    for a in arts:
        src = a.artifact_dir / "concordance" / "concordance_summary.json"
        if not src.exists():
            continue
        try:
            v = json.loads(src.read_text()).get("global_ccc")
            if v is not None:
                gcccs.append(float(v))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    mean_global_ccc = float(np.mean(gcccs)) if gcccs else None

    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "n_clusters": len(cluster_ids),
        "n_chroms": len(chroms),
        "n_artifacts": len(arts),
        "n_clusters_pass_coverage": n_pass,
        "mean_merged_r_per_rf_label": mean_merged_r,
        "mean_global_ccc": mean_global_ccc,                      # ★ v1.1
        "n_clusters_with_hla_flagged": n_with_hla,
        "n_regional_outliers_outside_mask": n_outliers_outside,
        "traffic_light_per_cluster": traffic,
        "total_wallclock_hours": round(total_wall_h, 2),
        "total_peak_rss_gb_max": round(total_peak, 2),
    }


def build_qc_dashboard(arts: list[ClusterArtifact], thresholds: dict) -> dict:
    """Per-cluster traffic light per QC dimension. V1: green / yellow / red rules
    are simple; thresholds is echoed so downstream tools know what was applied.
    """
    dimensions = ("coverage", "calibration", "concordance",
                  "structural", "hap_disagreement", "regional",
                  "rye_concordance")
    cal_lo, cal_hi = thresholds.get("calibration_slope_outside", [0.85, 1.15])
    hd_threshold = thresholds.get("hap_disagreement_yellow", 0.30)
    regional_threshold = thresholds.get("regional_significant_yellow", 10)
    # ★ v1.1: R10 thresholds (PLAN2.md §3.3 traffic-light rules).
    r10_pearson_threshold = thresholds.get("rye_pearson_pass", 0.95)
    r10_ccc_threshold = thresholds.get("rye_ccc_pass", 0.90)
    r10_yellow_margin = thresholds.get("rye_yellow_margin_pct", 0.05)

    by_cluster: dict[str, list[ClusterArtifact]] = defaultdict(list)
    for a in arts:
        by_cluster[a.cluster_id].append(a)

    per_cluster: dict[str, dict[str, str]] = {}
    for cid, group in by_cluster.items():
        verdicts: dict[str, str] = {}
        # coverage
        verdicts["coverage"] = "green" if all(g.manifest.get("coverage_passed") for g in group) else "red"
        # concordance: any merged_r < 0.7 → yellow; < 0.5 → red. Skip rows
        # where summed_mu == 0 — those are RF labels with no FLARE components
        # mapped (mid is always one for FLARE-source clusters since the FLARE
        # basis drops it per the R4 rule, and any cohort can legitimately
        # have zero samples of some ancestry). A degenerate non-test shouldn't
        # force red.
        min_r = 1.0
        for g in group:
            mg = g.artifact_dir / "soft_correlation" / "rf_merged_groups.tsv"
            if mg.exists():
                with open(mg) as f:
                    next(f)
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 3:
                            continue
                        try:
                            mu = float(parts[2])
                            if mu <= 0:
                                continue
                            r = float(parts[1])
                            min_r = min(min_r, r)
                        except (ValueError, IndexError):
                            pass
        verdicts["concordance"] = "red" if min_r < 0.5 else ("yellow" if min_r < 0.7 else "green")
        # calibration: any slope outside [cal_lo, cal_hi] → yellow; <0 → red
        worst_dev = 0.0
        any_negative = False
        for g in group:
            sm = g.artifact_dir / "calibration" / "slope_matrix.tsv"
            if not sm.exists():
                continue
            with open(sm) as f:
                header = f.readline().rstrip("\n").split("\t")
                slope_cols = [i for i, h in enumerate(header) if h.endswith("_slope")]
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    for ci in slope_cols:
                        v = parts[ci] if ci < len(parts) else "NA"
                        if v == "NA":
                            continue
                        try:
                            slope = float(v)
                            if slope < 0:
                                any_negative = True
                            worst_dev = max(worst_dev, abs(slope - 1.0))
                        except ValueError:
                            pass
        if any_negative:
            verdicts["calibration"] = "red"
        elif worst_dev > max(abs(cal_lo - 1.0), abs(cal_hi - 1.0)):
            verdicts["calibration"] = "yellow"
        else:
            verdicts["calibration"] = "green"
        # structural: mu_vs_global_diff failure → red
        struct_pass = True
        for g in group:
            mvg = g.artifact_dir / "model" / "mu_vs_global_diff.json"
            if mvg.exists():
                d = json.loads(mvg.read_text())
                if not d.get("overall_pass", True):
                    struct_pass = False
        verdicts["structural"] = "green" if struct_pass else "red"
        # hap_disagreement: cohort mean > threshold → yellow
        worst_hd = 0.0
        for g in group:
            hd = g.artifact_dir / "hap_disagreement" / "summary.json"
            if hd.exists():
                d = json.loads(hd.read_text())
                worst_hd = max(worst_hd, float(d.get("cohort_mean_disagreement", 0.0)))
        verdicts["hap_disagreement"] = "yellow" if worst_hd > hd_threshold else "green"
        # regional: n significant > threshold → yellow
        max_sig = 0
        for g in group:
            rg = g.artifact_dir / "regional" / "summary.json"
            if rg.exists():
                d = json.loads(rg.read_text())
                max_sig = max(max_sig, int(d.get("n_windows_significant", 0)))
        verdicts["regional"] = "yellow" if max_sig > regional_threshold else "green"

        # ★ v1.1: rye_concordance traffic light per PLAN2.md §3.3.
        # green = all μ≥0.01 labels pass r≥0.95 AND CCC≥0.90.
        # yellow = at least one passes r but not CCC, OR fails by ≤ 5 pct.
        # red = any fails by > 5 pct.
        worst_r_gap = 0.0  # how far below threshold the worst r is
        worst_ccc_gap = 0.0
        any_r_pass_ccc_fail = False
        any_present = False
        for g in group:
            cm = g.artifact_dir / "concordance" / "concordance_metrics.tsv"
            if not cm.exists():
                continue
            with open(cm) as f:
                header = f.readline().rstrip("\n").split("\t")
                try:
                    i_anc = header.index("ancestry")
                    i_mu = header.index("cluster_mu")
                    i_r = header.index("pearson_r")
                    i_c = header.index("ccc")
                except ValueError:
                    continue
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= max(i_anc, i_mu, i_r, i_c):
                        continue
                    try:
                        mu = float(parts[i_mu])
                        if mu < 0.01:
                            continue  # μ-gated; not a test
                        r_val = float(parts[i_r]) if parts[i_r] not in ("", "NA") else float("nan")
                        c_val = float(parts[i_c]) if parts[i_c] not in ("", "NA") else float("nan")
                    except ValueError:
                        continue
                    any_present = True
                    if not (r_val != r_val):  # not NaN
                        gap = r10_pearson_threshold - r_val
                        if gap > worst_r_gap:
                            worst_r_gap = gap
                    if not (c_val != c_val):
                        gap = r10_ccc_threshold - c_val
                        if gap > worst_ccc_gap:
                            worst_ccc_gap = gap
                    if r_val >= r10_pearson_threshold and c_val < r10_ccc_threshold:
                        any_r_pass_ccc_fail = True
        if not any_present:
            verdicts["rye_concordance"] = "skip"  # no rye_q provided for this cluster
        elif worst_r_gap > r10_yellow_margin or worst_ccc_gap > r10_yellow_margin:
            verdicts["rye_concordance"] = "red"
        elif worst_r_gap > 0 or worst_ccc_gap > 0 or any_r_pass_ccc_fail:
            verdicts["rye_concordance"] = "yellow"
        else:
            verdicts["rye_concordance"] = "green"

        per_cluster[cid] = verdicts

    return {
        "schema_version": SCHEMA_VERSION,
        "dimensions": list(dimensions),
        "per_cluster": per_cluster,
        "thresholds": {
            "calibration_slope_outside": [cal_lo, cal_hi],
            "hap_disagreement_yellow": hd_threshold,
            "regional_significant_yellow": regional_threshold,
            "rye_pearson_pass": r10_pearson_threshold,
            "rye_ccc_pass": r10_ccc_threshold,
            "rye_yellow_margin_pct": r10_yellow_margin,
        },
    }


# ── Bundling ──────────────────────────────────────────────────────────────


def stage_per_cluster_copies(arts: list[ClusterArtifact], bundle_root: Path) -> None:
    """Move the per-cluster artifact dirs into the bundle's per_cluster/ tree."""
    pc_root = bundle_root / "per_cluster"
    pc_root.mkdir(parents=True, exist_ok=True)
    for a in arts:
        dst = pc_root / a.cluster_id / a.chrom
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(a.artifact_dir, dst)


def write_cohort_manifest(
    arts: list[ClusterArtifact], bundle_root: Path, *,
    run_name: str, collation_mode: str, collation_config: dict, diff_against: str | None,
) -> dict:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "collation_mode": collation_mode,
        "n_clusters": len(set(a.cluster_id for a in arts)),
        "n_chroms": len(set(a.chrom for a in arts)),
        "n_artifacts": len(arts),
        "cluster_ids": sorted({a.cluster_id for a in arts}),
        "chroms": sorted({a.chrom for a in arts}),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256_per_artifact": {f"{a.cluster_id}.{a.chrom}": a.sha256 for a in arts},
        "diff_against": diff_against,
        "collation_config": collation_config,
    }
    (bundle_root / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def tar_bundle(bundle_root: Path, out_tarball: Path) -> None:
    out_tarball.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tarball, "w:gz") as tar:
        tar.add(bundle_root, arcname="cohort_bundle")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cluster-artifacts", type=Path, nargs="+", required=True)
    p.add_argument("--collation-config", type=Path, default=None,
                   help="JSON config; default = {mode: 'single_run'}")
    p.add_argument("--diff-against", type=Path, default=None,
                   help="Cohort bundle to diff against (mode=diff_runs)")
    p.add_argument("--schema-version", default=SCHEMA_VERSION)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-bundle", type=Path, required=True)
    p.add_argument("--out-summary", type=Path, required=True)
    p.add_argument("--staging-dir", type=Path, default=None,
                   help="Working dir for unpacked artifacts; default = <out-bundle>.work/")
    args = p.parse_args()

    if args.schema_version != SCHEMA_VERSION:
        raise RuntimeError(
            f"--schema-version {args.schema_version!r} != bundled "
            f"validation/schema.py SCHEMA_VERSION {SCHEMA_VERSION!r}"
        )

    config = {"mode": "single_run"}
    if args.collation_config is not None:
        config.update(json.loads(args.collation_config.read_text()))
    collation_mode = config.get("mode", "single_run")
    thresholds = {
        "calibration_slope_outside": config.get("fail_on_calibration_slope_outside", [0.85, 1.15]),
    }

    staging = args.staging_dir or args.out_bundle.with_suffix(".work")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # ── Load + validate every per-cluster artifact ──
    _phase(f"loading {len(args.cluster_artifacts)} artifact(s) into {staging}")
    arts: list[ClusterArtifact] = []
    for i, tarball in enumerate(args.cluster_artifacts):
        if not tarball.exists():
            raise FileNotFoundError(tarball)
        _log(f"[{i+1}/{len(args.cluster_artifacts)}] {tarball.name}")
        art = _load_artifact(tarball, staging / f"art_{i:03d}")
        if art.manifest.get("schema_version") != args.schema_version:
            raise RuntimeError(
                f"{tarball.name}: schema_version {art.manifest.get('schema_version')!r} "
                f"!= cohort {args.schema_version!r}"
            )
        arts.append(art)

    # Filter by collation config.
    include = config.get("include_clusters")
    exclude = set(config.get("exclude_clusters", []) or [])
    if include:
        include = set(include)
        arts = [a for a in arts if a.cluster_id in include]
    if exclude:
        arts = [a for a in arts if a.cluster_id not in exclude]
    if not arts:
        raise RuntimeError("no artifacts left after include/exclude filtering")
    _phase(f"collating {len(arts)} artifact(s) "
           f"({len(set(a.cluster_id for a in arts))} clusters × "
           f"{len(set(a.chrom for a in arts))} chroms)")

    # ── Build bundle on disk ──
    bundle_root = staging / "bundle"
    cohort_dir = bundle_root / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    _phase("collating long-form tables")
    collate_cohort_global(arts,           cohort_dir / "cohort_global.tsv")
    collate_coverage(arts,                cohort_dir / "coverage.tsv")
    collate_manifest(arts,                cohort_dir / "manifest.tsv")
    collate_tier1(arts,                   cohort_dir / "tier1_metrics.tsv")
    collate_soft_correlation_rf(arts,     cohort_dir / "soft_correlation_rf.tsv")
    collate_merged_groups_rf(arts,        cohort_dir / "merged_groups_rf.tsv")
    # ★ v1.1: Rye concordance metrics (optional, gated on rye_q per cluster).
    has_rye = collate_concordance_metrics(
        arts,                              cohort_dir / "concordance_metrics.tsv")
    collate_confusion_rf(arts,            cohort_dir / "confusion_rf.tsv")
    collate_calibration_slope(arts,       cohort_dir / "calibration_slope.tsv")
    collate_tract_length_stats(arts,      cohort_dir / "tract_length_stats.tsv")
    collate_switch_rate_stats(arts,       cohort_dir / "switch_rate_stats.tsv")
    collate_hap_disagreement(arts,        cohort_dir / "hap_disagreement.tsv")
    collate_regional_windows(arts,        cohort_dir / "regional_windows.tsv.gz")
    has_self_id = collate_self_id(arts,   cohort_dir / "self_id.tsv")

    _phase("running cross-cluster regional meta-analysis")
    regional_meta_analysis(cohort_dir / "regional_windows.tsv.gz",
                           cohort_dir / "regional_meta.tsv")

    _phase("computing cohort summary + QC dashboard")
    cohort_summary = build_cohort_summary(arts, cohort_dir, run_name=args.run_name)
    (bundle_root / "cohort_summary.json").write_text(json.dumps(cohort_summary, indent=2))
    args.out_summary.write_text(json.dumps(cohort_summary, indent=2))

    dashboard = build_qc_dashboard(arts, thresholds)
    (bundle_root / "cohort_qc_dashboard.json").write_text(json.dumps(dashboard, indent=2))

    _phase("writing per_cluster/ copies")
    stage_per_cluster_copies(arts, bundle_root)

    _phase("writing cohort_manifest.json")
    write_cohort_manifest(arts, bundle_root,
                          run_name=args.run_name, collation_mode=collation_mode,
                          collation_config=config,
                          diff_against=str(args.diff_against) if args.diff_against else None)

    _phase("validating bundle against cohort schema")
    issues = validate_cohort_bundle(bundle_root)
    n_err = report_issues(issues, label="cohort bundle")
    if n_err:
        raise RuntimeError(
            f"cohort bundle failed schema validation ({n_err} error(s)); "
            f"bundle dir kept at {bundle_root}"
        )

    _phase(f"tarring {bundle_root} → {args.out_bundle}")
    tar_bundle(bundle_root, args.out_bundle)
    _log(f"wrote {args.out_bundle} ({args.out_bundle.stat().st_size / 1e6:.1f} MB)")
    _log(f"wrote {args.out_summary}")
    _log(f"Rye concordance rows: {'present' if has_rye else 'absent'}; "
         f"self_id rows: {'present' if has_self_id else 'absent'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
