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
    """Concatenate per-cluster ``global.tsv`` into a cohort-wide file with
    named-column header.

    Schema v3.0.0+: the per-cluster ``global.tsv`` header carries the
    FLARE panel-population names verbatim from the VCF ``##ANCESTRY=``
    line (e.g. ``sample_id<TAB>eas<TAB>amr<TAB>eur<TAB>afr<TAB>sas``).
    The cohort file echoes those names with a ``cluster_id<TAB>chrom``
    prefix. All clusters in the cohort must share the same panel
    columns in the same order — mismatch is a hard error (collator
    cannot reconcile two panels into one wide table). v2 emitted a
    single meta-header ``ancestry_props_tab_separated`` because v2
    panel naming wasn't stable; v3 fixed naming, so cohort_global.tsv
    can now carry the real schema.
    """
    if not arts:
        return
    panel_cols: list[str] | None = None
    panel_source: tuple[str, str] | None = None
    rows: list[str] = []
    for art in arts:
        global_tsv = art.artifact_dir / "global.tsv"
        with open(global_tsv) as f:
            header = f.readline().rstrip("\n").split("\t")
            if not header or header[0] != "sample_id":
                raise RuntimeError(
                    f"{global_tsv}: first column must be 'sample_id', got "
                    f"{header[0]!r} (header: {header!r})"
                )
            cols = header[1:]
            if panel_cols is None:
                panel_cols = cols
                panel_source = (art.cluster_id, art.chrom)
            elif cols != panel_cols:
                src_cid, src_chrom = panel_source
                raise RuntimeError(
                    f"{global_tsv}: panel header {cols!r} disagrees with "
                    f"cohort panel {panel_cols!r} from "
                    f"{src_cid}/{src_chrom}/global.tsv — cohort_global.tsv "
                    f"requires every cluster to share the same FLARE panel"
                )
            for line in f:
                rows.append(f"{art.cluster_id}\t{art.chrom}\t{line.rstrip()}")

    header_line = "cluster_id\tchrom\tsample_id\t" + "\t".join(panel_cols)
    out_path.write_text(header_line + "\n" + "\n".join(rows) + "\n")


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


def collate_soft_correlation_rf(
    arts: list[ClusterArtifact], out_path: Path, *, mid_rule: str = "none",
) -> None:
    """Unpivot the wide rf_soft_correlation.tsv per cluster into long form.

    Category B (FLARE x RF). ``mid_rule`` selects MID handling on the
    RF axis:

    - ``"none"`` — pass MID through (legacy behaviour).
    - ``"drop"`` — omit the ``rf_label == "mid"`` column. Correct for
      a SP5-targeted bundle.
    - ``"fold_to_eur"`` — Pearson r doesn't sum, so folding MID's r
      into EUR's r is undefined; rather than invent a value, this
      rule degrades to ``"drop"`` here and is recorded in the
      manifest's ``transformations`` list.
    """
    if mid_rule not in ("none", "drop", "fold_to_eur"):
        raise ValueError(f"unknown mid_rule {mid_rule!r}")
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
                    if rf_lab == "mid" and mid_rule in ("drop", "fold_to_eur"):
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


def _collate_concordance_per_tool(
    arts: list[ClusterArtifact], src_name: str, out_path: Path,
) -> bool:
    """Concat per-cluster ``concordance/<src_name>`` into a cohort TSV.

    Used for both Rye (`concordance_metrics_rye.tsv`) and RF
    (`concordance_metrics_rf.tsv`). Same row shape per the shared
    `concordance.py` writer.
    """
    cols = ["cluster_id", "chrom", "ancestry", "cluster_mu", "n_samples",
            "pearson_r", "ccc", "cosine_mean", "mae_mean", "mae_median", "mae_p95",
            "jaccard_at_0.10", "jaccard_at_0.25", "jaccard_at_0.50", "pass"]
    _write_header_once(out_path, "\t".join(cols))
    any_present = False
    for art in arts:
        src = art.artifact_dir / "concordance" / src_name
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


def collate_concordance_metrics_rye(
    arts: list[ClusterArtifact], out_path: Path,
) -> bool:
    """★ v1.1 (optional, gated on rye_q)."""
    return _collate_concordance_per_tool(
        arts, "concordance_metrics_rye.tsv", out_path,
    )


def collate_concordance_metrics_rf(
    arts: list[ClusterArtifact], out_path: Path,
) -> bool:
    """Per-ancestry Pearson r + CCC for FLARE vs RF on the SP5 intersection.
    Same row shape as Rye; emitted whenever compare_to_rf ran in by_name
    mode and produced ``concordance/concordance_metrics_rf.tsv``."""
    return _collate_concordance_per_tool(
        arts, "concordance_metrics_rf.tsv", out_path,
    )


def _apply_mid_rule(
    cluster_id: str, chrom: str,
    popout_names: list[str], rf_counts: dict[str, list[int]],
    *, mid_rule: str,
) -> list[str]:
    """Apply the MID-handling rule to one cluster's confusion rows.

    ``rf_counts`` is ``{rf_label: [n_per_flare_call, ...]}``. Returns
    the long-form TSV rows for this cluster (one per (rf_label,
    flare_call) cell).
    """
    if mid_rule == "drop":
        rf_counts = {k: v for k, v in rf_counts.items() if k != "mid"}
    elif mid_rule == "fold_to_eur":
        mid = rf_counts.pop("mid", None)
        if mid is not None:
            eur = rf_counts.setdefault("eur", [0] * len(popout_names))
            for j in range(len(eur)):
                eur[j] = (eur[j] if j < len(eur) else 0) + (mid[j] if j < len(mid) else 0)
    elif mid_rule != "none":
        raise ValueError(f"unknown mid_rule {mid_rule!r}")

    rows = []
    for rf_label, vec in rf_counts.items():
        for j, name in enumerate(popout_names):
            n = vec[j] if j < len(vec) else 0
            rows.append(f"{cluster_id}\t{chrom}\t{rf_label}\t{name}\t{n}")
    return rows


def collate_confusion_rf(
    arts: list[ClusterArtifact], out_path: Path, *, mid_rule: str = "none",
) -> None:
    """Unpivot the wide rf_confusion_matrix.tsv into long (cluster, rf_label, flare_call, n).

    ``mid_rule`` (Phase 6 of the label-space retrofit) selects the
    RF-side MID handling rule. ``none`` passes MID rows through;
    ``drop`` removes them; ``fold_to_eur`` sums MID counts into the
    EUR row per (cluster, chrom, flare_call). The chosen rule is
    recorded in ``cohort_manifest.json.provenance.mid_rule``.
    """
    _write_header_once(out_path, "cluster_id\tchrom\trf_label\tflare_call\tn")
    for art in arts:
        src = art.artifact_dir / "confusion" / "rf_confusion_matrix.tsv"
        with open(src) as f:
            header = f.readline().rstrip("\n").split("\t")
            # header[0] = "rf_label", header[1..-1] = popout ancestry names, header[-1] = "total"
            popout_names = header[1:-1]
            rf_counts: dict[str, list[int]] = {}
            for line in f:
                # Comment lines (sidecar footnotes) must never become
                # rf_label rows. The legacy ``# n_low_confidence`` row
                # in ``rf_confusion_matrix.tsv`` has been retired, but
                # the guard stays so future sidecars can't pollute the
                # confusion table.
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                rf_label = parts[0]
                if rf_label == "total":
                    continue
                vec = []
                for idx in range(len(popout_names)):
                    try:
                        vec.append(int(parts[1 + idx]))
                    except (IndexError, ValueError):
                        vec.append(0)
                rf_counts[rf_label] = vec
        rows = _apply_mid_rule(
            art.cluster_id, art.chrom, popout_names, rf_counts,
            mid_rule=mid_rule,
        )
        _append_lines(out_path, rows)


def collate_confusion_rye(
    arts: list[ClusterArtifact], out_path: Path,
) -> bool:
    """Unpivot ``concordance/rye_confusion_matrix.tsv`` into long form:
    ``(cluster_id, chrom, flare_call, rye_call, n)``. Rye carries no MID
    column so no ``mid_rule`` is meaningful here. Returns True if any
    cluster had Rye data.

    The source confusion is FLARE-primary rows x Rye-primary cols (see
    ``compare_to_rye.py``'s ``write_hard_confusion`` call). We unpivot
    that orientation faithfully.
    """
    _write_header_once(
        out_path, "cluster_id\tchrom\tflare_call\trye_call\tn",
    )
    any_present = False
    for art in arts:
        src = art.artifact_dir / "concordance" / "rye_confusion_matrix.tsv"
        if not src.exists():
            continue
        any_present = True
        with open(src) as f:
            header = f.readline().rstrip("\n").split("\t")
            # header[0] = "flare_primary", header[1..-1] = rye label names,
            # header[-1] = "total".
            rye_labels = header[1:-1]
            rows_out: list[str] = []
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                flare_call = parts[0]
                if flare_call == "total":
                    continue
                for j, rye_label in enumerate(rye_labels):
                    try:
                        n = int(parts[1 + j])
                    except (IndexError, ValueError):
                        n = 0
                    rows_out.append(
                        f"{art.cluster_id}\t{art.chrom}\t"
                        f"{flare_call}\t{rye_label}\t{n}"
                    )
            _append_lines(out_path, rows_out)
    if not any_present:
        out_path.unlink(missing_ok=True)
    return any_present


def collate_calibration_slope(
    arts: list[ClusterArtifact], out_path: Path, *, mid_rule: str = "none",
) -> None:
    """Unpivot the wide slope_matrix.tsv into long form.

    Category B (FLARE component x RF label). ``mid_rule`` selects MID
    handling on the RF axis:

    - ``"none"`` — pass MID through (legacy behaviour).
    - ``"drop"`` — omit ``rf_label == "mid"`` cells.
    - ``"fold_to_eur"`` — calibration slope and max_cal are not
      summable across RF columns; folding is undefined here, so the
      rule degrades to ``"drop"`` and is recorded in the manifest's
      ``transformations`` list.
    """
    if mid_rule not in ("none", "drop", "fold_to_eur"):
        raise ValueError(f"unknown mid_rule {mid_rule!r}")
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
                    if rf_lab == "mid" and mid_rule in ("drop", "fold_to_eur"):
                        continue
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


def collate_switch_rate_per_hap(arts: list[ClusterArtifact], out_path: Path) -> None:
    """Category A (FLARE-only) per-haplotype switch rate.

    One row per (cluster, chrom, sample, hap). Carries the haplotype's
    dominant FLARE ancestry so the report can stratify by FLARE
    top-1 without re-deriving anything from ``cohort_global.tsv``.
    """
    cols = ["cluster_id", "chrom", "sample_id", "hap", "dominant_anc", "n_switches"]
    _write_header_once(out_path, "\t".join(cols))
    for art in arts:
        src = art.artifact_dir / "structural" / "switch_rate_per_hap.tsv"
        if not src.exists():
            raise RuntimeError(
                f"{src} missing; rebuild the per-cluster artifact under "
                f"schema v5 (write_structural_outputs emits per-hap rows)"
            )
        with open(src) as f:
            next(f)            # skip header
            rows = [f"{art.cluster_id}\t{art.chrom}\t{line.rstrip()}"
                    for line in f if line.strip()]
        _append_lines(out_path, rows)


def collate_hap_disagreement(arts: list[ClusterArtifact], out_path: Path) -> None:
    """Category A (FLARE-only) metric: hap1-vs-hap2 disagreement.

    Bundled keyed by **FLARE's per-sample top-1 ancestry**. RF never
    enters the schema. The per-sample TSV emitted by
    ``validate_per_site_metrics.write_hap_disagreement_outputs`` carries
    the raw rf_label + rf_max_prob for downstream filtering; the
    aggregate here is FLARE-keyed only.
    """
    cols = ["cluster_id", "chrom", "flare_top1", "n", "mean", "median"]
    _write_header_once(out_path, "\t".join(cols))
    rows = []
    for art in arts:
        d = json.loads((art.artifact_dir / "hap_disagreement" / "summary.json").read_text())
        for entry in d.get("per_flare_top1", []):
            rows.append("\t".join([
                art.cluster_id, art.chrom, entry["flare_top1"],
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
            cm = g.artifact_dir / "concordance" / "concordance_metrics_rye.tsv"
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


def _build_provenance_block(
    *, mid_rule: str, thresholds: dict, generated_at: str,
) -> dict:
    """Build the cohort_manifest.json ``provenance`` block (v3.0.0+).

    Records the transformations the collator baked into the bundle so
    the report (and downstream consumers) can render a faithful
    figure-shorthand tag without re-deriving anything. See
    ``my_notes/validation/COLLECTOR_FIXES.md`` §3.
    """
    from popout.labelspace.shorthand import format as _format_tag
    from popout.labelspace.matching import by_name as _by_name
    from popout.labelspace.registry import SP5, SP6

    # v3.0.0 FLARE bundle: labels.json is built via by_name → SP5
    # (MID isn't a FLARE panel population; only RF has MID). Tag the
    # cohort accordingly. SP5 is the target space; mid_rule names how
    # RF's MID column was handled on the RF side.
    flare_panel = SP5.members
    rf_panel = SP6.members
    flare_assignment = _by_name(flare_panel, SP5, source={"tool": "flare"})
    rf_assignment = _by_name(rf_panel, SP5, source={"tool": "rf"})
    tag = _format_tag(SP5, [flare_assignment, rf_assignment], mid_rule=mid_rule)

    transformations: list[dict] = [
        {"step": "flare_to_estimate",
         "input_format": "flare_global_anc_named_columns",
         "output_format": "popout_format_global_tsv"},
        {"step": "labels_via_by_name",
         "matching": "by_name",
         "target_space": "SP5"},
    ]
    if mid_rule in ("drop", "fold_to_eur"):
        # Category B tables that actually receive the rule. Only these
        # three are FLARE x RF cross-tabs where RF's MID column exists
        # as real data. Hap_disagreement, switch_rate, and tract_length
        # are category A (FLARE-only) and therefore never carry MID at
        # all; they don't need the rule and don't appear here.
        rule_applied_to = [
            "cohort/confusion_rf.tsv",
            "cohort/calibration_slope.tsv",
            "cohort/soft_correlation_rf.tsv",
        ]
        if mid_rule == "fold_to_eur":
            note = ("calibration_slope and soft_correlation_rf cells "
                    "are not summable across RF labels; the fold "
                    "degrades to drop for those tables.")
        else:
            note = None
        entry = {
            "step": "mid_rule",
            "rule": mid_rule,
            "applied_to": rule_applied_to,
        }
        if note:
            entry["note"] = note
        transformations.append(entry)

    return {
        "tag": tag,
        "target_space": "SP5",
        "mid_rule": mid_rule,
        "matching": {"flare": "by_name", "rf": "by_name"},
        "thresholds": dict(thresholds),
        "schema_version_built": SCHEMA_VERSION,
        "transformations": transformations,
        "generated_at": generated_at,
    }


def write_cohort_manifest(
    arts: list[ClusterArtifact], bundle_root: Path, *,
    run_name: str, collation_mode: str, collation_config: dict,
    diff_against: str | None, mid_rule: str, thresholds: dict,
) -> dict:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "collation_mode": collation_mode,
        "n_clusters": len(set(a.cluster_id for a in arts)),
        "n_chroms": len(set(a.chrom for a in arts)),
        "n_artifacts": len(arts),
        "cluster_ids": sorted({a.cluster_id for a in arts}),
        "chroms": sorted({a.chrom for a in arts}),
        "generated_at": generated_at,
        "sha256_per_artifact": {f"{a.cluster_id}.{a.chrom}": a.sha256 for a in arts},
        "diff_against": diff_against,
        "collation_config": collation_config,
        "provenance": _build_provenance_block(
            mid_rule=mid_rule,
            thresholds=thresholds,
            generated_at=generated_at,
        ),
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
    p.add_argument(
        "--mid-rule", choices=("none", "drop", "fold_to_eur"), default="none",
        help=(
            "How to handle the RF MID column when collating "
            "cohort/confusion_rf.tsv. FLARE's panel has no MID component "
            "(SP5); RF emits SP6 including MID. ``none`` (default) keeps "
            "MID rows as-is, ``drop`` removes them, ``fold_to_eur`` sums "
            "MID counts into the EUR row. The chosen rule is recorded in "
            "cohort_manifest.json.provenance.mid_rule and surfaces in "
            "every figure footer's shorthand tag."
        ),
    )
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
    collate_soft_correlation_rf(arts,     cohort_dir / "soft_correlation_rf.tsv",
                                mid_rule=args.mid_rule)
    collate_merged_groups_rf(arts,        cohort_dir / "merged_groups_rf.tsv")
    # Per-tool concordance (FLARE vs Rye and FLARE vs RF). Same row
    # shape from the shared concordance.py module. Rye block is optional
    # (gated on rye_q per cluster); RF block fires whenever
    # compare_to_rf ran by_name.
    has_rye = collate_concordance_metrics_rye(
        arts,                              cohort_dir / "concordance_metrics_rye.tsv")
    has_rf_conc = collate_concordance_metrics_rf(
        arts,                              cohort_dir / "concordance_metrics_rf.tsv")
    collate_confusion_rf(arts,            cohort_dir / "confusion_rf.tsv",
                         mid_rule=args.mid_rule)
    # Symmetric Rye-side hard-call confusion (cluster_id, chrom,
    # rye_label, flare_call, n). Gated on rye_q.
    has_rye_conf = collate_confusion_rye(
        arts,                              cohort_dir / "confusion_rye.tsv")
    collate_calibration_slope(arts,       cohort_dir / "calibration_slope.tsv",
                              mid_rule=args.mid_rule)
    collate_tract_length_stats(arts,      cohort_dir / "tract_length_stats.tsv")
    collate_switch_rate_stats(arts,       cohort_dir / "switch_rate_stats.tsv")
    collate_switch_rate_per_hap(arts,     cohort_dir / "switch_rate_per_hap.tsv")
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
    write_cohort_manifest(
        arts, bundle_root,
        run_name=args.run_name, collation_mode=collation_mode,
        collation_config=config,
        diff_against=str(args.diff_against) if args.diff_against else None,
        mid_rule=args.mid_rule,
        thresholds=thresholds,
    )

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
         f"RF concordance rows: {'present' if has_rf_conc else 'absent'}; "
         f"Rye confusion rows: {'present' if has_rye_conf else 'absent'}; "
         f"self_id rows: {'present' if has_self_id else 'absent'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
