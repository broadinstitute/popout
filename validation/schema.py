"""FLARE validation artifact schema — executable contract.

See validation/SCHEMA.md for the prose contract. This module provides:

  * SCHEMA_VERSION constant
  * REQUIRED_FILES / OPTIONAL_FILES manifests describing the per-cluster
    artifact layout and the cohort bundle layout
  * write_cluster_artifact / read_cluster_artifact tarball helpers
  * validate_cluster_artifact / validate_cohort_bundle that check a
    directory (or tarball) against the layout and return a list of
    `SchemaIssue`s

The orchestrator calls `validate_cluster_artifact` before tarring; the
collator calls it on every untarred artifact before concatenating. A
non-empty issue list is a hard failure (CLAUDE.md: no silent drops).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "1.1.0"


# ── Per-cluster artifact layout ───────────────────────────────────────────
#
# Paths are relative to the artifact root (i.e. inside the tarball, drop
# the leading "<cluster_id>/<chrom>/" prefix). Required files MUST be
# present in every artifact. Optional files are conditional on the
# corresponding input being supplied at orchestration time.

REQUIRED_CLUSTER_FILES: tuple[str, ...] = (
    "manifest.json",
    "tier1_metrics.tsv",
    "global.tsv",
    # coverage/
    "coverage/coverage_check.tsv",
    "coverage/per_chrom_consistency.tsv",
    # model/
    "model/mu_vs_global_diff.json",
    # soft_correlation/
    "soft_correlation/labels.json",
    "soft_correlation/rf_soft_correlation.tsv",
    "soft_correlation/rf_merged_groups.tsv",
    "soft_correlation/popout_composition.tsv",
    # confusion/
    "confusion/rf_confusion_matrix.tsv",
    # concordance/
    "concordance/SUMMARY.md",
    # NOTE: concordance/{concordance_metrics.tsv, concordance_summary.json,
    # rye_*} are v1.1 additions gated on rye_q — see OPTIONAL_CLUSTER_FILE_GROUPS.
    # calibration/
    "calibration/slope_matrix.tsv",
    "calibration/notes.txt",
    # structural/
    "structural/tract_length_summary.json",
    "structural/switch_rate_summary.json",
    # hap_disagreement/
    "hap_disagreement/per_sample.tsv",
    "hap_disagreement/summary.json",
    # regional/
    "regional/windows.tsv.gz",
    "regional/significant.bed",
    "regional/summary.json",
    # provenance/
    "provenance/flare_command_line.txt",
    "provenance/flare_log_tail.txt",
    "provenance/input_vcf_header.txt",
    "provenance/schema_version.txt",
    # NOTE: provenance/flare_qc.tsv is v1.1 optional (gated on flare_qc_tsv).
    # NOTE: provenance/ref_target_concordance.{tsv,json} are required v1.1
    # additions — R6 only needs ref_vcf + input_vcf which are always provided.
    "provenance/ref_target_concordance.tsv",     # ★ v1.1
    "provenance/ref_target_concordance_summary.json",  # ★ v1.1
)


# Optional files keyed by the input flag that gates their existence. When
# the orchestrator skips a step, the entire group is absent (no partial
# emit).
OPTIONAL_CLUSTER_FILE_GROUPS: dict[str, tuple[str, ...]] = {
    # ★ v1.1: renamed from admixture_q; gated on the caller-supplied Rye Q TSV.
    # When rye_q is provided, the orchestrator runs compare_to_rye.py and
    # emits the concordance metrics + Rye output family.
    "rye_q": (
        "concordance/concordance_metrics.tsv",
        "concordance/concordance_summary.json",
        "concordance/rye_full_matrix.tsv",
        "concordance/rye_merged_groups.tsv",
        "concordance/rye_confusion_matrix.tsv",
    ),
    "self_id": (
        "self_id/check.tsv",
        "self_id/summary.json",
    ),
    "region_bed": (
        "hap_disagreement/per_region.tsv",
    ),
    # F_ST tree is gated on having a popout-style .model.npz with per-site
    # allele frequencies; FLARE-source artifacts do not have this and the
    # orchestrator skips the step (see SCHEMA.md §1.4).
    "fst_tree": (
        "model/fst_matrix.tsv",
    ),
    # ★ v1.1: gated on the FLARE pipeline having emitted qc.tsv. Pre-pipeline
    # test fixtures lack qc.tsv; the orchestrator marks the qc-dependent
    # coverage checks SKIP and the per_chrom_consistency.tsv stays header-only.
    "flare_qc_tsv": (
        "provenance/flare_qc.tsv",
    ),
}


# Manifest.json keys that are required for downstream consumption.
REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "schema_version",
    "cluster_id",
    "chrom",
    "run_name",
    "flare_version",
    "flare_command_line",
    "input_vcf_sha",
    "n_samples",
    "n_markers",
    "n_ancestries",
    "coverage_passed",
    "steps",
    "total_wallclock_seconds",
    "peak_rss_gb",
    "cpu_wall_ratio",
    "generated_at",
    "optional_inputs",
)


# ── Cohort bundle layout ──────────────────────────────────────────────────

REQUIRED_COHORT_FILES: tuple[str, ...] = (
    "cohort_manifest.json",
    "cohort_summary.json",
    "cohort_qc_dashboard.json",
    "cohort/cohort_global.tsv",
    "cohort/coverage.tsv",
    "cohort/manifest.tsv",
    "cohort/tier1_metrics.tsv",
    "cohort/soft_correlation_rf.tsv",
    "cohort/merged_groups_rf.tsv",
    "cohort/confusion_rf.tsv",
    "cohort/calibration_slope.tsv",
    "cohort/tract_length_stats.tsv",
    "cohort/switch_rate_stats.tsv",
    "cohort/hap_disagreement.tsv",
    "cohort/regional_windows.tsv.gz",
    "cohort/regional_meta.tsv",
    "cohort/ref_target_concordance.tsv",   # ★ v1.1 — always emitted (R6 always runs)
)


OPTIONAL_COHORT_FILES: tuple[str, ...] = (
    "cohort/concordance_metrics.tsv",   # ★ v1.1 — present iff any artifact had rye_q
    "cohort/self_id.tsv",
    "cohort/fst_matrix.tsv",            # only for popout-source artifacts
)


REQUIRED_COHORT_MANIFEST_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_name",
    "collation_mode",
    "n_clusters",
    "n_chroms",
    "n_artifacts",
    "cluster_ids",
    "chroms",
    "generated_at",
    "sha256_per_artifact",
)


# ── Issue type ────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SchemaIssue:
    severity: str          # "error" or "warning"
    path: str              # relative path inside the artifact / bundle
    message: str

    def __str__(self) -> str:  # human-friendly for CLI dumps
        return f"[{self.severity.upper():>7}] {self.path}: {self.message}"


# ── Validation ────────────────────────────────────────────────────────────


def validate_cluster_artifact(root: Path) -> list[SchemaIssue]:
    """Validate an unpacked per-cluster artifact directory.

    `root` is the directory that contains `manifest.json` (i.e. the layer
    *below* `<cluster_id>/<chrom>/`). Caller should untar to a staging
    dir and pass the inner directory.
    """
    issues: list[SchemaIssue] = []

    # 1. All required files present.
    for rel in REQUIRED_CLUSTER_FILES:
        if not (root / rel).exists():
            issues.append(SchemaIssue("error", rel, "missing required file"))

    # 2. Manifest is well-formed JSON with the required keys and matches
    #    SCHEMA_VERSION.
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            issues.append(SchemaIssue("error", "manifest.json", f"invalid JSON: {e}"))
            manifest = None
        if isinstance(manifest, dict):
            for key in REQUIRED_MANIFEST_KEYS:
                if key not in manifest:
                    issues.append(SchemaIssue("error", "manifest.json", f"missing key {key!r}"))
            sv = manifest.get("schema_version")
            if sv != SCHEMA_VERSION:
                issues.append(SchemaIssue(
                    "error", "manifest.json",
                    f"schema_version {sv!r} != expected {SCHEMA_VERSION!r}",
                ))
            # If the manifest claims an optional input was supplied, the
            # corresponding files must be present (and vice versa).
            opt = manifest.get("optional_inputs", {})
            if isinstance(opt, dict):
                for flag, files in OPTIONAL_CLUSTER_FILE_GROUPS.items():
                    claimed = bool(opt.get(flag, False))
                    for rel in files:
                        present = (root / rel).exists()
                        if claimed and not present:
                            issues.append(SchemaIssue(
                                "error", rel,
                                f"optional input {flag!r} was claimed in manifest but file missing",
                            ))
                        elif present and not claimed:
                            issues.append(SchemaIssue(
                                "warning", rel,
                                f"file present but manifest says optional input {flag!r} was not supplied",
                            ))

    # 3. schema_version.txt mirrors manifest.json's value.
    sv_txt = root / "provenance" / "schema_version.txt"
    if sv_txt.exists():
        v = sv_txt.read_text().strip()
        if v != SCHEMA_VERSION:
            issues.append(SchemaIssue(
                "error", "provenance/schema_version.txt",
                f"value {v!r} != expected {SCHEMA_VERSION!r}",
            ))

    return issues


def validate_cohort_bundle(root: Path) -> list[SchemaIssue]:
    """Validate an unpacked cohort bundle directory."""
    issues: list[SchemaIssue] = []

    for rel in REQUIRED_COHORT_FILES:
        if not (root / rel).exists():
            issues.append(SchemaIssue("error", rel, "missing required file"))

    manifest_path = root / "cohort_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            issues.append(SchemaIssue("error", "cohort_manifest.json", f"invalid JSON: {e}"))
            manifest = None
        if isinstance(manifest, dict):
            for key in REQUIRED_COHORT_MANIFEST_KEYS:
                if key not in manifest:
                    issues.append(SchemaIssue("error", "cohort_manifest.json", f"missing key {key!r}"))
            sv = manifest.get("schema_version")
            if sv != SCHEMA_VERSION:
                issues.append(SchemaIssue(
                    "error", "cohort_manifest.json",
                    f"schema_version {sv!r} != expected {SCHEMA_VERSION!r}",
                ))

    return issues


# ── Tarball helpers ───────────────────────────────────────────────────────


def write_cluster_artifact(
    cluster_root: Path,
    out_tarball: Path,
    *,
    cluster_id: str,
    chrom: str,
) -> None:
    """Tar `cluster_root` into `out_tarball` under prefix `<cluster_id>/<chrom>/`.

    Caller is expected to have already populated `cluster_root` with the
    full §2.3 layout and called `validate_cluster_artifact(cluster_root)`
    to confirm completeness.
    """
    if not cluster_root.is_dir():
        raise NotADirectoryError(cluster_root)
    out_tarball.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{cluster_id}/{chrom}"
    with tarfile.open(out_tarball, "w:gz") as tar:
        tar.add(cluster_root, arcname=prefix)


def read_cluster_artifact(tarball: Path, dest_dir: Path) -> Path:
    """Untar a per-cluster artifact into `dest_dir`.

    Returns the path to the artifact root (`dest_dir/<cluster_id>/<chrom>`)
    by inspecting the tarball's top-level entries.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError(f"{tarball} is empty")
        top = members[0].name.split("/", 1)[0]
        tar.extractall(dest_dir)
    # Drill to the chrom dir. Layout is dest_dir/<cluster_id>/<chrom>/...
    cluster_dir = dest_dir / top
    chrom_dirs = [p for p in cluster_dir.iterdir() if p.is_dir()]
    if len(chrom_dirs) != 1:
        raise RuntimeError(
            f"{tarball}: expected exactly one chrom dir under {cluster_dir}, "
            f"got {[p.name for p in chrom_dirs]}"
        )
    return chrom_dirs[0]


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ── Convenience: stream issues to stderr ──────────────────────────────────


def report_issues(issues: Iterable[SchemaIssue], *, label: str) -> int:
    """Print issues to stderr; return count of errors."""
    import sys
    n_err = 0
    for iss in issues:
        if iss.severity == "error":
            n_err += 1
        print(f"  {iss}", file=sys.stderr)
    if n_err:
        print(f"{label}: {n_err} schema error(s)", file=sys.stderr)
    return n_err
