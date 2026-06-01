"""popout DX artifact schema — executable contract.

See ``validation/popout_dx/SCHEMA.md`` for the prose contract. Layout
mirrors ``validation/schema.py`` so cohort tooling can be written
generically against either suite.

Gating model
------------

``tools`` (manifest field) lists the comparison axes that were active
for the run. It is always a subset of ``{popout, flare, rye, rf}`` and
``popout`` must be present. ``optional_inputs`` keys mirror the
non-anchor tools and the local-mode flag — when a tool is in ``tools``
the corresponding ``optional_inputs`` flag is true and the gated files
must exist.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "1.0.0"


# ── Tool / mode vocabulary ────────────────────────────────────────────────

ANCHOR_TOOL = "popout"
COMPARISON_TOOLS: tuple[str, ...] = ("flare", "rye", "rf")
ALL_TOOLS: tuple[str, ...] = (ANCHOR_TOOL,) + COMPARISON_TOOLS
MODES: tuple[str, ...] = ("global", "global_local")


# ── Per-cluster artifact layout ───────────────────────────────────────────

REQUIRED_CLUSTER_FILES: tuple[str, ...] = (
    "manifest.json",
    "tier1_metrics.tsv",
    "popout.global.tsv",
    "labels.json",
    # global/ — pairwise summary + per-sample MAE always present (cohort
    # collation depends on stable columns even when a tool is absent).
    "global/pairwise_soft/per_sample_mae.tsv",
    "global/pairwise_soft/summary.json",
    # provenance/
    "provenance/schema_version.txt",
    "provenance/dx_config.json",
)


# Each key here is a string that appears in manifest.json["optional_inputs"]
# with a bool value. When the value is true, every file in the tuple must
# exist; when false, none of them may.
OPTIONAL_CLUSTER_FILE_GROUPS: dict[str, tuple[str, ...]] = {
    "flare": (
        "global/pairwise_hard/popout_vs_flare.confusion.tsv",
        "global/pairwise_soft/popout_vs_flare.metrics.tsv",
    ),
    "rye": (
        "global/pairwise_hard/popout_vs_rye.confusion.tsv",
        "global/pairwise_soft/popout_vs_rye.metrics.tsv",
    ),
    "rf": (
        "global/pairwise_hard/popout_vs_rf.confusion.tsv",
        "global/pairwise_soft/popout_vs_rf.metrics.tsv",
    ),
    "local_mode": (
        "local/selected_samples.tsv",
        "local/local_per_sample.tsv",
        "local/local_per_haplotype.tsv",
        "local/local_summary.json",
        "local/views/bp_confusion_segments.tsv.gz",
        "local/views/boundary_localization.tsv",
        "local/views/coarse_grid_summary.tsv",
    ),
}


REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "schema_version",
    "cluster_id",
    "chrom",
    "run_name",
    "mode",
    "tools",
    "n_samples",
    "n_ancestries_popout",
    "popout_run_dir",
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
    "cohort/manifest.tsv",
    "cohort/tier1_metrics.tsv",
    "cohort/per_sample_mae.tsv",
    "cohort/pairwise_soft_summary.tsv",
)


# Gated cohort files. Present iff at least one cluster in the bundle had
# the corresponding tool / mode active.
OPTIONAL_COHORT_FILE_GROUPS: dict[str, tuple[str, ...]] = {
    "flare": (
        "cohort/popout_vs_flare.confusion.tsv",
        "cohort/popout_vs_flare.metrics.tsv",
    ),
    "rye": (
        "cohort/popout_vs_rye.confusion.tsv",
        "cohort/popout_vs_rye.metrics.tsv",
    ),
    "rf": (
        "cohort/popout_vs_rf.confusion.tsv",
        "cohort/popout_vs_rf.metrics.tsv",
    ),
    "local_mode": (
        "cohort/local_per_sample.tsv",
        "cohort/bp_confusion_segments.tsv.gz",
        "cohort/boundary_localization.tsv",
        "cohort/coarse_grid_summary.tsv",
    ),
}


REQUIRED_COHORT_MANIFEST_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_name",
    "mode",
    "tools",
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
    severity: str   # "error" or "warning"
    path: str       # relative path inside the artifact / bundle
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper():>7}] {self.path}: {self.message}"


# ── Validation ────────────────────────────────────────────────────────────


def _validate_tools_field(tools: object, where: str) -> list[SchemaIssue]:
    issues: list[SchemaIssue] = []
    if not isinstance(tools, list):
        issues.append(SchemaIssue("error", where, f"tools must be a list, got {type(tools).__name__}"))
        return issues
    if ANCHOR_TOOL not in tools:
        issues.append(SchemaIssue("error", where, f"tools must include the anchor {ANCHOR_TOOL!r}"))
    extras = [t for t in tools if t != ANCHOR_TOOL]
    if not extras:
        issues.append(SchemaIssue("error", where, "tools must include at least one comparison tool besides popout"))
    for t in tools:
        if t not in ALL_TOOLS:
            issues.append(SchemaIssue("error", where, f"unknown tool {t!r}; allowed: {ALL_TOOLS}"))
    return issues


def validate_cluster_artifact(root: Path) -> list[SchemaIssue]:
    """Validate an unpacked per-cluster DX artifact directory."""
    issues: list[SchemaIssue] = []

    for rel in REQUIRED_CLUSTER_FILES:
        if not (root / rel).exists():
            issues.append(SchemaIssue("error", rel, "missing required file"))

    manifest_path = root / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            issues.append(SchemaIssue("error", "manifest.json", f"invalid JSON: {e}"))
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

            mode = manifest.get("mode")
            if mode not in MODES:
                issues.append(SchemaIssue("error", "manifest.json", f"mode {mode!r} not in {MODES}"))

            issues.extend(_validate_tools_field(manifest.get("tools"), "manifest.json:tools"))

            opt = manifest.get("optional_inputs", {})
            if isinstance(opt, dict):
                # local_mode flag must agree with the mode field.
                claims_local = bool(opt.get("local_mode", False))
                if mode == "global_local" and not claims_local:
                    issues.append(SchemaIssue(
                        "error", "manifest.json",
                        "mode=global_local but optional_inputs.local_mode is false",
                    ))
                if mode == "global" and claims_local:
                    issues.append(SchemaIssue(
                        "error", "manifest.json",
                        "mode=global but optional_inputs.local_mode is true",
                    ))
                # Comparison-tool flags must agree with the tools list.
                tools_list = manifest.get("tools", []) if isinstance(manifest.get("tools"), list) else []
                for tool in COMPARISON_TOOLS:
                    declared_in_tools = tool in tools_list
                    declared_in_opt = bool(opt.get(tool, False))
                    if declared_in_tools != declared_in_opt:
                        issues.append(SchemaIssue(
                            "error", "manifest.json",
                            f"tools-list / optional_inputs disagreement for {tool!r}: "
                            f"tools={declared_in_tools} optional_inputs={declared_in_opt}",
                        ))

                # Files gated on each flag.
                for flag, files in OPTIONAL_CLUSTER_FILE_GROUPS.items():
                    claimed = bool(opt.get(flag, False))
                    for rel in files:
                        present = (root / rel).exists()
                        if claimed and not present:
                            issues.append(SchemaIssue(
                                "error", rel,
                                f"optional input {flag!r} claimed in manifest but file missing",
                            ))
                        elif present and not claimed:
                            issues.append(SchemaIssue(
                                "warning", rel,
                                f"file present but optional input {flag!r} not claimed",
                            ))

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
    """Validate an unpacked DX cohort bundle directory."""
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
            mode = manifest.get("mode")
            if mode not in MODES:
                issues.append(SchemaIssue("error", "cohort_manifest.json", f"mode {mode!r} not in {MODES}"))
            issues.extend(_validate_tools_field(manifest.get("tools"), "cohort_manifest.json:tools"))

            tools_list = manifest.get("tools", []) if isinstance(manifest.get("tools"), list) else []
            present_flags = {
                "flare": "flare" in tools_list,
                "rye": "rye" in tools_list,
                "rf": "rf" in tools_list,
                "local_mode": mode == "global_local",
            }
            for flag, files in OPTIONAL_COHORT_FILE_GROUPS.items():
                expected = present_flags.get(flag, False)
                for rel in files:
                    present = (root / rel).exists()
                    if expected and not present:
                        issues.append(SchemaIssue(
                            "error", rel,
                            f"expected for {flag!r} active in cohort but file missing",
                        ))
                    elif present and not expected:
                        issues.append(SchemaIssue(
                            "warning", rel,
                            f"file present but {flag!r} not active in cohort",
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
    """Tar ``cluster_root`` into ``out_tarball`` under prefix ``<cluster_id>/<chrom>/``."""
    if not cluster_root.is_dir():
        raise NotADirectoryError(cluster_root)
    out_tarball.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{cluster_id}/{chrom}"
    with tarfile.open(out_tarball, "w:gz") as tar:
        tar.add(cluster_root, arcname=prefix)


def read_cluster_artifact(tarball: Path, dest_dir: Path) -> Path:
    """Untar a per-cluster DX artifact into ``dest_dir`` and return the inner chrom dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError(f"{tarball} is empty")
        top = members[0].name.split("/", 1)[0]
        tar.extractall(dest_dir)
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
