#!/usr/bin/env python3
"""FLARE per-cluster validation orchestrator (Stage 1 of the validation WDL).

See `validation/SCHEMA.md` for the artifact contract this writes and
`my_notes/validation/PLAN.md` §2.2 for the design.

The diagnostic sub-scripts (siblings in validation/scripts/) form a
DAG with two independent tracks once setup completes. We execute the
DAG via concurrent.futures.ThreadPoolExecutor sized to --max-workers so
that, e.g., ref_target_concordance can run while compare_to_rf is
producing labels.json.

Each step emits a structured stderr phase block on entry and exit,
captures wallclock + peak RSS via resource.getrusage deltas, and
redirects the sub-script's combined stdout/stderr to a per-step log
file. Step failures propagate unless the step is declared
nonfatal-on-exit; the coverage step is the only nonfatal one and sets
manifest.coverage_passed=False rather than aborting.

After the DAG drains, the orchestrator (a) computes derived files
(rf_merged_groups.tsv, calibration/notes.txt, provenance/*), (b) writes
manifest.json + tier1_metrics.tsv, (c) validates the artifact tree
against the schema, and (d) tars the result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Allow `import validation.schema` from this file's parent's parent
# (which is the repo root, both locally and inside the docker at /opt/).
VALIDATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATION_DIR.parent))
from validation.schema import (
    SCHEMA_VERSION,
    OPTIONAL_CLUSTER_FILE_GROUPS,
    report_issues,
    sha256_file,
    validate_cluster_artifact,
    write_cluster_artifact,
)


# Where the diagnostic sub-scripts live.
SCRIPTS_DIR = Path(__file__).resolve().parent


# ── Phase-boundary logging ────────────────────────────────────────────────


def _log(msg: str) -> None:
    """Single-line stderr message with ISO-second timestamp."""
    ts = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def _phase_open(step_no: int, name: str, cluster_id: str, chrom: str) -> None:
    print(
        f"===== flare_validate step {step_no:>2}: {name} | "
        f"cluster={cluster_id} chrom={chrom} =====",
        file=sys.stderr, flush=True,
    )


def _phase_close(
    step_no: int, name: str, *, wallclock: float, peak_rss_gb: float, exit_code: int, status: str,
) -> None:
    print(
        f"===== flare_validate step {step_no:>2}: {name} DONE | "
        f"wallclock={wallclock:.1f}s | peak_rss_gb={peak_rss_gb:.2f} | "
        f"exit={exit_code} | status={status} =====",
        file=sys.stderr, flush=True,
    )


# ── Step abstraction ──────────────────────────────────────────────────────


@dataclasses.dataclass
class Step:
    no: int                          # step number (1..N) for log messages
    name: str
    depends_on: tuple[str, ...]      # names of steps that must complete first
    runner: callable                 # callable(work, scratch_dir) -> None, raises on failure
    optional_input_flag: str | None = None  # if set, step is skipped when not provided

    # Filled in after execution.
    status: str = "pending"          # pending | ok | skipped | failed
    exit_code: int = 0
    wallclock_seconds: float = 0.0
    peak_rss_gb: float = 0.0

    # Whether a non-zero exit should abort the orchestrator (coverage = no).
    nonfatal: bool = False


@dataclasses.dataclass
class Workspace:
    """Bundle of paths the steps share."""

    work_root: Path                  # work/<cluster_id>/<chrom>/
    scratch_root: Path               # work/scratch/
    logs_root: Path                  # work/<cluster_id>/<chrom>/logs/  (kept in the tarball)
    intermediates: dict[str, Path] = dataclasses.field(default_factory=dict)
    optional_inputs: dict[str, bool] = dataclasses.field(default_factory=dict)

    def subdir(self, name: str) -> Path:
        d = self.work_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d


# ── Sub-script invocation helper ──────────────────────────────────────────


def _run_subprocess(cmd: list[str], log_path: Path, *, step_name: str = "") -> int:
    """Run cmd, tee stdout+stderr to both `log_path` AND the orchestrator's
    stderr in real time. Return exit code.

    Live tee to stderr is non-negotiable for Cromwell: per-step log files
    only become accessible after the task completes (Cromwell streams the
    task's stderr/stdout to GCS during the run, but anything under the
    execution dir's work tree is invisible until the task finishes).
    Without live tee, hung sub-steps look identical to slow-but-progressing
    sub-steps. Each line is prefixed `[step_name]` so concurrent DAG
    streams don't blend into nonsense.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[{step_name}] " if step_name else ""
    # PYTHONUNBUFFERED disables Python's block-buffering on stdout/stderr
    # so child print()s flush per-line. Without this, child output sits
    # in a 4 KB buffer through any long C call (pysam.tabix_index, etc.)
    # and the live-tee shows nothing for minutes. Belt-and-suspenders:
    # also tee the command line into stderr so the operator sees what's
    # about to run before any of its own output appears.
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd_str = "$ " + " ".join(str(c) for c in cmd)
    sys.stderr.write(prefix + cmd_str + "\n")
    sys.stderr.flush()
    with open(log_path, "w") as logf:
        logf.write(cmd_str + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,           # line-buffered on the parent's read side
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            sys.stderr.write(prefix + line)
            sys.stderr.flush()
        proc.wait()
    return proc.returncode


# ── Step runners (one per DAG node) ───────────────────────────────────────
#
# Each runner takes (args, ws, log_dir) and is responsible for invoking
# its sub-script and moving outputs into the schema-mandated locations.
# Runners raise CalledProcessError-equivalents (RuntimeError) on failure;
# the executor catches and records.


class StepFailed(RuntimeError):
    """Raised by a step runner when its underlying sub-script returns non-zero."""

    def __init__(self, name: str, exit_code: int, log_path: Path):
        super().__init__(f"step {name!r} failed (exit={exit_code}); see {log_path}")
        self.exit_code = exit_code
        self.log_path = log_path


def _check(step: str, log_path: Path, rc: int) -> None:
    if rc != 0:
        raise StepFailed(step, rc, log_path)


def step_per_site_metrics(args, ws: Workspace, log_dir: Path) -> None:
    """Step 1: validate_per_site_metrics.py — fused single-pass collector.

    Replaces the old vcf_to_tracts → {structural, hap_disagreement, regional}
    fan-out. Streams the FLARE anc.vcf.gz once through bcftools query,
    advances three accumulators in lock-step, and writes the
    schema-mandated summary files directly under
    structural/, hap_disagreement/, regional/, model/.
    No tracts.tsv.gz is ever written.
    """
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "validate_per_site_metrics.py"),
        "--anc-vcf",     str(args.anc_vcf),
        "--global-tsv",  str(ws.intermediates["global_tsv"]),
        "--flare-model", str(args.flare_model),
        "--rf-ancestry", str(args.rf_ancestry),
        "--chrom-sizes", str(args.chrom_sizes),
        "--out-root",    str(ws.work_root),
    ]
    if "labels_json" in ws.intermediates:
        cmd += ["--labels-json", str(ws.intermediates["labels_json"])]
    if args.region_masks_dir is not None and args.region_masks_dir.is_dir():
        for bed in sorted(args.region_masks_dir.glob("*.bed")):
            cmd += ["--region-mask-bed", str(bed)]
    rc = _run_subprocess(
        cmd, log_dir / "01_per_site_metrics.log", step_name="per_site_metrics",
    )
    _check("per_site_metrics", log_dir / "01_per_site_metrics.log", rc)


def step_render_collector_pngs(args, ws: Workspace, log_dir: Path) -> None:
    """Step 7: render_collector_pngs.py — produces the 5 PNGs the schema
    lists under structural/, hap_disagreement/, regional/. Reads only the
    summary JSONs/TSVs that step_per_site_metrics just wrote — no VCF
    access, no tract streaming."""
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "render_collector_pngs.py"),
        "--out-root", str(ws.work_root),
    ]
    if args.region_masks_dir is not None and args.region_masks_dir.is_dir():
        for bed in sorted(args.region_masks_dir.glob("*.bed")):
            cmd += ["--region-mask-bed", str(bed)]
    rc = _run_subprocess(
        cmd, log_dir / "07_render_collector_pngs.log", step_name="render_collector_pngs",
    )
    _check("render_collector_pngs", log_dir / "07_render_collector_pngs.log", rc)


def step_to_popout_format(args, ws: Workspace, log_dir: Path) -> None:
    """Step 2: flare_to_popout_format.py → popout-style global.tsv etc."""
    # The script wants a flare-prefix; we lay out a symlink farm in scratch
    # so the FLARE outputs all sit next to each other under one stem.
    stem = f"{args.cluster_id}.{args.chrom}"
    stage = ws.scratch_root / "flare_stage"
    stage.mkdir(parents=True, exist_ok=True)
    flare_prefix = stage / stem
    _stage_symlinks(args, flare_prefix)

    out_dir = ws.scratch_root / "popout_format"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "flare_to_popout_format.py"),
         "--flare-prefix", str(flare_prefix),
         "--out-dir", str(out_dir),
         "--run-prefix", stem],
        log_dir / "02_flare_to_popout_format.log", step_name="to_popout_format",
    )
    _check("flare_to_popout_format", log_dir / "02_flare_to_popout_format.log", rc)
    ws.intermediates["popout_prefix"] = out_dir / stem
    ws.intermediates["global_tsv"] = out_dir / f"{stem}.global.tsv"
    ws.intermediates["popout_model"] = out_dir / f"{stem}.model"
    ws.intermediates["popout_summary"] = out_dir / f"{stem}.summary.json"


def _stage_symlinks(args, flare_prefix: Path) -> None:
    """Symlink the FLARE outputs into a single stem so flare_to_popout_format
    can find them by suffix.
    """
    mapping = {
        ".anc.vcf.gz": args.anc_vcf,
        ".global.anc.gz": args.global_anc,
        ".model": args.flare_model,
        ".log": args.flare_log,
    }
    if args.flare_qc_tsv is not None:
        mapping[".qc.tsv"] = args.flare_qc_tsv
    if args.flare_summary is not None:
        mapping[".summary.json"] = args.flare_summary
    for suffix, src in mapping.items():
        link = flare_prefix.with_name(flare_prefix.name + suffix)
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(src.resolve())
        except OSError:
            # filesystem doesn't support symlinks; fall back to copy
            shutil.copy2(src, link)


def step_coverage(args, ws: Workspace, log_dir: Path) -> None:
    """Step 3: validate_coverage.py — writes coverage/.

    Coverage failure (exit 1) is nonfatal: the manifest records
    coverage_passed=False and downstream steps still run.
    """
    out = ws.subdir("coverage")
    # ★ v1.1: pass --qc-tsv only when present; validate_coverage emits SKIP
    # rows for qc-dependent checks when omitted (pre-pipeline fixtures).
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "validate_coverage.py"),
        "--global-tsv", str(ws.intermediates["global_tsv"]),
        "--input-samples", str(args.input_vcf),
        "--flare-log", str(args.flare_log),
        "--out-dir", str(out),
    ]
    if args.flare_qc_tsv is not None:
        cmd += ["--qc-tsv", str(args.flare_qc_tsv)]
    rc = _run_subprocess(cmd, log_dir / "03_validate_coverage.log", step_name="coverage")
    # Mark in workspace so the manifest writer can read it.
    ws.intermediates["coverage_exit"] = rc
    if rc != 0:
        # Nonfatal: log a warning but allow the DAG to proceed.
        _log(f"WARN: validate_coverage exited {rc}; continuing with coverage_passed=False")


def step_compare_to_rf(args, ws: Workspace, log_dir: Path) -> None:
    """Step 4: compare_to_rf.py — writes labels.json + concordance summary."""
    scratch = ws.scratch_root / "compare_to_rf"
    scratch.mkdir(parents=True, exist_ok=True)
    rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "compare_to_rf.py"),
         "--popout-global", str(ws.intermediates["global_tsv"]),
         "--rf-ancestry", str(args.rf_ancestry),
         "--out-dir", str(scratch)],
        log_dir / "04_compare_to_rf.log", step_name="compare_to_rf",
    )
    _check("compare_to_rf", log_dir / "04_compare_to_rf.log", rc)

    # Route outputs into the schema-mandated locations.
    soft = ws.subdir("soft_correlation")
    conf = ws.subdir("confusion")
    conc = ws.subdir("concordance")

    shutil.move(str(scratch / "labels.json"),               soft / "labels.json")
    shutil.move(str(scratch / "soft_correlation.tsv"),      soft / "rf_soft_correlation.tsv")
    shutil.move(str(scratch / "popout_composition.tsv"),    soft / "popout_composition.tsv")
    shutil.move(str(scratch / "confusion_matrix.tsv"),      conf / "rf_confusion_matrix.tsv")
    shutil.move(str(scratch / "SUMMARY.md"),                conc / "SUMMARY.md")
    pca_png = scratch / "pca_by_rf_label.png"
    if pca_png.exists():
        shutil.move(str(pca_png), conf / "pca_by_rf_label.png")

    ws.intermediates["labels_json"] = soft / "labels.json"

    # Derived: rf_merged_groups.tsv (long form of labels.json's merge_group_stats).
    labels = json.loads((soft / "labels.json").read_text())
    mgs = labels.get("merge_group_stats", {})
    with open(soft / "rf_merged_groups.tsv", "w") as f:
        f.write("rf_label\tmerged_r\tsummed_mu\tcomponent_indices\tcomponent_names\n")
        for rf_label, stats in mgs.items():
            indices = ",".join(str(i) for i in stats.get("indices", []))
            names = ",".join(stats.get("names", []))
            f.write(
                f"{rf_label}\t{stats.get('merged_r', 0.0):.6f}\t"
                f"{stats.get('summed_mu', 0.0):.6f}\t{indices}\t{names}\n"
            )


def step_compare_to_rye(args, ws: Workspace, log_dir: Path) -> None:
    """Step 5 (★ v1.1, optional): compare_to_rye.py.

    Outputs route into concordance/ (NOT soft_correlation/ as in v1.0).
    """
    scratch = ws.scratch_root / "compare_to_rye"
    scratch.mkdir(parents=True, exist_ok=True)
    rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "compare_to_rye.py"),
         "--global-tsv", str(ws.intermediates["global_tsv"]),
         "--rye-q", str(args.rye_q),
         "--labels-json", str(ws.intermediates["labels_json"]),
         "--out-dir", str(scratch)],
        log_dir / "05_compare_to_rye.log", step_name="compare_to_rye",
    )
    _check("compare_to_rye", log_dir / "05_compare_to_rye.log", rc)

    conc = ws.subdir("concordance")
    for name in ("concordance_metrics.tsv", "concordance_summary.json",
                 "rye_full_matrix.tsv", "rye_merged_groups.tsv",
                 "rye_confusion_matrix.tsv", "rye_admixture_comparison.png"):
        src = scratch / name
        if src.exists():
            shutil.move(str(src), conc / name)
    # rye_scatter_<label>.png — zero or more, one per ancestry with mu >= 0.01.
    for png in scratch.glob("rye_scatter_*.png"):
        shutil.move(str(png), conc / png.name)


def step_ref_target_concordance(args, ws: Workspace, log_dir: Path) -> None:
    """Step 5a (★ v1.1, R6): validate_ref_target_concordance.py.

    Outputs route into provenance/ per SCHEMA.md §1.13.
    """
    scratch = ws.scratch_root / "ref_target_concordance"
    scratch.mkdir(parents=True, exist_ok=True)
    rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "validate_ref_target_concordance.py"),
         "--ref-vcf", str(args.ref_vcf),
         "--input-vcf", str(args.input_vcf),
         "--chrom", args.chrom,
         "--out-dir", str(scratch)],
        log_dir / "05a_validate_ref_target_concordance.log", step_name="ref_target_concordance",
    )
    _check("validate_ref_target_concordance", log_dir / "05a_validate_ref_target_concordance.log", rc)
    prov = ws.subdir("provenance")
    for name in ("ref_target_concordance.tsv", "ref_target_concordance_summary.json"):
        src = scratch / name
        if src.exists():
            shutil.move(str(src), prov / name)


def step_self_id(args, ws: Workspace, log_dir: Path) -> None:
    """Step 6 (optional): validate_self_id.py."""
    out = ws.subdir("self_id")
    rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "validate_self_id.py"),
         "--global-tsv", str(ws.intermediates["global_tsv"]),
         "--self-id-tsv", str(args.self_id),
         "--labels-json", str(ws.intermediates["labels_json"]),
         "--out-dir", str(out)],
        log_dir / "06_validate_self_id.log", step_name="self_id",
    )
    _check("validate_self_id", log_dir / "06_validate_self_id.log", rc)


def step_plot_concordance(args, ws: Workspace, log_dir: Path) -> None:
    """Step 11: plot_concordance.py — writes the calibration figure set."""
    scratch = ws.scratch_root / "plot_concordance"
    scratch.mkdir(parents=True, exist_ok=True)
    extra: list[str] = []
    if args.popout_secondary_global and args.popout_secondary_labels:
        extra += [
            "--secondary-global", str(args.popout_secondary_global),
            "--secondary-labels", str(args.popout_secondary_labels),
        ]
    rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "plot_concordance.py"),
         "--popout-global", str(ws.intermediates["global_tsv"]),
         "--rf-ancestry", str(args.rf_ancestry),
         "--labels-json", str(ws.intermediates["labels_json"]),
         "--out-dir", str(scratch),
         *extra],
        log_dir / "11_plot_concordance.log", step_name="plot_concordance",
    )
    _check("plot_concordance", log_dir / "11_plot_concordance.log", rc)

    out = ws.subdir("calibration")
    # Move everything: rename slope matrix per schema, keep PNGs as-is.
    for src in scratch.iterdir():
        dst_name = "slope_matrix.tsv" if src.name == "calibration_slope_matrix.tsv" else src.name
        shutil.move(str(src), out / dst_name)


# ── Provenance helpers ────────────────────────────────────────────────────


def _write_provenance(args, ws: Workspace) -> None:
    out = ws.subdir("provenance")

    # flare_command_line.txt — best-effort: try the parsed Parameters block first.
    cmd_line = _extract_flare_command(args.flare_log)
    (out / "flare_command_line.txt").write_text(cmd_line + "\n")

    # flare_log_tail.txt — last 200 lines.
    log_lines = args.flare_log.read_text().splitlines()
    (out / "flare_log_tail.txt").write_text("\n".join(log_lines[-200:]) + "\n")

    # flare_qc.tsv — verbatim copy (★ v1.1: optional, gated on flare_qc_tsv).
    if args.flare_qc_tsv is not None:
        shutil.copy2(args.flare_qc_tsv, out / "flare_qc.tsv")

    # input_vcf_header.txt — bcftools view -h, with a python fallback.
    try:
        h = subprocess.run(
            ["bcftools", "view", "-h", str(args.input_vcf)],
            check=True, capture_output=True, text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        h = _pysam_vcf_header(args.input_vcf)
    (out / "input_vcf_header.txt").write_text(h)

    # schema_version.txt.
    (out / "schema_version.txt").write_text(SCHEMA_VERSION + "\n")


def _extract_flare_command(log_path: Path) -> str:
    """Best-effort reconstruction of the FLARE invocation from the log."""
    text = log_path.read_text()
    raw: dict[str, str] = {}
    in_params = False
    for line in text.splitlines():
        s = line.strip()
        if s == "Parameters":
            in_params = True
            continue
        if in_params:
            m = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.+)$", s)
            if m:
                raw[m.group(1)] = m.group(2).strip()
            elif s == "" or s.startswith("Statistics"):
                break
    if not raw:
        return "(no Parameters block found in log)"
    return "flare " + " ".join(f"--{k}={v}" for k, v in raw.items())


def _pysam_vcf_header(vcf: Path) -> str:
    try:
        import pysam
    except ImportError:
        return f"(bcftools and pysam both unavailable; cannot read header of {vcf})\n"
    return str(pysam.VariantFile(str(vcf)).header)


# ── Calibration notes ─────────────────────────────────────────────────────


def _write_calibration_notes(args, ws: Workspace) -> None:
    out = ws.subdir("calibration") / "notes.txt"
    # Look at the flare_log for --probs setting.
    probs_true = False
    try:
        text = args.flare_log.read_text()
        m = re.search(r"^\s*probs\s*:\s*(true|false)\s*$", text, re.MULTILINE)
        if m and m.group(1).lower() == "true":
            probs_true = True
    except Exception:
        pass
    if probs_true:
        out.write_text("probs=true, per-bin calibration curves available\n")
    else:
        out.write_text(
            "probs=false (FLARE default); calibration curves derived from hard "
            "calls only — slopes are still computed but error bars may be inflated\n"
        )


# ── Manifest writer ───────────────────────────────────────────────────────


def _parse_flare_log_for_manifest(log_path: Path) -> dict:
    out = {"flare_version": "unknown", "n_markers": 0, "n_ancestries": 0}
    text = log_path.read_text()
    m = re.search(r"Program\s*:\s*(.+)$", text, re.MULTILINE)
    if m:
        out["flare_version"] = m.group(1).strip()
    m = re.search(r"markers\s*:\s*(\d+)", text)
    if m:
        out["n_markers"] = int(m.group(1))
    return out


def _count_global_samples(global_tsv: Path) -> int:
    with open(global_tsv) as f:
        next(f)  # header
        return sum(1 for _ in f)


def _global_ncols(global_tsv: Path) -> int:
    with open(global_tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
    # First column is sample_id; remainder are ancestries.
    return len(header) - 1


def _write_manifest(
    args, ws: Workspace, steps: list[Step], total_wall: float, total_cpu_seconds: float,
) -> dict:
    log_info = _parse_flare_log_for_manifest(args.flare_log)
    n_samples = _count_global_samples(ws.intermediates["global_tsv"])
    n_ancestries = _global_ncols(ws.intermediates["global_tsv"])

    coverage_passed = ws.intermediates.get("coverage_exit", 0) == 0

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cluster_id": args.cluster_id,
        "chrom": args.chrom,
        "run_name": args.run_name or f"{args.cluster_id}.{args.chrom}",
        "flare_version": log_info["flare_version"],
        "flare_command_line": _extract_flare_command(args.flare_log),
        "panel_id": args.panel_id,
        "ref_panel_sha": (sha256_file(args.ref_panel) if args.ref_panel else None),
        "input_vcf_sha": sha256_file(args.input_vcf),
        "n_samples": n_samples,
        "n_markers": log_info["n_markers"],
        "n_ancestries": n_ancestries,
        "coverage_passed": coverage_passed,
        "steps": {
            s.name: {
                "no": s.no,
                "status": s.status,
                "exit": s.exit_code,
                "wallclock_seconds": s.wallclock_seconds,
                "peak_rss_gb": s.peak_rss_gb,
            } for s in steps
        },
        "total_wallclock_seconds": total_wall,
        "peak_rss_gb": max((s.peak_rss_gb for s in steps), default=0.0),
        "cpu_wall_ratio": (total_cpu_seconds / total_wall) if total_wall > 0 else 0.0,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "optional_inputs": ws.optional_inputs,
    }
    (ws.work_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ── Tier-1 metrics ────────────────────────────────────────────────────────


def _write_tier1_metrics(ws: Workspace, manifest: dict) -> None:
    """Write the 16-row tier1_metrics.tsv from the artifact's own outputs.

    Pulls from labels.json (merged-r per RF label), calibration slope_matrix,
    switch_rate_summary.json, hap_disagreement/summary.json, regional/summary.json.
    Any missing source becomes NA — the WDL emits it verbatim.
    """
    rows: list[tuple[str, str]] = []

    def add(key: str, value) -> None:
        rows.append((key, "NA" if value is None else str(value)))

    add("flare_validate.cluster_id", manifest["cluster_id"])
    add("flare_validate.chrom", manifest["chrom"])
    add("flare_validate.n_samples", manifest["n_samples"])
    add("flare_validate.n_markers", manifest["n_markers"])
    add("flare_validate.coverage_pass", str(manifest["coverage_passed"]).lower())

    # Merged-r per RF label from labels.json.
    labels_path = ws.work_root / "soft_correlation" / "labels.json"
    merged_r = {"afr": None, "amr": None, "eas": None, "eur": None, "sas": None}
    if labels_path.exists():
        try:
            mgs = json.loads(labels_path.read_text()).get("merge_group_stats", {})
            for lab in merged_r:
                if lab in mgs:
                    merged_r[lab] = round(float(mgs[lab].get("merged_r", 0.0)), 4)
        except json.JSONDecodeError:
            pass
    for lab in ("afr", "amr", "eas", "eur", "sas"):
        add(f"flare_validate.merged_r_{lab}", merged_r[lab])

    # Calibration slope max deviation from 1.0 across all (ancestry, RF label).
    # Reports NA when no slopes were computed (small cluster, sparse bins).
    slope_max_dev = None
    slope_tsv = ws.work_root / "calibration" / "slope_matrix.tsv"
    if slope_tsv.exists():
        max_dev = None
        with open(slope_tsv) as f:
            header = f.readline().rstrip("\n").split("\t")
            slope_cols = [i for i, h in enumerate(header) if h.endswith("_slope")]
            for line in f:
                parts = line.rstrip("\n").split("\t")
                for ci in slope_cols:
                    v = parts[ci] if ci < len(parts) else "NA"
                    if v == "NA":
                        continue
                    try:
                        dev = abs(float(v) - 1.0)
                        if max_dev is None or dev > max_dev:
                            max_dev = dev
                    except ValueError:
                        pass
        slope_max_dev = round(max_dev, 4) if max_dev is not None else None
    add("flare_validate.calibration_slope_max_dev", slope_max_dev)

    # Switch rate p99.
    sw = ws.work_root / "structural" / "switch_rate_summary.json"
    p99 = None
    if sw.exists():
        try:
            p99 = round(float(json.loads(sw.read_text()).get("p99", 0.0)), 2)
        except json.JSONDecodeError:
            pass
    add("flare_validate.switch_rate_p99", p99)

    # Hap disagreement mean.
    hd = ws.work_root / "hap_disagreement" / "summary.json"
    hd_mean = None
    if hd.exists():
        try:
            hd_mean = round(float(json.loads(hd.read_text()).get("cohort_mean_disagreement", 0.0)), 5)
        except json.JSONDecodeError:
            pass
    add("flare_validate.hap_disagreement_mean", hd_mean)

    # Regional significant count.
    rg = ws.work_root / "regional" / "summary.json"
    n_sig = None
    if rg.exists():
        try:
            n_sig = int(json.loads(rg.read_text()).get("n_windows_significant", 0))
        except json.JSONDecodeError:
            pass
    add("flare_validate.regional_significant_n", n_sig)

    # Resource metrics.
    add("flare_validate.peak_rss_gb", round(manifest["peak_rss_gb"], 2))
    add("flare_validate.cpu_wall_ratio", round(manifest["cpu_wall_ratio"], 3))

    # ★ v1.1: Rye concordance metrics from concordance/concordance_summary.json
    # + concordance/concordance_metrics.tsv. NA when rye_q not provided.
    conc_summary = ws.work_root / "concordance" / "concordance_summary.json"
    conc_metrics = ws.work_root / "concordance" / "concordance_metrics.tsv"
    global_ccc = None
    ccc_by_label: dict[str, float | None] = {lab: None for lab in ("afr", "amr", "eas", "eur", "sas")}
    if conc_summary.exists():
        try:
            d = json.loads(conc_summary.read_text())
            global_ccc = (
                round(float(d.get("global_ccc")), 4)
                if d.get("global_ccc") is not None else None
            )
        except (json.JSONDecodeError, TypeError):
            pass
    if conc_metrics.exists():
        with open(conc_metrics) as f:
            header = f.readline().rstrip("\n").split("\t")
            try:
                i_anc = header.index("ancestry")
                i_ccc = header.index("ccc")
                i_mu = header.index("cluster_mu")
            except ValueError:
                i_anc = i_ccc = i_mu = None
            if i_anc is not None:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= max(i_anc, i_ccc, i_mu):
                        continue
                    lab = parts[i_anc]
                    if lab not in ccc_by_label:
                        continue
                    try:
                        mu = float(parts[i_mu])
                        v = parts[i_ccc]
                        if v == "NA" or v == "" or mu < 0.01:
                            continue
                        ccc_by_label[lab] = round(float(v), 4)
                    except ValueError:
                        continue
    add("flare_validate.global_ccc", global_ccc)
    for lab in ("afr", "amr", "eas", "eur", "sas"):
        add(f"flare_validate.ccc_{lab}", ccc_by_label[lab])

    # ★ v1.1: R6 exact-overlap pct from provenance/ref_target_concordance_summary.json.
    r6_pct = None
    r6_summary = ws.work_root / "provenance" / "ref_target_concordance_summary.json"
    if r6_summary.exists():
        try:
            d = json.loads(r6_summary.read_text())
            r6_pct = round(float(d.get("exact_overlap_pct")), 4)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    add("flare_validate.ref_target_exact_overlap_pct", r6_pct)

    # Per-step wallclock. Already in manifest.json:steps.<name>.wallclock_seconds;
    # surfacing them as tier1 rows lets the WDL replay drop them into the
    # flare-validate W&B project as step.<name>.wallclock_seconds so a
    # cluster-by-cluster before/after chart is one click away.
    for step_name, step_info in manifest.get("steps", {}).items():
        wall = step_info.get("wallclock_seconds")
        add(f"step.{step_name}.wallclock_seconds",
            round(float(wall), 2) if wall is not None else None)

    with open(ws.work_root / "tier1_metrics.tsv", "w") as f:
        for k, v in rows:
            f.write(f"{k}\t{v}\n")


# ── DAG runner ────────────────────────────────────────────────────────────


def _run_dag(args, ws: Workspace, steps: list[Step]) -> tuple[float, float]:
    """Execute the DAG. Returns (total_wallclock_seconds, total_cpu_seconds)."""
    by_name = {s.name: s for s in steps}
    pending = {s.name for s in steps if s.status == "pending"}
    completed: set[str] = set()
    failed: list[str] = []

    overall_t0 = time.monotonic()

    def _step_done(s: Step) -> bool:
        return s.status in ("ok", "skipped", "failed")

    def _ready_now() -> list[Step]:
        ready = []
        for name in list(pending):
            s = by_name[name]
            if all(by_name[d].status in ("ok", "skipped") for d in s.depends_on):
                ready.append(s)
            elif any(by_name[d].status == "failed" for d in s.depends_on):
                # Upstream failed; mark skipped.
                s.status = "skipped"
                pending.discard(name)
        return ready

    def _execute(s: Step) -> Step:
        log_dir = ws.logs_root
        _phase_open(s.no, s.name, args.cluster_id, args.chrom)
        # Per-step wallclock + getrusage delta on RUSAGE_CHILDREN.
        t0 = time.monotonic()
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        step_log_path: Path | None = None
        try:
            s.runner(args, ws, log_dir)
            s.status = "ok"
            s.exit_code = 0
        except StepFailed as e:
            step_log_path = getattr(e, "log_path", None)
            if s.nonfatal:
                s.status = "ok"
                s.exit_code = e.exit_code
            else:
                s.status = "failed"
                s.exit_code = e.exit_code
        except Exception:
            s.status = "failed"
            s.exit_code = -1
            traceback.print_exc()
        finally:
            # On failure, dump the tail of the sub-script's log directly
            # into stderr so Cromwell preserves it next to the phase block
            # — saves the operator from digging through the execution dir
            # to figure out what went wrong.
            if s.status == "failed" and step_log_path is not None and step_log_path.exists():
                try:
                    lines = step_log_path.read_text().splitlines()
                except Exception:
                    lines = []
                tail = lines[-40:] if lines else []
                print(f"---- BEGIN tail of {step_log_path.name} ({len(tail)} lines) ----",
                      file=sys.stderr, flush=True)
                for line in tail:
                    print(line, file=sys.stderr, flush=True)
                print(f"---- END tail of {step_log_path.name} ----",
                      file=sys.stderr, flush=True)
            wall = time.monotonic() - t0
            ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            # ru_maxrss is high-water across all children since process start;
            # the per-step delta is the right proxy when steps run sequentially,
            # but it's noisy under parallel execution. Report the absolute
            # high-water at exit instead — the manifest's overall peak_rss_gb
            # is max over steps, which is what bucket-tuning actually needs.
            # Unit: Linux=KB, macOS=bytes.
            divisor = (1024 ** 3) if sys.platform == "darwin" else (1024 ** 2)
            peak_rss_gb = ru1.ru_maxrss / divisor
            s.wallclock_seconds = round(wall, 2)
            s.peak_rss_gb = round(peak_rss_gb, 3)
            _phase_close(s.no, s.name,
                         wallclock=wall, peak_rss_gb=peak_rss_gb,
                         exit_code=s.exit_code, status=s.status)
        return s

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        in_flight: dict[concurrent.futures.Future, Step] = {}

        def _submit_ready() -> None:
            for s in _ready_now():
                pending.discard(s.name)
                in_flight[pool.submit(_execute, s)] = s

        _submit_ready()
        while in_flight:
            done, _ = concurrent.futures.wait(
                in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                s = in_flight.pop(fut)
                completed.add(s.name)
                if s.status == "failed":
                    failed.append(s.name)
            _submit_ready()

    total_wall = time.monotonic() - overall_t0
    total_cpu = sum(s.wallclock_seconds for s in steps)  # approximation; actual CPU s would need RUSAGE.ru_utime

    if failed:
        # Emit a closing summary so Cromwell's tail-of-stderr always has
        # the actionable picture (failed step + path to its full log).
        print("===== flare_validate FAILED =====", file=sys.stderr, flush=True)
        for s in steps:
            if s.status != "failed":
                continue
            log_guess = ws.logs_root / f"{s.no:02d}_{s.name}.log"
            # The runners use slightly different log-name formats per step;
            # find anything in logs/ that matches the step name as a fallback.
            if not log_guess.exists():
                matches = sorted(ws.logs_root.glob(f"*{s.name}*.log"))
                if matches:
                    log_guess = matches[-1]
            print(f"  FAILED: step {s.no} {s.name} (exit={s.exit_code}, "
                  f"log={log_guess})", file=sys.stderr, flush=True)
        print(f"  work_dir kept at: {ws.work_root}", file=sys.stderr, flush=True)
        print("==================================", file=sys.stderr, flush=True)
        raise RuntimeError(f"step(s) failed: {', '.join(failed)}")
    return total_wall, total_cpu


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cluster-id", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--anc-vcf", type=Path, required=True)
    p.add_argument("--global-anc", type=Path, required=True,
                   help="FLARE <prefix>.global.anc.gz")
    p.add_argument("--flare-model", type=Path, required=True,
                   dest="flare_model", help="FLARE <prefix>.model")
    p.add_argument("--flare-log", type=Path, required=True)
    p.add_argument("--flare-qc-tsv", type=Path, default=None,
                   help="FLARE <prefix>.qc.tsv (★ v1.1 optional; SKIP-path on coverage when missing — used by pre-pipeline test fixtures)")
    p.add_argument("--flare-summary", type=Path, default=None,
                   help="FLARE <prefix>.summary.json (optional; FLARE does not emit one — "
                        "this flag exists for forward compatibility)")
    p.add_argument("--input-vcf", type=Path, required=True,
                   help="Per-cluster gt= VCF that fed Stage C")
    p.add_argument("--rf-ancestry", type=Path, required=True)
    p.add_argument("--region-masks-dir", type=Path, default=None,
                   help="Directory containing *.bed mask files (HLA, segdup, centromere, high-LD)")
    p.add_argument("--chrom-sizes", type=Path, required=True)
    # ★ v1.1: Rye Q replaces ADMIXTURE Q. Required by default in production
    # but stays as Path-optional so the orchestrator works in fixture-only
    # mode (no Rye → skip compare_to_rye step + emit no concordance files).
    # The WDL drives the path; --admixture-q / --admixture-fam are hard-dropped.
    p.add_argument("--rye-q", type=Path, default=None,
                   help="Rye Q TSV (★ v1.1; caller-supplied path; no default)")
    # ★ v1.1: R6 ref/target concordance audit. Required for the new step.
    p.add_argument("--ref-vcf", type=Path, default=None,
                   help="FLARE reference VCF for this chrom (★ v1.1; required for R6 audit)")
    p.add_argument("--self-id", type=Path, default=None)
    p.add_argument("--popout-secondary-global", type=Path, default=None)
    p.add_argument("--popout-secondary-labels", type=Path, default=None)
    p.add_argument("--ref-panel", type=Path, default=None,
                   help="Ref-panel TSV; only used to compute its sha256 for the manifest.")
    p.add_argument("--panel-id", default=None,
                   help="Free-text panel identifier (e.g. 'gnomad_90')")
    p.add_argument("--schema-version", default=SCHEMA_VERSION)
    p.add_argument("--out-tarball", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Staging dir; default = <out-tarball>.work/")
    p.add_argument("--run-name", default=None,
                   help="Magicwand run name; used in manifest only")
    p.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--keep-work-dir", action="store_true",
                   help="Do not delete the staging dir after tarring")
    args = p.parse_args()

    if args.schema_version != SCHEMA_VERSION:
        raise RuntimeError(
            f"--schema-version {args.schema_version!r} != bundled "
            f"validation/schema.py SCHEMA_VERSION {SCHEMA_VERSION!r}"
        )

    # ── Workspace setup ──
    work_dir = args.work_dir or args.out_tarball.with_suffix(".work")
    work_root = work_dir / args.cluster_id / args.chrom
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    scratch_root = work_dir / "scratch"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir()
    logs_root = work_root / "logs"
    logs_root.mkdir()

    ws = Workspace(work_root=work_root, scratch_root=scratch_root, logs_root=logs_root)
    ws.optional_inputs = {
        # ★ v1.1: rye_q replaces admixture_q (Rye = supervised ADMIXTURE).
        "rye_q": bool(args.rye_q),
        "self_id": bool(args.self_id),
        "popout_secondary": bool(args.popout_secondary_global and args.popout_secondary_labels),
        "region_bed": False,        # not wired into hap_disagreement at WDL level
        "fst_tree": False,          # FLARE: no per-site allele freq → skipped
        # ★ v1.1: gates the qc-dependent coverage checks (SKIP path when false).
        "flare_qc_tsv": bool(args.flare_qc_tsv),
    }

    # Validate paired optional inputs.
    if bool(args.popout_secondary_global) != bool(args.popout_secondary_labels):
        raise RuntimeError(
            "--popout-secondary-global and --popout-secondary-labels must be provided together"
        )
    # ★ v1.1: ref-vcf is required for R6 (always runs).
    if not args.ref_vcf:
        raise RuntimeError("--ref-vcf is required (R6 ref/target concordance audit)")

    # ── DAG definition ──
    # Tract-derived metrics (structural / hap_disagreement / regional) all
    # come from one fused single-pass step (`per_site_metrics`) that streams
    # the anc.vcf.gz exactly once via bcftools query. Render is a thin
    # downstream PNG step.
    steps: list[Step] = [
        Step(2, "to_popout_format",     (),                          step_to_popout_format),
        Step(3, "coverage",             ("to_popout_format",),       step_coverage, nonfatal=True),
        Step(4, "compare_to_rf",        ("to_popout_format", "coverage"), step_compare_to_rf),
        # ★ v1.1: R6 has no FLARE-output dependency — only needs ref_vcf + input_vcf.
        # Schedule at the "to_popout_format" depth so it runs in parallel with
        # coverage / compare_to_rf instead of serializing behind them.
        Step(5, "ref_target_concordance", ("to_popout_format",),     step_ref_target_concordance),
        # per_site_metrics waits for compare_to_rf so labels.json is ready
        # to feed ancestry names into windows.tsv.gz. compare_to_rf is
        # cheap relative to the heavy VCF pass.
        Step(1, "per_site_metrics",     ("to_popout_format", "compare_to_rf"), step_per_site_metrics),
        Step(7, "render_collector_pngs", ("per_site_metrics",),      step_render_collector_pngs),
        Step(11, "plot_concordance",    ("compare_to_rf",),          step_plot_concordance),
    ]
    # ★ v1.1: Rye (was admixture) concordance is per-cluster optional.
    if args.rye_q:
        steps.append(Step(6, "compare_to_rye", ("compare_to_rf",), step_compare_to_rye,
                          optional_input_flag="rye_q"))
    if args.self_id:
        steps.append(Step(10, "self_id", ("compare_to_rf",), step_self_id,
                          optional_input_flag="self_id"))

    # Sort by step number for stable logging.
    steps.sort(key=lambda s: s.no)

    _log(f"flare_validate orchestrator | schema={SCHEMA_VERSION} | "
         f"cluster={args.cluster_id} | chrom={args.chrom} | max_workers={args.max_workers} | "
         f"n_steps={len(steps)}")

    # ── Execute ──
    total_wall, total_cpu = _run_dag(args, ws, steps)

    # ── Derived files + provenance + manifest + tier1 ──
    _write_calibration_notes(args, ws)
    _write_provenance(args, ws)
    # Per-sample ancestry table — feeds the cohort_global.tsv at collation.
    shutil.copy2(ws.intermediates["global_tsv"], ws.work_root / "global.tsv")
    manifest = _write_manifest(args, ws, steps, total_wall, total_cpu)
    _write_tier1_metrics(ws, manifest)

    _log(f"total wallclock {total_wall:.1f}s, peak_rss_gb {manifest['peak_rss_gb']:.2f}, "
         f"coverage_passed={manifest['coverage_passed']}")

    # ── Validate against schema and tar ──
    issues = validate_cluster_artifact(work_root)
    n_err = report_issues(issues, label="cluster artifact")
    if n_err:
        raise RuntimeError(
            f"cluster artifact failed schema validation ({n_err} error(s)); "
            f"see stderr. Work dir kept at {work_root}"
        )

    write_cluster_artifact(work_root, args.out_tarball,
                           cluster_id=args.cluster_id, chrom=args.chrom)
    _log(f"wrote artifact tarball: {args.out_tarball} "
         f"({args.out_tarball.stat().st_size / 1e6:.1f} MB)")

    if not args.keep_work_dir:
        shutil.rmtree(scratch_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
