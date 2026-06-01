#!/usr/bin/env python3
"""popout DX per-shard orchestrator.

One invocation per ``(cluster_id, chrom)`` Cromwell shard. Looks up the
shard's row in the manifest produced by ``discover_runs.py``, builds a
DAG of small subprocess steps, runs them concurrently via
``ThreadPoolExecutor`` (perf contract #9), writes ``manifest.json`` and
``tier1_metrics.tsv`` (with per-step ``step.<name>.wallclock_seconds``
rows — perf contract #11), and validates the artifact against
``validation/popout_dx/schema.py``.

Mode gating
-----------
``--mode global``        : load + label-align + pairwise hard + pairwise soft
``--mode global_local``  : the above plus stratified picker → local FLARE
                           parse → local align/metrics → compare_tracts views

The cluster's sample roster is the sample-id column of FLARE's per-cluster
``global.tsv`` (which by construction contains exactly the cluster's
samples). Popout's whole-cohort files are subsetted to this roster
in-process by the consuming step scripts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Local schema validator.
from validation.popout_dx import schema as dx_schema


SCRIPTS_DIR = Path(__file__).resolve().parent
COMPARE_TO_RF = Path("/opt/validation/scripts/compare_to_rf.py")
if not COMPARE_TO_RF.exists():
    # Local dev fallback: gpulai checkout sibling of where this script lives.
    COMPARE_TO_RF = Path(__file__).resolve().parents[2] / "scripts" / "compare_to_rf.py"


# ── Step abstraction (same pattern as validation/scripts/run_cluster_validation.py) ─


class StepFailed(RuntimeError):
    def __init__(self, name: str, exit_code: int, log_path: Path):
        super().__init__(f"step {name!r} failed (exit={exit_code}); see {log_path}")
        self.exit_code = exit_code
        self.log_path = log_path


@dataclasses.dataclass
class Step:
    no: int
    name: str
    depends_on: tuple[str, ...]
    runner: callable
    nonfatal: bool = False
    status: str = "pending"            # pending | ok | skipped | failed
    exit_code: int = 0
    wallclock_seconds: float = 0.0
    peak_rss_gb: float = 0.0


@dataclasses.dataclass
class Workspace:
    work_root: Path                    # work/<cluster_id>/<chrom>/
    scratch_root: Path
    logs_root: Path
    intermediates: dict[str, Path] = dataclasses.field(default_factory=dict)
    optional_inputs: dict[str, bool] = dataclasses.field(default_factory=dict)

    def subdir(self, name: str) -> Path:
        d = self.work_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d


# ── Sub-script invocation ───────────────────────────────────────────────


def _run_subprocess(cmd: list[str], log_path: Path, *, step_name: str) -> int:
    """Tee subprocess stdout+stderr to log file + orchestrator stderr."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[{step_name}] "
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd_str = "$ " + " ".join(str(c) for c in cmd)
    sys.stderr.write(prefix + cmd_str + "\n"); sys.stderr.flush()
    with open(log_path, "w") as logf:
        logf.write(cmd_str + "\n\n"); logf.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line); logf.flush()
            sys.stderr.write(prefix + line); sys.stderr.flush()
        proc.wait()
    return proc.returncode


def _check(step_name: str, log_path: Path, rc: int) -> None:
    if rc != 0:
        raise StepFailed(step_name, rc, log_path)


# ── Manifest helpers ────────────────────────────────────────────────────


def load_manifest_row(manifest_tsv: Path, cluster_id: str, chrom: str) -> dict:
    """Return the row dict for the (cluster_id, chrom) shard.

    Accepts both header-bearing and headerless TSVs (the WDL-facing one is
    headerless so ``read_tsv()`` can scatter on it directly; the
    ``.with_header.tsv`` sibling exists for humans). Column order is the
    fixed :data:`discover_runs.TSV_COLUMNS`.
    """
    from validation.popout_dx.scripts.discover_runs import TSV_COLUMNS
    with open(manifest_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            if parts[0] == "cluster_id":
                continue   # header row, skip
            if len(parts) != len(TSV_COLUMNS):
                continue
            row = dict(zip(TSV_COLUMNS, parts))
            if row["cluster_id"] == cluster_id and row["chrom"] == chrom:
                return row
    raise SystemExit(
        f"run_dx_cluster: no row for cluster_id={cluster_id!r} chrom={chrom!r} in {manifest_tsv}"
    )


def derive_roster_from_flare_global(flare_global_tsv: Path) -> list[str]:
    samples: list[str] = []
    with open(flare_global_tsv) as f:
        f.readline()   # header
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0].strip():
                samples.append(parts[0].strip())
    if not samples:
        raise SystemExit(f"derive_roster: {flare_global_tsv} produced zero samples")
    return samples


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def subset_popout_global(
    popout_global_tsv: Path, roster: list[str], out_path: Path
) -> None:
    """Write a popout-format global.tsv restricted to roster, in roster order."""
    roster_set = set(roster)
    rows: dict[str, str] = {}
    header: str | None = None
    with open(popout_global_tsv) as f:
        header = f.readline()
        for line in f:
            parts = line.split("\t", 1)
            sid = parts[0].strip()
            if sid in roster_set:
                rows[sid] = line
    missing = [s for s in roster if s not in rows]
    if missing:
        raise SystemExit(
            f"subset_popout_global: {len(missing)} roster sample(s) absent from "
            f"{popout_global_tsv}; first: {missing[:5]}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as g:
        g.write(header)
        for s in roster:
            g.write(rows[s])


def subset_rf_ancestry(
    rf_ancestry_tsv: Path, roster: list[str], out_path: Path
) -> None:
    """Write an RF ancestry TSV restricted to roster, in roster order.
    Preserves the header line."""
    roster_set = set(roster)
    rows: dict[str, str] = {}
    header: str | None = None
    with open(rf_ancestry_tsv) as f:
        header = f.readline()
        hdr_lower = header.rstrip("\n").split("\t")
        try:
            id_col = next(
                i for i, h in enumerate(hdr_lower)
                if h.lower() in ("research_id", "sample_id", "sample")
            )
        except StopIteration:
            raise SystemExit(
                f"subset_rf_ancestry: no sample-id column in {rf_ancestry_tsv} header"
            )
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= id_col:
                continue
            sid = parts[id_col].strip()
            if sid in roster_set:
                rows[sid] = line
    missing = [s for s in roster if s not in rows]
    if missing:
        raise SystemExit(
            f"subset_rf_ancestry: {len(missing)} roster sample(s) absent from "
            f"{rf_ancestry_tsv}; first: {missing[:5]}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as g:
        g.write(header)
        for s in roster:
            g.write(rows[s])


# ── Step runners ────────────────────────────────────────────────────────


def step_load_inputs(args, ws: Workspace, log_dir: Path) -> None:
    """Materialise the cluster roster + per-cluster subset of popout/RF.

    Outputs (intermediates):
      ``roster_txt``        one sample_id per line
      ``popout_subset_tsv`` popout global.tsv subset to roster
      ``rf_subset_tsv``     RF ancestry subset to roster (only when rf is in tools)
    """
    row = ws.intermediates["manifest_row"]
    flare_global = Path(row["flare_global_tsv"])
    roster = derive_roster_from_flare_global(flare_global)
    roster_txt = ws.subdir("intermediates") / "cluster_roster.txt"
    write_lines(roster_txt, roster)
    ws.intermediates["roster_txt"] = roster_txt
    ws.intermediates["roster_size"] = str(len(roster))

    popout_global = Path(row["popout_global_tsv"])
    popout_subset = ws.subdir("intermediates") / "popout.cluster.global.tsv"
    subset_popout_global(popout_global, roster, popout_subset)
    ws.intermediates["popout_subset_tsv"] = popout_subset

    if ws.optional_inputs.get("rf", False):
        rf_path = Path(row["rf_ancestry_path"])
        rf_subset = ws.subdir("intermediates") / "rf.cluster.tsv"
        subset_rf_ancestry(rf_path, roster, rf_subset)
        ws.intermediates["rf_subset_tsv"] = rf_subset

    # Copy FLARE per-cluster global.tsv into the artifact root for completeness.
    # (popout.global.tsv at the artifact root is the cluster-subset version.)
    artifact_popout = ws.work_root / "popout.global.tsv"
    artifact_popout.write_text(popout_subset.read_text())


def step_align_labels(args, ws: Workspace, log_dir: Path) -> None:
    """Run compare_to_rf.py on the per-cluster popout subset to derive
    the popout → RF label alignment (labels.json). The other files
    compare_to_rf emits (confusion_matrix.tsv, soft_correlation.tsv,
    SUMMARY.md, popout_composition.tsv, pca_by_rf_label.png) go to a
    scratch dir; only labels.json is promoted to the artifact root.
    """
    if not ws.optional_inputs.get("rf", False):
        raise SystemExit(
            "step_align_labels: rf must be in tools to compute popout → RF labels.json"
        )
    scratch = ws.scratch_root / "align_labels"
    scratch.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "step_align_labels.log"
    cmd = [
        sys.executable, str(COMPARE_TO_RF),
        "--popout-global", str(ws.intermediates["popout_subset_tsv"]),
        "--rf-ancestry",   str(ws.intermediates["rf_subset_tsv"]),
        "--out-dir",       str(scratch),
    ]
    _check("step_align_labels", log_path, _run_subprocess(cmd, log_path, step_name="align_labels"))
    src = scratch / "labels.json"
    if not src.exists():
        raise StepFailed("step_align_labels", 1, log_path)
    dst = ws.work_root / "labels.json"
    dst.write_text(src.read_text())
    ws.intermediates["popout_labels_json"] = dst


def step_pairwise_hard(args, ws: Workspace, log_dir: Path) -> None:
    out_dir = ws.subdir("global") / "pairwise_hard"
    log_path = log_dir / "step_pairwise_hard.log"
    row = ws.intermediates["manifest_row"]
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "dx_pairwise_hard.py"),
        "--popout-global", str(ws.intermediates["popout_subset_tsv"]),
        "--popout-labels", str(ws.intermediates["popout_labels_json"]),
        "--out-dir",       str(out_dir),
    ]
    if ws.optional_inputs.get("flare", False):
        cmd += ["--flare-global", row["flare_global_tsv"],
                "--flare-labels", row["flare_labels_json"]]
    if ws.optional_inputs.get("rye", False):
        cmd += ["--rye-q", row["rye_q_path"]]
    if ws.optional_inputs.get("rf", False):
        cmd += ["--rf", str(ws.intermediates["rf_subset_tsv"])]
    _check("step_pairwise_hard", log_path, _run_subprocess(cmd, log_path, step_name="pairwise_hard"))


def step_pairwise_soft(args, ws: Workspace, log_dir: Path) -> None:
    out_dir = ws.subdir("global") / "pairwise_soft"
    log_path = log_dir / "step_pairwise_soft.log"
    row = ws.intermediates["manifest_row"]
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "dx_pairwise_soft.py"),
        "--popout-global", str(ws.intermediates["popout_subset_tsv"]),
        "--popout-labels", str(ws.intermediates["popout_labels_json"]),
        "--out-dir",       str(out_dir),
    ]
    if ws.optional_inputs.get("flare", False):
        cmd += ["--flare-global", row["flare_global_tsv"],
                "--flare-labels", row["flare_labels_json"]]
    if ws.optional_inputs.get("rye", False):
        cmd += ["--rye-q", row["rye_q_path"]]
    if ws.optional_inputs.get("rf", False):
        cmd += ["--rf", str(ws.intermediates["rf_subset_tsv"])]
    _check("step_pairwise_soft", log_path, _run_subprocess(cmd, log_path, step_name="pairwise_soft"))


def step_pick_samples(args, ws: Workspace, log_dir: Path) -> None:
    out_path = ws.subdir("local") / "selected_samples.tsv"
    log_path = log_dir / "step_pick_samples.log"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "pick_local_samples.py"),
        "--popout-global", str(ws.intermediates["popout_subset_tsv"]),
        "--labels",        str(ws.intermediates["popout_labels_json"]),
        "--cluster-roster", str(ws.intermediates["roster_txt"]),
        "--out",           str(out_path),
        "--cluster-id",    args.cluster_id,
        "--rng-seed",      str(args.local_rng_seed),
        "--per-bucket-n",  str(args.local_per_bucket_n),
        "--threshold",     str(args.local_threshold),
    ]
    _check("step_pick_samples", log_path, _run_subprocess(cmd, log_path, step_name="pick_samples"))
    ws.intermediates["selected_samples_tsv"] = out_path

    # Derive a plain sample_id list for the downstream parsers.
    sample_ids: list[str] = []
    with open(out_path) as f:
        f.readline()
        for line in f:
            sid = line.split("\t", 1)[0].strip()
            if sid:
                sample_ids.append(sid)
    selected_list = ws.subdir("intermediates") / "selected_samples_ids.txt"
    write_lines(selected_list, sample_ids)
    ws.intermediates["selected_ids_txt"] = selected_list


def step_parse_flare_local(args, ws: Workspace, log_dir: Path) -> None:
    """Parse FLARE anc.vcf.gz restricted to selected samples (bcftools)."""
    row = ws.intermediates["manifest_row"]
    if not row.get("flare_anc_vcf"):
        raise SystemExit(
            "step_parse_flare_local: manifest row has no flare_anc_vcf; "
            "the discover task must extract the per-cluster .anc.vcf.gz path "
            "into the manifest (v1.0.0 of discover_runs uses the cohort bundle's "
            "global.tsv slice for global mode; local mode requires the raw VCF)."
        )
    out_npz = ws.subdir("intermediates") / "flare.tractset.npz"
    log_path = log_dir / "step_parse_flare_local.log"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "dx_local_parse_flare.py"),
        "--vcf",          row["flare_anc_vcf"],
        "--samples-file", str(ws.intermediates["selected_ids_txt"]),
        "--out-npz",      str(out_npz),
        "--workspace",    str(ws.scratch_root / "flare_parse"),
    ]
    _check("step_parse_flare_local", log_path, _run_subprocess(cmd, log_path, step_name="parse_flare_local"))
    ws.intermediates["flare_npz"] = out_npz


def step_local_metrics(args, ws: Workspace, log_dir: Path) -> None:
    out_dir = ws.subdir("local")
    log_path = log_dir / "step_local_metrics.log"
    row = ws.intermediates["manifest_row"]
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "dx_local_align_metrics.py"),
        "--popout-tracts", row["popout_tracts"],
        "--popout-labels", str(ws.intermediates["popout_labels_json"]),
        "--flare-npz",     str(ws.intermediates["flare_npz"]),
        "--flare-labels",  row["flare_labels_json"],
        "--samples-file",  str(ws.intermediates["selected_ids_txt"]),
        "--chrom",         args.chrom,
        "--out-dir",       str(out_dir),
    ]
    _check("step_local_metrics", log_path, _run_subprocess(cmd, log_path, step_name="local_metrics"))


def step_local_views(args, ws: Workspace, log_dir: Path) -> None:
    out_dir = ws.subdir("local") / "views"
    log_path = log_dir / "step_local_views.log"
    row = ws.intermediates["manifest_row"]
    summary_in = ws.work_root / "local" / "local_summary.json"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "dx_local_views.py"),
        "--flare-npz",     str(ws.intermediates["flare_npz"]),
        "--flare-labels",  row["flare_labels_json"],
        "--popout-tracts", row["popout_tracts"],
        "--popout-labels", str(ws.intermediates["popout_labels_json"]),
        "--samples-file",  str(ws.intermediates["selected_ids_txt"]),
        "--chrom",         args.chrom,
        "--out-dir",       str(out_dir),
        "--coarse-grids-mb", *(str(x) for x in args.local_coarse_grids_mb),
        "--local-summary-in", str(summary_in),
    ]
    _check("step_local_views", log_path, _run_subprocess(cmd, log_path, step_name="local_views"))


# ── Provenance + manifest + tier1 writers ───────────────────────────────


def step_provenance(args, ws: Workspace, log_dir: Path) -> None:
    prov = ws.subdir("provenance")
    (prov / "schema_version.txt").write_text(dx_schema.SCHEMA_VERSION + "\n")
    # Copy the dx config through for reproducibility.
    if args.config_file is not None and Path(args.config_file).exists():
        dst = prov / "dx_config.json"
        dst.write_text(Path(args.config_file).read_text())


def _write_manifest(args, ws: Workspace, steps: list[Step], tools: list[str]) -> dict:
    mode = args.mode
    optional_inputs = {
        "flare":      "flare" in tools,
        "rye":        "rye" in tools,
        "rf":         "rf" in tools,
        "local_mode": mode == "global_local",
    }
    manifest = {
        "schema_version": dx_schema.SCHEMA_VERSION,
        "cluster_id": args.cluster_id,
        "chrom": args.chrom,
        "run_name": args.run_name,
        "mode": mode,
        "tools": tools,
        "n_samples": int(ws.intermediates.get("roster_size", 0)),
        "n_ancestries_popout": _read_n_ancestries_popout(ws),
        "popout_run_dir": str(Path(ws.intermediates["manifest_row"]["popout_global_tsv"]).parent),
        "steps": {
            s.name: {
                "wallclock_seconds": s.wallclock_seconds,
                "peak_rss_gb": s.peak_rss_gb,
                "exit": s.exit_code,
                "status": s.status,
            }
            for s in steps
        },
        "total_wallclock_seconds": round(sum(s.wallclock_seconds for s in steps), 2),
        "peak_rss_gb": round(max((s.peak_rss_gb for s in steps), default=0.0), 3),
        "cpu_wall_ratio": 1.0,   # placeholder; real ratio needs RUSAGE.ru_utime accumulation
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "optional_inputs": optional_inputs,
    }
    (ws.work_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _read_n_ancestries_popout(ws: Workspace) -> int:
    p = ws.intermediates.get("popout_subset_tsv")
    if p is None or not p.exists():
        return 0
    with open(p) as f:
        header = f.readline().rstrip("\n").split("\t")
    return max(0, len(header) - 1)


def _write_tier1_metrics(args, ws: Workspace, manifest: dict, tools: list[str]) -> None:
    rows: list[tuple[str, str]] = []
    def add(k: str, v) -> None:
        rows.append((k, "" if v is None else str(v)))

    add("popout_dx.cluster_id", args.cluster_id)
    add("popout_dx.chrom", args.chrom)
    add("popout_dx.mode", args.mode)
    add("popout_dx.n_samples", manifest.get("n_samples"))
    add("popout_dx.n_ancestries_popout", manifest.get("n_ancestries_popout"))
    add("popout_dx.peak_rss_gb", manifest.get("peak_rss_gb"))
    add("popout_dx.cpu_wall_ratio", manifest.get("cpu_wall_ratio"))

    soft_summary_path = ws.work_root / "global" / "pairwise_soft" / "summary.json"
    if soft_summary_path.exists():
        soft = json.loads(soft_summary_path.read_text())
        per_tool = soft.get("per_tool", {})
        for tool in ("flare", "rye", "rf"):
            t = per_tool.get(tool, {})
            add(f"popout_dx.mean_ccc_vs_{tool}", t.get("mean_ccc_eligible"))
            add(f"popout_dx.mean_pearson_r_vs_{tool}", t.get("mean_pearson_r_eligible"))
            add(f"popout_dx.n_pairs_passing_vs_{tool}", t.get("n_passing"))
            add(f"popout_dx.n_pairs_failing_vs_{tool}", t.get("n_failing"))

    if args.mode == "global_local":
        local_summary_path = ws.work_root / "local" / "local_summary.json"
        if local_summary_path.exists():
            local = json.loads(local_summary_path.read_text())
            add("popout_dx.local_bp_agreement_vs_flare", local.get("bp_agreement"))
            add("popout_dx.local_calibration_drift_fraction",
                local.get("calibration_drift_fraction"))
            add("popout_dx.local_boundary_match_switch_fraction",
                local.get("boundary_localization_match_switch_fraction"))

    for step_name, info in manifest.get("steps", {}).items():
        wall = info.get("wallclock_seconds")
        add(f"step.{step_name}.wallclock_seconds",
            round(float(wall), 2) if wall is not None else None)

    with open(ws.work_root / "tier1_metrics.tsv", "w") as f:
        for k, v in rows:
            f.write(f"{k}\t{v}\n")


# ── DAG runner (mirrors run_cluster_validation.py:_run_dag) ─────────────


def _phase_open(step_no: int, name: str, cluster_id: str, chrom: str) -> None:
    print(f"===== popout_dx step {step_no:>2}: {name} START "
          f"({cluster_id}/{chrom}) =====", file=sys.stderr, flush=True)


def _phase_close(step_no: int, name: str, *, wallclock: float, peak_rss_gb: float,
                 exit_code: int, status: str) -> None:
    print(f"===== popout_dx step {step_no:>2}: {name} DONE | "
          f"wallclock={wallclock:.1f}s | peak_rss_gb={peak_rss_gb:.2f} | "
          f"exit={exit_code} | status={status} =====",
          file=sys.stderr, flush=True)


def _run_dag(args, ws: Workspace, steps: list[Step]) -> tuple[float, list[str]]:
    by_name = {s.name: s for s in steps}
    pending = {s.name for s in steps if s.status == "pending"}
    failed: list[str] = []
    t0_all = time.monotonic()

    def _ready_now() -> list[Step]:
        ready: list[Step] = []
        for name in list(pending):
            s = by_name[name]
            if all(by_name[d].status in ("ok", "skipped") for d in s.depends_on):
                ready.append(s)
            elif any(by_name[d].status == "failed" for d in s.depends_on):
                s.status = "skipped"
                pending.discard(name)
        return ready

    def _execute(s: Step) -> Step:
        _phase_open(s.no, s.name, args.cluster_id, args.chrom)
        t0 = time.monotonic()
        try:
            s.runner(args, ws, ws.logs_root)
            s.status = "ok"
            s.exit_code = 0
        except StepFailed as e:
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
            wall = time.monotonic() - t0
            ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            divisor = (1024 ** 3) if sys.platform == "darwin" else (1024 ** 2)
            s.wallclock_seconds = round(wall, 2)
            s.peak_rss_gb = round(ru1.ru_maxrss / divisor, 3)
            _phase_close(s.no, s.name, wallclock=wall, peak_rss_gb=s.peak_rss_gb,
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
                if s.status == "failed":
                    failed.append(s.name)
            _submit_ready()

    return time.monotonic() - t0_all, failed


# ── DAG construction ────────────────────────────────────────────────────


def build_dag(args, ws: Workspace, tools: list[str]) -> list[Step]:
    steps: list[Step] = []
    no = 1

    def add(name: str, depends_on: tuple[str, ...], runner: callable, nonfatal: bool = False) -> None:
        nonlocal no
        steps.append(Step(no=no, name=name, depends_on=depends_on, runner=runner, nonfatal=nonfatal))
        no += 1

    add("load_inputs", (), step_load_inputs)
    add("align_labels", ("load_inputs",), step_align_labels)
    add("pairwise_hard", ("align_labels",), step_pairwise_hard)
    add("pairwise_soft", ("align_labels",), step_pairwise_soft)

    if args.mode == "global_local":
        add("pick_samples", ("align_labels",), step_pick_samples)
        add("parse_flare_local", ("pick_samples",), step_parse_flare_local)
        add("local_metrics", ("parse_flare_local",), step_local_metrics)
        add("local_views", ("parse_flare_local",), step_local_views)

    add("provenance", (), step_provenance)
    return steps


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-manifest-tsv", required=True, type=Path)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--mode", required=True, choices=("global", "global_local"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tools", required=True,
                    help="comma-separated subset of popout,flare,rye,rf (anchor=popout)")
    ap.add_argument("--config-file", type=Path, default=None,
                    help="dx_config.json, copied into provenance/ for reproducibility")
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--max-workers", type=int, default=4)

    # Local-mode parameters (ignored when --mode=global).
    ap.add_argument("--local-rng-seed", type=int, default=42)
    ap.add_argument("--local-per-bucket-n", type=int, default=25)
    ap.add_argument("--local-threshold", type=float, default=0.80)
    ap.add_argument("--local-coarse-grids-mb", type=int, nargs="+",
                    default=[1, 2, 5, 10, 20])

    ap.add_argument("--emit-tarball", type=Path, default=None,
                    help="if set, write <cluster_id>.<chrom>.popout_dx.v<schema>.tar.gz here")
    args = ap.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    if "popout" not in tools:
        raise SystemExit("--tools must include 'popout' (the anchor)")

    row = load_manifest_row(args.runs_manifest_tsv, args.cluster_id, args.chrom)

    work_root = args.work_dir / args.cluster_id / args.chrom
    work_root.mkdir(parents=True, exist_ok=True)
    logs_root = work_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    scratch_root = args.work_dir / "scratch" / args.cluster_id / args.chrom
    scratch_root.mkdir(parents=True, exist_ok=True)

    ws = Workspace(
        work_root=work_root,
        scratch_root=scratch_root,
        logs_root=logs_root,
        intermediates={"manifest_row": row},
        optional_inputs={t: True for t in tools if t != "popout"} | {"local_mode": args.mode == "global_local"},
    )

    steps = build_dag(args, ws, tools)
    total_wall, failed = _run_dag(args, ws, steps)

    manifest = _write_manifest(args, ws, steps, tools)
    _write_tier1_metrics(args, ws, manifest, tools)

    # Schema validation. Issues are dumped to stderr; a non-empty error
    # count fails the orchestrator (CLAUDE.md: never silently drop).
    issues = dx_schema.validate_cluster_artifact(work_root)
    n_err = dx_schema.report_issues(
        issues, label=f"{args.cluster_id}/{args.chrom}",
    )

    if args.emit_tarball is not None:
        dx_schema.write_cluster_artifact(
            work_root, args.emit_tarball,
            cluster_id=args.cluster_id, chrom=args.chrom,
        )
        print(f"popout_dx: wrote {args.emit_tarball}", file=sys.stderr)

    if failed:
        print(f"popout_dx: {len(failed)} step(s) failed: {failed}", file=sys.stderr)
        return 1
    if n_err:
        print(f"popout_dx: artifact failed schema validation ({n_err} error(s))", file=sys.stderr)
        return 1
    print(f"popout_dx: total wallclock {total_wall:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
