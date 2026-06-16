#!/usr/bin/env python3
"""Fused per-site collector — replaces the four-step
``vcf_to_tracts → structural/hap_disagreement/regional`` slice with a
single pass over the FLARE ancestry VCF.

Architecture
============

A `bcftools query` subprocess emits flat per-site text at htslib (C)
speed:

    bcftools query -f '%CHROM\\t%POS[\\t%AN1\\t%AN2]\\n' <anc.vcf.gz>

The Python side parses one line at a time. Every per-sample integer at
that site is converted with ``np.array(parts[2:], dtype=np.int8)`` — one
C call per record, not one Python dict lookup per cell. Three
accumulators advance in lock-step:

* **structural** — per-(sample, hap) open-tract state plus per-ancestry
  list of finished tract lengths in bp. Emits ``tract_length_summary.json``
  and ``switch_rate_summary.json``.
* **hap_disagreement** — per-sample (agree_bp, disagree_bp) and
  per-(sample, hap) ancestry-bp bincount (for dominant-anc-per-hap).
  Joined with the RF hard label at write time. Emits
  ``per_sample.tsv`` and ``summary.json``.
* **regional** — per-(window_bin, ancestry) bp counter; each site
  contributes ``width × (sum over haps of indicator(an == anc))`` to
  every sliding window that contains ``pos``. Emits ``windows.tsv.gz``,
  ``significant.bed``, and ``summary.json``.

A small mu-vs-global helper (formerly ``validate_structural.check_mu_agreement``)
also writes ``model/mu_vs_global_diff.json`` from ``global.tsv`` and the
FLARE ``.model`` text file.

No ``tracts.tsv.gz`` is written at any point.

Why ``bcftools query`` instead of pysam
---------------------------------------

The dominant cost in the old single-threaded pipeline was
``rec.samples[sample]; gt.get('AN1')`` — per-cell Python wrapper
construction at ~1 µs/cell. At 30 k samples × 587 k records that is
~5 hours per chromosome on one core. ``bcftools query`` writes the same
two integer arrays as text at C speed; ``np.array(..., dtype=np.int8)``
parses one line in a single C call.

Usage
-----

    python validate_per_site_metrics.py \\
        --anc-vcf      cluster_007.chr1.anc.vcf.gz \\
        --global-tsv   popout_format/cluster_007.chr1.global.tsv \\
        --flare-model  cluster_007.chr1.model \\
        --rf-ancestry  ancestry_preds.tsv \\
        --chrom-sizes  grch38.chrom.sizes \\
        --region-mask-bed centromere.bed \\
        --region-mask-bed segdup.bed \\
        --labels-json  soft_correlation/labels.json \\
        --out-root     work/cluster_007/chr1/
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


# ── Static-input loaders ──────────────────────────────────────────────────


def load_rf_hard_labels(path: Path) -> dict[str, tuple[str, float]]:
    """Return ``sample_id -> (rf_hard_label, rf_max_prob)``.

    The RF hard label is carried verbatim from the TSV's
    ``ancestry_pred`` column; ``rf_max_prob`` is the corresponding
    confidence. Downstream callers can apply a max-prob threshold if
    they want to gate on confidence, but **no relabelling happens
    here**: the collector never invents pseudo-labels like ``mixed``.
    """
    out: dict[str, tuple[str, float]] = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        rid_col = header.index("research_id")
        pred_col = header.index("ancestry_pred")
        prob_col = header.index("probabilities")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rid = parts[rid_col]
            pred = parts[pred_col]
            probs = ast.literal_eval(parts[prob_col])
            out[rid] = (pred, float(max(probs)))
    return out


def load_chrom_sizes(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom, size = line.split("\t")
            out[chrom] = int(size)
    return out


def load_region_bed(path: Path) -> list[tuple[str, int, int, str]]:
    out: list[tuple[str, int, int, str]] = []
    default_name = path.stem
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"BED line too short: {line!r}")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            rname = parts[3] if len(parts) >= 4 else default_name
            out.append((chrom, start, end, rname))
    return out


def load_labels_json(path: Path | None) -> dict[int, str]:
    """Return ancestry index -> display name. Empty dict when labels
    aren't available (fallback to 'ancestry_<i>' downstream)."""
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    ptr = raw.get("popout_to_rf_label") or {}
    if not ptr:
        return {}
    mapping = {int(k): v for k, v in ptr.items()}
    counts: dict[str, int] = {}
    for v in mapping.values():
        counts[v] = counts.get(v, 0) + 1
    out: dict[int, str] = {}
    for i, lab in mapping.items():
        out[i] = f"{lab}.{i}" if counts[lab] > 1 else lab
    return out


def read_model_text(path: Path) -> dict:
    """Light reimpl of popout.viz._loaders.read_model_text — kept local
    so this script has no popout dependency."""
    result: dict = {}
    with open(path) as f:
        for line in f:
            key, val = line.strip().split("\t", 1)
            if key == "n_ancestries":
                result[key] = int(val)
            elif key == "gen_since_admix":
                result[key] = float(val)
            elif key == "mu":
                result[key] = [float(x) for x in val.split(",")]
            else:
                result[key] = val
    return result


def read_global_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    """Returns (sample_names, proportions of shape (n_samples, K))."""
    sample_names: list[str] = []
    rows: list[list[float]] = []
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sample_names.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    return sample_names, np.array(rows, dtype=np.float64)


# ── VCF header probe ──────────────────────────────────────────────────────


def probe_vcf_header(anc_vcf: Path) -> tuple[list[str], str]:
    """Run `bcftools view -h` once to learn the sample list. Returns
    (samples, first_chrom_hint). The first_chrom_hint comes from a
    follow-up `bcftools query -f '%CHROM\\n' | head -1` since the header
    doesn't carry it."""
    p = subprocess.run(
        ["bcftools", "view", "-h", str(anc_vcf)],
        check=True, capture_output=True, text=True,
    )
    samples: list[str] = []
    for line in p.stdout.splitlines():
        if line.startswith("#CHROM"):
            cols = line.rstrip("\n").split("\t")
            samples = cols[9:]
            break
    if not samples:
        raise RuntimeError(f"no samples in VCF header of {anc_vcf}")

    # First-line peek for the chrom. This task is per-chrom so the file
    # should contain exactly one CHROM value; we re-validate during the
    # main stream.
    p2 = subprocess.run(
        ["bcftools", "query", "-f", "%CHROM\\n", str(anc_vcf)],
        check=True, capture_output=True, text=True,
    )
    first_line = p2.stdout.split("\n", 1)[0].strip()
    if not first_line:
        raise RuntimeError(f"VCF {anc_vcf} has no records")
    return samples, first_line


# ── Sliding-window helper ─────────────────────────────────────────────────


def make_windows(chrom_len: int, window_bp: int, step_bp: int) -> list[tuple[int, int]]:
    wins: list[tuple[int, int]] = []
    s = 0
    while s < chrom_len:
        e = min(s + window_bp, chrom_len)
        wins.append((s, e))
        if e == chrom_len:
            break
        s += step_bp
    return wins


# ── mu vs global agreement (formerly check_mu_agreement) ──────────────────

# FLARE's model.mu reports the post-EM converged values; the per-sample
# global mean is over the FINAL assignments. 0.05 absorbs post-EM
# consolidation drift without masking real issues.
MU_DIFF_THRESHOLD = 0.05


def write_mu_vs_global_diff(
    global_tsv: Path, flare_model: Path, out_dir: Path, anc_names: dict[int, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, props = read_global_tsv(global_tsv)
    global_mu = props.mean(axis=0)
    model_mu = np.array(read_model_text(flare_model)["mu"], dtype=np.float64)
    diff = np.abs(global_mu - model_mu)
    max_diff = float(diff.max())
    all_pass = bool(max_diff < MU_DIFF_THRESHOLD)
    per_anc = []
    for i in range(len(model_mu)):
        per_anc.append({
            "ancestry": i,
            "name": anc_names.get(i, f"ancestry_{i}"),
            "global_mu": float(global_mu[i]),
            "model_mu": float(model_mu[i]),
            "abs_diff": float(diff[i]),
            "pass": bool(diff[i] < MU_DIFF_THRESHOLD),
        })
    (out_dir / "mu_vs_global_diff.json").write_text(json.dumps({
        "max_abs_diff": max_diff,
        "threshold": MU_DIFF_THRESHOLD,
        "overall_pass": all_pass,
        "per_ancestry": per_anc,
    }, indent=2))


# ── The fused per-site collector ──────────────────────────────────────────


def _collect_slice(
    anc_vcf: str,
    sample_names: list[str],
    K: int,
    win_starts_list: list[int],
    win_ends_list: list[int],
    worker_id: int,
    progress_every: int,
) -> dict:
    """Stream the VCF for the given sample slice and return aggregated
    state. Module-level so ProcessPoolExecutor can pickle it.

    Each worker writes its sample list to a temp file and pipes
    `bcftools query -S samples.txt -f '%CHROM\\t%POS[\\t%AN1\\t%AN2]\\n'`
    into a per-record numpy loop. bcftools restricts the FORMAT columns
    to the worker's samples so wire bytes and per-record parse work both
    scale 1/N with the worker count.

    Returns a dict of (per-slice) arrays plus a (n_wins, K) bp_window_anc
    matrix that the cohort reducer SUMS across workers.
    """
    n_samples = len(sample_names)
    win_starts = np.asarray(win_starts_list, dtype=np.int64)
    win_ends = np.asarray(win_ends_list, dtype=np.int64)
    n_wins = win_starts.size

    NO_ANC = np.int8(-1)
    cur_anc_h1 = np.full(n_samples, NO_ANC, dtype=np.int8)
    cur_anc_h2 = np.full(n_samples, NO_ANC, dtype=np.int8)
    tract_start_h1 = np.zeros(n_samples, dtype=np.int64)
    tract_start_h2 = np.zeros(n_samples, dtype=np.int64)
    tract_end_h1 = np.zeros(n_samples, dtype=np.int64)
    tract_end_h2 = np.zeros(n_samples, dtype=np.int64)
    tract_count_h1 = np.zeros(n_samples, dtype=np.int32)
    tract_count_h2 = np.zeros(n_samples, dtype=np.int32)
    tract_lengths_bp: dict[int, list[int]] = {a: [] for a in range(K)}
    agree_bp = np.zeros(n_samples, dtype=np.int64)
    disagree_bp = np.zeros(n_samples, dtype=np.int64)
    bp_per_anc_h1 = np.zeros((n_samples, K), dtype=np.int64)
    bp_per_anc_h2 = np.zeros((n_samples, K), dtype=np.int64)
    bp_window_anc = np.zeros((n_wins, K), dtype=np.int64)
    sample_idx = np.arange(n_samples)

    def _close_tracts(
        closing_anc: np.ndarray,
        closing_starts: np.ndarray,
        closing_ends: np.ndarray,
    ) -> None:
        if closing_anc.size == 0:
            return
        lengths = closing_ends - closing_starts + 1
        for a in np.unique(closing_anc):
            mask = closing_anc == a
            tract_lengths_bp[int(a)].extend(lengths[mask].tolist())
            t_starts = closing_starts[mask]
            t_ends_half_open = closing_ends[mask] + 1
            for ts, te in zip(t_starts.tolist(), t_ends_half_open.tolist()):
                lo = np.searchsorted(win_ends, ts, side="right")
                hi = np.searchsorted(win_starts, te, side="left")
                if hi <= lo:
                    continue
                ov = np.minimum(te, win_ends[lo:hi]) - np.maximum(ts, win_starts[lo:hi])
                bp_window_anc[lo:hi, int(a)] += ov

    # Write the slice's sample names to a temp file for bcftools -S.
    sl_fd, slice_path = tempfile.mkstemp(
        prefix=f"per_site_metrics_w{worker_id}_", suffix=".samples.txt"
    )
    try:
        with os.fdopen(sl_fd, "w") as f:
            for s in sample_names:
                f.write(s + "\n")

        cmd = [
            "bcftools", "query",
            "-S", slice_path,
            "-f", r"%CHROM\t%POS[\t%AN1\t%AN2]" + "\n",
            str(anc_vcf),
        ]
        print(f"[worker {worker_id}] $ " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 16)
        assert proc.stdout is not None

        prev_pos: int | None = None
        n_records = 0
        chrom_seen: str | None = None

        for line in proc.stdout:
            if not line or line == "\n":
                continue
            parts = line.rstrip("\n").split("\t")
            c = parts[0]
            pos = int(parts[1])
            flat = np.array(parts[2:], dtype=np.int8)
            if flat.size != 2 * n_samples:
                raise RuntimeError(
                    f"[worker {worker_id}] record at {c}:{pos} has "
                    f"{flat.size} cell-fields, expected {2 * n_samples}"
                )
            an1 = flat[0::2]
            an2 = flat[1::2]

            if chrom_seen is None:
                chrom_seen = c
            elif c != chrom_seen:
                raise RuntimeError(
                    f"[worker {worker_id}] per-cluster VCF expected to "
                    f"hold one chromosome; saw {chrom_seen!r} and {c!r}"
                )

            width = 0 if prev_pos is None else (pos - prev_pos)

            if width:
                agree_mask = an1 == an2
                agree_bp[agree_mask] += width
                disagree_bp[~agree_mask] += width
                bp_per_anc_h1[sample_idx, an1] += width
                bp_per_anc_h2[sample_idx, an2] += width

            first_h1 = cur_anc_h1 == NO_ANC
            first_h2 = cur_anc_h2 == NO_ANC
            change_h1 = (~first_h1) & (cur_anc_h1 != an1)
            change_h2 = (~first_h2) & (cur_anc_h2 != an2)

            if change_h1.any():
                _close_tracts(
                    cur_anc_h1[change_h1],
                    tract_start_h1[change_h1],
                    tract_end_h1[change_h1],
                )
                tract_count_h1[change_h1] += 1
                tract_start_h1[change_h1] = pos
                tract_end_h1[change_h1] = pos
                cur_anc_h1[change_h1] = an1[change_h1]
            if change_h2.any():
                _close_tracts(
                    cur_anc_h2[change_h2],
                    tract_start_h2[change_h2],
                    tract_end_h2[change_h2],
                )
                tract_count_h2[change_h2] += 1
                tract_start_h2[change_h2] = pos
                tract_end_h2[change_h2] = pos
                cur_anc_h2[change_h2] = an2[change_h2]

            extend_h1 = (~first_h1) & ~change_h1
            extend_h2 = (~first_h2) & ~change_h2
            if extend_h1.any():
                tract_end_h1[extend_h1] = pos
            if extend_h2.any():
                tract_end_h2[extend_h2] = pos

            if first_h1.any():
                tract_count_h1[first_h1] = 1
                tract_start_h1[first_h1] = pos
                tract_end_h1[first_h1] = pos
                cur_anc_h1[first_h1] = an1[first_h1]
            if first_h2.any():
                tract_count_h2[first_h2] = 1
                tract_start_h2[first_h2] = pos
                tract_end_h2[first_h2] = pos
                cur_anc_h2[first_h2] = an2[first_h2]

            prev_pos = pos
            n_records += 1
            if progress_every and n_records % progress_every == 0:
                print(f"[worker {worker_id}]   ... {n_records:,} records "
                      f"(pos={pos:,})", flush=True)

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"[worker {worker_id}] bcftools query exited with {rc}"
            )
        if n_records == 0:
            raise RuntimeError(
                f"[worker {worker_id}] no records emitted from {anc_vcf}"
            )

        # End-of-chrom flush
        open_h1 = cur_anc_h1 != NO_ANC
        if open_h1.any():
            _close_tracts(
                cur_anc_h1[open_h1],
                tract_start_h1[open_h1],
                tract_end_h1[open_h1],
            )
        open_h2 = cur_anc_h2 != NO_ANC
        if open_h2.any():
            _close_tracts(
                cur_anc_h2[open_h2],
                tract_start_h2[open_h2],
                tract_end_h2[open_h2],
            )
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass

    return {
        "worker_id": worker_id,
        "sample_names": sample_names,
        "chrom_seen": chrom_seen,
        "n_records": n_records,
        "tract_lengths_bp": tract_lengths_bp,
        "tract_count_h1": tract_count_h1,
        "tract_count_h2": tract_count_h2,
        "agree_bp": agree_bp,
        "disagree_bp": disagree_bp,
        "bp_per_anc_h1": bp_per_anc_h1,
        "bp_per_anc_h2": bp_per_anc_h2,
        "bp_window_anc": bp_window_anc,
    }


def _partition_samples(samples: list[str], n_workers: int) -> list[list[str]]:
    """Contiguous, balanced split. Each worker gets ceil(N/W) or floor(N/W) samples."""
    n = len(samples)
    base, extra = divmod(n, n_workers)
    parts: list[list[str]] = []
    s = 0
    for w in range(n_workers):
        size = base + (1 if w < extra else 0)
        parts.append(samples[s:s + size])
        s += size
    return parts


def run_collector(args: argparse.Namespace) -> None:
    out_root: Path = args.out_root
    structural_out = out_root / "structural"
    hap_out = out_root / "hap_disagreement"
    regional_out = out_root / "regional"
    model_out = out_root / "model"
    for d in (structural_out, hap_out, regional_out, model_out):
        d.mkdir(parents=True, exist_ok=True)

    # ── Static inputs ──
    print(f"loading RF hard labels from {args.rf_ancestry}", flush=True)
    rf_hard_labels = load_rf_hard_labels(args.rf_ancestry)
    print(f"  {len(rf_hard_labels):,} samples with RF labels", flush=True)

    chrom_sizes = load_chrom_sizes(args.chrom_sizes)

    masks: list[tuple[str, int, int, str]] = []
    for bed in args.region_mask_bed:
        masks.extend(load_region_bed(bed))
    print(f"loaded {len(masks)} mask intervals from "
          f"{len(args.region_mask_bed)} BED(s)", flush=True)

    anc_names = load_labels_json(args.labels_json)

    model_info = read_model_text(args.flare_model)
    K = int(model_info["n_ancestries"])
    model_T = float(model_info.get("gen_since_admix") or 0.0) or None
    print(f"FLARE K = {K}; model gen_since_admix = {model_T}", flush=True)

    write_mu_vs_global_diff(args.global_tsv, args.flare_model, model_out, anc_names)
    print(f"  wrote {model_out / 'mu_vs_global_diff.json'}", flush=True)

    # ── Probe VCF for samples + chrom ──
    samples, first_chrom = probe_vcf_header(args.anc_vcf)
    n_samples = len(samples)
    print(f"VCF samples: {n_samples}; first chrom: {first_chrom}", flush=True)
    if first_chrom not in chrom_sizes:
        raise RuntimeError(
            f"chrom {first_chrom!r} not in chrom-sizes file {args.chrom_sizes}"
        )

    windows = make_windows(chrom_sizes[first_chrom], args.window_bp, args.step_bp)
    n_wins = len(windows)
    win_starts_list = [w[0] for w in windows]
    win_ends_list = [w[1] for w in windows]
    print(f"  {n_wins} sliding windows "
          f"(window={args.window_bp/1e6:g} Mb, step={args.step_bp/1e6:g} Mb)",
          flush=True)

    if K > 127:
        raise RuntimeError(f"FLARE K={K} exceeds int8 capacity")

    # ── Decide workers ──
    # Keep at least ~300 samples per slice so the bcftools spawn +
    # per-process numpy import isn't a larger fraction of wallclock than
    # the work itself. Above that the per-record dominance of the
    # numpy ops makes the speedup ~linear in `workers`.
    MIN_SAMPLES_PER_SLICE = 300
    workers = max(1, int(args.workers))
    workers = min(workers, max(1, n_samples // MIN_SAMPLES_PER_SLICE))
    print(f"  fan-out: {workers} worker(s) over {n_samples} samples", flush=True)

    partitions = _partition_samples(samples, workers)
    # Progress prints only matter for the single-worker case; in fan-out
    # mode the per-worker tail interleaves.
    progress_every = 50_000 if workers == 1 else 0

    if workers == 1:
        results = [_collect_slice(
            anc_vcf=str(args.anc_vcf),
            sample_names=partitions[0],
            K=K,
            win_starts_list=win_starts_list,
            win_ends_list=win_ends_list,
            worker_id=0,
            progress_every=progress_every,
        )]
    else:
        results = [None] * workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _collect_slice,
                    str(args.anc_vcf),
                    partitions[w],
                    K,
                    win_starts_list,
                    win_ends_list,
                    w,
                    progress_every,
                ): w
                for w in range(workers)
            }
            for fut in concurrent.futures.as_completed(futures):
                w = futures[fut]
                results[w] = fut.result()
                print(f"  worker {w} done: "
                      f"{results[w]['n_records']:,} records, "
                      f"{sum(len(v) for v in results[w]['tract_lengths_bp'].values()):,} "
                      f"tracts", flush=True)

    # ── Cohort-level reduction ──
    # Sanity: every worker must agree on chrom and n_records (they all
    # read the same per-cluster file).
    chrom_seen = results[0]["chrom_seen"]
    n_records = results[0]["n_records"]
    for r in results[1:]:
        if r["chrom_seen"] != chrom_seen:
            raise RuntimeError(
                f"worker chrom mismatch: {chrom_seen!r} vs {r['chrom_seen']!r}"
            )
        if r["n_records"] != n_records:
            raise RuntimeError(
                f"worker n_records mismatch: {n_records} vs {r['n_records']}"
            )
    print(f"  cohort: {n_records:,} records streamed across {workers} worker(s)",
          flush=True)

    # Concatenate per-sample arrays in worker order (matches the original
    # sample order because partitions are contiguous).
    cohort_samples = [s for r in results for s in r["sample_names"]]
    if cohort_samples != samples:
        raise RuntimeError(
            "cohort sample-name concat does not match VCF header order"
        )
    tract_count_h1 = np.concatenate([r["tract_count_h1"] for r in results])
    tract_count_h2 = np.concatenate([r["tract_count_h2"] for r in results])
    agree_bp = np.concatenate([r["agree_bp"] for r in results])
    disagree_bp = np.concatenate([r["disagree_bp"] for r in results])
    bp_per_anc_h1 = np.concatenate([r["bp_per_anc_h1"] for r in results], axis=0)
    bp_per_anc_h2 = np.concatenate([r["bp_per_anc_h2"] for r in results], axis=0)
    bp_window_anc = sum(r["bp_window_anc"] for r in results)

    # Per-ancestry tract length lists: extend.
    tract_lengths_bp: dict[int, list[int]] = {a: [] for a in range(K)}
    for r in results:
        for a, lst in r["tract_lengths_bp"].items():
            tract_lengths_bp[a].extend(lst)

    n_tracts_total = sum(len(v) for v in tract_lengths_bp.values())
    print(f"  total tracts: {n_tracts_total:,}", flush=True)

    write_structural_outputs(
        tract_lengths_bp, tract_count_h1, tract_count_h2,
        samples, bp_per_anc_h1, bp_per_anc_h2,
        anc_names, model_T, K, structural_out,
    )
    write_hap_disagreement_outputs(
        samples, rf_hard_labels, anc_names,
        agree_bp, disagree_bp,
        bp_per_anc_h1, bp_per_anc_h2, hap_out,
    )
    write_regional_outputs(
        chrom_seen, windows, bp_window_anc, n_samples, K, anc_names,
        masks, args.fdr_q, regional_out,
    )


# ── Output writers ────────────────────────────────────────────────────────


def write_structural_outputs(
    tract_lengths_bp: dict[int, list[int]],
    tract_count_h1: np.ndarray,
    tract_count_h2: np.ndarray,
    samples: list[str],
    bp_per_anc_h1: np.ndarray,
    bp_per_anc_h2: np.ndarray,
    anc_names: dict[int, str],
    model_T: float | None,
    K: int,
    out_dir: Path,
) -> None:
    """Emit tract_length_summary.json + switch_rate_summary.json +
    switch_rate_per_hap.tsv.

    The per-hap TSV is the durable, FLARE-keyed artifact for switch
    rate: one row per (sample, hap) with the haplotype's switch
    count and the haplotype's dominant FLARE ancestry. This lets the
    report stratify by FLARE top-1 without re-deriving anything from
    cohort_global.tsv.
    """
    # ── tract_length_summary.json ──
    per_anc = []
    n_tracts_total = 0
    for a in sorted(tract_lengths_bp.keys()):
        lengths_bp = np.array(tract_lengths_bp[a], dtype=np.float64)
        n = int(lengths_bp.size)
        n_tracts_total += n
        if n == 0:
            mean_Mb = 0.0
            median_Mb = 0.0
        else:
            lengths_Mb = lengths_bp / 1e6
            mean_Mb = float(lengths_Mb.mean())
            median_Mb = float(np.median(lengths_Mb))
        if n >= 100 and mean_Mb > 0:
            exp_fit_rate = float(1.0 / mean_Mb)
            implied_T = 100.0 / (K * mean_Mb)
        else:
            exp_fit_rate = None
            implied_T = None
        per_anc.append({
            "ancestry": int(a),
            "name": anc_names.get(int(a), f"ancestry_{a}"),
            "n_tracts": n,
            "mean_Mb": mean_Mb,
            "median_Mb": median_Mb,
            "exp_fit_rate": exp_fit_rate,
            "implied_T_gen": implied_T,
            "model_T_gen": model_T,
        })

    note = ("stats computed per-ancestry; exp_fit_rate is exponential MLE "
            "1/mean_Mb; implied_T_gen assumes 1 cM/Mb. exp_fit_rate is null "
            "for ancestries with n_tracts < 100.")
    (out_dir / "tract_length_summary.json").write_text(json.dumps({
        "n_tracts_total": n_tracts_total,
        "per_ancestry": per_anc,
        "note": note,
    }, indent=2))
    print(f"  wrote {out_dir / 'tract_length_summary.json'}", flush=True)

    # ── switch_rate_summary.json ──
    # switches per hap = tract_count - 1; only count haps that saw ≥1 site.
    saw_h1 = tract_count_h1 > 0
    saw_h2 = tract_count_h2 > 0
    sw_h1 = tract_count_h1[saw_h1] - 1
    sw_h2 = tract_count_h2[saw_h2] - 1
    switches = np.concatenate([sw_h1, sw_h2])
    n_haps = int(switches.size)
    if n_haps == 0:
        raise RuntimeError("no haplotypes saw any data; cannot compute switch rate")

    max_sw = int(switches.max())
    bins_summary = [0, 3, 10, 20, 50, 100, max_sw + 1]
    # Deduplicate adjacent equal edges (when max_sw < 100 the +1 edge can
    # collide with the prior fixed edges); keep ascending unique.
    edges = sorted(set(bins_summary))
    histogram = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        count = int(((switches >= lo) & (switches < hi)).sum())
        histogram.append({"bin_lo": int(lo), "bin_hi": int(hi), "count": count})

    (out_dir / "switch_rate_summary.json").write_text(json.dumps({
        "n_haplotypes": n_haps,
        "mean": float(switches.mean()),
        "median": float(np.median(switches)),
        "p99": float(np.percentile(switches, 99)),
        "min": int(switches.min()),
        "max": int(switches.max()),
        "histogram": histogram,
    }, indent=2))

    # ── switch_rate_per_hap.tsv ──
    # FLARE-keyed durable artifact: one row per (sample, hap). The
    # haplotype's dominant ancestry is FLARE's argmax over
    # bp_per_anc. Rows where the hap saw zero data are omitted
    # (consistent with the cohort aggregates above).
    dom_h1_idx = bp_per_anc_h1.argmax(axis=1)
    dom_h2_idx = bp_per_anc_h2.argmax(axis=1)
    with open(out_dir / "switch_rate_per_hap.tsv", "w") as f:
        f.write("sample_id\thap\tdominant_anc\tn_switches\n")
        for i, sid in enumerate(samples):
            if saw_h1[i]:
                f.write(f"{sid}\t1\t{anc_names.get(int(dom_h1_idx[i]), f'ancestry_{int(dom_h1_idx[i])}')}\t{int(tract_count_h1[i]) - 1}\n")
            if saw_h2[i]:
                f.write(f"{sid}\t2\t{anc_names.get(int(dom_h2_idx[i]), f'ancestry_{int(dom_h2_idx[i])}')}\t{int(tract_count_h2[i]) - 1}\n")
    print(f"  wrote {out_dir / 'switch_rate_per_hap.tsv'}", flush=True)
    print(f"  wrote {out_dir / 'switch_rate_summary.json'}", flush=True)


def write_hap_disagreement_outputs(
    samples: list[str],
    rf_hard_labels: dict[str, tuple[str, float]],
    anc_names: list[str],
    agree_bp: np.ndarray,
    disagree_bp: np.ndarray,
    bp_per_anc_h1: np.ndarray,
    bp_per_anc_h2: np.ndarray,
    out_dir: Path,
) -> None:
    """Emit ``per_sample.tsv`` + ``summary.json`` for hap disagreement.

    The per-sample TSV is the durable, FLARE-keyed artifact: it carries
    every sample's hap1-vs-hap2 disagreement fraction along with
    FLARE's per-hap dominant ancestry name (``dom_h1``, ``dom_h2``)
    and FLARE's per-sample top-1 (the dominant of ``dom_h1``,
    ``dom_h2``). The RF label and its max-prob ride along on every row
    for downstream filtering, but no metric is bucketed against RF and
    no relabelling happens.

    The summary aggregates by FLARE's per-sample top-1 only (no
    `per_rf_label` block). RF's MID column can never appear because
    FLARE has no MID class — this is a category A metric.

    A sample with no RF label is a broken join and raises.
    """
    total_bp = agree_bp + disagree_bp
    if (total_bp == 0).any():
        # A sample with zero bp means the VCF held no records spanning
        # that sample — refuse rather than emit divide-by-zero rows.
        idx = int(np.where(total_bp == 0)[0][0])
        raise RuntimeError(
            f"sample {samples[idx]!r} has zero covered bp; "
            f"hap_disagreement undefined"
        )

    # FLARE per-hap dominant ancestry (argmax over the FLARE bp-per-
    # ancestry breakdown). FLARE per-sample top-1 = argmax over
    # (bp_per_anc_h1 + bp_per_anc_h2).
    dom_h1_idx = bp_per_anc_h1.argmax(axis=1)
    dom_h2_idx = bp_per_anc_h2.argmax(axis=1)
    top1_idx = (bp_per_anc_h1 + bp_per_anc_h2).argmax(axis=1)
    disagree_frac = disagree_bp / total_bp
    agree_frac = 1.0 - disagree_frac

    # Refuse the silent "unjoined" fallback: every sample needs a real
    # RF label. A missing join is a collector bug, not a data point.
    missing = [sid for sid in samples if sid not in rf_hard_labels]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples missing from RF label table; "
            f"first few: {missing[:5]!r}. The collector does not invent "
            f"'unjoined' rows — fix the RF join upstream."
        )

    # Per-sample TSV (durable; FLARE-keyed; RF columns carried for
    # filtering, not for stratification).
    cols = [
        "sample_id",
        "flare_top1",
        "dominant_anc_h1",
        "dominant_anc_h2",
        "agreement_bp_frac",
        "disagreement_bp_frac",
        "total_bp",
        "rf_hard_label",
        "rf_max_prob",
    ]
    per_top1: dict[str, list[float]] = defaultdict(list)
    with open(out_dir / "per_sample.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for i, sid in enumerate(samples):
            rf_label, rf_p = rf_hard_labels[sid]
            top1_name = anc_names[int(top1_idx[i])]
            row = (
                sid,
                top1_name,
                anc_names[int(dom_h1_idx[i])],
                anc_names[int(dom_h2_idx[i])],
                f"{float(agree_frac[i]):.6f}",
                f"{float(disagree_frac[i]):.6f}",
                int(total_bp[i]),
                rf_label,
                f"{rf_p:.4f}",
            )
            f.write("\t".join(str(x) for x in row) + "\n")
            per_top1[top1_name].append(float(disagree_frac[i]))
    print(f"  wrote {out_dir / 'per_sample.tsv'}", flush=True)

    cohort_mean = float(disagree_frac.mean())
    per_flare_top1 = []
    for k in sorted(per_top1.keys()):
        vals = per_top1[k]
        if not vals:
            continue
        per_flare_top1.append({
            "flare_top1": k,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
        })

    (out_dir / "summary.json").write_text(json.dumps({
        "cohort_mean_disagreement": cohort_mean,
        "n_samples": int(len(samples)),
        "per_flare_top1": per_flare_top1,
    }, indent=2))
    print(f"  wrote {out_dir / 'summary.json'}", flush=True)


def write_regional_outputs(
    chrom: str,
    windows: list[tuple[int, int]],
    bp_window_anc: np.ndarray,
    n_samples: int,
    K: int,
    anc_names: dict[int, str],
    masks: list[tuple[str, int, int, str]],
    fdr_q: float,
    out_dir: Path,
) -> None:
    """Emit windows.tsv.gz + significant.bed + summary.json."""
    win_lens = np.array([(e - s) for s, e in windows], dtype=np.float64)
    n_haps = 2 * n_samples
    denom = win_lens[:, None] * n_haps  # (n_wins, 1)
    mean_anc = bp_window_anc / denom  # (n_wins, K)

    chrom_mean = mean_anc.mean(axis=0)  # (K,)
    chrom_sd = mean_anc.std(axis=0, ddof=1)  # (K,)

    rows: list[dict] = []
    for wi, (ws, we) in enumerate(windows):
        # Window-level mask names: any mask interval that intersects [ws, we).
        mask_names: list[str] = []
        for m_chrom, m_s, m_e, m_name in masks:
            if m_chrom != chrom:
                continue
            if min(m_e, we) > max(m_s, ws):
                mask_names.append(m_name)
        mask_label = ",".join(sorted(set(mask_names))) if mask_names else ""
        for a in range(K):
            if chrom_sd[a] == 0 or np.isnan(chrom_sd[a]):
                z = 0.0
                p_raw = 1.0
            else:
                z = float((mean_anc[wi, a] - chrom_mean[a]) / chrom_sd[a])
                p_raw = float(2.0 * norm.sf(abs(z)))
            rows.append({
                "chrom": chrom,
                "start": ws,
                "end": we,
                "ancestry": a,
                "mean_anc": float(mean_anc[wi, a]),
                "z": z,
                "p": p_raw,
                "mask_region": mask_label,
            })

    # BH-FDR across all (window, ancestry) tests.
    p_arr = np.array([r["p"] for r in rows])
    _, q_arr, _, _ = multipletests(p_arr, alpha=fdr_q, method="fdr_bh")
    for r, q in zip(rows, q_arr):
        r["q"] = float(q)
        r["ancestry_name"] = anc_names.get(r["ancestry"], f"ancestry {r['ancestry']}")

    # ── windows.tsv.gz ──
    cols = ["chrom", "start", "end", "ancestry_name", "mean_anc", "z", "p", "q", "mask_region"]
    with gzip.open(out_dir / "windows.tsv.gz", "wt") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write(
                f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['ancestry_name']}\t"
                f"{r['mean_anc']:.4f}\t{r['z']:+.2f}\t{r['p']:.2e}\t{r['q']:.2e}\t"
                f"{r['mask_region']}\n"
            )
    print(f"  wrote {out_dir / 'windows.tsv.gz'} ({len(rows)} rows)", flush=True)

    # ── significant.bed ──
    sig = [r for r in rows if r["q"] < fdr_q]
    with open(out_dir / "significant.bed", "w") as f:
        f.write("#chrom\tstart\tend\tname\n")
        for r in sig:
            name = f"{r['ancestry_name']}|z{r['z']:+.2f}|q{r['q']:.1e}"
            if r["mask_region"]:
                name += f"|{r['mask_region']}"
            f.write(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{name}\n")
    print(f"  wrote {out_dir / 'significant.bed'} ({len(sig)} significant)", flush=True)

    # ── summary.json ──
    per_anc_summary = []
    for a in sorted({r["ancestry"] for r in rows}):
        a_sig = [r for r in sig if r["ancestry"] == a]
        peak = None
        if a_sig:
            peak_row = max(a_sig, key=lambda r: abs(r["z"]))
            peak = {
                "chrom": peak_row["chrom"],
                "start": int(peak_row["start"]),
                "end": int(peak_row["end"]),
                "z": float(peak_row["z"]),
                "q": float(peak_row["q"]),
                "mask_region": peak_row["mask_region"],
            }
        per_anc_summary.append({
            "ancestry": int(a),
            "name": anc_names.get(int(a), f"ancestry {a}"),
            "n_significant": len(a_sig),
            "peak_window": peak,
        })

    def _mask_count(token: str) -> int:
        token = token.lower()
        return sum(1 for r in sig if token in r["mask_region"].lower())

    (out_dir / "summary.json").write_text(json.dumps({
        "n_windows_total": len(rows),
        "n_windows_significant": len(sig),
        "fdr_q_threshold": float(fdr_q),
        "per_ancestry": per_anc_summary,
        "hla_overlap_n": _mask_count("hla"),
        "centromere_overlap_n": _mask_count("centromere"),
        "segdup_overlap_n": _mask_count("segdup"),
        "high_ld_overlap_n": _mask_count("high"),
        "outside_mask_n": sum(1 for r in sig if not r["mask_region"]),
    }, indent=2))
    print(f"  wrote {out_dir / 'summary.json'}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--anc-vcf", type=Path, required=True,
                   help="FLARE <prefix>.anc.vcf.gz")
    p.add_argument("--global-tsv", type=Path, required=True,
                   help="popout-format global ancestry TSV (for mu vs global check)")
    p.add_argument("--flare-model", type=Path, required=True,
                   help="FLARE <prefix>.model text file")
    p.add_argument("--rf-ancestry", type=Path, required=True,
                   help="RF ancestry predictions (research_id, ancestry_pred, probabilities)")
    p.add_argument("--chrom-sizes", type=Path, required=True)
    p.add_argument("--region-mask-bed", type=Path, action="append", default=[],
                   help="BED of named regions to overlay (repeatable)")
    p.add_argument("--labels-json", type=Path, default=None,
                   help="labels.json from compare_to_rf.py for ancestry names (optional)")
    p.add_argument("--window-bp", type=int, default=1_000_000)
    p.add_argument("--step-bp", type=int, default=250_000)
    p.add_argument("--fdr-q", type=float, default=0.05)
    p.add_argument("--out-root", type=Path, required=True,
                   help="Artifact work root; writes <out-root>/{structural,hap_disagreement,regional,model}/")
    p.add_argument("--workers", type=int, default=1,
                   help="Per-sample fan-out. Each worker subsets the VCF via "
                        "`bcftools query -S` to its sample slice and streams "
                        "independently; the master reduces. 1 = legacy single "
                        "process. The orchestrator passes the WDL-allocated cpu "
                        "count by default.")
    args = p.parse_args()

    if args.step_bp > args.window_bp:
        raise ValueError("--step-bp must be <= --window-bp")
    for path in (args.anc_vcf, args.global_tsv, args.flare_model,
                 args.rf_ancestry, args.chrom_sizes):
        if not path.exists():
            raise FileNotFoundError(path)

    run_collector(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
