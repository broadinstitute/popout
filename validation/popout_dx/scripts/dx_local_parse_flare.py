#!/usr/bin/env python3
"""popout DX — sampled-subset FLARE parser (bcftools-streamed).

Builds a :class:`popout.benchmark.common.TractSet` from a FLARE
``.anc.vcf.gz``, restricted to a sample subset (typically the
stratified picker's output, ~175 samples per cluster). API-compatible
with what ``popout.benchmark.align`` and ``popout.benchmark.metrics``
expect, so downstream local-mode steps can use those libraries
unchanged.

Performance contract (see ``../PERFORMANCE_CONTRACT.md``):

  #1  Stream via ``bcftools query`` — never pysam per-cell.
  #2  No per-record Python objects; ``np.array(parts[2:], dtype=np.uint16)``
      parses the whole row in one C call.
  #4  ``bcftools query -S samples.txt`` restricts FORMAT output to the
      sampled subset at the C layer; never reads the whole VCF in Python.
  #5  No ProcessPoolExecutor — the subset size (~175) is well below the
      300-sample fan-out threshold; spawn overhead would dominate.
"""

from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from popout.benchmark.common import MISSING_LABEL, TractSet, load_ancestry_header


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"dx_local_parse_flare: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── Header parsing (one short subprocess; not a hot path) ────────────────


def read_ancestry_header(vcf_path: Path) -> dict[int, str]:
    """Parse ``##ANCESTRY=<...>`` from the VCF header via ``bcftools view -h``."""
    res = subprocess.run(
        ["bcftools", "view", "-h", str(vcf_path)],
        check=True, capture_output=True, text=True,
    )
    for line in res.stdout.splitlines():
        if line.startswith("##ANCESTRY="):
            return load_ancestry_header(line.strip())
    die(f"{vcf_path}: no ##ANCESTRY= header found")


def list_vcf_samples(vcf_path: Path) -> list[str]:
    res = subprocess.run(
        ["bcftools", "query", "-l", str(vcf_path)],
        check=True, capture_output=True, text=True,
    )
    return [s for s in res.stdout.split("\n") if s]


# ── The hot loop ─────────────────────────────────────────────────────────


def stream_and_build_calls(
    vcf_path: Path,
    samples_subset_file: Path,
    n_samples: int,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Stream ``bcftools query`` output; return ``(chrom, positions, calls)``.

    ``calls`` is ``(2 * n_samples, n_sites)`` uint16 in interleaved
    sample-major order (column for sample i, hap 0 at row 2*i; hap 1 at
    row 2*i+1). Matches the layout produced by
    ``popout.benchmark.parsers.flare.parse_flare``.

    Inner loop honours perf contract #1 and #2: split + numpy parse, no
    per-record object construction, no dict lookups.
    """
    fmt = "%CHROM\\t%POS[\\t%AN1\\t%AN2]\\n"
    expected_cells = 2 * n_samples
    cmd = [
        "bcftools", "query",
        "-S", str(samples_subset_file),
        "-f", fmt,
        str(vcf_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    chrom_seen: str | None = None
    positions: list[int] = []
    cols: list[np.ndarray] = []   # one numpy array per VCF record

    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\n").split("\t")
        # Tolerate ``.`` for missing AN1/AN2 by mapping to MISSING_LABEL.
        # Replace in-place; this is O(2*n_samples) per record but only
        # for the rare-missing case (FLARE's outputs are dense).
        if "." in parts[2:]:
            parts = parts[:2] + [
                str(MISSING_LABEL) if p == "." else p for p in parts[2:]
            ]
        if chrom_seen is None:
            chrom_seen = parts[0]
        elif parts[0] != chrom_seen:
            die(
                f"{vcf_path}: multiple chromosomes in VCF ({chrom_seen!r} and "
                f"{parts[0]!r}); FLARE per-cluster .anc.vcf.gz should be single-chrom"
            )
        positions.append(int(parts[1]))
        row = np.array(parts[2:], dtype=np.uint16)
        if row.shape[0] != expected_cells:
            die(
                f"{vcf_path}: row at pos {parts[1]} has {row.shape[0]} cells "
                f"!= expected {expected_cells} (2 * {n_samples} subset samples)"
            )
        cols.append(row)

    rc = proc.wait()
    if rc != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        die(f"bcftools query exited {rc}: {stderr.strip()}")

    if not positions:
        die(f"{vcf_path}: bcftools query produced zero records for the requested samples")
    if chrom_seen is None:
        die(f"{vcf_path}: bcftools query produced rows without chrom")

    # (n_sites, 2*n_samples) → (2*n_samples, n_sites) to match the
    # popout.benchmark.parsers.flare layout.
    calls = np.stack(cols, axis=0).T
    site_positions = np.array(positions, dtype=np.int64)
    return chrom_seen, site_positions, calls


# ── Public entry: build a TractSet ───────────────────────────────────────


def parse_flare_subset(
    vcf_path: Path,
    selected_samples: list[str],
    samples_subset_file: Path,
) -> TractSet:
    """Build a ``TractSet`` from a FLARE VCF restricted to ``selected_samples``.

    ``samples_subset_file`` is a workspace path the caller controls; this
    function writes the VCF-order intersection of ``selected_samples`` and
    the VCF's actual sample list to it, then invokes bcftools.

    Missing samples (in selected but not in VCF) are an error — silent
    drops would mask upstream sample-id drift.
    """
    label_map = read_ancestry_header(vcf_path)
    vcf_samples = list_vcf_samples(vcf_path)
    vcf_set = set(vcf_samples)
    missing = [s for s in selected_samples if s not in vcf_set]
    if missing:
        die(
            f"{vcf_path}: {len(missing)} selected sample(s) absent from VCF; "
            f"first: {missing[:5]}"
        )
    # Preserve VCF order — bcftools query -S emits FORMAT fields in the
    # order they appear in the VCF, not the order of the samples file.
    selected_set = set(selected_samples)
    ordered = [s for s in vcf_samples if s in selected_set]
    samples_subset_file.write_text("\n".join(ordered) + "\n")

    chrom, positions, calls = stream_and_build_calls(
        vcf_path, samples_subset_file, n_samples=len(ordered),
    )

    hap_ids = np.array(
        [f"{s}:{h}" for s in ordered for h in (0, 1)],
        dtype=object,
    )

    ts = TractSet(
        tool_name="flare",
        chrom=chrom,
        hap_ids=hap_ids,
        site_positions=positions,
        calls=calls,
        label_map=label_map,
    )
    ts.validate()
    return ts


# ── CLI (for standalone testing / orchestrator invocation) ───────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf", required=True, type=Path)
    ap.add_argument("--samples-file", required=True, type=Path,
                    help="one sample_id per line; the selected subset to parse")
    ap.add_argument("--out-npz", required=True, type=Path,
                    help="numpy-savez output with hap_ids, site_positions, calls, label_map_json, chrom, tool_name")
    ap.add_argument("--workspace", type=Path, default=None,
                    help="temp dir for the bcftools samples subset file; defaults to out-npz dir")
    args = ap.parse_args()

    selected = [s for s in args.samples_file.read_text().splitlines() if s.strip()]
    if not selected:
        die(f"--samples-file {args.samples_file} is empty")

    workspace = args.workspace or args.out_npz.parent
    workspace.mkdir(parents=True, exist_ok=True)
    subset_file = workspace / "flare_subset_samples.txt"

    ts = parse_flare_subset(args.vcf, selected, subset_file)

    import json
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        hap_ids=ts.hap_ids,
        site_positions=ts.site_positions,
        calls=ts.calls,
        label_map_json=json.dumps({int(k): v for k, v in ts.label_map.items()}),
        chrom=ts.chrom,
        tool_name=ts.tool_name,
    )
    print(
        f"dx_local_parse_flare: wrote {args.out_npz} "
        f"({ts.calls.shape[0]} hap rows × {ts.calls.shape[1]} sites, chrom={ts.chrom})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
