#!/usr/bin/env python3
"""tbi_byte_range.py — derive BGZF-block-aligned [byte_start, byte_end] for a region.

Reads the tabix .tbi linear index (via parse_tbi() from generate_partitions.py) and
walks to the bins covering the requested region. Prints two integers to stdout:

    byte_start byte_end

byte_start is the file offset of the BGZF block containing the first record in the
region; byte_end is one less than the file offset of the BGZF block just after the
region, so the half-open range [byte_start, byte_end] in HTTP-Range terms covers
exactly the BGZF blocks that hold the region's records.

Used by throughput_probe.sh so the WDL caller only needs to supply --region; the
byte offsets are computed Terra-side from the localized .tbi instead of being
fed in via the inputs JSON.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_GP_PATH = Path("/usr/local/bin/generate_partitions.py")


def _load_gp():
    spec = importlib.util.spec_from_file_location("gp", _GP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gp"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tbi-path", required=True, type=Path)
    ap.add_argument("--region", required=True, help="e.g. chr21:25000000-25210000")
    args = ap.parse_args()

    gp = _load_gp()

    if ":" not in args.region or "-" not in args.region.split(":", 1)[1]:
        raise SystemExit(f"bad --region {args.region!r}; expected chr:start-end")
    chrom, rest = args.region.split(":", 1)
    start_bp, end_bp = (int(x) for x in rest.split("-", 1))
    if start_bp < 1 or end_bp < start_bp:
        raise SystemExit(f"bad --region {args.region!r}; positions must be 1-based and start <= end")

    indexes = gp.parse_tbi(args.tbi_path)
    if chrom not in indexes:
        raise SystemExit(f"chrom {chrom!r} not present in {args.tbi_path}; "
                         f"have: {list(indexes.keys())!r}")

    linear = indexes[chrom]
    n = len(linear)
    if n == 0:
        raise SystemExit(f"chrom {chrom!r} has empty linear index in {args.tbi_path}")

    BIN = gp.TBI_LINEAR_BIN_BP  # 16384
    foffs = [(v >> 16) for v in linear]

    start_bin = (start_bp - 1) // BIN
    end_bin   = (end_bp - 1) // BIN

    if start_bin >= n:
        raise SystemExit(
            f"start position {start_bp} beyond last bin ({n-1}, covering up to bp {n*BIN}) for {chrom}"
        )
    if end_bin >= n:
        raise SystemExit(
            f"end position {end_bp} beyond last bin ({n-1}, covering up to bp {n*BIN}) for {chrom}; "
            f"pick a region inside the .tbi's known bins, or get the file size some other way"
        )

    byte_start = foffs[start_bin]
    if end_bin + 1 >= n:
        raise SystemExit(
            f"region end is in the last bin (no foffs[end_bin+1] to derive byte_end from); "
            f"pick a region with end_bin <= {n-2}"
        )
    byte_end = foffs[end_bin + 1] - 1

    if byte_end < byte_start:
        raise SystemExit(
            f"computed byte_end ({byte_end}) < byte_start ({byte_start}); "
            f"the region likely spans an empty stretch of the index. Pick a denser region."
        )

    print(f"{byte_start} {byte_end}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
