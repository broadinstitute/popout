#!/usr/bin/env python3
"""
filter_global_anc.py

Drop rows of a FLARE `.global.anc.gz` whose first column (SAMPLE) appears in a
drop-list file, preserving the file's header verbatim.

FLARE's real header (from ref_panel population names) looks like:

    SAMPLE  eas  amr  eur  afr  sas
    1000618 0.906 0 0.019 0.002 0.072
    ...

Per the NIH DATA FIDELITY rule in CLAUDE.md, the header line is copied byte
for byte from the input into the output. No column renaming, no reordering,
no reconstruction of population names from any external source.

Exit non-zero if:
  - header line is missing or does not start with 'SAMPLE'
  - the drop-list references samples not present in the input (belt-and-
    suspenders check; the preflight has already scoped this per-cluster)
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path


def read_drop_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line)
    return ids


def filter_stream(in_path: Path, out_path: Path, drops: set[str],
                  strict_present: bool) -> tuple[int, int, set[str]]:
    """Copy header verbatim, drop rows whose column-1 sample is in `drops`.

    Returns (kept_rows, dropped_rows, seen_drops).
    """
    kept = 0
    dropped = 0
    seen: set[str] = set()

    with gzip.open(in_path, "rb") as fin, gzip.open(out_path, "wb") as fout:
        header = fin.readline()
        if not header:
            raise RuntimeError(f"{in_path}: empty file, no header")
        first_field = header.split(b"\t", 1)[0]
        if first_field != b"SAMPLE":
            raise RuntimeError(
                f"{in_path}: first header field {first_field!r} != b'SAMPLE'; "
                "refusing to filter a file whose shape we do not recognise"
            )
        fout.write(header)

        for raw in fin:
            sid = raw.split(b"\t", 1)[0].decode("ascii")
            if sid in drops:
                seen.add(sid)
                dropped += 1
                continue
            fout.write(raw)
            kept += 1

    if strict_present:
        missing = drops - seen
        if missing:
            raise RuntimeError(
                f"{in_path}: {len(missing)} drop-list sample(s) not present in "
                f"input: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )

    return kept, dropped, seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in",       dest="in_path",  required=True, type=Path)
    ap.add_argument("--out",      dest="out_path", required=True, type=Path)
    ap.add_argument("--drop-ids", dest="drops",    required=True, type=Path,
                    help="one research ID per line, '#' comments allowed")
    ap.add_argument("--strict-present", action="store_true",
                    help="fail if a drop-list ID is not present in the input")
    args = ap.parse_args()

    drops = read_drop_ids(args.drops)
    if not drops:
        raise SystemExit(f"{args.drops}: no drop IDs parsed; refusing to run")

    kept, dropped, seen = filter_stream(
        args.in_path, args.out_path, drops, args.strict_present
    )
    print(
        f"filter_global_anc: {args.in_path} -> {args.out_path}: "
        f"kept={kept} dropped={dropped} seen={len(seen)}/{len(drops)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
