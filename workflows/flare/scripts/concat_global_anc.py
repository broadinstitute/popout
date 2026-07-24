#!/usr/bin/env python3
"""
concat_global_anc.py

Row-concatenate multiple FLARE `.global.anc.gz` files (one per cluster for a
single chromosome) into a single per-chrom `.global.anc.gz` covering the
union of samples.

Every input MUST share a byte-identical header. The header is FLARE's real
`SAMPLE<TAB><pop1><TAB>...<popN>` string (population names come from the
ref_panel) and must be preserved verbatim per the NIH DATA FIDELITY rule in
CLAUDE.md. A mismatched header across inputs is a hard fail — it is
evidence of a ref_panel discrepancy between clusters, and a naive concat
would silently misalign columns.

Inputs are consumed in the order given; the plan is to pass them in
cluster_id-sorted order so the output row order is deterministic.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",   dest="out_path", required=True, type=Path)
    ap.add_argument("--input", dest="inputs",   required=True, type=Path,
                    action="append",
                    help="repeat for each per-cluster global.anc.gz; "
                         "processed in the order given")
    args = ap.parse_args()

    if len(args.inputs) < 1:
        raise SystemExit("need at least one --input")

    canonical_header: bytes | None = None
    total_rows = 0

    with gzip.open(args.out_path, "wb") as fout:
        for i, in_path in enumerate(args.inputs):
            with gzip.open(in_path, "rb") as fin:
                header = fin.readline()
                if not header:
                    raise RuntimeError(f"{in_path}: empty file, no header")
                if not header.split(b"\t", 1)[0] == b"SAMPLE":
                    raise RuntimeError(
                        f"{in_path}: first header field is not b'SAMPLE'; "
                        "refusing to concatenate a file of unknown shape"
                    )
                if canonical_header is None:
                    canonical_header = header
                    fout.write(header)
                elif header != canonical_header:
                    raise RuntimeError(
                        f"{in_path}: header mismatch\n"
                        f"  expected: {canonical_header!r}\n"
                        f"  got:      {header!r}\n"
                        "byte-identical headers are required; refusing to "
                        "silently misalign ancestry columns"
                    )

                rows = 0
                for raw in fin:
                    fout.write(raw)
                    rows += 1
                total_rows += rows
                print(f"  [{i+1}/{len(args.inputs)}] {in_path.name}: "
                      f"{rows} data rows")

    print(f"concat_global_anc: {args.out_path}: "
          f"{len(args.inputs)} inputs, {total_rows} data rows total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
