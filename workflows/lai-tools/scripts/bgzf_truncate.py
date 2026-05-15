#!/usr/bin/env python3
"""bgzf_truncate.py — truncate a BGZF file to the last complete block.

Used by `prep_local_slice` in `throughput_probe.sh`. The script concatenates
the first 64 MiB of a remote VCF with a region byte slice; byte 67108864
will almost never coincide with a BGZF block boundary, so the seam in the
concatenated file is mid-deflate and breaks strict BSIZE-following readers.

Walking BGZF block headers via the 18-byte gzip+FEXTRA layout (magic
``1f 8b 08 04`` + ``BC`` subfield + BSIZE u16 LE at offset 16) lets us
truncate to the last byte of the last complete block. The result is a
spec-clean BGZF prefix that any reader (htslib or otherwise) can consume.

Usage:
    bgzf_truncate.py INPUT OUTPUT

Reads INPUT in full into memory (we run this on ~64 MiB inputs in the
throughput probe; not designed for streaming-on-disk truncation).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

_HEADER_LEN = 18                                # 12-byte gzip + 6-byte FEXTRA
_MAGIC = b"\x1f\x8b\x08\x04"                    # ID1 ID2 CM FLG (FEXTRA only)
_BC = b"BC"                                     # FEXTRA subfield ID for BGZF


def last_complete_block_end(data: bytes) -> int:
    """Return the byte offset (== length of valid prefix) up to and
    including the last complete BGZF block in ``data``.

    Raises if the input doesn't start with a BGZF magic — that's an
    operator-level configuration error, not something to silently truncate.
    """
    if len(data) < _HEADER_LEN:
        raise ValueError(f"input is only {len(data)} bytes; needs >= {_HEADER_LEN}")
    if data[:4] != _MAGIC:
        raise ValueError(
            f"input does not start with BGZF magic; got {data[:4].hex()}"
        )

    pos = 0
    last_good_end = 0
    while pos + _HEADER_LEN <= len(data):
        if data[pos:pos + 4] != _MAGIC:
            # Lost alignment — return up through the last good block.
            break
        if data[pos + 12:pos + 14] != _BC:
            break
        bsize = struct.unpack_from("<H", data, pos + 16)[0]
        block_end = pos + bsize + 1
        if block_end > len(data):
            break                                # incomplete tail block
        pos = block_end
        last_good_end = pos
    return last_good_end


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} INPUT OUTPUT", file=sys.stderr)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    data = src.read_bytes()
    end = last_complete_block_end(data)
    if end == 0:
        raise SystemExit(f"{src}: no complete BGZF blocks found in {len(data)} bytes")
    Path(dst).write_bytes(data[:end])
    print(
        f"bgzf_truncate: {src} ({len(data)} B) -> {dst} ({end} B); "
        f"dropped {len(data) - end} B of partial tail",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
