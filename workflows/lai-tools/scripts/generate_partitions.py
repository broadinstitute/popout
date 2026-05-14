#!/usr/bin/env python3
"""
generate_partitions.py — byte-balanced genomic partitions for streaming scatter.

Reads a tabix index (.tbi) and emits genomic partitions sized so each covers
~target bytes in the input file. Used to scatter `bcftools +split` across
many preemptible workers, each streaming only its slice of a biobank-scale
VCF from gs://.

Why bytes-from-index instead of bp or records:
  - Bytes-in-index correlate with both record density and per-record size.
  - High-density loci (HLA, KIR) get small-bp but normal-byte partitions
    automatically — no special-case BED files needed.

Outputs:
  --out-manifest    TSV: partition_id, chrom, start, end, estimated_bytes
  --out-regions     one chrN:start-end per line, matching manifest order
  --out-region-ids  one partition_id per line, matching manifest order

All three are in largest-bytes-first order for Cromwell scheduling.

Tabix v1 format reference: https://samtools.github.io/hts-specs/tabix.pdf
"""
from __future__ import annotations

import argparse
import gzip
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


# Tabix linear-index granularity, hardcoded in htslib.
TBI_LINEAR_BIN_BP = 1 << 14  # 16384


@dataclass(frozen=True)
class Partition:
    partition_id: str
    chrom: str
    start: int           # 1-based inclusive
    end: int             # 1-based inclusive
    estimated_bytes: int


def _file_offset(voff: int) -> int:
    """Strip the 16-bit within-block offset; return the BGZF block file offset."""
    return voff >> 16


# Byte estimator: file_offset(voff_end) - file_offset(voff_start), i.e. the
# compressed BGZF byte distance between two records. The voff packs the
# 48-bit compressed file offset in the high bits and a 16-bit uncompressed
# within-block offset in the low bits — at biobank scale per-bin spans are
# multi-MB to multi-GB so the file_offset diff is the only signal that
# scales sensibly with the user's `--target-bytes-per-partition`. Using
# the full voff as the byte estimator would over-count by up to 65536x
# (every within-block tick contributes one unit) and produces ~one
# partition per linear-index bin on real biobank inputs.
#
# Single-block fixtures: file_offset diffs are zero across all bins, so the
# algorithm emits one partition for the whole contig. That's the right
# answer for a file that fits in one bgzf block — there's nothing to scatter.


def parse_tbi(tbi_path: Path) -> dict[str, list[int]]:
    """Parse a .tbi v1 index file. Returns {contig_name: [linear_index_voffs]}."""
    with gzip.open(tbi_path, "rb") as f:
        data = f.read()

    pos = 0

    def read(fmt: str):
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, data, pos)
        pos += size
        return vals

    magic = data[pos:pos + 4]
    pos += 4
    if magic != b"TBI\x01":
        raise ValueError(f"{tbi_path}: bad magic {magic!r}; not a tabix v1 index")

    (n_ref, _preset, _sc, _bc, _ec, _meta, _skip, l_nm) = read("<8i")
    names_blob = data[pos:pos + l_nm]
    pos += l_nm
    contigs = [n.decode("ascii") for n in names_blob.split(b"\0") if n]
    if len(contigs) != n_ref:
        raise ValueError(
            f"{tbi_path}: header declares {n_ref} contigs but names list has {len(contigs)}"
        )

    result: dict[str, list[int]] = {}
    for contig in contigs:
        (n_bin,) = read("<i")
        for _ in range(n_bin):
            (_bin_id, n_chunk) = read("<Ii")
            pos += 16 * n_chunk  # skip cnk_beg + cnk_end pairs
        (n_intv,) = read("<i")
        ioff = list(read(f"<{n_intv}Q")) if n_intv else []
        result[contig] = ioff

    return result


def partition_chrom(
    chrom: str,
    linear_index: list[int],
    target_bytes: int,
    max_bytes: int,
    starting_partition_idx: int = 1,
) -> tuple[list[Partition], int]:
    """Emit byte-balanced partitions for one contig from its linear index entries.

    Each entry i covers bp range [i*16384+1, (i+1)*16384] (1-based, inclusive).
    Its value is the virtual offset of the first record whose start lies in
    that bin (or, if the bin is empty, the voff of the next non-empty bin).
    The byte span of [bin_a, bin_b] = file_offset(bin_{b+1}) - file_offset(bin_a).

    Returns (partitions, next_partition_idx).
    """
    n = len(linear_index)
    if n == 0:
        return [], starting_partition_idx

    # Precompute compressed file offsets (high 48 bits of each voff).
    foffs = [_file_offset(v) for v in linear_index]

    partitions: list[Partition] = []
    idx = starting_partition_idx
    start_bin = 0

    while start_bin < n:
        # Empty bins have voff_i == voff_{i+1} (tabix propagates the next
        # non-empty bin's voff backward into empty trailing bins). Compare
        # full voffs here, not file offsets — multiple non-empty bins can
        # share a file offset if their records all live in the same bgzf
        # block, and those are NOT empty bins.
        while start_bin < n - 1 and linear_index[start_bin] == linear_index[start_bin + 1]:
            start_bin += 1
        # NOTE: must use `>= n`, not `>= n - 1`. The last bin (index n-1)
        # can hold real records (its bp range is [(n-1)*16384+1, n*16384],
        # whose lower portion lies within the chromosome). If a previous
        # partition's end_bin landed at n-2 (because byte_span crossed
        # target there), start_bin advances to n-1 and we MUST still
        # process it — otherwise records with POS in bin n-1 vanish.
        if start_bin >= n:
            break

        start_foff = foffs[start_bin]
        end_bin = start_bin
        byte_span = 0

        # Walk forward until we cross target_bytes (or run out of bins).
        # Byte span is the compressed-file distance between bins.
        while end_bin < n - 1:
            byte_span = foffs[end_bin + 1] - start_foff
            if byte_span >= target_bytes:
                break
            end_bin += 1
        else:
            byte_span = foffs[n - 1] - start_foff
            end_bin = n - 1

        # Cap subdivision at 1 bin per sub-partition: we can't usefully
        # subdivide more finely than the tabix index resolution, and a single
        # huge byte_span across just one bin is almost always an artifact of
        # a BGZF block boundary (the voff jump represents file structure,
        # not actual record density). Subdividing through it would create
        # bogus 1-bp partitions.
        n_bins_in_span = end_bin - start_bin + 1
        if byte_span > max_bytes and n_bins_in_span > 1:
            n_sub = min(n_bins_in_span, math.ceil(byte_span / max_bytes))
            total_bp = n_bins_in_span * TBI_LINEAR_BIN_BP
            bp_per_sub = total_bp // n_sub
            for k in range(n_sub):
                sub_start_bp = start_bin * TBI_LINEAR_BIN_BP + k * bp_per_sub + 1
                if k < n_sub - 1:
                    sub_end_bp = start_bin * TBI_LINEAR_BIN_BP + (k + 1) * bp_per_sub
                else:
                    sub_end_bp = (end_bin + 1) * TBI_LINEAR_BIN_BP
                partitions.append(Partition(
                    partition_id=f"{chrom}_p{idx:04d}",
                    chrom=chrom,
                    start=sub_start_bp,
                    end=sub_end_bp,
                    estimated_bytes=byte_span // n_sub,
                ))
                idx += 1
        else:
            partitions.append(Partition(
                partition_id=f"{chrom}_p{idx:04d}",
                chrom=chrom,
                start=start_bin * TBI_LINEAR_BIN_BP + 1,
                end=(end_bin + 1) * TBI_LINEAR_BIN_BP,
                estimated_bytes=byte_span,
            ))
            idx += 1

        start_bin = end_bin + 1

    return partitions, idx


def generate_partitions(
    tbi_path: Path,
    target_bytes: int,
    max_bytes: int,
    chromosomes: list[str] | None = None,
) -> list[Partition]:
    """Top-level entry: parse .tbi, emit partitions for the requested contigs."""
    linear_indexes = parse_tbi(tbi_path)
    available = list(linear_indexes.keys())

    if chromosomes is not None:
        missing = [c for c in chromosomes if c not in linear_indexes]
        if missing:
            raise ValueError(
                f"chromosomes not present in {tbi_path}: {missing!r}; "
                f"available: {available!r}"
            )
        targets = chromosomes
    else:
        targets = available

    all_partitions: list[Partition] = []
    next_idx = 1
    for chrom in targets:
        chrom_partitions, next_idx = partition_chrom(
            chrom=chrom,
            linear_index=linear_indexes[chrom],
            target_bytes=target_bytes,
            max_bytes=max_bytes,
            starting_partition_idx=next_idx,
        )
        all_partitions.extend(chrom_partitions)

    # Largest-first ordering: Cromwell schedules in submission order, so put
    # the long-runners up front.
    all_partitions.sort(key=lambda p: -p.estimated_bytes)
    return all_partitions


def write_outputs(partitions: list[Partition], manifest: Path, regions: Path, region_ids: Path) -> None:
    with manifest.open("w") as f:
        f.write("partition_id\tchrom\tstart\tend\testimated_bytes\n")
        for p in partitions:
            f.write(f"{p.partition_id}\t{p.chrom}\t{p.start}\t{p.end}\t{p.estimated_bytes}\n")
    with regions.open("w") as f:
        for p in partitions:
            f.write(f"{p.chrom}:{p.start}-{p.end}\n")
    with region_ids.open("w") as f:
        for p in partitions:
            f.write(f"{p.partition_id}\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tbi-path", required=True, type=Path,
                   help="local path to a .tbi index (Cromwell will localize this)")
    p.add_argument("--chromosomes", nargs="+", default=None,
                   help="restrict to these contigs (default: all in the index)")
    p.add_argument("--target-bytes-per-partition", type=int, default=10 * 1024**3,
                   help="byte target per partition (default: 10 GB)")
    p.add_argument("--max-bytes-per-partition", type=int, default=30 * 1024**3,
                   help="hard cap; partitions larger than this get subdivided BP-uniformly (default: 30 GB)")
    p.add_argument("--out-manifest", required=True, type=Path)
    p.add_argument("--out-regions", required=True, type=Path)
    p.add_argument("--out-region-ids", required=True, type=Path)
    args = p.parse_args()

    if args.target_bytes_per_partition <= 0:
        raise SystemExit("--target-bytes-per-partition must be positive")
    if args.max_bytes_per_partition < args.target_bytes_per_partition:
        raise SystemExit("--max-bytes-per-partition must be >= --target-bytes-per-partition")

    partitions = generate_partitions(
        tbi_path=args.tbi_path,
        target_bytes=args.target_bytes_per_partition,
        max_bytes=args.max_bytes_per_partition,
        chromosomes=args.chromosomes,
    )

    print(f"Emitted {len(partitions)} partitions", file=sys.stderr)
    if partitions:
        sizes = sorted((p.estimated_bytes for p in partitions), reverse=True)
        print(f"  largest:  {sizes[0]:>15,} bytes", file=sys.stderr)
        print(f"  median:   {sizes[len(sizes) // 2]:>15,} bytes", file=sys.stderr)
        print(f"  smallest: {sizes[-1]:>15,} bytes", file=sys.stderr)

    write_outputs(partitions, args.out_manifest, args.out_regions, args.out_region_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
