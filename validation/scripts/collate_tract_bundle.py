#!/usr/bin/env python3
"""Stage 1 collator: concatenate extract_tract_events shards into a cohort
bundle.

Inputs: a list of per-shard tarballs produced by
``extract_tract_events.py``. Each tarball contains ``tracts.parquet``,
``transitions.parquet``, ``per_sample_totals.parquet``,
``site_positions.parquet``, ``samples.parquet``, ``panel.json``, and
``provenance.json``.

Output layout under ``--out-dir``:

  tracts/chrom=<c>/part_<shard_idx>.parquet
  transitions/chrom=<c>/part_<shard_idx>.parquet
  site_positions/chrom=<c>/part_<shard_idx>.parquet
  per_sample_totals.parquet     (single file, concat across shards)
  samples.parquet               (single file, concat + dedup on sample_id)
  panel.json                    (canonical copy; must match across shards)
  provenance.jsonl              (one JSON object per shard)
  cohort_manifest.json          (shard count, sample count, tract count)

Panel.json byte-identity across shards is a hard invariant: two shards that
disagree on panel columns cannot be reconciled into one honest bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


_HIVE_PARTITIONED = ("tracts", "transitions", "site_positions")
_SINGLE_FILES = ("per_sample_totals.parquet",)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _untar(tarball: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tf:
        # Refuse absolute paths / .. path segments to keep the extractor
        # boundary honest.
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                sys.exit(f"FATAL: unsafe tar member {m.name!r} in {tarball}")
        tf.extractall(dst, filter="data")


def _partition_by_chrom(
    shard_dir: Path,
    stem: str,
    out_root: Path,
    shard_idx: int,
) -> dict[str, int]:
    """Split shard's ``<stem>.parquet`` into per-chrom files under
    ``<out_root>/<stem>/chrom=<c>/part_<i>.parquet``. Return per-chrom row
    counts."""
    src = shard_dir / f"{stem}.parquet"
    if not src.exists():
        sys.exit(f"FATAL: shard {shard_dir} missing {src.name}")
    table = pq.read_table(src)
    chroms = table["chrom"].to_pylist()
    per_chrom: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        per_chrom[c].append(i)
    counts: dict[str, int] = {}
    for c, idxs in per_chrom.items():
        sub = table.take(pa.array(idxs, type=pa.int64()))
        dst_dir = out_root / stem / f"chrom={c}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            sub, dst_dir / f"part_{shard_idx:05d}.parquet",
            compression="zstd",
        )
        counts[c] = sub.num_rows
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-tarball", type=Path, action="append", required=True,
                    dest="shard_tarballs",
                    help="per-shard tarball from extract_tract_events; repeat")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        sys.exit(f"FATAL: --out-dir {args.out_dir} is not empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_totals_frames: list[pa.Table] = []
    samples_frames: list[pa.Table] = []
    per_shard_row_counts: dict[str, dict[str, int]] = defaultdict(dict)
    canonical_panel: str | None = None
    provenance_lines: list[str] = []
    all_chroms: set[str] = set()
    all_sample_ids: set[str] = set()

    for idx, tarball in enumerate(args.shard_tarballs):
        if not tarball.exists():
            sys.exit(f"FATAL: shard tarball not found: {tarball}")
        print(f"[collate] shard {idx}: {tarball.name}", file=sys.stderr)
        with tempfile.TemporaryDirectory(prefix="collate_") as tmp:
            tmp_root = Path(tmp)
            _untar(tarball, tmp_root)

            # Shard may have unpacked into an intermediate directory
            # (e.g. tar was created with -C). Find the one containing panel.json.
            candidates = list(tmp_root.rglob("panel.json"))
            if not candidates:
                sys.exit(f"FATAL: no panel.json in {tarball}")
            if len(candidates) > 1:
                sys.exit(
                    f"FATAL: multiple panel.json in {tarball}: {candidates}"
                )
            shard_dir = candidates[0].parent

            # Panel byte-identity.
            panel_text = (shard_dir / "panel.json").read_text()
            if canonical_panel is None:
                canonical_panel = panel_text
                (args.out_dir / "panel.json").write_text(panel_text)
            elif panel_text != canonical_panel:
                sys.exit(
                    f"FATAL: shard {tarball} panel.json differs from "
                    "canonical. A cohort cannot span two panel orderings."
                )

            # Partition tracts / transitions / site_positions by chrom.
            for stem in _HIVE_PARTITIONED:
                counts = _partition_by_chrom(
                    shard_dir, stem, args.out_dir, idx
                )
                per_shard_row_counts[stem][tarball.name] = sum(counts.values())
                all_chroms.update(counts.keys())

            # Concat targets: per_sample_totals + samples.
            pst = pq.read_table(shard_dir / "per_sample_totals.parquet")
            per_sample_totals_frames.append(pst)
            samp = pq.read_table(shard_dir / "samples.parquet")
            samples_frames.append(samp)
            all_sample_ids.update(samp["sample_id"].to_pylist())

            # Provenance.
            prov = json.loads((shard_dir / "provenance.json").read_text())
            prov["_shard_tarball_sha256"] = _sha256(tarball)
            prov["_shard_index"] = idx
            provenance_lines.append(json.dumps(prov))

    # Write concatenated single-file artefacts.
    pst_all = pa.concat_tables(per_sample_totals_frames, promote_options="default")
    pq.write_table(
        pst_all, args.out_dir / "per_sample_totals.parquet",
        compression="zstd",
    )
    samples_all = pa.concat_tables(samples_frames, promote_options="default")
    pq.write_table(
        samples_all, args.out_dir / "samples.parquet",
        compression="zstd",
    )

    # Provenance jsonl + manifest.
    (args.out_dir / "provenance.jsonl").write_text(
        "\n".join(provenance_lines) + "\n"
    )
    per_stem_totals = {
        stem: sum(per_shard_row_counts[stem].values())
        for stem in _HIVE_PARTITIONED
    }
    manifest = {
        "run_name": args.run_name,
        "n_shards": len(args.shard_tarballs),
        "n_samples": len(all_sample_ids),
        "chroms": sorted(all_chroms),
        "row_counts": per_stem_totals,
        "panel": json.loads(canonical_panel or "{}"),
    }
    (args.out_dir / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"\n[collate] wrote cohort bundle to {args.out_dir}", file=sys.stderr)
    print(
        f"  shards={len(args.shard_tarballs)} samples={len(all_sample_ids):,} "
        f"tracts={per_stem_totals['tracts']:,} "
        f"transitions={per_stem_totals['transitions']:,}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
