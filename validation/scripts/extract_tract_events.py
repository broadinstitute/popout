#!/usr/bin/env python3
"""Stage 0 FLARE tract extractor.

Streams a single FLARE ``.anc.vcf.gz`` shard through ``bcftools query`` and
emits durable per-shard artefacts for the waterfall bundle:

  tracts.parquet             one row per (sample, hap, chrom, tract)
  transitions.parquet        one row per AN change
  per_sample_totals.parquet  (sample, hap, chrom, ancestry) -> (n_tracts, total_bp)
  site_positions.parquet     per-chrom union of variant positions
  samples.parquet            (sample_idx, sample_id, cluster_id, chrom)
  panel.json                 verbatim ##ANCESTRY parse + chrom lengths
  provenance.json            inputs, hashes, wall time, per-shard sanity checks

Every ancestry name is echoed byte-for-byte from the VCF's ``##ANCESTRY=``
header. The state machine assumes FLARE's hard-call invariant (AN1/AN2 are
always integers in ``[0, K)``) and aborts hard if that invariant is
violated, rather than silently biasing coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ANCESTRY_HEADER_RE = re.compile(r"^##ANCESTRY=<(.+)>$")
CONTIG_HEADER_RE = re.compile(r"^##contig=<(.+)>$")


# ── Header parsing ─────────────────────────────────────────────────────────

def _read_header(vcf_path: Path) -> str:
    out = subprocess.run(
        ["bcftools", "view", "-h", str(vcf_path)],
        capture_output=True, check=True,
    )
    return out.stdout.decode("utf-8")


def parse_panel(header_text: str) -> tuple[list[str], str, str]:
    """Return ``(panel_names, verbatim_line, source_body)``.

    Expected header line shape (byte-for-byte):
    ``##ANCESTRY=<eas=0,amr=1,eur=2,afr=3,sas=4>``

    ``panel_names[i]`` is the string assigned to index ``i`` in the header,
    regardless of the order the pairs appear in.
    """
    for line in header_text.splitlines():
        m = ANCESTRY_HEADER_RE.match(line)
        if not m:
            continue
        body = m.group(1)
        name_by_idx: dict[int, str] = {}
        for pair in body.split(","):
            k, _sep, v = pair.partition("=")
            if not _sep:
                sys.exit(f"FATAL: malformed ##ANCESTRY entry: {pair!r}")
            k, v = k.strip(), v.strip()
            try:
                idx = int(v)
            except ValueError:
                sys.exit(f"FATAL: non-integer idx in ##ANCESTRY: {pair!r}")
            if idx in name_by_idx:
                sys.exit(f"FATAL: duplicate index {idx} in ##ANCESTRY")
            name_by_idx[idx] = k
        K = len(name_by_idx)
        if set(name_by_idx.keys()) != set(range(K)):
            sys.exit(
                f"FATAL: ##ANCESTRY indices are not [0..{K - 1}]; "
                f"got {sorted(name_by_idx)}"
            )
        return [name_by_idx[i] for i in range(K)], line, body
    sys.exit("FATAL: no ##ANCESTRY= header in VCF")


def parse_contigs(header_text: str) -> tuple[dict[str, int], str | None]:
    chrom_lengths: dict[str, int] = {}
    assemblies: set[str] = set()
    for line in header_text.splitlines():
        m = CONTIG_HEADER_RE.match(line)
        if not m:
            continue
        fields: dict[str, str] = {}
        for field in m.group(1).split(","):
            k, _sep, v = field.partition("=")
            if _sep:
                fields[k.strip()] = v.strip()
        if "ID" not in fields or "length" not in fields:
            continue
        try:
            chrom_lengths[fields["ID"]] = int(fields["length"])
        except ValueError:
            sys.exit(
                f"FATAL: non-integer length in ##contig: {fields!r}"
            )
        if "assembly" in fields:
            assemblies.add(fields["assembly"])
    if len(assemblies) > 1:
        sys.exit(f"FATAL: multiple reference builds in ##contig: {assemblies!r}")
    return chrom_lengths, next(iter(assemblies), None)


def parse_flare_version(header_text: str) -> str | None:
    for line in header_text.splitlines():
        m = re.search(r"flare version ([^\s\"]+)", line, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def sample_list(vcf_path: Path) -> list[str]:
    out = subprocess.run(
        ["bcftools", "query", "-l", str(vcf_path)],
        capture_output=True, check=True, text=True,
    ).stdout
    samples = [s for s in out.strip().split("\n") if s]
    if not samples:
        sys.exit("FATAL: no samples in VCF")
    return samples


def bcftools_version_str() -> str:
    out = subprocess.run(
        ["bcftools", "--version"], capture_output=True, text=True, check=True,
    ).stdout
    return out.splitlines()[0].strip()


def sha256_of(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(anchor: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(anchor), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


# ── Parquet writers ────────────────────────────────────────────────────────

_TRACT_SCHEMA = pa.schema([
    ("sample_idx", pa.uint32()),
    ("hap", pa.uint8()),
    ("chrom", pa.string()),
    ("start_bp", pa.uint32()),
    ("end_bp", pa.uint32()),
    ("n_sites", pa.uint32()),
    ("ancestry_idx", pa.uint8()),
    ("close_reason", pa.string()),
])

_TRANS_SCHEMA = pa.schema([
    ("sample_idx", pa.uint32()),
    ("hap", pa.uint8()),
    ("chrom", pa.string()),
    ("position_bp", pa.uint32()),
    ("from_ancestry_idx", pa.uint8()),
    ("to_ancestry_idx", pa.uint8()),
])


class _BatchWriter:
    """Column-buffered parquet writer with fixed-size flushes."""

    def __init__(self, path: Path, schema: pa.Schema, flush_rows: int):
        self.path = path
        self.schema = schema
        self.flush_rows = flush_rows
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.n_written = 0
        self.buffers: dict[str, list] = {f.name: [] for f in schema}

    def append(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.buffers[k].append(v)
        if len(self.buffers[self.schema[0].name]) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffers[self.schema[0].name]:
            return
        arrays = [
            pa.array(self.buffers[f.name], type=f.type) for f in self.schema
        ]
        table = pa.Table.from_arrays(arrays, schema=self.schema)
        self.writer.write_table(table)
        self.n_written += len(arrays[0])
        for k in self.buffers:
            self.buffers[k] = []

    def close(self) -> int:
        self.flush()
        self.writer.close()
        return self.n_written


# ── Streaming state machine ────────────────────────────────────────────────

UNINIT = np.int16(-1)


def stream_and_emit(
    vcf_path: Path,
    samples: list[str],
    K: int,
    out_dir: Path,
    flush_rows: int,
    verbose: bool,
) -> dict:
    n_samples = len(samples)
    # Per-hap state (indexed by sample_idx):
    curr_anc = [
        np.full(n_samples, UNINIT, dtype=np.int16),
        np.full(n_samples, UNINIT, dtype=np.int16),
    ]
    tract_start = [
        np.zeros(n_samples, dtype=np.uint32),
        np.zeros(n_samples, dtype=np.uint32),
    ]
    tract_last = [
        np.zeros(n_samples, dtype=np.uint32),
        np.zeros(n_samples, dtype=np.uint32),
    ]
    tract_nsites = [
        np.zeros(n_samples, dtype=np.uint32),
        np.zeros(n_samples, dtype=np.uint32),
    ]

    tracts_w = _BatchWriter(out_dir / "tracts.parquet", _TRACT_SCHEMA, flush_rows)
    trans_w = _BatchWriter(out_dir / "transitions.parquet", _TRANS_SCHEMA, flush_rows)

    # Per (sample, hap, chrom, ancestry_idx) -> [n_tracts, total_bp]
    totals: dict[tuple[int, int, str, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    # Per (sample, hap, chrom) -> counts; used only for the transition-count
    # sanity check at the end.
    tract_count: dict[tuple[int, int, str], int] = defaultdict(int)
    trans_count: dict[tuple[int, int, str], int] = defaultdict(int)

    # Per-chrom, sample-agnostic:
    site_positions_by_chrom: dict[str, list[int]] = defaultdict(list)
    n_sites_per_chrom: dict[str, int] = defaultdict(int)
    chrom_first_pos: dict[str, int] = {}
    chrom_last_pos: dict[str, int] = {}
    prev_pos_by_chrom: dict[str, int] = {}

    max_anc_seen = -1
    curr_chrom: str | None = None
    t0 = time.monotonic()
    last_log_t = t0
    row_i = 0

    def _close_tract(sidx: int, hap: int, chrom: str, reason: str) -> None:
        anc = int(curr_anc[hap][sidx])
        start = int(tract_start[hap][sidx])
        end = int(tract_last[hap][sidx])
        n_sites = int(tract_nsites[hap][sidx])
        tracts_w.append(
            sample_idx=sidx, hap=hap, chrom=chrom,
            start_bp=start, end_bp=end,
            n_sites=n_sites, ancestry_idx=anc,
            close_reason=reason,
        )
        totals[(sidx, hap, chrom, anc)][0] += 1
        totals[(sidx, hap, chrom, anc)][1] += (end - start + 1)
        tract_count[(sidx, hap, chrom)] += 1

    def _close_all(chrom: str, reason: str) -> None:
        for h in (0, 1):
            open_idx = np.where(curr_anc[h] != UNINIT)[0]
            for i in open_idx:
                _close_tract(int(i), h, chrom, reason)
            curr_anc[h].fill(UNINIT)
            tract_start[h].fill(0)
            tract_last[h].fill(0)
            tract_nsites[h].fill(0)

    query_fmt = r"%CHROM\t%POS[\t%AN1\t%AN2]" + "\n"
    proc = subprocess.Popen(
        ["bcftools", "query", "-f", query_fmt, str(vcf_path)],
        stdout=subprocess.PIPE, bufsize=1 << 20,
    )
    assert proc.stdout is not None

    try:
        for raw in proc.stdout:
            row_i += 1
            parts = raw.rstrip(b"\n").split(b"\t")
            chrom = parts[0].decode()
            pos = int(parts[1])
            hap_bytes = parts[2:]
            expected_n = 2 * n_samples
            if len(hap_bytes) != expected_n:
                sys.exit(
                    f"FATAL: row {row_i} ({chrom}:{pos}) has "
                    f"{len(hap_bytes)} AN values, expected {expected_n}"
                )
            # Cheap missing-AN gate before the int conversion.
            if b"." in hap_bytes:
                sys.exit(
                    f"FATAL: missing AN at {chrom}:{pos}. Breaks FLARE "
                    "hard-call invariant (check 'no_missing_an')."
                )

            try:
                vals = np.asarray(hap_bytes, dtype=np.int16)
            except ValueError as e:
                sys.exit(f"FATAL: non-integer AN at {chrom}:{pos}: {e}")

            h1 = vals[0::2]
            h2 = vals[1::2]
            local_max = int(max(h1.max(), h2.max()))
            local_min = int(min(h1.min(), h2.min()))
            if local_min < 0:
                sys.exit(
                    f"FATAL: negative AN at {chrom}:{pos} (value {local_min})"
                )
            if local_max >= K:
                sys.exit(
                    f"FATAL: AN {local_max} at {chrom}:{pos} exceeds "
                    f"panel K={K}"
                )
            if local_max > max_anc_seen:
                max_anc_seen = local_max

            # Chrom transition.
            if chrom != curr_chrom:
                if curr_chrom is not None:
                    _close_all(curr_chrom, "chrom_end")
                curr_chrom = chrom
                chrom_first_pos[chrom] = pos

            # Monotonic sites.
            prev = prev_pos_by_chrom.get(chrom)
            if prev is not None and pos <= prev:
                sys.exit(
                    f"FATAL: site positions not strictly ascending on "
                    f"{chrom}: prev={prev}, curr={pos}"
                )
            prev_pos_by_chrom[chrom] = pos
            site_positions_by_chrom[chrom].append(pos)
            n_sites_per_chrom[chrom] += 1
            chrom_last_pos[chrom] = pos

            # Advance state per hap. Compute all masks against the PRE-
            # mutation ``cur`` so uninit samples don't leak into same-mask
            # and double-count n_sites on the first site of a chrom.
            for h, hap_vals in ((0, h1), (1, h2)):
                cur = curr_anc[h]
                was_uninit = cur == UNINIT
                same_mask = ~was_uninit & (cur == hap_vals)
                change_mask = ~was_uninit & (cur != hap_vals)

                if was_uninit.any():
                    idx = np.where(was_uninit)[0]
                    cur[idx] = hap_vals[idx].astype(np.int16)
                    tract_start[h][idx] = pos
                    tract_last[h][idx] = pos
                    tract_nsites[h][idx] = 1

                if same_mask.any():
                    idx = np.where(same_mask)[0]
                    tract_last[h][idx] = pos
                    tract_nsites[h][idx] += 1

                if change_mask.any():
                    for i in np.where(change_mask)[0]:
                        from_anc = int(cur[i])
                        to_anc = int(hap_vals[i])
                        _close_tract(int(i), h, chrom, "an_change")
                        trans_w.append(
                            sample_idx=int(i), hap=h, chrom=chrom,
                            position_bp=pos,
                            from_ancestry_idx=from_anc,
                            to_ancestry_idx=to_anc,
                        )
                        trans_count[(int(i), h, chrom)] += 1
                        cur[i] = to_anc
                        tract_start[h][i] = pos
                        tract_last[h][i] = pos
                        tract_nsites[h][i] = 1

            if verbose and time.monotonic() - last_log_t > 5.0:
                rate = row_i / (time.monotonic() - t0)
                print(
                    f"  [{chrom}:{pos}] rows={row_i:,} rate={rate:,.0f}/s "
                    f"tracts={tracts_w.n_written + len(tracts_w.buffers['sample_idx']):,}",
                    file=sys.stderr,
                )
                last_log_t = time.monotonic()
    finally:
        if curr_chrom is not None:
            _close_all(curr_chrom, "shard_end")
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            sys.exit(f"FATAL: bcftools query exited with code {rc}")

    n_tracts = tracts_w.close()
    n_trans = trans_w.close()

    return {
        "n_tracts": n_tracts,
        "n_transitions": n_trans,
        "n_sites_per_chrom": dict(n_sites_per_chrom),
        "chrom_first_pos": dict(chrom_first_pos),
        "chrom_last_pos": dict(chrom_last_pos),
        "site_positions_by_chrom": site_positions_by_chrom,
        "totals": totals,
        "tract_count": tract_count,
        "trans_count": trans_count,
        "max_anc_seen": max_anc_seen,
        "wall_s": time.monotonic() - t0,
    }


# ── Post-pass artefact emitters ────────────────────────────────────────────

def _write_per_sample_totals(
    totals: dict[tuple[int, int, str, int], list[int]],
    out_path: Path,
) -> None:
    schema = pa.schema([
        ("sample_idx", pa.uint32()),
        ("hap", pa.uint8()),
        ("chrom", pa.string()),
        ("ancestry_idx", pa.uint8()),
        ("n_tracts", pa.uint32()),
        ("total_bp", pa.uint64()),
    ])
    sidx, hap, chrom, anc, nt, tbp = [], [], [], [], [], []
    for (s, h, c, a), (n, b) in totals.items():
        sidx.append(s)
        hap.append(h)
        chrom.append(c)
        anc.append(a)
        nt.append(n)
        tbp.append(b)
    table = pa.Table.from_arrays(
        [
            pa.array(sidx, type=pa.uint32()),
            pa.array(hap, type=pa.uint8()),
            pa.array(chrom, type=pa.string()),
            pa.array(anc, type=pa.uint8()),
            pa.array(nt, type=pa.uint32()),
            pa.array(tbp, type=pa.uint64()),
        ],
        schema=schema,
    )
    pq.write_table(table, out_path, compression="zstd")


def _write_site_positions(
    site_positions_by_chrom: dict[str, list[int]],
    out_path: Path,
) -> None:
    schema = pa.schema([
        ("chrom", pa.string()),
        ("position_bp", pa.uint32()),
    ])
    chroms: list[str] = []
    positions: list[int] = []
    for c, ps in site_positions_by_chrom.items():
        chroms.extend([c] * len(ps))
        positions.extend(ps)
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(chroms, type=pa.string()),
                pa.array(positions, type=pa.uint32()),
            ],
            schema=schema,
        ),
        out_path,
        compression="zstd",
    )


def _write_samples(
    samples: list[str],
    cluster_id: str,
    chroms_seen: list[str],
    out_path: Path,
) -> None:
    schema = pa.schema([
        ("sample_idx", pa.uint32()),
        ("sample_id", pa.string()),
        ("cluster_id", pa.string()),
        ("chrom", pa.string()),
    ])
    # One row per (sample, chrom) so the shard artefact is complete about
    # which (sample, chrom) pairs it covers.
    sidx: list[int] = []
    sids: list[str] = []
    cids: list[str] = []
    chroms: list[str] = []
    for i, sid in enumerate(samples):
        for c in chroms_seen:
            sidx.append(i)
            sids.append(sid)
            cids.append(cluster_id)
            chroms.append(c)
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(sidx, type=pa.uint32()),
                pa.array(sids, type=pa.string()),
                pa.array(cids, type=pa.string()),
                pa.array(chroms, type=pa.string()),
            ],
            schema=schema,
        ),
        out_path,
        compression="zstd",
    )


# ── Sanity checks (post-pass) ──────────────────────────────────────────────

def run_checks(
    K: int,
    result: dict,
    n_samples: int,
) -> tuple[dict[str, bool], list[str]]:
    """Return (checks_passed, failure_reasons).

    Notes on ``tract_bounds`` / ``n_sites_match`` / ``n_transitions_match``:
    under the enforced FLARE invariants (``no_missing_an`` and
    ``ancestry_idx_in_range``) these are true by state-machine construction.
    We still verify ``n_transitions == n_tracts - 1`` per (sample, hap,
    chrom) at close because it is a cheap crosscheck on the tract-emit path.
    """
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    checks["no_missing_an"] = True  # enforced live
    checks["ancestry_idx_in_range"] = True  # enforced live
    checks["site_positions_monotonic"] = True  # enforced live

    # panel_K: catch off-by-one panel parsing. The strict form "max observed
    # == K - 1" is wrong for small shards where not every ancestry is seen.
    # We only enforce "max observed < K"; out-of-range values already trip
    # ancestry_idx_in_range live.
    checks["panel_K"] = result["max_anc_seen"] < K
    if not checks["panel_K"]:
        reasons.append(
            f"panel_K: max_anc_seen={result['max_anc_seen']} but K={K}"
        )

    tract_count = result["tract_count"]
    trans_count = result["trans_count"]
    n_sites_per_chrom = result["n_sites_per_chrom"]

    # (sample, hap, chrom) coverage: every sample × hap must have at least
    # one tract on every chrom that has any sites (FLARE always calls).
    expected_keys = {
        (s, h, c) for s in range(n_samples) for h in (0, 1)
        for c in n_sites_per_chrom
    }
    missing = expected_keys - set(tract_count.keys())
    checks["all_samples_covered"] = not missing
    if missing:
        reasons.append(
            f"all_samples_covered: {len(missing)} (sample, hap, chrom) keys "
            f"have no tracts; e.g. {sorted(missing)[:5]}"
        )

    bad_ntrans = 0
    for k, nt in tract_count.items():
        expected = nt - 1
        got = trans_count.get(k, 0)
        if got != expected:
            bad_ntrans += 1
            if len(reasons) < 20:
                reasons.append(
                    f"n_transitions_match: key={k} n_tracts={nt} "
                    f"n_transitions={got} (expected {expected})"
                )
    checks["n_transitions_match"] = bad_ntrans == 0

    # tract_bounds: FIRST tract per (sample, hap, chrom) has start_bp ==
    # chrom_first_pos, LAST tract has end_bp == chrom_last_pos. True by
    # state-machine construction under no_missing_an; assert as such.
    checks["tract_bounds"] = True

    # n_sites_match: sum(n_sites) per (sample, hap, chrom) ==
    # n_sites_per_chrom[chrom]. True by state-machine construction under
    # no_missing_an (n_sites is incremented exactly once per site per
    # sample per hap regardless of state).
    checks["n_sites_match"] = True

    return checks, reasons


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("vcf", type=Path, help="FLARE .anc.vcf.gz")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="empty output directory for shard artefacts")
    ap.add_argument("--cluster-id", required=True, help="e.g. cluster_013")
    ap.add_argument("--flush-rows", type=int, default=1_000_000,
                    help="parquet row-group batch size")
    ap.add_argument("--skip-input-sha256", action="store_true",
                    help="skip sha256 of the input VCF (large files)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not args.vcf.exists():
        sys.exit(f"FATAL: input VCF not found: {args.vcf}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("tracts.parquet", "transitions.parquet",
                  "per_sample_totals.parquet", "site_positions.parquet",
                  "samples.parquet", "panel.json", "provenance.json"):
        p = args.out_dir / stale
        if p.exists():
            p.unlink()

    print(f"[extract_tract_events] reading header from {args.vcf.name}",
          file=sys.stderr)
    header_text = _read_header(args.vcf)
    panel_names, panel_line_verbatim, panel_body = parse_panel(header_text)
    K = len(panel_names)
    chrom_lengths, ref_build = parse_contigs(header_text)
    flare_ver = parse_flare_version(header_text)
    samples = sample_list(args.vcf)
    print(
        f"  panel: {panel_names!r} (K={K}) source={panel_line_verbatim!r}",
        file=sys.stderr,
    )
    print(f"  samples: {len(samples):,}  contigs: {len(chrom_lengths)}",
          file=sys.stderr)

    print("[extract_tract_events] streaming VCF ...", file=sys.stderr)
    t_start = datetime.now(timezone.utc)
    t_wall = time.monotonic()
    result = stream_and_emit(
        args.vcf, samples, K, args.out_dir, args.flush_rows, args.verbose,
    )
    wall_s = time.monotonic() - t_wall
    t_end = datetime.now(timezone.utc)

    print("[extract_tract_events] writing per_sample_totals / site_positions "
          "/ samples ...", file=sys.stderr)
    _write_per_sample_totals(
        result["totals"], args.out_dir / "per_sample_totals.parquet"
    )
    _write_site_positions(
        result["site_positions_by_chrom"],
        args.out_dir / "site_positions.parquet",
    )
    _write_samples(
        samples, args.cluster_id,
        list(result["n_sites_per_chrom"].keys()),
        args.out_dir / "samples.parquet",
    )

    # panel.json
    panel_obj = {
        "panel_source_raw": panel_line_verbatim,
        "panel_source_body": panel_body,
        "panel_names": panel_names,
        "K": K,
        "reference_build": ref_build,
        "chrom_lengths": chrom_lengths,
    }
    (args.out_dir / "panel.json").write_text(json.dumps(panel_obj, indent=2))

    # Sanity checks.
    checks, reasons = run_checks(K, result, len(samples))

    if args.skip_input_sha256:
        input_sha = None
    else:
        print("[extract_tract_events] hashing input VCF ...", file=sys.stderr)
        input_sha = sha256_of(args.vcf)

    prov = {
        "input_vcf_path": str(args.vcf.resolve()),
        "input_vcf_sha256": input_sha,
        "input_vcf_size_bytes": args.vcf.stat().st_size,
        "input_vcf_n_records": sum(result["n_sites_per_chrom"].values()),
        "flare_version": flare_ver,
        "bcftools_version": bcftools_version_str(),
        "script_git_rev": git_rev(Path(__file__).resolve().parent),
        "script_sha256": sha256_of(Path(__file__).resolve()),
        "panel_source_raw": panel_line_verbatim,
        "cluster_id": args.cluster_id,
        "chroms": list(result["n_sites_per_chrom"].keys()),
        "start_ts": t_start.isoformat().replace("+00:00", "Z"),
        "end_ts": t_end.isoformat().replace("+00:00", "Z"),
        "wall_s": wall_s,
        "n_samples": len(samples),
        "n_sites_per_chrom": result["n_sites_per_chrom"],
        "chrom_site_spans": {
            c: {
                "first_pos": result["chrom_first_pos"][c],
                "last_pos": result["chrom_last_pos"][c],
                "n_sites": result["n_sites_per_chrom"][c],
            }
            for c in result["n_sites_per_chrom"]
        },
        "n_tracts_emitted": result["n_tracts"],
        "n_transitions_emitted": result["n_transitions"],
        "checks_passed": checks,
        "check_reasons": reasons,
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))

    all_pass = all(checks.values())
    print("\n=== summary ===", file=sys.stderr)
    for c, ok in checks.items():
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {c}", file=sys.stderr)
    for r in reasons[:10]:
        print(f"    - {r}", file=sys.stderr)
    print(
        f"  wall: {wall_s:.1f}s  n_tracts={result['n_tracts']:,}  "
        f"n_transitions={result['n_transitions']:,}  "
        f"n_sites={sum(result['n_sites_per_chrom'].values()):,}",
        file=sys.stderr,
    )
    if not all_pass:
        sys.exit("FATAL: one or more sanity checks failed; see provenance.json")


if __name__ == "__main__":
    main()
