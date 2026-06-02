#!/usr/bin/env python3
"""popout DX — local-mode View A / View B / View C inputs.

Lifts the per-hap crosstool walker, boundary-match algorithm, and
coarse-grid sweep from
``popout/diagnostics/scripts/compare_tracts.py:135-237`` and emits per-hap
TSVs sized for cohort collation. No rendering — the laptop renderer
consumes these directly.

Outputs (all in ``--out-dir/``):

  ``bp_confusion_segments.tsv.gz`` — one row per ``(sample, hap)`` overlap
                                     segment with both tools' RF labels
                                     (View A material)
  ``boundary_localization.tsv``    — one row per FLARE switch (View B)
  ``coarse_grid_summary.tsv``      — one row per ``(sample, hap, resolution_mb)``
                                     with diagonal-fraction + off-diagonal
                                     label pairs (View C)

Vocabulary in ``local_summary.json`` follows
``popout/diagnostics/GLOSSARY.md``:

  bp-agreement                       — fraction of bp where both tools call
                                       the same RF label
  calibration drift                  — disagreements that DO NOT resolve at
                                       any coarser grid
  boundary-localization error        — disagreements where the same label
                                       pair appears within
                                       ``BOUNDARY_MATCH_MAX_BP`` on both
                                       tools (same switch, off by a bit)
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# popout/benchmark is installed in the docker.
from popout.benchmark.common import MISSING_LABEL, TractSet
from popout.viz._loaders import read_labels_json


RF_LABEL_ORDER: tuple[str, ...] = ("afr", "amr", "eas", "eur", "mid", "sas")
RF_INDEX: dict[str, int] = {lbl: i for i, lbl in enumerate(RF_LABEL_ORDER)}

DEFAULT_COARSE_GRIDS_MB: tuple[int, ...] = (1, 2, 5, 10, 20)
BOUNDARY_MATCH_MAX_BP: int = 5_000_000


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"dx_local_views: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── Tract-list extraction ───────────────────────────────────────────────


def flare_tractset_to_tract_lists(
    ts: TractSet,
) -> dict[str, dict[int, list[tuple[str, int, int, int]]]]:
    """Convert a FLARE TractSet (from dx_local_parse_flare) into the
    ``{sample -> {hap -> [(chrom, start_bp, end_bp, anc), ...]}}`` shape
    that the compare_tracts kernels expect.

    Sample/hap come from ``hap_ids`` of the form ``"<sample>:<hap>"``.
    Ancestry codes are the FLARE-native integers from ``ts.label_map``.
    Adjacent same-label sites are run-length encoded to one tract.
    """
    out: dict[str, dict[int, list[tuple[str, int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row_idx, hap_id in enumerate(ts.hap_ids.tolist()):
        sample, hap_str = hap_id.rsplit(":", 1)
        hap = int(hap_str)
        # to_tracts() output is per-hap; filter rows for this hap_idx.
        row = ts.calls[row_idx]
        if len(row) == 0:
            continue
        start = 0
        cur = int(row[0])
        for t in range(1, len(row)):
            v = int(row[t])
            if v != cur:
                if cur != MISSING_LABEL:
                    out[sample][hap].append((
                        ts.chrom,
                        int(ts.site_positions[start]),
                        int(ts.site_positions[t - 1]),
                        cur,
                    ))
                start = t
                cur = v
        if cur != MISSING_LABEL:
            out[sample][hap].append((
                ts.chrom,
                int(ts.site_positions[start]),
                int(ts.site_positions[len(row) - 1]),
                cur,
            ))
    # Sort each hap's tracts by start_bp (already sorted by construction
    # but make it explicit for safety).
    for sample in out:
        for hap in out[sample]:
            out[sample][hap].sort(key=lambda t: t[1])
    return {s: dict(h) for s, h in out.items()}


def stream_popout_tract_lists(
    tracts_path: Path,
    sample_filter: set[str],
    chrom: str,
) -> dict[str, dict[int, list[tuple[str, int, int, int]]]]:
    """Stream popout's whole-cohort tracts.tsv.gz; build per-(sample, hap)
    tract lists for the requested samples on the requested chrom."""
    out: dict[str, dict[int, list[tuple[str, int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    opener = gzip.open if str(tracts_path).endswith(".gz") else open
    with opener(tracts_path, "rt") as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        try:
            ic = header.index("chrom")
            isb = header.index("start_bp")
            ieb = header.index("end_bp")
            isa = header.index("sample")
            ihp = header.index("haplotype")
            ian = header.index("ancestry")
        except ValueError:
            die(f"{tracts_path}: header missing required columns; got {header}")
        max_col = max(ic, isb, ieb, isa, ihp, ian)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max_col:
                continue
            file_chrom = parts[ic]
            if not file_chrom.startswith("chr"):
                file_chrom = "chr" + file_chrom
            if file_chrom != chrom:
                continue
            sam = parts[isa]
            if sam not in sample_filter:
                continue
            hap = int(parts[ihp])
            out[sam][hap].append((
                file_chrom,
                int(parts[isb]),
                int(parts[ieb]),
                int(parts[ian]),
            ))
    for sample in out:
        for hap in out[sample]:
            out[sample][hap].sort(key=lambda t: t[1])
    return {s: dict(h) for s, h in out.items()}


def src_label_lut(labels: dict) -> dict[int, str]:
    """Build ``{src_int_code -> rf_label_name}`` from a labels.json's
    ``popout_to_rf_label`` (works for both popout and FLARE label files)."""
    p2rf = labels.get("popout_to_rf_label")
    if not isinstance(p2rf, dict) or not p2rf:
        die("labels.json missing popout_to_rf_label")
    out: dict[int, str] = {}
    for k, v in p2rf.items():
        out[int(k)] = str(v)
        if v not in RF_INDEX:
            die(f"labels.json popout_to_rf_label[{k}] = {v!r} not in canonical RF set")
    return out


# ── Compare-tracts kernels (lifted verbatim from
#    popout/diagnostics/scripts/compare_tracts.py:135-237) ────────────────


def crosstool_merge_walk(
    flare_tracts: list[tuple[str, int, int, int]],
    popout_tracts: list[tuple[str, int, int, int]],
) -> list[tuple[str, int, int, int, int]]:
    """Lockstep walker; yields ``(chrom, seg_start, seg_end, anc_flare, anc_popout)``
    for every bp segment covered by both tools.
    """
    segments: list[tuple[str, int, int, int, int]] = []
    chroms = sorted(set(t[0] for t in flare_tracts) | set(t[0] for t in popout_tracts))
    for chrom in chroms:
        f = [t for t in flare_tracts if t[0] == chrom]
        p = [t for t in popout_tracts if t[0] == chrom]
        if not f or not p:
            raise RuntimeError(
                f"chrom {chrom} present on only one tool "
                f"(flare={len(f)} popout={len(p)}); refusing to compare"
            )
        i = j = 0
        cur = max(f[0][1], p[0][1])
        while i < len(f) and j < len(p):
            _, _, a_end, a_anc = f[i]
            _, _, b_end, b_anc = p[j]
            seg_end = min(a_end, b_end)
            if cur <= seg_end:
                segments.append((chrom, cur, seg_end, a_anc, b_anc))
            cur = seg_end + 1
            if a_end <= b_end:
                i += 1
            if b_end <= a_end:
                j += 1
    return segments


def hap_switches(tracts: list[tuple[str, int, int, int]]) -> list[tuple[int, int, int]]:
    """``[(switch_bp, anc_before, anc_after)]`` at each ancestry change."""
    out: list[tuple[int, int, int]] = []
    for prev, curr in zip(tracts, tracts[1:]):
        if prev[3] != curr[3]:
            out.append((curr[1], prev[3], curr[3]))
    return out


def match_boundaries(
    flare_h: list[tuple[str, int, int, int]],
    popout_h: list[tuple[str, int, int, int]],
    idx_to_rf_f: dict[int, str],
    idx_to_rf_p: dict[int, str],
    *,
    max_offset_bp: int = BOUNDARY_MATCH_MAX_BP,
) -> list[tuple[int, str, str, int | None, bool]]:
    """For each FLARE switch, find the nearest popout switch with the same
    flanking RF-label pair within ``max_offset_bp``.
    """
    f_sw = hap_switches(flare_h)
    p_sw = hap_switches(popout_h)
    rows: list[tuple[int, str, str, int | None, bool]] = []
    if not p_sw:
        for bp, a_b, a_a in f_sw:
            rows.append((bp, idx_to_rf_f[a_b], idx_to_rf_f[a_a], None, False))
        return rows
    p_bps = np.array([s[0] for s in p_sw], dtype=np.int64)
    for bp, a_b, a_a in f_sw:
        rf_b = idx_to_rf_f[a_b]
        rf_a = idx_to_rf_f[a_a]
        matched: int | None = None
        order = np.argsort(np.abs(p_bps - bp))
        for j in order:
            offset = int(abs(p_bps[j] - bp))
            if offset > max_offset_bp:
                break
            p_bp, p_b, p_a = p_sw[int(j)]
            if (idx_to_rf_p[p_b], idx_to_rf_p[p_a]) == (rf_b, rf_a):
                matched = p_bp
                break
        rows.append((bp, rf_b, rf_a, matched, matched is not None))
    return rows


def dominant_in_window(
    tracts: list[tuple[str, int, int, int]],
    w_start: int,
    w_end: int,
    idx_to_rf: dict[int, str],
) -> str | None:
    """bp-weighted argmax RF label over the (w_start, w_end) intersection."""
    bp_by_rf: dict[str, int] = defaultdict(int)
    for _, s, e, a in tracts:
        s_use = max(s, w_start)
        e_use = min(e, w_end)
        if s_use > e_use:
            continue
        bp_by_rf[idx_to_rf[a]] += e_use - s_use + 1
    if not bp_by_rf:
        return None
    return max(bp_by_rf, key=lambda k: bp_by_rf[k])


# ── Output writers ───────────────────────────────────────────────────────


def write_bp_confusion_segments(
    rows_iter,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("sample\thap\tchrom\tseg_start_bp\tseg_end_bp\tflare_rf_label\tpopout_rf_label\n")
        for row in rows_iter:
            f.write("\t".join(str(x) for x in row) + "\n")


def write_boundary_localization(
    rows_iter,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("sample\thap\tchrom\tflare_switch_pos\tflare_left_label\tflare_right_label\t"
                "nearest_popout_switch_pos\tdistance_bp\tflanking_label_match\n")
        for row in rows_iter:
            f.write("\t".join("" if x is None else str(x) for x in row) + "\n")


def write_coarse_grid_summary(
    rows_iter,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("sample\thap\tchrom\tresolution_mb\tdiagonal_fraction\toff_diagonal_pairs_json\n")
        for row in rows_iter:
            f.write("\t".join(str(x) for x in row) + "\n")


# ── Main ────────────────────────────────────────────────────────────────


def coarse_grid_sweep_hap(
    flare_tracts: list[tuple[str, int, int, int]],
    popout_tracts: list[tuple[str, int, int, int]],
    idx_to_rf_f: dict[int, str],
    idx_to_rf_p: dict[int, str],
    chrom_len_bp: int,
    resolutions_mb: list[int],
) -> list[tuple[int, float, str]]:
    """One row per resolution: (resolution_mb, diagonal_fraction, off_diagonal_pairs_json)."""
    rows: list[tuple[int, float, str]] = []
    for res_mb in resolutions_mb:
        win = res_mb * 1_000_000
        diag = 0
        off = 0
        off_pairs: dict[tuple[str, str], int] = defaultdict(int)
        w = 0
        while w < chrom_len_bp:
            w_end = min(w + win - 1, chrom_len_bp)
            f_lbl = dominant_in_window(flare_tracts, w, w_end, idx_to_rf_f)
            p_lbl = dominant_in_window(popout_tracts, w, w_end, idx_to_rf_p)
            if f_lbl is not None and p_lbl is not None:
                if f_lbl == p_lbl:
                    diag += 1
                else:
                    off += 1
                    off_pairs[(f_lbl, p_lbl)] += 1
            w += win
        total = diag + off
        frac = float(diag / total) if total > 0 else float("nan")
        off_list = [
            {"flare": pair[0], "popout": pair[1], "n_windows": n}
            for pair, n in sorted(off_pairs.items())
        ]
        rows.append((res_mb, frac, json.dumps(off_list)))
    return rows


def chrom_length_for(chrom: str) -> int:
    """Hard-coded GRCh38 autosome + chrX/Y lengths so this script does not
    pull in popout.viz._style.CHROM_LENGTHS_GRCH38 (avoids a matplotlib
    import at script start)."""
    table = {
        "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559,
        "chr4": 190_214_555, "chr5": 181_538_259, "chr6": 170_805_979,
        "chr7": 159_345_973, "chr8": 145_138_636, "chr9": 138_394_717,
        "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
        "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
        "chr16": 90_338_345, "chr17": 83_257_441, "chr18": 80_373_285,
        "chr19": 58_617_616, "chr20": 64_444_167, "chr21": 46_709_983,
        "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
    }
    if chrom not in table:
        die(f"chrom {chrom!r} not in GRCh38 autosome/sex-chrom table")
    return table[chrom]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flare-npz", required=True, type=Path,
                    help="output of dx_local_parse_flare.py")
    ap.add_argument("--flare-labels", required=True, type=Path)
    ap.add_argument("--popout-tracts", required=True, type=Path)
    ap.add_argument("--popout-labels", required=True, type=Path)
    ap.add_argument("--samples-file", required=True, type=Path,
                    help="one sample_id per line (the picker's selection)")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--coarse-grids-mb", type=int, nargs="+",
                    default=list(DEFAULT_COARSE_GRIDS_MB))
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--local-summary-in", type=Path, default=None,
                    help="optional local_summary.json from dx_local_align_metrics "
                         "to update with calibration_drift / boundary_localization "
                         "fractions in place")
    args = ap.parse_args()

    selected = [s for s in args.samples_file.read_text().splitlines() if s.strip()]
    if not selected:
        die(f"--samples-file {args.samples_file} is empty")
    selected_set = set(selected)

    # Load FLARE TractSet → tract lists; popout streaming → tract lists.
    flare_data = np.load(args.flare_npz, allow_pickle=True)
    ts = TractSet(
        tool_name=str(flare_data["tool_name"]),
        chrom=str(flare_data["chrom"]),
        hap_ids=np.asarray(flare_data["hap_ids"], dtype=object),
        site_positions=np.asarray(flare_data["site_positions"], dtype=np.int64),
        calls=np.asarray(flare_data["calls"], dtype=np.uint16),
        label_map={int(k): v for k, v in json.loads(str(flare_data["label_map_json"])).items()},
    )
    if ts.chrom != args.chrom:
        die(f"--chrom {args.chrom!r} != FLARE TractSet chrom {ts.chrom!r}")
    flare_lists = flare_tractset_to_tract_lists(ts)
    popout_lists = stream_popout_tract_lists(args.popout_tracts, selected_set, args.chrom)

    flare_labels = read_labels_json(args.flare_labels)
    popout_labels = read_labels_json(args.popout_labels)
    idx_to_rf_f = src_label_lut(flare_labels)
    idx_to_rf_p = src_label_lut(popout_labels)

    chrom_len = chrom_length_for(args.chrom)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Walk each (sample, hap) once and emit rows for all three outputs.
    seg_rows: list[tuple] = []
    bnd_rows: list[tuple] = []
    grid_rows: list[tuple] = []

    # Boundary tallies for calibration-drift vs boundary-localization split.
    n_flare_switches = 0
    n_boundary_matched = 0

    missing_flare = sorted(s for s in selected_set if s not in flare_lists)
    if missing_flare:
        die(f"{len(missing_flare)} selected sample(s) absent from FLARE TractSet; "
            f"first: {missing_flare[:5]}")
    missing_popout = sorted(s for s in selected_set if s not in popout_lists)
    if missing_popout:
        die(f"{len(missing_popout)} selected sample(s) absent from popout tracts on "
            f"{args.chrom}; first: {missing_popout[:5]}")

    for sample in selected:
        for hap in (0, 1):
            ft = flare_lists.get(sample, {}).get(hap, [])
            pt = popout_lists.get(sample, {}).get(hap, [])
            if not ft or not pt:
                continue

            # View A
            for seg in crosstool_merge_walk(ft, pt):
                _, s_start, s_end, a_f, a_p = seg
                seg_rows.append((
                    sample, hap, args.chrom, s_start, s_end,
                    idx_to_rf_f.get(a_f, str(a_f)),
                    idx_to_rf_p.get(a_p, str(a_p)),
                ))

            # View B
            for bnd in match_boundaries(ft, pt, idx_to_rf_f, idx_to_rf_p):
                bp, rf_b, rf_a, matched_bp, is_matched = bnd
                distance = "" if matched_bp is None else str(abs(matched_bp - bp))
                bnd_rows.append((
                    sample, hap, args.chrom, bp, rf_b, rf_a,
                    matched_bp, distance, is_matched,
                ))
                n_flare_switches += 1
                if is_matched:
                    n_boundary_matched += 1

            # View C
            for res_mb, frac, off_json in coarse_grid_sweep_hap(
                ft, pt, idx_to_rf_f, idx_to_rf_p, chrom_len, args.coarse_grids_mb
            ):
                grid_rows.append((sample, hap, args.chrom, res_mb, frac, off_json))

    write_bp_confusion_segments(seg_rows, args.out_dir / "bp_confusion_segments.tsv.gz")
    write_boundary_localization(bnd_rows, args.out_dir / "boundary_localization.tsv")
    write_coarse_grid_summary(grid_rows, args.out_dir / "coarse_grid_summary.tsv")

    # Roll-ups for the local_summary.json calibration vs boundary split.
    # bp_total / bp_agree from segments → bp_agreement (already computed by
    # dx_local_align_metrics, but recompute here from segments for the
    # boundary-fraction denominator).
    bp_total = 0
    bp_agree = 0
    bp_disagree = 0
    for sample, hap, chrom, s, e, rf_f, rf_p in seg_rows:
        n = max(0, e - s + 1)
        bp_total += n
        if rf_f == rf_p:
            bp_agree += n
        else:
            bp_disagree += n

    # Boundary localization fraction: of FLARE switches, fraction matched on
    # popout within BOUNDARY_MATCH_MAX_BP. This is a switch-count fraction, NOT
    # a bp fraction — the bp-fraction-of-disagreements requires per-segment
    # attribution which the laptop renderer derives from segments + boundaries.
    boundary_match_switch_fraction = (
        float(n_boundary_matched / n_flare_switches) if n_flare_switches > 0 else None
    )
    # Coarse-grid sweep: take the diagonal fraction at the largest requested
    # resolution as a coarse "if we collapse boundaries, how much disagreement
    # remains?" signal — calibration_drift_fraction ≈ 1 - largest-grid-diagonal.
    if grid_rows:
        max_res = max(r[3] for r in grid_rows)
        diag_max = [r[4] for r in grid_rows if r[3] == max_res and not np.isnan(r[4])]
        calibration_drift_fraction = (
            float(1.0 - np.mean(diag_max)) if diag_max else None
        )
    else:
        calibration_drift_fraction = None

    if args.local_summary_in is not None and args.local_summary_in.exists():
        summary = json.loads(args.local_summary_in.read_text())
        summary["bp_agreement_from_segments"] = (
            float(bp_agree / bp_total) if bp_total > 0 else None
        )
        summary["calibration_drift_fraction"] = calibration_drift_fraction
        summary["boundary_localization_match_switch_fraction"] = boundary_match_switch_fraction
        summary["n_flare_switches"] = n_flare_switches
        summary["n_boundary_matched"] = n_boundary_matched
        args.local_summary_in.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"dx_local_views: wrote {len(seg_rows)} segments, "
        f"{len(bnd_rows)} boundaries, {len(grid_rows)} coarse-grid rows "
        f"(switches matched: {n_boundary_matched}/{n_flare_switches})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
