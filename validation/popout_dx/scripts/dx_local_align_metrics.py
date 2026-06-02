#!/usr/bin/env python3
"""popout DX — local-mode align + per-(sample, hap) metrics.

Aligns popout's whole-cohort tracts subset to FLARE's per-cluster-per-chrom
site grid via ``popout.benchmark.align.align_sites(strategy="project_a_onto_b")``
— popout's tract membership is projected onto FLARE's high-resolution
marker positions. Then computes:

  ``local_per_sample.tsv``     one row per (sample, chrom): n_sites_compared,
                                agree_pct, jaccard_tracts
  ``local_per_haplotype.tsv``  one row per (sample, hap, chrom): agree_pct,
                                per_ancestry_r2 (JSON-encoded)
  ``local_summary.json``       chrom-level bp_agreement plus placeholders
                                for calibration_drift / boundary_localization
                                (those are produced by dx_local_views.py
                                from the segments TSV)

Inputs
------
``--popout-tracts``     popout whole-cohort tracts.tsv.gz
``--popout-labels``     popout-side labels.json
``--flare-npz``         output of dx_local_parse_flare.py (TractSet npz)
``--flare-labels``      FLARE-side per-cluster labels.json (from cohort bundle)
``--samples-file``      one sample_id per line (the picker's selection)
``--chrom``             chrom to filter popout's tracts.tsv.gz on
``--out-dir``           emits the three files above
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

# popout/benchmark is installed in the docker.
from popout.benchmark.common import MISSING_LABEL, TractSet
from popout.benchmark.align import align_haps, align_sites
from popout.benchmark.metrics import (
    per_site_accuracy,
    per_haplotype_accuracy,
    per_ancestry_r2,
)
from popout.viz._loaders import read_labels_json


# Phase 4 of the label-space retrofit: SP6 is the canonical superpop
# space from popout.labelspace.registry.
from popout.labelspace.registry import SP6 as _SP6
RF_LABELS_CANONICAL: tuple[str, ...] = _SP6.members
RF_NAME_TO_CODE: dict[str, int] = {n: i for i, n in enumerate(RF_LABELS_CANONICAL)}


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"dx_local_align_metrics: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── popout tracts subset → TractSet on a target site grid ────────────────


def load_popout_tracts_subset(
    tracts_path: Path,
    selected_samples: list[str],
    chrom: str,
    site_grid: np.ndarray,
) -> TractSet:
    """Stream popout's whole-cohort tracts.tsv.gz; for each tract whose
    sample is in ``selected_samples`` and chrom matches, write its
    ancestry into every site in ``site_grid`` that falls within
    ``[start_bp, end_bp]``.

    Returns a TractSet on the supplied site grid with popout's native
    integer label codes.
    """
    selected_set = set(selected_samples)
    sample_to_idx = {s: i for i, s in enumerate(selected_samples)}
    n_haps = 2 * len(selected_samples)
    calls = np.full((n_haps, len(site_grid)), MISSING_LABEL, dtype=np.uint16)
    label_codes: set[int] = set()

    seen_samples: set[str] = set()
    n_kept = 0
    n_total = 0

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
            n_total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max_col:
                continue
            file_chrom = parts[ic]
            if not file_chrom.startswith("chr"):
                file_chrom = "chr" + file_chrom
            if file_chrom != chrom:
                continue
            sam = parts[isa]
            if sam not in selected_set:
                continue
            hap = int(parts[ihp])
            if hap not in (0, 1):
                die(f"{tracts_path}: tract for {sam!r} has invalid haplotype {hap}")
            hap_row = 2 * sample_to_idx[sam] + hap
            start = int(parts[isb])
            end = int(parts[ieb])
            anc = int(parts[ian])
            label_codes.add(anc)
            mask = (site_grid >= start) & (site_grid <= end)
            calls[hap_row, mask] = anc
            seen_samples.add(sam)
            n_kept += 1

    missing = sorted(selected_set - seen_samples)
    if missing:
        die(
            f"{tracts_path}: {len(missing)} selected sample(s) had no tracts on "
            f"chrom {chrom}; first: {missing[:5]}"
        )

    label_map = {code: str(code) for code in sorted(label_codes)}
    hap_ids = np.array(
        [f"{s}:{h}" for s in selected_samples for h in (0, 1)],
        dtype=object,
    )
    ts = TractSet(
        tool_name="popout",
        chrom=chrom,
        hap_ids=hap_ids,
        site_positions=site_grid.copy(),
        calls=calls,
        label_map=label_map,
    )
    ts.validate()
    print(
        f"dx_local_align_metrics: loaded popout tracts ({n_kept} kept / {n_total} total) "
        f"for {len(selected_samples)} samples on {chrom}",
        file=sys.stderr,
    )
    return ts


# ── Remap a TractSet's labels into canonical RF codes ────────────────────


def remap_to_rf_codes(ts: TractSet, labels_json: dict) -> TractSet:
    """Translate ``ts.calls`` from tool-native integer codes to canonical
    RF integer codes via the popout/FLARE ``popout_to_rf_label`` mapping.

    Phase 3 of the label-space retrofit routes through
    :func:`popout.labelspace.project.project_tracts`.
    """
    p2rf = labels_json.get("popout_to_rf_label")
    if not isinstance(p2rf, dict) or not p2rf:
        die("labels.json missing or empty popout_to_rf_label")

    # Build a transient Assignment that mirrors the legacy mapping shape.
    from popout.labelspace import Assignment, get as _get_space, project_tracts
    target = _get_space("SP6")
    component_to_label: dict[int, str] = {}
    for k, v in p2rf.items():
        try:
            src = int(k)
        except (TypeError, ValueError):
            die(f"labels.json popout_to_rf_label key {k!r} is not int-coercible")
        if v not in target.members:
            die(f"labels.json popout_to_rf_label[{src}] = {v!r} is not a canonical RF label")
        component_to_label[src] = v
    label_to_components: dict[str, list[int]] = {}
    for k, v in component_to_label.items():
        label_to_components.setdefault(v, []).append(k)

    assignment = Assignment(
        target_space=target, source={"tool": ts.tool_name},
        method="postS", input_space="hard_call",
        component_to_label=component_to_label,
        label_to_components=label_to_components,
        subcomponent_names={}, diagnostics={}, provenance={},
    )
    try:
        return project_tracts(ts, assignment, missing_label=MISSING_LABEL)
    except ValueError as e:
        die(str(e))


# ── TractSet loader from dx_local_parse_flare's npz ──────────────────────


def load_flare_npz(npz_path: Path) -> TractSet:
    d = np.load(npz_path, allow_pickle=True)
    label_map = {int(k): v for k, v in json.loads(str(d["label_map_json"])).items()}
    ts = TractSet(
        tool_name=str(d["tool_name"]),
        chrom=str(d["chrom"]),
        hap_ids=np.asarray(d["hap_ids"], dtype=object),
        site_positions=np.asarray(d["site_positions"], dtype=np.int64),
        calls=np.asarray(d["calls"], dtype=np.uint16),
        label_map=label_map,
    )
    ts.validate()
    return ts


# ── Output writers ───────────────────────────────────────────────────────


def write_per_sample(
    selected_samples: list[str],
    chrom: str,
    popout_rf: TractSet,
    flare_rf: TractSet,
    out_path: Path,
) -> dict[str, dict]:
    """Write local_per_sample.tsv. Returns the parsed-back dict for
    summary computation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = (popout_rf.calls != MISSING_LABEL) & (flare_rf.calls != MISSING_LABEL)
    agree = popout_rf.calls == flare_rf.calls
    sample_rows: dict[str, dict] = {}
    with open(out_path, "w") as f:
        f.write("sample\tchrom\tn_sites_compared\tagree_pct\tjaccard_tracts\n")
        for i, sam in enumerate(selected_samples):
            h0, h1 = 2 * i, 2 * i + 1
            v0 = valid[h0]; v1 = valid[h1]
            n_compared = int(v0.sum() + v1.sum())
            if n_compared == 0:
                agree_pct = float("nan")
                jaccard = float("nan")
            else:
                n_agree = int(agree[h0, v0].sum() + agree[h1, v1].sum())
                agree_pct = n_agree / n_compared
                # Per-sample tract Jaccard: union over haps of (label-agreeing
                # site mask) vs union of (any-valid-site mask). Hap-merged.
                inter = int(((agree[h0] & v0) | (agree[h1] & v1)).sum())
                union = int((v0 | v1).sum())
                jaccard = inter / union if union > 0 else float("nan")
            sample_rows[sam] = {
                "n_sites_compared": n_compared,
                "agree_pct": agree_pct,
                "jaccard_tracts": jaccard,
            }
            agree_pct_s = "NA" if np.isnan(agree_pct) else f"{agree_pct:.6f}"
            jaccard_s = "NA" if np.isnan(jaccard) else f"{jaccard:.6f}"
            f.write(f"{sam}\t{chrom}\t{n_compared}\t{agree_pct_s}\t{jaccard_s}\n")
    return sample_rows


def write_per_haplotype(
    selected_samples: list[str],
    chrom: str,
    popout_rf: TractSet,
    flare_rf: TractSet,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = (popout_rf.calls != MISSING_LABEL) & (flare_rf.calls != MISSING_LABEL)
    agree = popout_rf.calls == flare_rf.calls

    # Per-hap r² over RF labels: we restrict to the per-label fractions on
    # this single hap and compute the (degenerate) r² across labels. The
    # population-level per_ancestry_r2 from popout.benchmark.metrics works
    # across HAPS not labels; here we want a per-hap diagnostic, so compute
    # per-label fraction agreement explicitly.
    K = len(RF_LABELS_CANONICAL)
    with open(out_path, "w") as f:
        f.write("sample\thap\tchrom\tagree_pct\tper_ancestry_fraction_match_json\n")
        for i, sam in enumerate(selected_samples):
            for h in (0, 1):
                row = 2 * i + h
                v = valid[row]
                if v.sum() == 0:
                    agree_pct_s = "NA"
                    frac_dict = {}
                else:
                    agree_pct_s = f"{agree[row, v].mean():.6f}"
                    frac_dict = {}
                    for k_code, k_name in zip(range(K), RF_LABELS_CANONICAL):
                        pop_is_k = (popout_rf.calls[row, v] == k_code).mean()
                        fla_is_k = (flare_rf.calls[row, v] == k_code).mean()
                        # Per-hap fraction match: 1 - |fraction_diff|. Bounded
                        # in [0, 1]. A per-label diagnostic per hap.
                        frac_dict[k_name] = round(1.0 - abs(float(pop_is_k) - float(fla_is_k)), 6)
                f.write(f"{sam}\t{h}\t{chrom}\t{agree_pct_s}\t{json.dumps(frac_dict)}\n")


def write_summary(
    chrom: str,
    popout_rf: TractSet,
    flare_rf: TractSet,
    sample_rows: dict[str, dict],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = (popout_rf.calls != MISSING_LABEL) & (flare_rf.calls != MISSING_LABEL)
    bp_agreement = float(per_site_accuracy(popout_rf, flare_rf))
    # Population per-ancestry r² (across haps).
    per_anc_r2 = per_ancestry_r2(popout_rf, flare_rf)
    per_anc_r2_named = {
        RF_LABELS_CANONICAL[k]: (None if np.isnan(v) else float(v))
        for k, v in per_anc_r2.items()
    }
    # calibration_drift_fraction and boundary_localization_error_fraction
    # are computed downstream in dx_local_views.py from the segments TSV;
    # we emit them as null here so the schema slot is present.
    summary = {
        "chrom": chrom,
        "n_haps_compared": int(popout_rf.n_haps),
        "n_sites_grid": int(popout_rf.n_sites),
        "n_valid_cells": int(valid.sum()),
        "bp_agreement": bp_agreement,
        "calibration_drift_fraction": None,
        "boundary_localization_error_fraction": None,
        "per_ancestry_r2_mean": per_anc_r2_named,
        "per_sample_agree_pct_mean": float(np.nanmean(
            [r["agree_pct"] for r in sample_rows.values()]
        )) if sample_rows else None,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--popout-tracts", required=True, type=Path)
    ap.add_argument("--popout-labels", required=True, type=Path)
    ap.add_argument("--flare-npz", required=True, type=Path)
    ap.add_argument("--flare-labels", required=True, type=Path)
    ap.add_argument("--samples-file", required=True, type=Path)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    selected = [s for s in args.samples_file.read_text().splitlines() if s.strip()]
    if not selected:
        die(f"--samples-file {args.samples_file} is empty")

    flare_ts = load_flare_npz(args.flare_npz)
    if flare_ts.chrom != args.chrom:
        die(f"--chrom {args.chrom!r} != FLARE TractSet chrom {flare_ts.chrom!r}")

    popout_ts = load_popout_tracts_subset(
        args.popout_tracts, selected, args.chrom, flare_ts.site_positions
    )

    # The popout subset was already built on FLARE's grid → align_sites is a
    # no-op; align_haps gives us the matched order.
    popout_aligned, flare_aligned = align_haps(popout_ts, flare_ts)

    popout_labels = read_labels_json(args.popout_labels)
    flare_labels = read_labels_json(args.flare_labels)
    popout_rf = remap_to_rf_codes(popout_aligned, popout_labels)
    flare_rf = remap_to_rf_codes(flare_aligned, flare_labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = write_per_sample(
        selected, args.chrom, popout_rf, flare_rf, args.out_dir / "local_per_sample.tsv"
    )
    write_per_haplotype(
        selected, args.chrom, popout_rf, flare_rf, args.out_dir / "local_per_haplotype.tsv"
    )
    write_summary(
        args.chrom, popout_rf, flare_rf, sample_rows, args.out_dir / "local_summary.json"
    )
    print(
        f"dx_local_align_metrics: wrote per-sample / per-hap / summary "
        f"({len(selected)} samples, {flare_rf.n_sites} sites)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
