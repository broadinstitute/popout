"""Ancestry labeling via correlation with 1KG superpopulation frequencies.

Algorithm:
    1. Load inferred allele frequencies + site positions from .model.npz
    2. Load 1KG superpopulation frequencies, intersect sites by position
    3. Pearson correlation matrix (K_inferred x K_ref)
    4. Optimal assignment via Hungarian algorithm (linear_sum_assignment)
    5. If K_inferred > K_ref, merge multiple inferred ancestries into one label
    6. Rewrite .global.tsv and .tracts.tsv.gz with population labels

Usage:
    popout label --model PREFIX.model.npz --global PREFIX.global.tsv \\
                 --tracts PREFIX.tracts.tsv.gz --out PREFIX.labeled \\
                 [--reference REF.tsv.gz] [--genome GRCh38]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .fetch_ref import load_ref_frequencies, resolve_ref_path

log = logging.getLogger(__name__)


@dataclass
class LabelResult:
    """Result of ancestry labeling."""

    label_map: dict[int, str]
    """Maps inferred ancestry index → population name."""

    merge_map: dict[str, list[int]]
    """Maps population name → list of inferred ancestry indices (>1 = merged)."""

    correlations: np.ndarray
    """Pearson correlation matrix (K_inferred, K_ref)."""

    ref_names: list[str]
    """Reference population names."""

    n_overlapping_sites: int
    """Number of sites used for correlation."""


def label_ancestries(
    model_npz_path: str | Path,
    ref_path: str | Path,
) -> LabelResult:
    """Label inferred ancestries by correlating with reference frequencies.

    Parameters
    ----------
    model_npz_path : path to .model.npz file (must contain pos_bp, pos_cm, chrom)
    ref_path : path to reference frequency TSV

    Returns
    -------
    LabelResult with assignment and correlation info.
    """
    model_npz_path = Path(model_npz_path)
    m = np.load(model_npz_path)

    allele_freq = m["allele_freq"]  # (K_inf, T_model)
    pos_bp = m["pos_bp"]            # (T_model,)
    chrom = str(m["chrom"])

    K_inf = allele_freq.shape[0]
    log.info("Model: K=%d, T=%d sites on %s", K_inf, allele_freq.shape[1], chrom)

    ref_freq, ref_pos, ref_names = load_ref_frequencies(ref_path, chrom=chrom)
    K_ref = ref_freq.shape[0]

    # Intersect sites by position
    _common, model_idx, ref_idx = np.intersect1d(pos_bp, ref_pos, return_indices=True)
    n_overlap = len(_common)
    log.info("Overlapping sites: %d (model=%d, ref=%d)", n_overlap, len(pos_bp), len(ref_pos))

    if n_overlap < 10:
        raise ValueError(
            f"Only {n_overlap} overlapping sites between model and reference. "
            f"Need at least 10 for reliable correlation."
        )
    if n_overlap < 100:
        log.warning("Only %d overlapping sites — correlation may be unreliable", n_overlap)

    # Subset to overlapping sites
    freq_inf = allele_freq[:, model_idx]   # (K_inf, n_overlap)
    freq_ref = ref_freq[:, ref_idx]        # (K_ref, n_overlap)

    # Pearson correlation matrix
    corr = _correlation_matrix(freq_inf, freq_ref)  # (K_inf, K_ref)

    # Assign labels
    label_map, merge_map = _assign_labels(corr, ref_names)

    log.info("Label assignment:")
    for idx, name in sorted(label_map.items()):
        log.info("  ancestry_%d -> %s (r=%.3f)", idx, name, corr[idx, ref_names.index(name)])

    return LabelResult(
        label_map=label_map,
        merge_map=merge_map,
        correlations=corr,
        ref_names=ref_names,
        n_overlapping_sites=n_overlap,
    )


def _correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows of a and rows of b.

    Parameters
    ----------
    a : (M, T) array
    b : (N, T) array

    Returns
    -------
    corr : (M, N) array of Pearson correlations
    """
    combined = np.vstack([a, b])  # (M+N, T)
    full_corr = np.corrcoef(combined)  # (M+N, M+N)
    M = a.shape[0]
    return full_corr[:M, M:]  # (M, N)


def _assign_labels(
    corr: np.ndarray,
    ref_names: list[str],
) -> tuple[dict[int, str], dict[str, list[int]]]:
    """Assign population labels to inferred ancestries.

    Uses Hungarian algorithm for optimal 1-to-1 matching when K_inf <= K_ref.
    When K_inf > K_ref, maps each inferred ancestry to its best-correlated
    reference (multiple inferred may map to the same reference = merge).

    Returns
    -------
    label_map : {inferred_idx: pop_name}
    merge_map : {pop_name: [inferred_idx, ...]}
    """
    K_inf, K_ref = corr.shape

    if K_inf <= K_ref:
        # Hungarian algorithm on cost = -correlation
        row_ind, col_ind = linear_sum_assignment(-corr)
        label_map = {int(r): ref_names[int(c)] for r, c in zip(row_ind, col_ind)}
    else:
        # More inferred than reference: each inferred gets its max-corr ref
        best_ref = np.argmax(corr, axis=1)
        label_map = {int(i): ref_names[int(best_ref[i])] for i in range(K_inf)}

    # Build merge map
    merge_map: dict[str, list[int]] = {}
    for idx, name in sorted(label_map.items()):
        merge_map.setdefault(name, []).append(idx)

    # Sort each merge group by correlation strength (strongest first)
    for name, indices in merge_map.items():
        ref_col = ref_names.index(name)
        indices.sort(key=lambda i: -corr[i, ref_col])

    return label_map, merge_map


def rewrite_global_tsv(
    in_path: str | Path,
    out_path: str | Path,
    label_result: LabelResult,
) -> None:
    """Rewrite global ancestry TSV with population labels.

    Merged ancestries have their proportions summed.
    """
    in_path, out_path = Path(in_path), Path(out_path)

    with open(in_path) as fin:
        reader = csv.reader(fin, delimiter="\t")
        header = next(reader)
        rows = list(reader)

    # header: sample, ancestry_0, ancestry_1, ...
    n_anc_cols = len(header) - 1
    anc_col_start = 1

    # Build new columns based on merge_map (ordered by first occurrence in label_map)
    seen_names: list[str] = []
    for idx in range(n_anc_cols):
        name = label_result.label_map.get(idx, f"unknown_{idx}")
        if name not in seen_names:
            seen_names.append(name)

    new_header = ["sample"] + seen_names

    new_rows = []
    for row in rows:
        sample = row[0]
        old_vals = [float(v) for v in row[anc_col_start:]]
        new_vals = {}
        for idx, val in enumerate(old_vals):
            name = label_result.label_map.get(idx, f"unknown_{idx}")
            new_vals[name] = new_vals.get(name, 0.0) + val
        new_rows.append([sample] + [f"{new_vals.get(n, 0.0):.6f}" for n in seen_names])

    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(new_header)
        writer.writerows(new_rows)

    log.info("Wrote labeled global ancestry to %s", out_path)


def rewrite_tracts_tsv(
    in_path: str | Path,
    out_path: str | Path,
    label_result: LabelResult,
) -> None:
    """Rewrite tracts TSV with population labels.

    Adjacent tracts that map to the same label after remapping are merged.
    """
    in_path, out_path = Path(in_path), Path(out_path)

    in_opener = gzip.open if in_path.suffix == ".gz" else open
    out_opener = gzip.open if out_path.suffix == ".gz" else open

    with in_opener(in_path, "rt") as fin, out_opener(out_path, "wt") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")

        header = next(reader)
        # Columns: #chrom, start_bp, end_bp, sample, haplotype, ancestry, n_sites, [mean_posterior]
        has_posterior = len(header) > 7
        writer.writerow(header)

        # Track previous tract per (sample, haplotype) for merging
        prev: dict[tuple[str, str], list[str]] = {}

        def _flush(key: tuple[str, str]) -> list[str] | None:
            return prev.pop(key, None)

        def _remap_ancestry(anc_str: str) -> str:
            try:
                idx = int(anc_str)
                return label_result.label_map.get(idx, anc_str)
            except ValueError:
                return anc_str

        for row in reader:
            if len(row) < 7:
                continue
            chrom, start, end, sample, hap, ancestry, n_sites = row[:7]
            posterior = row[7] if has_posterior else None

            new_ancestry = _remap_ancestry(ancestry)
            key = (sample, hap)

            if key in prev:
                prev_row = prev[key]
                prev_ancestry = prev_row[5]
                if prev_ancestry == new_ancestry and prev_row[0] == chrom:
                    # Merge: extend end, sum n_sites, average posteriors
                    prev_row[2] = end
                    prev_row[6] = str(int(prev_row[6]) + int(n_sites))
                    if has_posterior and posterior is not None:
                        old_n = int(prev_row[6]) - int(n_sites)
                        new_n = int(n_sites)
                        total_n = old_n + new_n
                        if total_n > 0:
                            old_post = float(prev_row[7])
                            new_post = float(posterior)
                            prev_row[7] = f"{(old_post * old_n + new_post * new_n) / total_n:.6f}"
                    continue

                # Different ancestry or chrom — flush previous
                writer.writerow(prev[key])

            # Store current as pending
            new_row = [chrom, start, end, sample, hap, new_ancestry, n_sites]
            if has_posterior:
                new_row.append(posterior if posterior else "0.0")
            prev[key] = new_row

        # Flush remaining
        for row in prev.values():
            writer.writerow(row)

    log.info("Wrote labeled tracts to %s", out_path)


def write_label_report(out_path: str | Path, label_result: LabelResult) -> None:
    """Write labeling metadata as JSON."""
    out_path = Path(out_path)
    report = {
        "label_map": {str(k): v for k, v in label_result.label_map.items()},
        "merge_map": label_result.merge_map,
        "ref_names": label_result.ref_names,
        "n_overlapping_sites": label_result.n_overlapping_sites,
        "correlations": label_result.correlations.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote label report to %s", out_path)


def label_main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``popout label``."""
    parser = argparse.ArgumentParser(
        description="Label inferred ancestries using 1KG superpopulation reference",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to .model.npz file",
    )
    parser.add_argument(
        "--global", dest="global_tsv", required=True,
        help="Path to .global.tsv file",
    )
    parser.add_argument(
        "--tracts", required=True,
        help="Path to .tracts.tsv.gz file",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output prefix (produces .global.tsv, .tracts.tsv.gz, .labels.json)",
    )
    parser.add_argument(
        "--reference", default=None,
        help="Path to reference frequency TSV (default: auto-resolve from cache)",
    )
    parser.add_argument(
        "--genome", default="GRCh38",
        help="Reference genome build (default: GRCh38)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    ref_path = resolve_ref_path(args.genome, ref_arg=args.reference)
    result = label_ancestries(args.model, ref_path)

    out = args.out
    rewrite_global_tsv(args.global_tsv, f"{out}.global.tsv", result)
    rewrite_tracts_tsv(args.tracts, f"{out}.tracts.tsv.gz", result)
    write_label_report(f"{out}.labels.json", result)

    print(f"Labeling complete. Output prefix: {out}")
    for idx, name in sorted(result.label_map.items()):
        ref_col = result.ref_names.index(name)
        print(f"  ancestry_{idx} -> {name} (r={result.correlations[idx, ref_col]:.3f})")
