"""Parser for popout tracts.tsv.gz output."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional

import numpy as np

from popout.benchmark.common import TractSet


def parse_popout(
    tracts_path: str | Path,
    global_path: Optional[str | Path] = None,
    site_positions: Optional[np.ndarray] = None,
) -> TractSet:
    """Parse popout's tracts TSV into a TractSet.

    Parameters
    ----------
    tracts_path : path to .tracts.tsv or .tracts.tsv.gz
    global_path : optional path to .global.tsv (unused, for API consistency)
    site_positions : optional (T,) int array of bp positions to evaluate on.
        If not provided, uses the union of tract start/end positions.
    """
    tracts_path = Path(tracts_path)
    opener = gzip.open if tracts_path.suffix == ".gz" else open

    # Read all tracts
    tracts: list[dict] = []
    chrom: Optional[str] = None
    with opener(tracts_path, "rt") as f:
        header = f.readline().strip().lstrip("#").split("\t")
        # Expected columns: chrom, start_bp, end_bp, sample, haplotype, ancestry, n_sites
        col_idx = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.strip().split("\t")
            row_chrom = parts[col_idx["chrom"]]
            if chrom is None:
                chrom = row_chrom
            tract = {
                "chrom": row_chrom,
                "start_bp": int(parts[col_idx["start_bp"]]),
                "end_bp": int(parts[col_idx["end_bp"]]),
                "sample": parts[col_idx["sample"]],
                "haplotype": int(parts[col_idx["haplotype"]]),
                "ancestry": int(parts[col_idx["ancestry"]]),
            }
            tracts.append(tract)

    if not tracts:
        raise ValueError(f"No tracts found in {tracts_path}")

    # Build haplotype IDs in sorted order
    hap_set: set[str] = set()
    for t in tracts:
        hap_id = f"{t['sample']}:{t['haplotype']}"
        hap_set.add(hap_id)
    hap_ids_sorted = sorted(hap_set)
    hap_to_idx = {h: i for i, h in enumerate(hap_ids_sorted)}

    # Determine site positions
    if site_positions is None:
        pos_set: set[int] = set()
        for t in tracts:
            pos_set.add(t["start_bp"])
            pos_set.add(t["end_bp"])
        site_positions = np.array(sorted(pos_set), dtype=np.int64)

    # Build ancestry labels
    anc_codes = sorted(set(t["ancestry"] for t in tracts))
    label_map = {code: str(code) for code in anc_codes}

    # Build dense calls array
    n_haps = len(hap_ids_sorted)
    n_sites = len(site_positions)
    calls = np.zeros((n_haps, n_sites), dtype=np.uint16)

    for t in tracts:
        hap_id = f"{t['sample']}:{t['haplotype']}"
        h_idx = hap_to_idx[hap_id]
        # Find sites within [start_bp, end_bp]
        mask = (site_positions >= t["start_bp"]) & (site_positions <= t["end_bp"])
        calls[h_idx, mask] = t["ancestry"]

    ts = TractSet(
        tool_name="popout",
        chrom=chrom or "unknown",
        hap_ids=np.array(hap_ids_sorted, dtype=object),
        site_positions=site_positions,
        calls=calls,
        label_map=label_map,
    )
    ts.validate()
    return ts
