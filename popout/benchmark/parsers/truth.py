"""Parser for popout simulator ground-truth output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from popout.benchmark.common import TractSet


def parse_truth(
    truth_path: str | Path,
    site_positions: Optional[np.ndarray] = None,
) -> TractSet:
    """Parse ground-truth ancestry from popout's simulator.

    Expects an .npz file with:
        - true_ancestry: (H, T) int8 array of ground-truth labels
        - pos_bp: (T,) int64 array of site positions
        - n_ancestries: int (optional, inferred from data if absent)
        - chrom: str (optional, defaults to "sim")
    """
    truth_path = Path(truth_path)
    data = np.load(truth_path, allow_pickle=True)

    true_ancestry = data["true_ancestry"]  # (H, T)
    pos_bp = data["pos_bp"] if "pos_bp" in data else site_positions
    if pos_bp is None:
        raise ValueError("No pos_bp in truth file and no site_positions provided")

    n_haps = true_ancestry.shape[0]
    chrom = str(data["chrom"]) if "chrom" in data else "sim"

    # Determine number of ancestries
    if "n_ancestries" in data:
        n_anc = int(data["n_ancestries"])
    else:
        n_anc = int(true_ancestry.max()) + 1

    label_map = {i: f"anc_{i}" for i in range(n_anc)}

    # If site_positions provided, subset/reindex
    if site_positions is not None and not np.array_equal(pos_bp, site_positions):
        # Find intersection indices
        common = np.intersect1d(pos_bp, site_positions)
        idx_in_truth = np.searchsorted(pos_bp, common)
        true_ancestry = true_ancestry[:, idx_in_truth]
        pos_bp = common

    # Build haplotype IDs: sample_idx:hap_idx convention
    hap_ids = np.array(
        [f"S{i // 2:04d}:{i % 2}" for i in range(n_haps)],
        dtype=object,
    )

    calls = true_ancestry.astype(np.uint16)

    ts = TractSet(
        tool_name="truth",
        chrom=chrom,
        hap_ids=hap_ids,
        site_positions=np.asarray(pos_bp, dtype=np.int64),
        calls=calls,
        label_map=label_map,
    )
    ts.validate()
    return ts


def tractset_from_arrays(
    true_ancestry: np.ndarray,
    pos_bp: np.ndarray,
    sample_names: Optional[list[str]] = None,
    chrom: str = "sim",
) -> TractSet:
    """Build a TractSet directly from simulator arrays (convenience for tests).

    Parameters
    ----------
    true_ancestry : (H, T) int array of ground-truth labels
    pos_bp : (T,) int64 site positions
    sample_names : optional list of sample names (length H/2)
    chrom : chromosome name
    """
    n_haps = true_ancestry.shape[0]
    n_anc = int(true_ancestry.max()) + 1

    if sample_names is not None:
        hap_ids = np.array(
            [f"{sample_names[i // 2]}:{i % 2}" for i in range(n_haps)],
            dtype=object,
        )
    else:
        hap_ids = np.array(
            [f"S{i // 2:04d}:{i % 2}" for i in range(n_haps)],
            dtype=object,
        )

    label_map = {i: f"anc_{i}" for i in range(n_anc)}

    ts = TractSet(
        tool_name="truth",
        chrom=chrom,
        hap_ids=hap_ids,
        site_positions=np.asarray(pos_bp, dtype=np.int64),
        calls=true_ancestry.astype(np.uint16),
        label_map=label_map,
    )
    ts.validate()
    return ts
