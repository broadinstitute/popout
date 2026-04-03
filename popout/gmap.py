"""Genetic map loading and chromosome normalization utilities.

Shared by both VCF and PGEN readers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .datatypes import GeneticMap

log = logging.getLogger(__name__)


def normalise_chrom(chrom: str) -> str:
    """Strip 'chr' prefix for consistent keying."""
    if chrom.startswith("chr"):
        return chrom[3:]
    return chrom


def load_genetic_map(path: str | Path) -> dict[str, GeneticMap]:
    """Load a genetic recombination map.

    Auto-detects two formats:

    **PLINK .map** (4 columns, no header)::

        chr  id  cM  bp

    **HapMap** (4 columns, with header)::

        chr  position(bp)  rate(cM/Mb)  Map(cM)

    Returns a dict keyed by chromosome name (without 'chr' prefix).
    """
    maps: dict[str, GeneticMap] = {}
    path = Path(path)
    current_chrom = None
    bp_buf: list[int] = []
    cm_buf: list[float] = []

    with open(path) as fh:
        first_line = fh.readline()
        parts = first_line.split()

        # Detect format: PLINK .map has no header (first field is a chrom name/number)
        # HapMap has a header line (first field is typically "chr" or "Chromosome")
        is_plink = len(parts) == 4 and parts[1] == "." or _is_chrom_name(parts[0])

        if is_plink:
            # Re-process first line (it's data, not a header)
            lines = [first_line] + fh.readlines()
        else:
            # Skip header
            lines = fh.readlines()

        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue

            if is_plink:
                # PLINK format: chr, id, cM, bp
                chrom = normalise_chrom(parts[0])
                cm = float(parts[2])
                bp = int(parts[3])
            else:
                # HapMap format: chr, bp, rate, cM
                chrom = normalise_chrom(parts[0])
                bp = int(parts[1])
                cm = float(parts[3])

            if chrom != current_chrom:
                if current_chrom is not None:
                    maps[current_chrom] = GeneticMap(
                        np.array(bp_buf, dtype=np.int64),
                        np.array(cm_buf, dtype=np.float64),
                    )
                current_chrom = chrom
                bp_buf, cm_buf = [], []
            bp_buf.append(bp)
            cm_buf.append(cm)

        if current_chrom is not None:
            maps[current_chrom] = GeneticMap(
                np.array(bp_buf, dtype=np.int64),
                np.array(cm_buf, dtype=np.float64),
            )
    log.info("Loaded genetic map for %d chromosomes from %s", len(maps), path)
    return maps


def _is_chrom_name(s: str) -> bool:
    """Check if a string looks like a chromosome name (1-22, X, Y, or chr-prefixed)."""
    s = normalise_chrom(s)
    return s.isdigit() or s in ("X", "Y", "M", "MT")


def load_genetic_map_per_chrom(directory: str | Path) -> dict[str, GeneticMap]:
    """Load per-chromosome map files from a directory.

    Expects files named like: plink.chr1.GRCh38.map (or similar patterns).
    Falls back to trying each file and extracting the chromosome from content.
    """
    directory = Path(directory)
    maps: dict[str, GeneticMap] = {}
    for f in sorted(directory.glob("*.map")):
        file_maps = load_genetic_map(f)
        maps.update(file_maps)
    return maps
