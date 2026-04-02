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
    """Load a HapMap-format recombination map.

    Expected columns (whitespace-delimited, header line):
        chr  position(bp)  rate(cM/Mb)  Map(cM)

    Returns a dict keyed by chromosome name (without 'chr' prefix).
    """
    maps: dict[str, GeneticMap] = {}
    path = Path(path)
    current_chrom = None
    bp_buf: list[int] = []
    cm_buf: list[float] = []

    with open(path) as fh:
        header = fh.readline()
        for line in fh:
            parts = line.split()
            chrom, bp, _rate, cm = parts[0], int(parts[1]), parts[2], float(parts[3])
            chrom = normalise_chrom(chrom)
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
