"""Download and cache 1000 Genomes superpopulation allele frequency reference.

The reference file is a gzipped TSV with columns:
    #chrom  pos  ref  alt  EUR  EAS  AMR  AFR  SAS

This module handles downloading, caching, and loading the reference for
use by ``popout label``.

Usage:
    popout fetch-ref [--genome GRCh38|GRCh37] [--dest PATH]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SUPERPOP_NAMES = ["EUR", "EAS", "AMR", "AFR", "SAS"]

REF_URLS: dict[str, str] = {
    # Placeholder — replace with hosted URL after running scripts/build_1kg_ref.py
    # "GRCh38": "https://storage.googleapis.com/popout-ref/1kg_superpop_freq.GRCh38.tsv.gz",
}

_CACHE_DIR = Path.home() / ".popout" / "ref"


def resolve_ref_path(
    genome: str = "GRCh38",
    ref_arg: str | None = None,
) -> Path:
    """Resolve reference frequency file path.

    Priority:
        1. Explicit ``ref_arg`` path
        2. Local cache (``~/.popout/ref/{genome}/1kg_superpop_freq.tsv.gz``)
        3. Auto-download to local cache

    Returns
    -------
    Path to the gzipped TSV reference file.
    """
    if ref_arg is not None:
        p = Path(ref_arg)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Reference file not found: {p}")

    cache = _CACHE_DIR / genome / "1kg_superpop_freq.tsv.gz"
    if cache.is_file():
        log.info("Using cached reference: %s", cache)
        return cache

    if genome not in REF_URLS:
        raise FileNotFoundError(
            f"No cached reference for {genome} and no download URL configured. "
            f"Provide a reference file via --reference, or build one with "
            f"scripts/build_1kg_ref.py and place it at {cache}"
        )

    log.info("Reference not found locally — downloading %s...", genome)
    return fetch_ref(genome, dest=cache)


def fetch_ref(genome: str = "GRCh38", dest: Path | None = None) -> Path:
    """Download 1KG superpopulation frequency reference.

    Parameters
    ----------
    genome : reference genome build
    dest : target file path (default: ``~/.popout/ref/{genome}/1kg_superpop_freq.tsv.gz``)

    Returns
    -------
    Path to the downloaded file.
    """
    if genome not in REF_URLS:
        raise ValueError(
            f"No download URL for genome build {genome!r}. "
            f"Available: {', '.join(REF_URLS) or '(none — build with scripts/build_1kg_ref.py)'}"
        )

    url = REF_URLS[genome]
    if dest is None:
        dest = _CACHE_DIR / genome / "1kg_superpop_freq.tsv.gz"

    dest.parent.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s reference from %s", genome, url)
    data = urllib.request.urlopen(url).read()
    log.info("Downloaded %.1f MB", len(data) / 1e6)

    dest.write_bytes(data)
    log.info("Saved reference to %s", dest)
    return dest


def load_ref_frequencies(
    ref_path: str | Path,
    chrom: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load reference superpopulation frequencies from TSV.

    Parameters
    ----------
    ref_path : path to gzipped TSV
    chrom : if given, filter to this chromosome only

    Returns
    -------
    freq : (n_pops, n_sites) float32 array
    pos_bp : (n_sites,) int64 array
    pop_names : list of population names from the header
    """
    ref_path = Path(ref_path)
    positions = []
    freq_rows = []
    pop_names = None

    opener = gzip.open if ref_path.suffix == ".gz" else open
    with opener(ref_path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                # Header row: #chrom  pos  ref  alt  EUR  EAS  AMR  AFR  SAS
                pop_names = row[4:]
                continue
            row_chrom = row[0]
            if chrom is not None and row_chrom != chrom:
                # Normalise: try with/without "chr" prefix
                norm_row = row_chrom.replace("chr", "")
                norm_query = chrom.replace("chr", "")
                if norm_row != norm_query:
                    continue
            positions.append(int(row[1]))
            freq_rows.append([float(x) for x in row[4:]])

    if pop_names is None:
        pop_names = SUPERPOP_NAMES

    if not positions:
        raise ValueError(
            f"No sites loaded from {ref_path}"
            + (f" for chrom={chrom}" if chrom else "")
        )

    pos_bp = np.array(positions, dtype=np.int64)
    freq = np.array(freq_rows, dtype=np.float32).T  # (n_pops, n_sites)

    log.info("Loaded %d reference sites (%d populations)", len(positions), freq.shape[0])
    return freq, pos_bp, pop_names


def fetch_ref_main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``popout fetch-ref``."""
    parser = argparse.ArgumentParser(
        description="Download 1000 Genomes superpopulation frequency reference",
    )
    parser.add_argument(
        "--genome", choices=["GRCh38", "GRCh37"], default="GRCh38",
        help="Reference genome build (default: GRCh38)",
    )
    parser.add_argument(
        "--dest", default=None,
        help="Destination file path (default: ~/.popout/ref/GENOME/1kg_superpop_freq.tsv.gz)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    dest = Path(args.dest) if args.dest else None
    result = fetch_ref(args.genome, dest=dest)
    print(result)
