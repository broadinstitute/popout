"""Download and cache 1000 Genomes superpopulation allele-frequency TSV.

The file is a gzipped TSV with columns:
    #chrom  pos  ref  alt  EUR  EAS  AMR  AFR  SAS

This module handles downloading, caching, and loading the per-superpop
allele-frequency table used by ``popout label`` and the priors framework's
identity scoring. (It is NOT the genome FASTA — that's a separate
"reference" concept; this file is the 1KG superpop frequency table only.)

Usage:
    popout fetch-superpop-freqs [--genome GRCh38|GRCh37] [--dest PATH]
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

SUPERPOP_FREQS_URLS: dict[str, str] = {
    # Placeholder — replace with hosted URL after running scripts/build_1kg_ref.py
    # "GRCh38": "https://storage.googleapis.com/popout-ref/1kg_superpop_freq.GRCh38.tsv.gz",
}

_CACHE_DIR = Path.home() / ".popout" / "superpop_freqs"


def resolve_superpop_freqs_path(
    genome: str = "GRCh38",
    arg: str | None = None,
) -> Path:
    """Resolve the 1KG superpop-frequency TSV path.

    Priority:
        1. Explicit ``arg`` path (e.g. from ``--superpop-freqs``)
        2. Local cache (``~/.popout/superpop_freqs/{genome}/1kg_superpop_freq.tsv.gz``)
        3. Auto-download to local cache

    Returns
    -------
    Path to the gzipped TSV file.
    """
    if arg is not None:
        p = Path(arg)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Superpop frequencies file not found: {p}")

    cache = _CACHE_DIR / genome / "1kg_superpop_freq.tsv.gz"
    if cache.is_file():
        log.info("Using cached superpop frequencies: %s", cache)
        return cache

    if genome not in SUPERPOP_FREQS_URLS:
        raise FileNotFoundError(
            f"No cached superpop frequencies for {genome} and no download URL configured. "
            f"Provide a file via --superpop-freqs, or build one with "
            f"scripts/build_1kg_ref.py and place it at {cache}"
        )

    log.info("Superpop frequencies not found locally — downloading %s...", genome)
    return fetch_superpop_freqs(genome, dest=cache)


def fetch_superpop_freqs(genome: str = "GRCh38", dest: Path | None = None) -> Path:
    """Download the 1KG superpop allele-frequency TSV.

    Parameters
    ----------
    genome : genome build (GRCh38 or GRCh37)
    dest : target file path (default: ``~/.popout/superpop_freqs/{genome}/1kg_superpop_freq.tsv.gz``)

    Returns
    -------
    Path to the downloaded file.
    """
    if genome not in SUPERPOP_FREQS_URLS:
        raise ValueError(
            f"No download URL for genome build {genome!r}. "
            f"Available: {', '.join(SUPERPOP_FREQS_URLS) or '(none — build with scripts/build_1kg_ref.py)'}"
        )

    url = SUPERPOP_FREQS_URLS[genome]
    if dest is None:
        dest = _CACHE_DIR / genome / "1kg_superpop_freq.tsv.gz"

    dest.parent.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s superpop frequencies from %s", genome, url)
    data = urllib.request.urlopen(url).read()
    log.info("Downloaded %.1f MB", len(data) / 1e6)

    dest.write_bytes(data)
    log.info("Saved superpop frequencies to %s", dest)
    return dest


def load_superpop_frequencies(
    superpop_freqs_path: str | Path,
    chrom: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load 1KG superpopulation frequencies from TSV.

    Parameters
    ----------
    superpop_freqs_path : path to gzipped TSV
    chrom : if given, filter to this chromosome only

    Returns
    -------
    freq : (n_pops, n_sites) float32 array
    pos_bp : (n_sites,) int64 array
    pop_names : list of population names from the header
    """
    superpop_freqs_path = Path(superpop_freqs_path)
    positions = []
    freq_rows = []
    pop_names = None

    opener = gzip.open if superpop_freqs_path.suffix == ".gz" else open
    with opener(superpop_freqs_path, "rt") as f:
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
            f"No sites loaded from {superpop_freqs_path}"
            + (f" for chrom={chrom}" if chrom else "")
        )

    pos_bp = np.array(positions, dtype=np.int64)
    freq = np.array(freq_rows, dtype=np.float32).T  # (n_pops, n_sites)

    log.info("Loaded %d sites (%d populations)", len(positions), freq.shape[0])
    return freq, pos_bp, pop_names


def fetch_superpop_freqs_main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``popout fetch-superpop-freqs``."""
    parser = argparse.ArgumentParser(
        description="Download 1000 Genomes superpopulation allele-frequency TSV",
    )
    parser.add_argument(
        "--genome", choices=["GRCh38", "GRCh37"], default="GRCh38",
        help="Genome build (default: GRCh38)",
    )
    parser.add_argument(
        "--dest", default=None,
        help="Destination file path (default: ~/.popout/superpop_freqs/GENOME/1kg_superpop_freq.tsv.gz)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    dest = Path(args.dest) if args.dest else None
    result = fetch_superpop_freqs(args.genome, dest=dest)
    print(result)
