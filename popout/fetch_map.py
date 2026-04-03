"""Download and cache Beagle/plink genetic recombination maps.

Maps are per-chromosome HapMap-format files distributed as zips from
the Browning lab.  This module handles downloading, extracting, caching,
and resolving the map directory for use by popout and vcf2pgen.

Usage:
    popout fetch-map [--genome GRCh38|GRCh37|GRCh36] [--dest DIR]
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

MAP_URLS = {
    "GRCh38": "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip",
    "GRCh37": "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh37.map.zip",
    "GRCh36": "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh36.map.zip",
}

# Where to look for pre-existing maps
_CACHE_DIR = Path.home() / ".popout" / "maps"


def resolve_map_dir(
    genome: str = "GRCh38",
    map_arg: str | None = None,
) -> Path:
    """Resolve genetic map to a directory of .map files.

    Priority:
        1. Explicit ``map_arg`` (file or directory path)
        2. Local cache (``~/.popout/maps/{genome}/``)
        3. Auto-download to local cache

    Returns
    -------
    Path to a directory containing per-chromosome ``.map`` files.
    """
    # 1. Explicit path
    if map_arg is not None:
        p = Path(map_arg)
        if p.is_dir():
            return p
        if p.is_file():
            return p  # single file — caller handles via load_genetic_map()
        raise FileNotFoundError(f"--map path does not exist: {p}")

    # 2. Cache
    cache = _CACHE_DIR / genome
    if cache.is_dir() and any(cache.glob("*.map")):
        log.info("Using cached genetic map: %s", cache)
        return cache

    # 3. Auto-download
    log.info("Genetic map not found locally — downloading %s maps...", genome)
    return fetch_map(genome, dest=cache)


def fetch_map(genome: str = "GRCh38", dest: Path | None = None) -> Path:
    """Download and extract genetic map zip.

    Parameters
    ----------
    genome : reference genome build (GRCh38, GRCh37, GRCh36)
    dest : target directory (default: ``~/.popout/maps/{genome}/``)

    Returns
    -------
    Path to directory containing extracted ``.map`` files.
    """
    if genome not in MAP_URLS:
        raise ValueError(
            f"Unknown genome build: {genome!r}. "
            f"Choose from: {', '.join(MAP_URLS)}"
        )

    url = MAP_URLS[genome]
    if dest is None:
        dest = _CACHE_DIR / genome

    dest.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s from %s", genome, url)
    data = urllib.request.urlopen(url).read()
    log.info("Downloaded %.1f MB", len(data) / 1e6)

    # Extract only the no_chr_in_chrom_field/ .map files (normalised names)
    z = zipfile.ZipFile(io.BytesIO(data))
    extracted = 0
    for member in z.namelist():
        if member.startswith("no_chr_in_chrom_field/") and member.endswith(".map"):
            filename = Path(member).name
            with z.open(member) as src, open(dest / filename, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    log.info("Extracted %d map files to %s", extracted, dest)
    return dest


def fetch_map_main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``popout fetch-map``."""
    parser = argparse.ArgumentParser(
        description="Download Beagle/plink genetic recombination maps",
    )
    parser.add_argument(
        "--genome", choices=list(MAP_URLS), default="GRCh38",
        help="Reference genome build (default: GRCh38)",
    )
    parser.add_argument(
        "--dest", default=None,
        help=f"Destination directory (default: ~/.popout/maps/GENOME/)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    dest = Path(args.dest) if args.dest else None
    result = fetch_map(args.genome, dest=dest)
    print(result)
