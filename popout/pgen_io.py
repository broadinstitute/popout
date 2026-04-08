"""PGEN I/O for biobank-scale phased data.

Reads PLINK2 PGEN/PVAR/PSAM file sets and constructs the binary haplotype
matrix that feeds the rest of the pipeline.  Designed for per-chromosome
file layouts (e.g. AoU) and WGS-scale data with optional site thinning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pgenlib

from .datatypes import ChromData, GeneticMap
from .gmap import normalise_chrom

log = logging.getLogger(__name__)

# Number of variants to read per batch to limit int32 memory overhead
_READ_CHUNK = 2000
# Number of variants to scan per batch for MAF filtering
_COUNT_CHUNK = 50_000


# ---------------------------------------------------------------------------
# .pvar / .psam parsing
# ---------------------------------------------------------------------------

@dataclass
class _PvarRecord:
    """Parsed .pvar metadata for candidate variants on one chromosome."""

    variant_idx: np.ndarray   # (N,) uint32 — index into the .pgen file
    pos_bp: np.ndarray        # (N,) int64
    site_ids: list[str]       # (N,)
    ref: list[str]            # (N,)
    alt: list[str]            # (N,)


def _parse_pvar(
    pvar_path: Path,
    chromosomes: Optional[set[str]] = None,
) -> dict[str, _PvarRecord]:
    """Parse a .pvar file, returning per-chromosome variant metadata.

    Only biallelic SNPs (single-char REF and ALT) are included.

    Parameters
    ----------
    pvar_path : path to .pvar or .pvar.zst file
    chromosomes : if provided, restrict to these chromosomes (normalised names)

    Returns
    -------
    dict mapping normalised chromosome name → _PvarRecord
    """
    records: dict[str, dict] = {}  # chrom → {variant_idx, pos_bp, ids, ref, alt}

    with open(pvar_path) as fh:
        # Skip ## header lines, find the #CHROM line
        col_names = None
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                col_names = line.lstrip("#").strip().split("\t")
                break

        if col_names is None:
            raise ValueError(f"No header line found in {pvar_path}")

        # Find column indices
        col_idx = {name: i for i, name in enumerate(col_names)}
        chrom_col = col_idx.get("CHROM", col_idx.get("chrom", 0))
        pos_col = col_idx.get("POS", col_idx.get("pos", 1))
        id_col = col_idx.get("ID", col_idx.get("id", 2))
        ref_col = col_idx.get("REF", col_idx.get("ref", 3))
        alt_col = col_idx.get("ALT", col_idx.get("alt", 4))

        variant_idx = 0
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            chrom_raw = parts[chrom_col]
            chrom = normalise_chrom(chrom_raw)

            if chromosomes is not None and chrom not in chromosomes:
                variant_idx += 1
                continue

            ref = parts[ref_col]
            alt = parts[alt_col]

            # Biallelic SNPs only: single-char REF and ALT, no commas in ALT
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                variant_idx += 1
                continue

            if chrom not in records:
                records[chrom] = {
                    "variant_idx": [], "pos_bp": [], "ids": [],
                    "ref": [], "alt": [],
                }
            rec = records[chrom]
            rec["variant_idx"].append(variant_idx)
            rec["pos_bp"].append(int(parts[pos_col]))
            rec["ids"].append(parts[id_col] if parts[id_col] != "." else f"{chrom_raw}:{parts[pos_col]}")
            rec["ref"].append(ref)
            rec["alt"].append(alt)

            variant_idx += 1

    result = {}
    for chrom, rec in records.items():
        result[chrom] = _PvarRecord(
            variant_idx=np.array(rec["variant_idx"], dtype=np.uint32),
            pos_bp=np.array(rec["pos_bp"], dtype=np.int64),
            site_ids=rec["ids"],
            ref=rec["ref"],
            alt=rec["alt"],
        )
    return result


def _parse_psam(psam_path: Path) -> list[str]:
    """Parse a .psam file, returning sample IIDs in order."""
    samples: list[str] = []
    iid_col = 0  # default

    with open(psam_path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                cols = line.lstrip("#").strip().split("\t")
                # Find IID column (might be FID\tIID or just IID)
                if "IID" in cols:
                    iid_col = cols.index("IID")
                elif "iid" in cols:
                    iid_col = cols.index("iid")
                elif len(cols) >= 2:
                    iid_col = 1  # FID IID convention
                continue

            parts = line.strip().split("\t")
            if parts:
                samples.append(parts[iid_col])

    return samples


def get_sample_names(psam_path: str | Path) -> list[str]:
    """Public API to get sample names from a .psam file."""
    return _parse_psam(Path(psam_path))


# ---------------------------------------------------------------------------
# Per-chromosome PGEN file discovery
# ---------------------------------------------------------------------------

def _find_pgen_files(
    path: str | Path,
    chromosomes: Optional[list[str]] = None,
) -> dict[str, tuple[Path, Path, Path]]:
    """Discover per-chromosome PGEN file sets.

    Parameters
    ----------
    path : directory containing per-chromosome PGEN files,
           or a single prefix (e.g. 'data/chr1' → data/chr1.pgen)
    chromosomes : restrict to these chromosome names

    Returns
    -------
    dict mapping chromosome name → (pgen_path, pvar_path, psam_path)
    """
    path = Path(path)
    result: dict[str, tuple[Path, Path, Path]] = {}

    if path.is_dir():
        # Discover per-chromosome files in directory
        pgen_files = sorted(path.glob("*.pgen"))
        if not pgen_files:
            raise FileNotFoundError(f"No .pgen files found in {path}")

        for pgen in pgen_files:
            # Use string replacement instead of Path.with_suffix() to handle
            # multi-dot filenames like chr20.aou.v9.phased.pgen correctly.
            pgen_str = str(pgen)
            stem = pgen_str[:-len(".pgen")]
            pvar = _find_pvar_str(stem)
            psam = Path(stem + ".psam")
            if not psam.exists():
                # Try shared .psam in directory
                shared_psam = list(path.glob("*.psam"))
                if shared_psam:
                    psam = shared_psam[0]

            if pvar is None or not psam.exists():
                log.warning("Incomplete file set for %s, skipping", pgen)
                continue

            # Extract chromosome from .pvar content (first data line)
            chrom = _chrom_from_pvar(pvar)
            if chrom is None:
                log.warning("Cannot determine chromosome for %s, skipping", pvar)
                continue

            chrom_norm = normalise_chrom(chrom)
            if chromosomes is not None:
                chroms_norm = {normalise_chrom(c) for c in chromosomes}
                if chrom_norm not in chroms_norm:
                    continue

            result[chrom_norm] = (pgen, pvar, psam)
    else:
        # Single prefix: path is like 'data/cohort' or 'data/chr1'
        pgen = path.with_suffix(".pgen") if not path.suffix == ".pgen" else path
        if not pgen.exists():
            raise FileNotFoundError(f"PGEN file not found: {pgen}")
        stem = str(pgen)[:-len(".pgen")]
        pvar = _find_pvar_str(stem)
        psam = Path(stem + ".psam")

        if pvar is None:
            raise FileNotFoundError(f"No .pvar file found for prefix {stem}")
        if not psam.exists():
            raise FileNotFoundError(f"No .psam file found: {psam}")

        # Single file may contain multiple chromosomes
        chrom = _chrom_from_pvar(pvar)
        if chrom:
            result[normalise_chrom(chrom)] = (pgen, pvar, psam)
        else:
            # Multi-chromosome file — use "all" as key, iter_chromosomes will handle
            result["_multi"] = (pgen, pvar, psam)

    return result


def _find_pvar_str(stem: str) -> Optional[Path]:
    """Find .pvar or .pvar.zst for a stem string (handles multi-dot names)."""
    for suffix in [".pvar", ".pvar.zst"]:
        p = Path(stem + suffix)
        if p.exists():
            return p
    return None


def _chrom_from_pvar(pvar_path: Path) -> Optional[str]:
    """Extract chromosome name from first data line of .pvar."""
    with open(pvar_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 2)
            if parts:
                return parts[0]
    return None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _apply_maf_mac_filter(
    reader: pgenlib.PgenReader,
    variant_idxs: np.ndarray,
    n_samples: int,
    min_maf: float,
    min_mac: int,
) -> np.ndarray:
    """Filter variant indices by MAF and MAC using pgenlib.count().

    Returns the subset of variant_idxs that pass both filters.
    """
    n_haps = 2 * n_samples
    passing = []
    cnt_buf = np.empty(4, dtype=np.uint32)

    for i in range(0, len(variant_idxs), _COUNT_CHUNK):
        chunk = variant_idxs[i:i + _COUNT_CHUNK]
        for vidx in chunk:
            reader.count(int(vidx), cnt_buf)
            hom_ref, het, hom_alt, missing = cnt_buf
            n_called = int(hom_ref + het + hom_alt)
            if n_called == 0:
                continue
            ac = int(het + 2 * hom_alt)
            total_alleles = 2 * n_called
            mac = min(ac, total_alleles - ac)
            maf = mac / total_alleles
            if maf >= min_maf and mac >= min_mac:
                passing.append(int(vidx))

    return np.array(passing, dtype=np.uint32)


def _thin_sites(pos_cm: np.ndarray, min_spacing_cm: float) -> np.ndarray:
    """Return boolean mask keeping sites spaced >= min_spacing_cm apart.

    Uses a greedy forward pass.  Always keeps the first and last site.
    """
    n = len(pos_cm)
    if n <= 2:
        return np.ones(n, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    last_cm = pos_cm[0]

    for i in range(1, n):
        if pos_cm[i] - last_cm >= min_spacing_cm:
            keep[i] = True
            last_cm = pos_cm[i]

    # Always keep the last site
    keep[-1] = True
    return keep


# ---------------------------------------------------------------------------
# Genotype reading
# ---------------------------------------------------------------------------

def _read_genotypes(
    reader: pgenlib.PgenReader,
    variant_idxs: np.ndarray,
    n_haps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read phased alleles for selected variants.

    Reads in chunks to limit int32 memory overhead.  Uses variant-major
    mode (hap_maj=0) to avoid a pgenlib phase-corruption bug that affects
    hap_maj=1 for large reads, then transposes each chunk.

    Drops sites with any missing alleles (-9).

    Returns
    -------
    geno : (n_haps, n_passing_sites) uint8
    site_ok : (n_sites,) bool — mask of sites without missing data
    """
    n_sites = len(variant_idxs)
    if n_sites == 0:
        return np.empty((n_haps, 0), dtype=np.uint8), np.array([], dtype=bool)

    # Pre-allocate output: (n_haps, n_sites) uint8
    geno = np.empty((n_haps, n_sites), dtype=np.uint8)
    site_ok = np.ones(n_sites, dtype=bool)
    n_missing_sites = 0

    for start in range(0, n_sites, _READ_CHUNK):
        end = min(start + _READ_CHUNK, n_sites)
        chunk_idxs = variant_idxs[start:end]
        n_chunk = len(chunk_idxs)

        # Read variant-major: (n_chunk, n_haps), then transpose
        buf = np.empty((n_chunk, n_haps), dtype=np.int32)
        reader.read_alleles_list(chunk_idxs, buf, hap_maj=0)

        # Check for missing alleles (-9)
        missing_mask = buf < 0
        if missing_mask.any():
            sites_with_missing = missing_mask.any(axis=1)  # (n_chunk,)
            n_bad = int(sites_with_missing.sum())
            n_missing_sites += n_bad
            site_ok[start:end] &= ~sites_with_missing
            buf[missing_mask] = 0

        # Transpose to hap-major and cast
        geno[:, start:end] = buf.T.astype(np.uint8)

    if n_missing_sites > 0:
        log.info("  Dropped %d sites with missing genotypes", n_missing_sites)
        geno = geno[:, site_ok]

    return geno, site_ok


# ---------------------------------------------------------------------------
# Main iterator
# ---------------------------------------------------------------------------

def iter_chromosomes(
    path: str | Path,
    gmap: dict[str, GeneticMap],
    min_maf: float = 0.0,
    min_mac: int = 0,
    chromosomes: Optional[list[str]] = None,
    thin_cm: Optional[float] = None,
    stats=None,
) -> Iterator[ChromData]:
    """Stream phased haplotype data one chromosome at a time from PGEN files.

    Parameters
    ----------
    path : directory of per-chromosome PGEN files, or a single prefix
    gmap : genetic maps keyed by normalised chromosome name
    min_maf : minimum minor allele frequency filter (default 0 = skip;
              plink2 should handle MAF/MAC filtering before popout)
    min_mac : minimum minor allele count filter (default 0 = skip)
    chromosomes : restrict to these chromosomes (default: autosomes 1-22)
    thin_cm : if set, thin sites to this minimum cM spacing (e.g. 0.02 for WGS)

    Yields
    ------
    ChromData for each chromosome.
    """
    path = Path(path)

    # Discover per-chromosome files
    pgen_files = _find_pgen_files(path, chromosomes)
    if not pgen_files:
        raise FileNotFoundError(f"No PGEN files found at {path}")

    # Determine processing order (autosomes 1-22 by default)
    if chromosomes is not None:
        chrom_order = [normalise_chrom(c) for c in chromosomes]
    else:
        # Sort numerically for autosomes
        available = sorted(
            pgen_files.keys(),
            key=lambda c: (int(c) if c.isdigit() else 99, c),
        )
        chrom_order = available

    # Get sample count from first .psam
    first_chrom = next(c for c in chrom_order if c in pgen_files)
    _, _, psam_path = pgen_files[first_chrom]
    sample_names = _parse_psam(psam_path)
    n_samples = len(sample_names)
    n_haps = 2 * n_samples
    log.info("PGEN input: %d samples (%d haplotypes)", n_samples, n_haps)
    if thin_cm is not None:
        log.info("Site thinning: %.3f cM minimum spacing", thin_cm)

    for chrom in chrom_order:
        if chrom not in pgen_files:
            log.warning("No PGEN file for chromosome %s, skipping", chrom)
            continue

        if chrom not in gmap:
            log.warning("No genetic map for chromosome %s, skipping", chrom)
            continue

        pgen_path, pvar_path, _ = pgen_files[chrom]
        log.info("Reading chromosome %s from %s", chrom, pgen_path.name)

        if stats is not None:
            stats.timer_start(f"io/chr{chrom}")

        cd = _read_one_chromosome(
            pgen_path=pgen_path,
            pvar_path=pvar_path,
            chrom=chrom,
            gmap=gmap[chrom],
            n_samples=n_samples,
            n_haps=n_haps,
            min_maf=min_maf,
            min_mac=min_mac,
            thin_cm=thin_cm,
            stats=stats,
        )

        if stats is not None:
            stats.timer_stop(f"io/chr{chrom}", chrom=chrom)

        if cd is None:
            continue

        yield cd


def _read_one_chromosome(
    pgen_path: Path,
    pvar_path: Path,
    chrom: str,
    gmap: GeneticMap,
    n_samples: int,
    n_haps: int,
    min_maf: float,
    min_mac: int,
    thin_cm: Optional[float],
    stats=None,
) -> Optional[ChromData]:
    """Read and filter one chromosome from a PGEN file set.

    Assumes the PGEN file contains only biallelic variants.  Multiallelic
    PGENs will crash pgenlib (v0.94 doesn't support multiallelic+phase).
    The WDL pre-filters with: plink2 --max-alleles 2 --make-pgen
    """

    # --- Pass 1: parse .pvar for biallelic SNP candidates ---
    chrom_set = {chrom}
    pvar_data = _parse_pvar(pvar_path, chromosomes=chrom_set)
    if chrom not in pvar_data:
        log.warning("  No biallelic SNPs found in %s", pvar_path)
        return None

    pvar = pvar_data[chrom]
    n_candidates = len(pvar.variant_idx)
    log.info("  %d biallelic SNP candidates", n_candidates)
    if stats is not None:
        stats.emit("io/sites_biallelic", n_candidates, chrom=chrom)

    # Interpolate genetic positions
    pos_cm = gmap.interpolate(pvar.pos_bp)

    # Site thinning (before MAF filter to reduce count() calls)
    if thin_cm is not None:
        keep = _thin_sites(pos_cm, thin_cm)
        n_thinned = int(keep.sum())
        log.info("  After thinning (%.3f cM): %d → %d sites", thin_cm, n_candidates, n_thinned)
        if stats is not None:
            stats.emit("io/sites_after_thinning", n_thinned, chrom=chrom)
        pvar = _PvarRecord(
            variant_idx=pvar.variant_idx[keep],
            pos_bp=pvar.pos_bp[keep],
            site_ids=[s for s, k in zip(pvar.site_ids, keep) if k],
            ref=[r for r, k in zip(pvar.ref, keep) if k],
            alt=[a for a, k in zip(pvar.alt, keep) if k],
        )
        pos_cm = pos_cm[keep]

    # --- Pass 1b: MAF/MAC filtering ---
    try:
        reader = pgenlib.PgenReader(bytes(str(pgen_path), encoding="utf-8"))
    except RuntimeError as e:
        if "multiallelic" in str(e).lower() or "allele_idx_offsets" in str(e).lower():
            raise RuntimeError(
                f"PGEN file {pgen_path.name} contains multiallelic variants, "
                "which pgenlib cannot read with phased data. Pre-filter with:\n"
                "  plink2 --pfile <prefix> --max-alleles 2 --make-pgen --out <prefix_biallelic>\n"
                "The popout WDL does this automatically."
            ) from e
        raise

    # Validate phase
    if not reader.hardcall_phase_present():
        reader.close()

        raise ValueError(
            f"PGEN file {pgen_path} does not contain phased genotypes. "
            "popout requires phased input. Re-run phasing (e.g. SHAPEIT5) or "
            "convert with: plink2 --vcf phased.vcf.gz --make-pgen phased-list"
        )

    if min_maf > 0 or min_mac > 0:
        passing_idxs = _apply_maf_mac_filter(
            reader, pvar.variant_idx, n_samples, min_maf, min_mac,
        )
        n_passing = len(passing_idxs)
        log.info("  After MAF/MAC filter: %d sites", n_passing)
        if stats is not None:
            stats.emit("io/sites_after_maf_mac", n_passing, chrom=chrom)

        if n_passing == 0:
            reader.close()
            log.warning("  No sites passed filters on chromosome %s", chrom)
            return None

        # Build index mapping: passing_idxs → positions in original pvar arrays
        passing_set = set(passing_idxs.tolist())
        keep_mask = np.array([int(v) in passing_set for v in pvar.variant_idx], dtype=bool)
        final_pos_bp = pvar.pos_bp[keep_mask]
        final_pos_cm = pos_cm[keep_mask]
        final_site_ids = [s for s, k in zip(pvar.site_ids, keep_mask) if k]
    else:
        log.info("  MAF/MAC filter skipped (plink2 pre-filtered)")
        passing_idxs = pvar.variant_idx
        final_pos_bp = pvar.pos_bp
        final_pos_cm = pos_cm
        final_site_ids = list(pvar.site_ids)

    # --- Pass 2: read phased genotypes ---
    geno, site_ok = _read_genotypes(reader, passing_idxs, n_haps)
    reader.close()

    # If some sites were dropped for missing data, filter metadata too
    if not site_ok.all():
        final_pos_bp = final_pos_bp[site_ok]
        final_pos_cm = final_pos_cm[site_ok]
        final_site_ids = [s for s, k in zip(final_site_ids, site_ok) if k]

    if geno.shape[1] == 0:
        log.warning("  No sites remaining after missing-data filter on chromosome %s", chrom)
        return None

    cd = ChromData(
        geno=geno,
        pos_bp=final_pos_bp,
        pos_cm=final_pos_cm,
        chrom=chrom,
        site_ids=np.array(final_site_ids) if final_site_ids else None,
    )
    cm_span = float(final_pos_cm[-1] - final_pos_cm[0]) if len(final_pos_cm) > 1 else 0.0
    log.info(
        "  Chromosome %s: %d sites, %d haplotypes, %.1f cM",
        chrom, cd.n_sites, cd.n_haps, cm_span,
    )
    if stats is not None:
        stats.emit("io/sites_final", cd.n_sites, chrom=chrom)
        stats.emit("io/genetic_length_cm", round(cm_span, 2), chrom=chrom)
    return cd
