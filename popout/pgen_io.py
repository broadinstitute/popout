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
                # Try shared .psam in directory. Sort so the choice is
                # filesystem-order-independent and reproducible.
                shared_psam = sorted(path.glob("*.psam"))
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
    must_keep_idxs: set[int] | None = None,
) -> np.ndarray:
    """Filter variant indices by MAF and MAC using pgenlib.count().

    ``must_keep_idxs`` (e.g. AIM panel positions) bypass the MAF/MAC
    threshold: those indices are always retained when present in
    ``variant_idxs``, even at cohort MAF below ``min_maf``. Used to
    protect AIM panel sites whose target-population frequencies are
    by design extreme in *some* superpops (so cohort MAF can be
    near-zero for non-target ancestries).

    Returns the subset of variant_idxs that pass both filters
    (or are in ``must_keep_idxs``).
    """
    n_haps = 2 * n_samples
    passing = []
    cnt_buf = np.empty(4, dtype=np.uint32)

    for i in range(0, len(variant_idxs), _COUNT_CHUNK):
        chunk = variant_idxs[i:i + _COUNT_CHUNK]
        for vidx in chunk:
            v_int = int(vidx)
            if must_keep_idxs is not None and v_int in must_keep_idxs:
                passing.append(v_int)
                continue
            reader.count(v_int, cnt_buf)
            hom_ref, het, hom_alt, missing = cnt_buf
            n_called = int(hom_ref + het + hom_alt)
            if n_called == 0:
                continue
            ac = int(het + 2 * hom_alt)
            total_alleles = 2 * n_called
            mac = min(ac, total_alleles - ac)
            maf = mac / total_alleles
            if maf >= min_maf and mac >= min_mac:
                passing.append(v_int)

    return np.array(passing, dtype=np.uint32)


def _thin_sites(
    pos_cm: np.ndarray,
    min_spacing_cm: float,
    must_keep_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean mask keeping sites spaced >= min_spacing_cm apart.

    Uses a greedy forward pass.  Always keeps the first and last site.

    ``must_keep_mask`` (e.g. AIM panel positions) overrides spacing:
    those sites are kept regardless of how close they are to other
    kept sites. They still update ``last_cm`` so subsequent sites
    measure spacing from the most recently kept site.
    """
    n = len(pos_cm)
    if n <= 2:
        return np.ones(n, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    last_cm = pos_cm[0]
    if must_keep_mask is not None and must_keep_mask[0]:
        # First site is already kept; nothing extra to do.
        pass

    for i in range(1, n):
        forced = must_keep_mask is not None and bool(must_keep_mask[i])
        if forced or pos_cm[i] - last_cm >= min_spacing_cm:
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
    protect_positions: Optional[dict[str, np.ndarray]] = None,
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
    protect_positions : optional ``{normalized_chrom: sorted pos_bp int64
        array}`` (e.g. from
        :func:`popout.prior_spec.panel_protect_positions`).  Listed
        positions bypass cM thinning and the MAF/MAC filter so AIM
        panel markers survive cohort-shape filtering.

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

        chrom_protect = (
            protect_positions.get(normalise_chrom(chrom))
            if protect_positions is not None else None
        )
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
            protect_positions=chrom_protect,
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
    protect_positions: Optional[np.ndarray] = None,
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

    # AIM-panel protection mask (aligned to pvar.pos_bp before thinning).
    must_keep_mask = (
        np.isin(pvar.pos_bp, protect_positions)
        if protect_positions is not None and len(protect_positions) > 0
        else None
    )
    if must_keep_mask is not None:
        n_protect = int(must_keep_mask.sum())
        log.info("  AIM panel protection: %d/%d sites flagged", n_protect, n_candidates)

    # Site thinning (before MAF filter to reduce count() calls)
    if thin_cm is not None:
        keep = _thin_sites(pos_cm, thin_cm, must_keep_mask=must_keep_mask)
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
        if must_keep_mask is not None:
            must_keep_mask = must_keep_mask[keep]

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
        must_keep_idxs = (
            {int(v) for v in pvar.variant_idx[must_keep_mask]}
            if must_keep_mask is not None else None
        )
        passing_idxs = _apply_maf_mac_filter(
            reader, pvar.variant_idx, n_samples, min_maf, min_mac,
            must_keep_idxs=must_keep_idxs,
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


# ---------------------------------------------------------------------------
# AIM panel sidecar PGEN (Phase 2 — option H)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelGeno:
    """Consolidated AIM-panel-only genotype block.

    Built upstream by ``extract_panel_geno.wdl`` from each chromosome's
    AIM panel positions × every cohort haplotype, then merged into a
    single multi-chromosome PGEN. ``read_panel_geno`` loads it whole.

    Multi-allelic positions in the cohort PGEN are split into per-alt
    biallelic rows by ``bcftools norm -m -any`` upstream, so a single
    ``(chrom, pos)`` may appear in multiple rows here, each with a
    different alt allele. The ref/alt arrays let the consumer resolve
    rows by matching the panel TSV's expected ``(ref, alt)``.

    Attributes
    ----------
    geno : (H, M_panel) uint8
        Alt-allele dosage per (haplotype, panel locus). Haplotype order
        matches the seed chromosome's PGEN psam (asserted at load).
    chrom : (M_panel,) object — str per locus (normalized, no "chr" prefix)
    pos_bp : (M_panel,) int64
    ref : (M_panel,) object — single-character ref allele
    alt : (M_panel,) object — single-character alt allele
    """

    geno: np.ndarray
    chrom: np.ndarray
    pos_bp: np.ndarray
    ref: np.ndarray
    alt: np.ndarray


def read_panel_geno(
    pgen_prefix: str | Path,
    expected_sample_iids: list[str],
) -> PanelGeno:
    """Open the consolidated AIM-panel-only PGEN and return a haplotype-
    aligned :class:`PanelGeno`.

    The panel PGEN must have been built from the same upstream cohort as
    the seed-chrom PGEN — its psam must list samples in identical order.
    Mismatch raises :class:`ValueError` immediately; there is no safe
    way to reorder genotypes after the fact.

    Parameters
    ----------
    pgen_prefix : path to the panel ``.pgen`` (with sibling ``.pvar`` and
        ``.psam``) or its prefix without suffix.
    expected_sample_iids : sample IIDs from the seed-chrom psam, in
        cohort order.
    """
    p = Path(pgen_prefix)
    pgen_path = p if p.suffix == ".pgen" else Path(str(p) + ".pgen")
    if not pgen_path.exists():
        raise FileNotFoundError(f"Panel PGEN not found: {pgen_path}")
    stem = str(pgen_path)[: -len(".pgen")]
    pvar_path = _find_pvar_str(stem)
    if pvar_path is None:
        raise FileNotFoundError(f"No .pvar for panel PGEN prefix {stem}")
    psam_path = Path(stem + ".psam")
    if not psam_path.exists():
        raise FileNotFoundError(f"No .psam for panel PGEN prefix {stem}")

    # Hap-set check + permutation: the panel and cohort psams must
    # contain the SAME set of sample IIDs. plink2 --pmerge-list
    # reorders samples by IID, so the panel psam is typically NOT in
    # cohort order. We build a permutation (panel-row → cohort-row)
    # and apply it to the geno matrix after read.
    actual_iids = _parse_psam(psam_path)
    expected_list = list(expected_sample_iids)
    panel_set = set(actual_iids)
    cohort_set = set(expected_list)

    if len(panel_set) != len(actual_iids):
        raise ValueError(
            f"Panel psam {psam_path} contains duplicate IIDs "
            f"({len(actual_iids)} rows but only {len(panel_set)} unique)."
        )
    if len(cohort_set) != len(expected_list):
        raise ValueError(
            f"Cohort psam contains duplicate IIDs "
            f"({len(expected_list)} rows but only {len(cohort_set)} unique)."
        )

    if panel_set != cohort_set:
        missing_in_panel = sorted(cohort_set - panel_set)[:5]
        missing_in_cohort = sorted(panel_set - cohort_set)[:5]
        raise ValueError(
            f"Panel PGEN sample SET does not match cohort PGEN.\n"
            f"  panel psam: {psam_path}  ({len(actual_iids)} samples)\n"
            f"  cohort psam: {len(expected_list)} samples\n"
            f"  in cohort but not panel ({len(cohort_set - panel_set)} total, "
            f"first 5): {missing_in_panel}\n"
            f"  in panel but not cohort ({len(panel_set - cohort_set)} total, "
            f"first 5): {missing_in_cohort}\n"
            f"Re-run extract_panel_geno.wdl against the same cohort."
        )

    n_samples = len(actual_iids)
    n_haps = 2 * n_samples

    # Build the haplotype-row permutation: cohort hap row 2*i, 2*i+1
    # comes from panel hap row 2*j, 2*j+1 where j is panel's index
    # for cohort_iids[i]. Same-set check above guarantees every iid
    # is present.
    panel_iid_to_idx = {iid: j for j, iid in enumerate(actual_iids)}
    sample_perm = np.array(
        [panel_iid_to_idx[iid] for iid in expected_list], dtype=np.int64,
    )
    if not np.array_equal(sample_perm, np.arange(n_samples)):
        log.info(
            "Panel psam in different order than cohort psam; building "
            "haplotype permutation (panel → cohort).",
        )
    hap_perm = np.empty(n_haps, dtype=np.int64)
    hap_perm[0::2] = 2 * sample_perm
    hap_perm[1::2] = 2 * sample_perm + 1

    # Parse the pvar — multi-chrom panel is supported (all chroms'
    # AIM positions live in one file).
    pvar_data = _parse_pvar(pvar_path)
    if not pvar_data:
        raise ValueError(
            f"Panel PGEN {pvar_path} has no biallelic SNP variants. "
            f"Phase 2 cannot proceed without panel positions."
        )

    # Concatenate per-chrom records into parallel arrays sorted by
    # variant_idx (= pgen file order). Reading in pgen-file order
    # avoids any internal sort by pgenlib.
    all_var_idx: list[int] = []
    all_chrom: list[str] = []
    all_pos: list[int] = []
    all_ref: list[str] = []
    all_alt: list[str] = []
    for chrom_norm, rec in pvar_data.items():
        for vi, p, r, a in zip(
            rec.variant_idx.tolist(), rec.pos_bp.tolist(), rec.ref, rec.alt,
        ):
            all_var_idx.append(int(vi))
            all_chrom.append(chrom_norm)
            all_pos.append(int(p))
            all_ref.append(str(r))
            all_alt.append(str(a))
    order = np.argsort(np.array(all_var_idx, dtype=np.uint32))
    var_idx_arr = np.array([all_var_idx[i] for i in order], dtype=np.uint32)
    chrom_arr = np.array([all_chrom[i] for i in order], dtype=object)
    pos_arr = np.array([all_pos[i] for i in order], dtype=np.int64)
    ref_arr = np.array([all_ref[i] for i in order], dtype=object)
    alt_arr = np.array([all_alt[i] for i in order], dtype=object)
    n_panel = len(var_idx_arr)
    log.info(
        "Loaded panel PGEN %s: %d biallelic panel positions across %d chroms",
        pgen_path.name, n_panel, len(set(chrom_arr.tolist())),
    )

    try:
        reader = pgenlib.PgenReader(bytes(str(pgen_path), encoding="utf-8"))
    except RuntimeError as e:
        if "multiallelic" in str(e).lower() or "allele_idx_offsets" in str(e).lower():
            raise RuntimeError(
                f"Panel PGEN {pgen_path.name} contains multiallelic variants, "
                "which pgenlib cannot read with phased data. Re-run "
                "extract_panel_geno.wdl with --max-alleles 2 in the per-chrom "
                "extract step (panel positions multi-allelic in the cohort "
                "are dropped from the sidecar)."
            ) from e
        raise

    if not reader.hardcall_phase_present():
        reader.close()
        raise ValueError(
            f"Panel PGEN {pgen_path} does not contain phased genotypes. "
            "Phase 2 reads alt-dosage per haplotype; unphased panel input "
            "is not supported."
        )

    try:
        geno, site_ok = _read_genotypes(reader, var_idx_arr, n_haps)
    finally:
        reader.close()

    # Drop sites with missing data (consistent with cohort read).
    if not bool(site_ok.all()):
        chrom_arr = chrom_arr[site_ok]
        pos_arr = pos_arr[site_ok]
        ref_arr = ref_arr[site_ok]
        alt_arr = alt_arr[site_ok]
        log.info(
            "Panel PGEN: dropped %d sites with missing genotypes; %d remain",
            int((~site_ok).sum()), len(chrom_arr),
        )

    # Apply the panel→cohort hap permutation. fancy-indexing copies
    # contiguously; at biobank scale (~1M haps × ~80 panel sites × 1
    # byte) this is ~80 MB and runs in a fraction of a second.
    if not np.array_equal(hap_perm, np.arange(n_haps)):
        geno = geno[hap_perm, :]

    log.info(
        "Panel PGEN ready: %d haps × %d sites (sample order matches cohort)",
        n_haps, len(chrom_arr),
    )
    return PanelGeno(
        geno=geno, chrom=chrom_arr, pos_bp=pos_arr,
        ref=ref_arr, alt=alt_arr,
    )
