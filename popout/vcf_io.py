"""VCF I/O and genetic map loading.

Reads phased VCF files via pysam and constructs the binary haplotype
matrix that feeds the rest of the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Iterator

import numpy as np
import pysam

from .datatypes import ChromData, GeneticMap
from .gmap import normalise_chrom, load_genetic_map, load_genetic_map_per_chrom

log = logging.getLogger(__name__)

# Re-export for backward compatibility
_normalise_chrom = normalise_chrom


# ---------------------------------------------------------------------------
# VCF reading
# ---------------------------------------------------------------------------

def iter_chromosomes(
    vcf_path: str | Path,
    gmap: dict[str, GeneticMap],
    min_maf: float = 0.005,
    min_mac: int = 50,
    chromosomes: Optional[list[str]] = None,
    stats=None,
) -> Iterator[ChromData]:
    """Stream phased haplotype data one chromosome at a time.

    Parameters
    ----------
    vcf_path : path to indexed, phased VCF/BCF
    gmap : genetic maps keyed by chromosome
    min_maf : minimum minor allele frequency filter
    min_mac : minimum minor allele count filter
    chromosomes : restrict to these chromosomes (default: all autosomes)

    Yields
    ------
    ChromData for each chromosome.
    """
    vcf_path = str(vcf_path)
    vcf = pysam.VariantFile(vcf_path)
    n_samples = len(vcf.header.samples)
    n_haps = 2 * n_samples

    if chromosomes is None:
        chromosomes = [str(c) for c in range(1, 23)]
        # also try chr-prefixed
        contigs = set(vcf.header.contigs)
        if "chr1" in contigs:
            chromosomes = [f"chr{c}" for c in range(1, 23)]

    log.info("Reading %d samples (%d haplotypes) from %s", n_samples, n_haps, vcf_path)

    for chrom in chromosomes:
        chrom_norm = normalise_chrom(chrom)
        if chrom_norm not in gmap:
            log.warning("No genetic map for %s, skipping", chrom)
            continue

        geno_rows, pos_bp_list, site_ids = _read_chrom(
            vcf, chrom, n_samples, n_haps, min_maf, min_mac
        )

        if len(geno_rows) == 0:
            log.warning("No sites passed filters on %s", chrom)
            continue

        geno = np.array(geno_rows, dtype=np.uint8).T  # (n_haps, n_sites)
        pos_bp = np.array(pos_bp_list, dtype=np.int64)
        pos_cm = gmap[chrom_norm].interpolate(pos_bp)

        cd = ChromData(
            geno=geno,
            pos_bp=pos_bp,
            pos_cm=pos_cm,
            chrom=chrom,
            site_ids=np.array(site_ids) if site_ids else None,
        )
        cm_span = float(pos_cm[-1] - pos_cm[0]) if len(pos_cm) > 1 else 0.0
        log.info(
            "Chromosome %s: %d sites, %d haplotypes, %.1f cM",
            chrom, cd.n_sites, cd.n_haps, cm_span,
        )
        if stats is not None:
            stats.emit("io/sites_final", cd.n_sites, chrom=normalise_chrom(chrom))
            stats.emit("io/genetic_length_cm", round(cm_span, 2), chrom=normalise_chrom(chrom))
        yield cd

    vcf.close()


def _read_chrom(
    vcf: pysam.VariantFile,
    chrom: str,
    n_samples: int,
    n_haps: int,
    min_maf: float,
    min_mac: int,
) -> tuple[list[list[int]], list[int], list[str]]:
    """Read all biallelic SNPs on one chromosome, apply MAF/MAC filters."""
    geno_rows: list[list[int]] = []
    pos_bp_list: list[int] = []
    site_ids: list[str] = []
    n_skipped = 0

    try:
        records = vcf.fetch(chrom)
    except ValueError:
        return geno_rows, pos_bp_list, site_ids

    for rec in records:
        # Biallelic SNPs only
        if len(rec.alleles) != 2:
            n_skipped += 1
            continue
        if len(rec.alleles[0]) != 1 or len(rec.alleles[1]) != 1:
            n_skipped += 1
            continue

        # Extract phased alleles
        haps = _extract_haplotypes(rec, n_samples)
        if haps is None:
            n_skipped += 1
            continue

        # MAF / MAC filter
        ac = sum(haps)
        mac = min(ac, n_haps - ac)
        maf = mac / n_haps
        if maf < min_maf or mac < min_mac:
            n_skipped += 1
            continue

        geno_rows.append(haps)
        pos_bp_list.append(rec.pos)
        site_ids.append(rec.id or f"{chrom}:{rec.pos}")

    if n_skipped:
        log.debug("Chromosome %s: skipped %d sites", chrom, n_skipped)
    return geno_rows, pos_bp_list, site_ids


def _extract_haplotypes(rec: pysam.VariantRecord, n_samples: int) -> Optional[list[int]]:
    """Pull phased haplotypes from a VCF record.

    Returns a list of length 2*n_samples (hap0_sample0, hap1_sample0, ...),
    or None if any genotype is missing or unphased.
    """
    haps: list[int] = []
    for sample in rec.samples.values():
        gt = sample["GT"]
        if gt[0] is None or gt[1] is None:
            return None
        if not sample.phased:
            return None
        haps.append(int(gt[0]))
        haps.append(int(gt[1]))
    return haps
