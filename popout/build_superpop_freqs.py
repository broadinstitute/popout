"""Build 1KG superpopulation allele-frequency TSV from Phase 3 VCFs.

Generates the per-superpop frequency file used by ``popout label`` and
the priors framework's identity scoring. Reads 1000 Genomes Phase 3
VCFs and computes per-superpopulation allele frequencies for biallelic
SNPs.

Output format (gzipped TSV):
    #chrom  pos  ref  alt  EUR  EAS  AMR  AFR  SAS

Usage:
    popout build-superpop-freqs --vcf chr20.vcf.gz chr21.vcf.gz --out 1kg_superpop_freq.GRCh38.tsv.gz
    popout build-superpop-freqs --vcf-dir /path/to/1kg/vcfs --out 1kg_superpop_freq.GRCh38.tsv.gz
    popout build-superpop-freqs --download --genome GRCh38 --out 1kg_superpop_freq.GRCh38.tsv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# 1KG VCF download URLs per genome build
KG_VCF_URLS: dict[str, dict[str, str]] = {
    "GRCh37": {
        "base": (
            "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
        ),
        "pattern": (
            "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b"
            ".20130502.genotypes.vcf.gz"
        ),
    },
    "GRCh38": {
        "base": (
            "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/working/"
            "20220422_3202_phased_SNV_INDEL_SV"
        ),
        "pattern": (
            "1kGP_high_coverage_Illumina.chr{chrom}"
            ".filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        ),
    },
}

# 1KG Phase 3 superpopulations and their constituent populations
SUPERPOPS = {
    "EUR": ["CEU", "TSI", "FIN", "GBR", "IBS"],
    "EAS": ["CHB", "JPT", "CHS", "CDX", "KHV"],
    "AMR": ["MXL", "PUR", "CLM", "PEL"],
    "AFR": ["YRI", "LWK", "GWD", "MSL", "ESN", "ACB", "ASW"],
    "SAS": ["GIH", "PJL", "BEB", "STU", "ITU"],
}
SUPERPOP_ORDER = ["EUR", "EAS", "AMR", "AFR", "SAS"]

PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "integrated_call_samples_v3.20130502.ALL.panel"
)


def load_sample_panel(panel_path: str | Path | None = None) -> dict[str, str]:
    """Load sample → superpopulation mapping.

    Downloads the panel file if not provided.

    Returns
    -------
    dict mapping sample ID → superpopulation code (EUR, AFR, etc.)
    """
    if panel_path is None:
        log.info("Downloading sample panel from 1KG FTP...")
        data = urllib.request.urlopen(PANEL_URL).read().decode()
        lines = data.strip().split("\n")
    else:
        with open(panel_path) as f:
            lines = f.read().strip().split("\n")

    # Build population → superpopulation lookup
    pop_to_superpop = {}
    for sp, pops in SUPERPOPS.items():
        for p in pops:
            pop_to_superpop[p] = sp

    sample_to_superpop = {}
    for line in lines[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) >= 3:
            sample, pop, _superpop = parts[0], parts[1], parts[2]
            if pop in pop_to_superpop:
                sample_to_superpop[sample] = pop_to_superpop[pop]

    log.info("Loaded %d samples across %d superpopulations",
             len(sample_to_superpop), len(set(sample_to_superpop.values())))
    return sample_to_superpop


def process_vcf(
    vcf_path: str | Path,
    sample_superpop: dict[str, str],
    out_writer: csv.writer,
    min_global_maf: float = 0.01,
) -> int:
    """Process one VCF file and write frequency rows.

    Returns number of variants written.
    """
    import pysam

    vcf = pysam.VariantFile(str(vcf_path))
    samples = list(vcf.header.samples)

    # Build sample index → superpopulation
    superpop_indices: dict[str, list[int]] = {sp: [] for sp in SUPERPOP_ORDER}
    for i, s in enumerate(samples):
        sp = sample_superpop.get(s)
        if sp in superpop_indices:
            superpop_indices[sp].append(i)

    for sp in SUPERPOP_ORDER:
        log.info("  %s: %d samples", sp, len(superpop_indices[sp]))

    n_written = 0
    for rec in vcf:
        # Skip non-biallelic and non-SNP
        if len(rec.alts) != 1:
            continue
        if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
            continue

        chrom = rec.chrom
        pos = rec.pos
        ref = rec.ref
        alt = rec.alts[0]

        # Compute allele frequency per superpopulation
        freqs = []
        total_ac, total_an = 0, 0
        for sp in SUPERPOP_ORDER:
            ac, an = 0, 0
            for idx in superpop_indices[sp]:
                gt = rec.samples[samples[idx]]["GT"]
                if gt is None:
                    continue
                for allele in gt:
                    if allele is not None:
                        an += 1
                        if allele > 0:
                            ac += 1
            freq = ac / an if an > 0 else 0.0
            freqs.append(freq)
            total_ac += ac
            total_an += an

        # Filter by global MAF
        if total_an == 0:
            continue
        global_af = total_ac / total_an
        global_maf = min(global_af, 1.0 - global_af)
        if global_maf < min_global_maf:
            continue

        out_writer.writerow([chrom, pos, ref, alt] + [f"{f:.4f}" for f in freqs])
        n_written += 1

    vcf.close()
    return n_written


def download_kg_vcfs(
    genome: str = "GRCh38",
    dest_dir: Path | None = None,
    chromosomes: list[str] | None = None,
) -> list[Path]:
    """Download 1KG VCFs for the specified genome build.

    Parameters
    ----------
    genome : reference genome build
    dest_dir : directory to download into (default: tempdir)
    chromosomes : list of chromosome names (default: autosomes 1-22)

    Returns
    -------
    List of paths to downloaded VCF files.
    """
    if genome not in KG_VCF_URLS:
        raise ValueError(
            f"No VCF URLs for genome {genome!r}. "
            f"Available: {list(KG_VCF_URLS)}"
        )

    urls = KG_VCF_URLS[genome]
    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)]

    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix="popout_kg_"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for chrom in chromosomes:
        filename = urls["pattern"].format(chrom=chrom)
        url = f"{urls['base']}/{filename}"
        dest = dest_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            log.info("Already downloaded: %s", dest.name)
        else:
            log.info("Downloading chr%s from %s ...", chrom, urls["base"])
            with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            log.info("  -> %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        paths.append(dest)

    return paths


def build_superpop_freqs_main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``popout build-superpop-freqs``."""
    parser = argparse.ArgumentParser(
        description="Build 1KG superpopulation allele-frequency TSV for popout",
    )
    parser.add_argument(
        "--vcf", nargs="+",
        help="Input VCF file(s) (can use shell glob)",
    )
    parser.add_argument(
        "--vcf-dir",
        help="Directory containing per-chromosome VCF files",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Auto-download 1KG VCFs from public FTP (requires internet)",
    )
    parser.add_argument(
        "--genome", choices=["GRCh38", "GRCh37"], default="GRCh38",
        help="Genome build for --download (default: GRCh38)",
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Chromosomes to process (default: all autosomes 1-22)",
    )
    parser.add_argument(
        "--panel", default=None,
        help="1KG sample panel file (auto-downloaded if not provided)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output file path (e.g., 1kg_superpop_freq.GRCh38.tsv.gz)",
    )
    parser.add_argument(
        "--min-maf", type=float, default=0.01,
        help="Minimum global MAF filter (default: 0.01)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Collect VCF paths
    vcf_paths = []
    if args.download:
        log.info("Auto-downloading 1KG VCFs for %s ...", args.genome)
        vcf_paths = download_kg_vcfs(
            genome=args.genome,
            chromosomes=args.chromosomes,
        )
    elif args.vcf:
        vcf_paths = [Path(p) for p in args.vcf]
    elif args.vcf_dir:
        vcf_dir = Path(args.vcf_dir)
        vcf_paths = sorted(vcf_dir.glob("*.vcf.gz"))
    else:
        parser.error("Provide --vcf, --vcf-dir, or --download")

    if not vcf_paths:
        log.error("No VCF files found")
        sys.exit(1)

    log.info("Processing %d VCF file(s)", len(vcf_paths))

    # Load sample panel
    sample_superpop = load_sample_panel(args.panel)

    # Write output
    out_path = Path(args.out)
    opener = gzip.open if out_path.suffix == ".gz" else open

    total_written = 0
    with opener(out_path, "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["#chrom", "pos", "ref", "alt"] + SUPERPOP_ORDER)

        for vcf_path in vcf_paths:
            log.info("Processing %s", vcf_path.name)
            n = process_vcf(vcf_path, sample_superpop, writer, min_global_maf=args.min_maf)
            total_written += n
            log.info("  -> %d variants", n)

    log.info("Total: %d variants written to %s", total_written, out_path)


if __name__ == "__main__":
    build_superpop_freqs_main()
