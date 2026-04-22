"""Convert popout native outputs to FLARE-compatible ancestry VCF.

Input:
  {prefix}.tracts.tsv.gz          per-tract calls (required)
  {prefix}.model.npz              ancestry_names (optional; overridden by --ancestry-names)
  {prefix}.chr{N}.decode.parquet   per-chrom pos_bp, calls, optional max_post

Output:
  {out}.anc.vcf.gz                per-site AN1/AN2 (and ANP1/ANP2 if --probs)
  {out}.global.anc.gz             per-sample ancestry fractions

The output format follows FLARE (browning-lab/flare) so that downstream
tools expecting FLARE output interoperate with popout results.

ANP1/ANP2 are K-valued per hap: max_post at the called ancestry,
(1-max_post)/(K-1) at the others. This preserves FLARE's tuple shape;
the approximation is documented in the FORMAT Description.
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pysam

log = logging.getLogger(__name__)


def convert_to_vcf(args) -> None:
    """Main entry point for 'popout convert --to vcf'."""
    prefix = Path(args.popout_prefix)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve ancestry names.
    ancestry_names = _resolve_ancestry_names(args, prefix)
    K = len(ancestry_names)
    log.info("Ancestry names (K=%d): %s", K, ancestry_names)

    # 2. Determine which chromosomes to convert.
    available_chroms = _scan_decode_parquet_chroms(prefix)
    if not available_chroms:
        raise ValueError(
            f"No decode.parquet files found matching {prefix}.chr*.decode.parquet. "
            f"Run popout with --write-dense-decode or --probs first."
        )
    requested = (
        [c.strip() for c in args.chroms.split(",")]
        if args.chroms else available_chroms
    )
    missing = set(requested) - set(available_chroms)
    if missing:
        raise ValueError(
            f"Requested chromosomes not found in popout outputs: {sorted(missing)}. "
            f"Available: {available_chroms}"
        )

    # 3. Open input VCF, build output header.
    vcf_in = pysam.VariantFile(args.input_vcf)
    header = _build_output_header(
        vcf_in.header, ancestry_names, write_probs=args.probs,
    )
    vcf_out = pysam.VariantFile(str(out_path), "wz", header=header)

    total_written = 0
    total_skipped = 0
    for chrom in requested:
        decode_path = prefix.parent / f"{prefix.name}.chr{chrom}.decode.parquet"
        written, skipped = _convert_chrom(
            vcf_in, vcf_out, decode_path, chrom,
            write_probs=args.probs,
            thinned_sites_mode=args.thinned_sites,
            n_ancestries=K,
        )
        total_written += written
        total_skipped += skipped
        log.info(
            "Chromosome %s: wrote %d sites, %s %d input sites not processed by popout",
            chrom, written,
            "skipped" if args.thinned_sites == "skip" else "filled-missing for",
            skipped,
        )

    vcf_out.close()
    vcf_in.close()

    # 4. Companion global ancestry file.
    _write_global_anc(prefix, out_path, ancestry_names)

    log.info(
        "Wrote %s (%d sites) and companion global ancestry file",
        out_path, total_written,
    )


def _resolve_ancestry_names(args, prefix: Path) -> list[str]:
    """Resolve ancestry names from CLI flag, model.npz, or defaults."""
    from .names import parse_ancestry_names

    model_path = prefix.parent / f"{prefix.name}.model.npz"
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")
    model_data = np.load(str(model_path), allow_pickle=True)
    K = int(model_data["n_ancestries"])

    if args.ancestry_names is not None:
        return parse_ancestry_names(args.ancestry_names, K)

    if "ancestry_names" in model_data:
        names = list(model_data["ancestry_names"])
        if len(names) == K:
            return names

    return [f"anc_{i}" for i in range(K)]


def _natural_sort_key(s: str):
    """Sort key for natural ordering: chr1, chr2, ..., chr10, chr22, chrX."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]


def _scan_decode_parquet_chroms(prefix: Path) -> list[str]:
    """Find available decode.parquet files and extract chromosome names."""
    pattern = f"{prefix.name}.chr*.decode.parquet"
    parent = prefix.parent
    files = sorted(parent.glob(pattern))
    chroms = []
    for f in files:
        # Extract chrom from {prefix.name}.chr{CHROM}.decode.parquet
        stem = f.name
        # Remove prefix and suffix
        after_prefix = stem[len(prefix.name) + 4:]  # skip "{name}.chr"
        chrom = after_prefix.removesuffix(".decode.parquet")
        chroms.append(chrom)
    return sorted(chroms, key=_natural_sort_key)


def _build_output_header(
    in_header: pysam.VariantHeader,
    names: list[str],
    write_probs: bool,
) -> pysam.VariantHeader:
    """Build the output VCF header with ANCESTRY and FORMAT lines."""
    header = pysam.VariantHeader()

    # Copy contigs from input
    for rec in in_header.records:
        if rec.type == "CONTIG":
            header.add_record(rec)

    # Add ANCESTRY line: single line matching FLARE format
    anc_pairs = ",".join(f"{name}={i}" for i, name in enumerate(names))
    header.add_line(f"##ANCESTRY=<{anc_pairs}>")

    # FORMAT fields
    header.add_line(
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">'
    )
    header.add_line(
        '##FORMAT=<ID=AN1,Number=1,Type=Integer,'
        'Description="Most probable ancestry for haplotype 1">'
    )
    header.add_line(
        '##FORMAT=<ID=AN2,Number=1,Type=Integer,'
        'Description="Most probable ancestry for haplotype 2">'
    )
    if write_probs:
        header.add_line(
            '##FORMAT=<ID=ANP1,Number=.,Type=Float,'
            'Description="Posterior ancestry probabilities for haplotype 1; '
            'popout emits max_post at the called ancestry and '
            '(1-max_post)/(K-1) for others">'
        )
        header.add_line(
            '##FORMAT=<ID=ANP2,Number=.,Type=Float,'
            'Description="Posterior ancestry probabilities for haplotype 2; '
            'popout emits max_post at the called ancestry and '
            '(1-max_post)/(K-1) for others">'
        )

    # Copy samples from input
    for s in in_header.samples:
        header.add_sample(s)

    return header


def _convert_chrom(
    vcf_in: pysam.VariantFile,
    vcf_out: pysam.VariantFile,
    decode_path: Path,
    chrom: str,
    *,
    write_probs: bool,
    thinned_sites_mode: str,
    n_ancestries: int,
) -> tuple[int, int]:
    """Convert one chromosome. Returns (written, skipped) site counts."""
    from .output import read_decode_parquet

    data = read_decode_parquet(str(decode_path))
    calls = data["calls"]          # (H, T) uint8
    pos_bp = data["pos_bp"]        # (T,) int64
    max_post = data.get("max_post")  # (H, T) float16 or None

    if write_probs and max_post is None:
        log.warning(
            "chrom %s: --probs requested but decode.parquet has no max_post; "
            "ANP1/ANP2 will be omitted. Re-run popout with --probs.",
            chrom,
        )

    # Build pos_bp → decode index map
    pos_to_idx = {int(bp): i for i, bp in enumerate(pos_bp)}

    sample_ids = list(vcf_in.header.samples)
    n_samples = len(sample_ids)
    K = n_ancestries

    written = 0
    skipped = 0

    for rec in vcf_in.fetch(chrom):
        idx = pos_to_idx.get(rec.pos)

        if idx is None:
            if thinned_sites_mode == "skip":
                skipped += 1
                continue
            # fill-missing: emit record with missing ancestry calls
            new_rec = vcf_out.new_record()
            new_rec.contig = rec.contig
            new_rec.pos = rec.pos
            new_rec.alleles = rec.alleles
            for si, sample in enumerate(sample_ids):
                gt = rec.samples[sample]["GT"]
                new_rec.samples[sample]["GT"] = gt
                new_rec.samples[sample]["AN1"] = None
                new_rec.samples[sample]["AN2"] = None
                if write_probs and max_post is not None:
                    new_rec.samples[sample]["ANP1"] = None
                    new_rec.samples[sample]["ANP2"] = None
            vcf_out.write(new_rec)
            skipped += 1
            written += 1
            continue

        new_rec = vcf_out.new_record()
        new_rec.contig = rec.contig
        new_rec.pos = rec.pos
        new_rec.alleles = rec.alleles

        for si, sample in enumerate(sample_ids):
            gt = rec.samples[sample]["GT"]
            new_rec.samples[sample]["GT"] = gt
            an1 = int(calls[2 * si, idx])
            an2 = int(calls[2 * si + 1, idx])
            new_rec.samples[sample]["AN1"] = an1
            new_rec.samples[sample]["AN2"] = an2

            if write_probs and max_post is not None:
                mp1 = float(max_post[2 * si, idx])
                mp2 = float(max_post[2 * si + 1, idx])
                anp1 = _expand_max_post(mp1, an1, K)
                anp2 = _expand_max_post(mp2, an2, K)
                new_rec.samples[sample]["ANP1"] = anp1
                new_rec.samples[sample]["ANP2"] = anp2

        vcf_out.write(new_rec)
        written += 1

    return written, skipped


def _expand_max_post(mp: float, called_anc: int, K: int) -> tuple[float, ...]:
    """Expand scalar max_post into K-valued probability vector.

    Places max_post at the called ancestry's index and distributes
    (1 - max_post) / (K - 1) uniformly over the other ancestries.
    """
    off_anc = (1.0 - mp) / max(K - 1, 1)
    probs = [round(off_anc, 4)] * K
    probs[called_anc] = round(mp, 4)
    return tuple(probs)


def _write_global_anc(prefix: Path, out_path: Path, ancestry_names: list[str]) -> None:
    """Write companion .global.anc.gz with renamed ancestry columns."""
    global_tsv = prefix.parent / f"{prefix.name}.global.tsv"
    if not global_tsv.exists():
        log.warning("Global ancestry file not found: %s", global_tsv)
        return

    # Derive output path: strip .anc.vcf.gz from out_path, append .global.anc.gz
    out_str = str(out_path)
    if out_str.endswith(".anc.vcf.gz"):
        stem = out_str[:-len(".anc.vcf.gz")]
    elif out_str.endswith(".vcf.gz"):
        stem = out_str[:-len(".vcf.gz")]
    else:
        stem = out_str
    global_out = f"{stem}.global.anc.gz"

    with open(global_tsv, "r") as f_in:
        header_line = f_in.readline().strip()
        cols = header_line.split("\t")
        # Rename ancestry_0, ancestry_1, ... to ancestry_names
        new_cols = [cols[0]]  # "sample"
        for i, name in enumerate(ancestry_names):
            new_cols.append(name)
        # If there are more columns than expected, keep them
        if len(cols) > len(ancestry_names) + 1:
            new_cols.extend(cols[len(ancestry_names) + 1:])

        with gzip.open(global_out, "wt") as f_out:
            f_out.write("\t".join(new_cols) + "\n")
            for line in f_in:
                f_out.write(line)

    log.info("Wrote global ancestry to %s", global_out)
