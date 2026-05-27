#!/usr/bin/env python3
"""Derive a popout-schema tracts.tsv.gz from a FLARE ancestry VCF.

FLARE writes per-site hard ancestry calls as FORMAT/AN1 and FORMAT/AN2
(integer ancestry indices, 0..K-1). When FLARE is run with --probs=true
the file also carries FORMAT/AP1, FORMAT/AP2 per-ancestry posterior
arrays; with --probs=false those are absent.

This script streams the VCF in chromosome order and emits a tract row
each time AN1 or AN2 changes for a sample. Output columns match the
popout `popout.viz._loaders.read_tracts` contract exactly:

    #chrom  start_bp  end_bp  sample  haplotype  ancestry  n_sites  mean_posterior

`mean_posterior` is the mean of FORMAT/AP{1,2}[ancestry] across the
sites that compose the tract, or NaN when --probs=false.

Usage:
    python flare_vcf_to_tracts.py \\
        --anc-vcf /path/to/null_cluster_sample_list.chr1.anc.vcf.gz \\
        --out     /path/to/null_cluster.chr1.tracts.tsv.gz
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path

import pysam


HEADER = "#chrom\tstart_bp\tend_bp\tsample\thaplotype\tancestry\tn_sites\tmean_posterior\n"


class OpenTract:
    __slots__ = ("chrom", "start_bp", "end_bp", "ancestry", "n_sites", "post_sum")

    def __init__(self, chrom: str, pos: int, ancestry: int, posterior: float | None):
        self.chrom = chrom
        self.start_bp = pos
        self.end_bp = pos
        self.ancestry = ancestry
        self.n_sites = 1
        # post_sum accumulates posteriors only when probs=true; NaN otherwise.
        self.post_sum = posterior if posterior is not None else float("nan")

    def extend(self, pos: int, posterior: float | None) -> None:
        self.end_bp = pos
        self.n_sites += 1
        if posterior is not None:
            # post_sum starts at first posterior; running sum thereafter.
            if math.isnan(self.post_sum):
                self.post_sum = posterior
            else:
                self.post_sum += posterior


def flush(out, sample: str, hap: int, t: OpenTract) -> None:
    if math.isnan(t.post_sum):
        mean_post = "nan"
    else:
        mean_post = f"{t.post_sum / t.n_sites:.6f}"
    out.write(
        f"{t.chrom}\t{t.start_bp}\t{t.end_bp}\t{sample}\t{hap}\t"
        f"{t.ancestry}\t{t.n_sites}\t{mean_post}\n"
    )


def derive(anc_vcf: Path, out_path: Path) -> None:
    vcf = pysam.VariantFile(str(anc_vcf))
    fmt = vcf.header.formats
    if "AN1" not in fmt or "AN2" not in fmt:
        raise RuntimeError(
            f"VCF {anc_vcf} is missing FORMAT/AN1 or FORMAT/AN2 "
            f"(found: {list(fmt.keys())}); not a FLARE ancestry VCF."
        )
    has_probs = "AP1" in fmt and "AP2" in fmt

    samples = list(vcf.header.samples)
    n_samples = len(samples)
    if n_samples == 0:
        raise RuntimeError(f"VCF {anc_vcf} has no samples.")
    print(f"  samples: {n_samples}  has_probs: {has_probs}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    open_tracts: dict[tuple[int, int], OpenTract] = {}
    n_records = 0
    n_tracts = 0
    current_chrom: str | None = None

    with gzip.open(out_path, "wt") as out:
        out.write(HEADER)

        for rec in vcf:
            chrom = rec.chrom
            pos = rec.pos

            # Chromosome boundary: flush every open tract before continuing.
            if current_chrom is not None and chrom != current_chrom:
                for (si, hap), t in open_tracts.items():
                    flush(out, samples[si], hap, t)
                    n_tracts += 1
                open_tracts.clear()
            current_chrom = chrom

            for si, sample in enumerate(samples):
                gt = rec.samples[sample]
                an1 = gt.get("AN1")
                an2 = gt.get("AN2")
                if an1 is None or an2 is None:
                    raise RuntimeError(
                        f"sample={sample} pos={chrom}:{pos} missing AN1 or AN2"
                    )

                if has_probs:
                    ap1 = gt["AP1"]
                    ap2 = gt["AP2"]
                    p1 = float(ap1[int(an1)])
                    p2 = float(ap2[int(an2)])
                else:
                    p1 = None
                    p2 = None

                for hap, anc, posterior in ((1, int(an1), p1), (2, int(an2), p2)):
                    key = (si, hap)
                    t = open_tracts.get(key)
                    if t is None or t.ancestry != anc:
                        if t is not None:
                            flush(out, sample, hap, t)
                            n_tracts += 1
                        open_tracts[key] = OpenTract(chrom, pos, anc, posterior)
                    else:
                        t.extend(pos, posterior)

            n_records += 1
            if n_records % 50000 == 0:
                print(f"  ... processed {n_records} records ({n_tracts} tracts so far)")

        # End-of-file flush.
        for (si, hap), t in open_tracts.items():
            flush(out, samples[si], hap, t)
            n_tracts += 1

    print(f"  processed {n_records} records -> {n_tracts} tracts")
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--anc-vcf", type=Path, required=True,
                   help="Path to FLARE <prefix>.anc.vcf.gz")
    p.add_argument("--out", type=Path, required=True,
                   help="Output tracts.tsv.gz path")
    args = p.parse_args()
    if not args.anc_vcf.exists():
        raise FileNotFoundError(args.anc_vcf)
    if args.out.suffix != ".gz":
        raise ValueError("--out must end in .gz (file is gzipped)")
    derive(args.anc_vcf, args.out)


if __name__ == "__main__":
    main()
