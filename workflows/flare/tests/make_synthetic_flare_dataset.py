#!/usr/bin/env python3
"""Generate a tiny synthetic dataset that exercises every stage of
flare_pipeline.wdl end-to-end under miniwdl.

Dimensions are deliberately minimal but structurally faithful:
  - 2 chromosomes (chr20, chr21) so stage C (apply on non-model-chrom) runs
  - 24 target samples partitioned across 4 clusters (3 admixed + 1 null)
  - 15 reference samples across 3 populations (POP_A, POP_B, POP_C)
  - ~500 biallelic SNPs per chromosome, phased
  - PLINK 4-col genetic maps, ref-panel TSV, cluster sample-list files,
    and a ready-to-run miniwdl inputs.json

Whole dataset weighs <5 MB and generates in a few seconds.

Usage:
  python workflows/flare/tests/make_synthetic_flare_dataset.py \
      --out data/synthetic_flare/
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import pysam

CHROMOSOMES   = ["chr20", "chr21"]
POPULATIONS   = ["POP_A", "POP_B", "POP_C"]
REF_PER_POP   = 5
N_TARGET      = 24
VARIANTS_PER_CHROM = 500
START_BP      = 1_000_000
SPACING_BP    = 50_000  # 500 variants × 50 kb = 25 Mb per chrom
CM_PER_MB     = 1.0     # 1 cM/Mb is a fine first approximation

CLUSTERS = [
    ("cluster_a",    6),
    ("cluster_b",    6),
    ("cluster_c",    6),
    ("null_cluster", 6),
]


def write_phased_vcf(
    path: Path,
    chrom: str,
    contig_lengths: dict[str, int],
    samples: list[str],
    haplotypes: list[list[tuple[int, int]]],
    positions: list[int],
) -> None:
    """haplotypes[v][s] = (allele1, allele2). Writes bgzipped VCF + .tbi.

    The header declares every contig in CHROMOSOMES (not just `chrom`) so
    that `bcftools concat --naive` can stitch per-chrom outputs together
    without complaining about missing contigs. FLARE preserves the input
    header in its output VCF, so this propagates through the pipeline.
    """
    header = pysam.VariantHeader()
    header.add_meta("FILTER", items=[("ID", "PASS"), ("Description", "All filters passed")])
    header.add_meta("FORMAT", items=[("ID", "GT"), ("Number", "1"), ("Type", "String"), ("Description", "Genotype")])
    for c in CHROMOSOMES:
        header.contigs.add(c, length=contig_lengths[c])
    for s in samples:
        header.add_sample(s)

    with pysam.VariantFile(str(path), "wz", header=header) as vcf:
        for v, pos in enumerate(positions):
            rec = vcf.new_record()
            rec.contig = chrom
            rec.pos    = pos
            rec.id     = f"{chrom}_{pos}"
            rec.ref    = "A"
            rec.alts   = ("G",)
            rec.qual   = 100
            for s_idx, sample in enumerate(samples):
                a1, a2 = haplotypes[v][s_idx]
                rec.samples[sample]["GT"] = (a1, a2)
                rec.samples[sample].phased = True
            vcf.write(rec)

    pysam.tabix_index(str(path), preset="vcf", force=True)


def upload_gt_to_gcs(out_dir: Path, gcs_prefix: str) -> dict[str, str]:
    """Upload the per-chrom gt.vcf.gz + .tbi to gs:// so the streaming pipeline
    can read them via libcurl. Returns {chrom: gs_url} for the .vcf.gz."""
    if not gcs_prefix.endswith("/"):
        gcs_prefix = gcs_prefix + "/"
    urls: dict[str, str] = {}
    paths = []
    for chrom in CHROMOSOMES:
        for ext in ("vcf.gz", "vcf.gz.tbi"):
            local = out_dir / f"{chrom}.gt.{ext}"
            paths.append(str(local))
        urls[chrom] = f"{gcs_prefix}{chrom}.gt.vcf.gz"
    print(f"Uploading {len(paths)} files to {gcs_prefix} …", file=sys.stderr)
    subprocess.run(["gcloud", "storage", "cp", *paths, gcs_prefix], check=True)
    return urls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument("--gcs-prefix", default=None,
                    help="if set (e.g. gs://my-bucket/path/), upload the gt VCFs + indices to this "
                         "prefix and emit inputs.json with gs:// URLs in aou_phased_vcf_urls so "
                         "miniwdl can exercise the streaming split pipeline end-to-end")
    args = ap.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clusters").mkdir(exist_ok=True)

    rng = random.Random(args.seed)

    # ---- sample IDs ----
    ref_samples_by_pop = {
        pop: [f"REF_{pop}_{i:02d}" for i in range(REF_PER_POP)]
        for pop in POPULATIONS
    }
    ref_samples = [s for pop in POPULATIONS for s in ref_samples_by_pop[pop]]
    target_samples = [f"TGT_{i:03d}" for i in range(N_TARGET)]

    # Assign each target sample a Dirichlet-ish ancestry weight vector. We
    # do this once per sample, used for both chromosomes so the simulated
    # ancestries are coherent across the genome (FLARE will still want to
    # call local switches, but at least there's a global signal to fit).
    target_weights: list[list[float]] = []
    for _ in target_samples:
        raw = [rng.gammavariate(0.5, 1.0) for _ in POPULATIONS]
        s = sum(raw)
        target_weights.append([x / s for x in raw])

    # ---- partition target samples into clusters ----
    shuffled = target_samples.copy()
    rng.shuffle(shuffled)
    cluster_assignments: dict[str, list[str]] = {}
    cursor = 0
    for cluster_id, size in CLUSTERS:
        cluster_assignments[cluster_id] = shuffled[cursor:cursor + size]
        cursor += size
    assert cursor == N_TARGET, f"cluster sizes sum to {cursor}, expected {N_TARGET}"

    # ---- write cluster sample-list files ----
    for cluster_id, members in cluster_assignments.items():
        with (out_dir / "clusters" / f"{cluster_id}.tsv").open("w") as f:
            for s in members:
                f.write(s + "\n")

    # ---- write ref-panel TSV (sample\tpopulation, no header) ----
    with (out_dir / "ref.panel").open("w") as f:
        for pop in POPULATIONS:
            for s in ref_samples_by_pop[pop]:
                f.write(f"{s}\t{pop}\n")

    # All chroms share the same variant layout in this synthetic dataset,
    # so contig lengths are constant.
    _positions = [START_BP + i * SPACING_BP for i in range(VARIANTS_PER_CHROM)]
    _contig_length = _positions[-1] + SPACING_BP
    contig_lengths = {c: _contig_length for c in CHROMOSOMES}

    # ---- per-chromosome data ----
    for chrom in CHROMOSOMES:
        positions = _positions

        # Per-population AFs. Beta(0.3, 0.3) gives a U-shape so most variants
        # are differentially fixed across populations -> highly informative.
        ref_afs = [
            {pop: rng.betavariate(0.3, 0.3) for pop in POPULATIONS}
            for _ in range(VARIANTS_PER_CHROM)
        ]

        # Reference haplotypes: drawn straight from each sample's population AF.
        ref_haps: list[list[tuple[int, int]]] = []
        for v in range(VARIANTS_PER_CHROM):
            row: list[tuple[int, int]] = []
            for pop in POPULATIONS:
                af = ref_afs[v][pop]
                for _ in ref_samples_by_pop[pop]:
                    a1 = 1 if rng.random() < af else 0
                    a2 = 1 if rng.random() < af else 0
                    row.append((a1, a2))
            ref_haps.append(row)

        # Target haplotypes: each haplotype draws a population per chunk of
        # ~50 variants (so FLARE has some local ancestry switches to call),
        # then samples its allele from that population's AF.
        target_haps: list[list[tuple[int, int]]] = [[] for _ in range(VARIANTS_PER_CHROM)]
        chunk_size = 50
        for s_idx, weights in enumerate(target_weights):
            for hap_idx in (0, 1):
                # For each chunk, pick a population by sample's ancestry weights.
                v = 0
                while v < VARIANTS_PER_CHROM:
                    pop = rng.choices(POPULATIONS, weights=weights, k=1)[0]
                    chunk_end = min(v + chunk_size, VARIANTS_PER_CHROM)
                    for vv in range(v, chunk_end):
                        af = ref_afs[vv][pop]
                        allele = 1 if rng.random() < af else 0
                        if hap_idx == 0:
                            target_haps[vv].append((allele, 0))  # placeholder a2
                        else:
                            a1, _ = target_haps[vv][s_idx]
                            target_haps[vv][s_idx] = (a1, allele)
                    v = chunk_end

        # Write the two VCFs.
        ref_vcf_path = out_dir / f"{chrom}.ref.vcf.gz"
        gt_vcf_path  = out_dir / f"{chrom}.gt.vcf.gz"
        write_phased_vcf(ref_vcf_path, chrom, contig_lengths, ref_samples,    ref_haps,    positions)
        write_phased_vcf(gt_vcf_path,  chrom, contig_lengths, target_samples, target_haps, positions)

        # Write the genetic map: chr <tab> . <tab> cM <tab> bp.
        map_path = out_dir / f"{chrom}.map"
        with map_path.open("w") as f:
            for pos in positions:
                cm = (pos - START_BP) * CM_PER_MB / 1_000_000
                f.write(f"{chrom}\t.\t{cm:.6f}\t{pos}\n")

        print(f"  {chrom}: ref={ref_vcf_path.name} gt={gt_vcf_path.name} map={map_path.name} variants={VARIANTS_PER_CHROM}", file=sys.stderr)

    # ---- inputs.json ----
    abs = lambda p: str((out_dir / p).resolve())
    # aou_phased_vcf_urls is Array[String] in the new (streaming) pipeline —
    # gs:// URLs when --gcs-prefix is set, local file paths otherwise (for
    # quick offline inspection of the inputs JSON shape).
    if args.gcs_prefix:
        gt_urls = upload_gt_to_gcs(out_dir, args.gcs_prefix)
        aou_phased_vcf_urls = [gt_urls[c] for c in CHROMOSOMES]
    else:
        aou_phased_vcf_urls = [abs(f"{c}.gt.vcf.gz") for c in CHROMOSOMES]

    inputs = {
        "flare_pipeline.chromosomes":            CHROMOSOMES,
        "flare_pipeline.aou_phased_vcf_urls":    aou_phased_vcf_urls,
        "flare_pipeline.aou_phased_vcf_indices": [abs(f"{c}.gt.vcf.gz.tbi")   for c in CHROMOSOMES],
        "flare_pipeline.ref_vcfs":               [abs(f"{c}.ref.vcf.gz")      for c in CHROMOSOMES],
        "flare_pipeline.ref_vcf_indices":        [abs(f"{c}.ref.vcf.gz.tbi")  for c in CHROMOSOMES],
        "flare_pipeline.genetic_maps":           [abs(f"{c}.map")             for c in CHROMOSOMES],
        "flare_pipeline.ref_panel":              abs("ref.panel"),
        "flare_pipeline.cluster_ids":            [c for c, _ in CLUSTERS],
        "flare_pipeline.cluster_sample_lists":   [abs(f"clusters/{c}.tsv") for c, _ in CLUSTERS],
        "flare_pipeline.model_chromosome":       "chr20",
        "flare_pipeline.seed":                   12345,
        "flare_pipeline.probs":                  False,
        "flare_pipeline.do_concat":              True,
        # FLARE defaults (min-mac=50, min-maf=0.005) are tuned for biobank-
        # scale ref panels; our 15-sample synthetic panel needs much looser
        # thresholds so the variants aren't all filtered out.
        "flare_pipeline.min_mac":                2,
        "flare_pipeline.min_maf":                0.01,
        # Partition target tuned for the synthetic fixture: each chrom has
        # ~2 BGZF blocks with very small within-block voff diffs, so the
        # partitioner emits ~1-2 partitions per chrom — enough to exercise
        # the parallel scatter path without spinning up hundreds of tasks
        # on a tiny dataset. Pipeline inputs are in MB (WDL Int is 32-bit
        # in Cromwell/Rawls, can't hold raw byte counts of biobank scale).
        "flare_pipeline.target_mb_per_partition": 100,    # ~100 MB voff units
        "flare_pipeline.max_mb_per_partition":    1000,   # ~1 GB voff units
    }
    inputs_path = out_dir / "inputs.json"
    with inputs_path.open("w") as f:
        json.dump(inputs, f, indent=2)
        f.write("\n")

    print(f"\nWrote {inputs_path}", file=sys.stderr)
    print(f"Run: miniwdl run workflows/flare/wdl/flare_pipeline.wdl -i {inputs_path} --dir /tmp/miniwdl_flare_smoke/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
