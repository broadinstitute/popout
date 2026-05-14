version 1.0

## End-to-end FLARE local-ancestry inference pipeline for biobank-scale
## cohorts (e.g. AoU v9 phased VCFs split across 22 autosomes and K
## ancestry clusters).
##
## Four stages, in order:
##
##   A. bcftools +split:   scatter[chrom]    -> K cluster sub-VCFs per chrom
##   B. flare em=true:     scatter[cluster]  -> one model + anc VCF on the
##                                              model chromosome
##   C. flare em=false:    scatter[cluster x non-model-chrom]
##                                           -> anc VCF using model from B
##   D. bcftools concat:   scatter[cluster]  -> one WGS anc VCF per cluster
##                                              (optional, do_concat input)
##
## Total task invocations for the AoU v9 + k=15 plan: 22 + 15 + 15*21 + 15 = 367.
## A single Cromwell run handles that scale; call-caching covers retries.
##
## Recommendations from the design doc surface here as input defaults
## rather than hardwires, so they can be overridden per run without
## editing the WDL.

import "../../bcftools/wdl/bcftools_split_streaming.wdl" as split_stream_wf
import "../../bcftools/wdl/bcftools_concat.wdl" as concat_wf
import "../../lai-tools/wdl/generate_genomic_partitions.wdl" as partitions_wf
import "./flare.wdl" as flare_wf

workflow flare_pipeline {
  input {
    # ---- Per-chromosome arrays (position-aligned, length N_chroms) ---
    # AoU phased VCFs are passed as gs:// URLs (NOT File) — Cromwell does
    # not localize them. htslib streams via libcurl, fetching only the
    # bytes for each region partition via tabix Range requests.
    # The .tbi index IS localized (it's tiny, ~300 KB per chrom) because
    # the partition planner reads it directly.
    Array[String] chromosomes
    Array[String] aou_phased_vcf_urls
    Array[File]   aou_phased_vcf_indices
    Array[File]   ref_vcfs
    Array[File]   ref_vcf_indices
    Array[File]   genetic_maps

    # ---- Per-cluster arrays (position-aligned, length K) -------------
    # cluster_ids must match the basename (extension stripped) of the
    # corresponding sample-list file — bcftools_split_streaming uses
    # cluster_ids as output basenames so the per-cluster array is
    # deterministically ordered.
    Array[String] cluster_ids
    Array[File]   cluster_sample_lists

    # ---- Reference panel (shared across all FLARE calls) -------------
    File ref_panel

    # ---- FLARE config (Sharon/Liz defaults from the 3/31 spec) -------
    String  model_chromosome = "chr20"
    Int     seed             = 12345
    Boolean probs            = false
    Float?  min_maf
    Int?    min_mac
    Int?    gen

    # ---- Streaming-split partitioning knobs --------------------------
    # Sizes in MB so the literals fit in WDL Int (32-bit in Cromwell/Rawls).
    # The partition task converts MB -> bytes via bash before calling the
    # Python script. Default 10 GB target; partitions larger than 30 GB
    # get subdivided.
    Int     target_mb_per_partition = 10240    # 10 GB
    Int     max_mb_per_partition    = 30720    # 30 GB

    # ---- Stage D toggle ----------------------------------------------
    Boolean do_concat        = true

    # ---- Observability -----------------------------------------------
    String? wandb_api_key
  }

  # Resolve model_chromosome to a position in `chromosomes`. Used to pick
  # the right ref VCF / genetic map in stage B and to gate stage C.
  call find_chrom_index {
    input:
      chromosomes = chromosomes,
      target      = model_chromosome
  }
  Int model_chr_idx = find_chrom_index.idx

  # =========================================================================
  # Stage A: streaming region scatter, gather per (chrom, cluster).
  #
  # For each chrom we (1) plan byte-balanced partitions from its tabix index,
  # (2) scatter `bcftools +split --regions` across those partitions reading
  # from gs:// (no localization), (3) gather the per-region per-cluster
  # sub-VCFs back into one per-chrom per-cluster sub-VCF via naive concat.
  #
  # This replaces the prior monolithic per-chrom `bcftools_split` task that
  # ran ~16 h wallclock at <15% CPU on a localized 366 GB chr1. The redesign
  # parallelizes the split across ~30 partitions per chrom and never pays
  # the localization cost.
  # =========================================================================
  scatter (i in range(length(chromosomes))) {

    # A.1 plan partitions for this chromosome (reads only the .tbi).
    call partitions_wf.generate_genomic_partitions as plan_partitions {
      input:
        vcf_index               = aou_phased_vcf_indices[i],
        chromosomes             = [chromosomes[i]],
        target_mb_per_partition = target_mb_per_partition,
        max_mb_per_partition    = max_mb_per_partition
    }

    # A.2 stream-split each partition. Each task pulls only its region
    # via tabix Range requests; no Cromwell localization of the VCF.
    scatter (r in range(length(plan_partitions.regions))) {
      call split_stream_wf.bcftools_split_streaming as split_region {
        input:
          vcf_url       = aou_phased_vcf_urls[i],
          region        = plan_partitions.regions[r],
          region_id     = plan_partitions.region_ids[r],
          sample_groups = cluster_sample_lists,
          cluster_ids   = cluster_ids,
          wandb_api_key = wandb_api_key
      }
    }

    # split_region.subset_vcfs has shape [partition][cluster]; transpose
    # to [cluster][partition] so we can gather each cluster's regions.
    Array[Array[File]] regions_by_cluster = transpose(split_region.subset_vcfs)

    # A.3 gather: per (chrom, cluster) concat the partition sub-VCFs back
    # into one per-chrom-per-cluster sub-VCF. naive concat is safe because
    # every region's output shares the input VCF's header.
    scatter (c in range(length(cluster_ids))) {
      call concat_wf.bcftools_concat as gather_regions {
        input:
          vcfs          = regions_by_cluster[c],
          output_prefix = cluster_ids[c] + "." + chromosomes[i],
          output_type   = "z",
          write_index   = true,
          naive         = true,
          wandb_api_key = wandb_api_key
      }
    }
  }

  # gather_regions.concat_vcf has shape [chrom][cluster]; transpose to
  # [cluster][chrom] for downstream FLARE consumption (unchanged shape).
  # FLARE auto-discovers <gt_vcf>.tbi from the localized File, so we don't
  # need to wire the index array through the pipeline.
  Array[Array[File]] by_cluster_vcfs = transpose(gather_regions.concat_vcf)

  # =========================================================================
  # Stages B + C + D: cluster scatter. Keeping these in one outer scatter
  # gives each cluster shard direct access to its own train output, so
  # stage C doesn't need to cross-index `train.out_model` by cluster.
  # =========================================================================
  scatter (c in range(length(cluster_ids))) {

    # ---- Stage B: train (em=true) on the model chromosome ----
    call flare_wf.flare as train {
      input:
        ref_vcf       = ref_vcfs[model_chr_idx],
        gt_vcf        = by_cluster_vcfs[c][model_chr_idx],
        map_file      = genetic_maps[model_chr_idx],
        ref_panel     = ref_panel,
        output_prefix = cluster_ids[c] + "." + model_chromosome,
        em            = true,
        probs         = probs,
        seed          = seed,
        min_maf       = min_maf,
        min_mac       = min_mac,
        gen           = gen,
        wandb_api_key = wandb_api_key
    }

    # ---- Stage C: apply (em=false) on the other 21 chromosomes ----
    scatter (ci in range(length(chromosomes))) {
      if (ci != model_chr_idx) {
        call flare_wf.flare as apply {
          input:
            ref_vcf       = ref_vcfs[ci],
            gt_vcf        = by_cluster_vcfs[c][ci],
            map_file      = genetic_maps[ci],
            ref_panel     = ref_panel,
            output_prefix = cluster_ids[c] + "." + chromosomes[ci],
            model         = train.out_model,
            em            = false,
            probs         = probs,
            seed          = seed,
            min_maf       = min_maf,
            min_mac       = min_mac,
            gen           = gen,
            wandb_api_key = wandb_api_key
        }
      }
    }
    # After the inner scatter, apply.anc_vcf is Array[File?] of length 22 —
    # None at model_chr_idx, File elsewhere. Same for apply.global_anc / log.

    # Re-assemble a full per-chromosome anc-VCF array by slotting train's
    # output back in at model_chr_idx. select_first picks the File when the
    # apply slot is set and falls back to train.anc_vcf otherwise (i.e.
    # exactly at the model chromosome position).
    scatter (ci in range(length(chromosomes))) {
      File chrom_anc_vcf   = select_first([apply.anc_vcf[ci],   train.anc_vcf])
      File chrom_qc_report = select_first([apply.qc_report[ci], train.qc_report])
    }

    # ---- Stage D (optional): concat to WGS anc VCF per cluster ----
    if (do_concat) {
      call concat_wf.bcftools_concat as concat_cluster {
        input:
          vcfs          = chrom_anc_vcf,
          output_prefix = cluster_ids[c] + ".wgs.anc",
          output_type   = "z",
          write_index   = true,
          naive         = true,    # FLARE writes identical headers across chroms
          wandb_api_key = wandb_api_key
      }
    }
  }

  output {
    # Stage B outputs
    Array[File] cluster_models              = train.out_model
    Array[File] cluster_model_chr_anc_vcfs  = train.anc_vcf
    Array[File] cluster_global_anc          = train.global_anc

    # Stage C outputs (per cluster, in chromosome order, model chrom slot
    # carries the stage-B anc VCF — same files as cluster_model_chr_anc_vcfs).
    Array[Array[File]] cluster_anc_vcfs_per_chrom    = chrom_anc_vcf
    Array[Array[File]] cluster_qc_reports_per_chrom  = chrom_qc_report

    # Stage D outputs (Array[File?] — None entries when do_concat=false).
    Array[File?] cluster_wgs_anc_vcfs = concat_cluster.concat_vcf
  }
}

# =========================================================================
# Helper tasks
# =========================================================================

task find_chrom_index {
  input {
    Array[String] chromosomes
    String        target
  }
  command <<<
    set -euo pipefail
    chroms=(~{sep=' ' chromosomes})
    target="~{target}"
    for i in "${!chroms[@]}"; do
      if [ "${chroms[$i]}" = "$target" ]; then
        echo "$i"
        exit 0
      fi
    done
    echo "ERROR: target chromosome '$target' not found in chromosomes list: ${chroms[*]}" >&2
    exit 1
  >>>
  output {
    Int idx = read_int(stdout())
  }
  runtime {
    docker: "ubuntu:24.04"
    cpu:    1
    memory: "1 GB"
    disks:  "local-disk 5 HDD"
  }
}

task pair_by_basename {
  input {
    Array[File]   files
    Array[String] expected_basenames
  }
  command <<<
    set -euo pipefail
    python3 <<'PYEOF'
import os, sys

files_str    = """~{sep='\t' files}"""
expected_str = """~{sep='\t' expected_basenames}"""

files    = [f for f in files_str.split('\t') if f]
expected = [e for e in expected_str.split('\t') if e]

# Match each file to an expected basename by progressively stripping
# extensions until a candidate matches. Handles `cluster_01.vcf.gz` ->
# `cluster_01` (two strips) and `cluster_01.vcf.gz.tbi` -> `cluster_01`
# (three strips) without hard-coding the suffix.
expected_set = set(expected)
def match(filepath: str) -> str | None:
    base = os.path.basename(filepath)
    while base:
        if base in expected_set:
            return base
        if '.' not in base:
            return None
        base = base.rsplit('.', 1)[0]
    return None

by_basename = {}
unmatched   = []
for f in files:
    m = match(f)
    if m is None:
        unmatched.append(f)
    else:
        if m in by_basename:
            sys.exit(f"ERROR: two files match basename '{m}': {by_basename[m]!r} and {f!r}")
        by_basename[m] = f

if unmatched:
    sys.exit(f"ERROR: {len(unmatched)} files did not match any expected basename: {unmatched}")

missing = [e for e in expected if e not in by_basename]
if missing:
    sys.exit(f"ERROR: no file matched expected basenames {missing}; available: {sorted(by_basename.keys())}")

with open("paired.txt", "w") as out:
    for e in expected:
        out.write(by_basename[e] + "\n")
PYEOF
  >>>
  output {
    Array[File] paired = read_lines("paired.txt")
  }
  runtime {
    docker: "python:3.12-slim"
    cpu:    1
    memory: "1 GB"
    disks:  "local-disk 5 HDD"
  }
}
