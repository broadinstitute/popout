version 1.0

## End-to-end FLARE local-ancestry inference pipeline for biobank-scale
## cohorts (e.g. AoU v9 phased VCFs split across 22 autosomes and K
## ancestry clusters).
##
## Four stages, in order:
##
##   A. plink2 --pfile --keep --export vcf bgz:
##                         scatter[chrom]    -> K cluster sub-VCFs per chrom
##                         (one task per chrom, loops over clusters)
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

import "../../bcftools/wdl/bcftools_concat.wdl" as concat_wf
import "../../plink2/wdl/plink2_export_clusters.wdl" as plink2_export_wf
import "./flare.wdl" as flare_wf

workflow flare_pipeline {
  input {
    # ---- Per-chromosome arrays (position-aligned, length N_chroms) ---
    # PGEN triplets per chromosome — plink2 reads the column-major binary
    # once per chrom, then per-cluster `--keep --export` is cheap. All
    # three Files are localized by Cromwell.
    Array[String] chromosomes
    Array[File]   aou_pgen
    Array[File]   aou_pvar
    Array[File]   aou_psam
    Array[File]   ref_vcfs
    Array[File]   ref_vcf_indices
    Array[File]   genetic_maps

    # ---- Per-cluster arrays (position-aligned, length K) -------------
    # cluster_ids must match the basename (extension stripped) of the
    # corresponding sample-list file — plink2_export_clusters uses
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
  # Stage A: per-chromosome plink2 export, K clusters per task.
  #
  # plink2 reads the chrom's PGEN matrix once (column-major binary), then
  # writes K bgzipped VCFs via `--keep <cluster> --export vcf-4.2 bgz`.
  # One task per chromosome — no sub-chrom scatter, no inter-partition
  # gather, no streaming-bytes drama.
  # =========================================================================
  scatter (i in range(length(chromosomes))) {
    call plink2_export_wf.plink2_export_clusters as stage_a {
      input:
        pgen          = aou_pgen[i],
        pvar          = aou_pvar[i],
        psam          = aou_psam[i],
        sample_groups = cluster_sample_lists,
        cluster_ids   = cluster_ids,
        wandb_api_key = wandb_api_key
    }
  }

  # stage_a.subset_vcfs has shape [chrom][cluster]; transpose to
  # [cluster][chrom] for downstream FLARE consumption.
  # FLARE auto-discovers <gt_vcf>.tbi from the localized File, so we don't
  # need to wire the index array through the pipeline.
  Array[Array[File]] by_cluster_vcfs = transpose(stage_a.subset_vcfs)

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

