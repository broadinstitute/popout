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
##   D. bcftools merge:    scatter[chrom]    -> one per-chrom anc VCF with
##                                              the union of all clusters'
##                                              samples (optional, do_merge)
##
## Total task invocations for the AoU v9 + k=15 plan: 22 + 15 + 15*21 + 22 = 374.
## A single Cromwell run handles that scale; call-caching covers retries.
##
## Recommendations from the design doc surface here as input defaults
## rather than hardwires, so they can be overridden per run without
## editing the WDL.

import "../../bcftools/wdl/bcftools_merge.wdl" as merge_wf
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
    # min_maf / min_mac pinned to FLARE's documented defaults (flare.wdl:40-41)
    # so they're never None at evaluation time — Cromwell trips on optional
    # lookups inside the Stage C nested scatter, and concretizing here keeps
    # the call sites simple. Override via inputs.json when needed.
    String  model_chromosome = "chr20"
    Int     seed             = 12345
    Boolean probs            = false
    Float   min_maf          = 0.005
    Int     min_mac          = 50
    Int?    gen

    # ---- Stage D toggle ----------------------------------------------
    Boolean do_merge         = true

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
  # Stage A: per-(chrom × cluster) plink2 export.
  #
  # Each shard runs one plink2 --keep --export for a single (chrom, cluster)
  # pair. K=16 clusters now scatter in parallel per chrom instead of looping
  # serially inside one big VM. Right-sized at 4 CPU / 16 GB HDD, preemptible
  # — see plink2_export_clusters.wdl for the W&B-backed sizing rationale.
  # =========================================================================
  scatter (i in range(length(chromosomes))) {
    scatter (c in range(length(cluster_ids))) {
      call plink2_export_wf.plink2_export_clusters as export_cluster_vcfs {
        input:
          pgen                = aou_pgen[i],
          pvar                = aou_pvar[i],
          psam                = aou_psam[i],
          cluster_sample_list = cluster_sample_lists[c],
          cluster_id          = cluster_ids[c],
          wandb_api_key       = wandb_api_key
      }
    }
  }

  # The nested scatter gives export_cluster_vcfs.subset_vcf shape
  # [chrom][cluster] directly; transpose to [cluster][chrom] for downstream
  # FLARE consumption.
  # FLARE auto-discovers <gt_vcf>.tbi from the localized File, so we don't
  # need to wire the index array through the pipeline.
  Array[Array[File]] by_cluster_vcfs = transpose(export_cluster_vcfs.subset_vcf)

  # =========================================================================
  # Stages B + C: cluster scatter. Keeping these in one outer scatter gives
  # each cluster shard direct access to its own fit_ancestry_model output, so
  # stage C doesn't need to cross-index the model by cluster.
  # =========================================================================
  scatter (c in range(length(cluster_ids))) {

    # ---- Stage B: fit_ancestry_model (FLARE em=true) on the model chromosome ----
    call flare_wf.flare as fit_ancestry_model {
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

    # ---- Stage C: infer_ancestry (FLARE em=false) on the other chromosomes ----
    # min_maf / min_mac are restored here — FLARE re-filters variants by
    # ref-allele-count at apply time too (AdmixReader.minMac validates against
    # ref sample count regardless of em mode). seed and gen are genuinely
    # training-only and stay dropped.
    scatter (ci in range(length(chromosomes))) {
      if (ci != model_chr_idx) {
        call flare_wf.flare as infer_ancestry {
          input:
            ref_vcf       = ref_vcfs[ci],
            gt_vcf        = by_cluster_vcfs[c][ci],
            map_file      = genetic_maps[ci],
            ref_panel     = ref_panel,
            output_prefix = cluster_ids[c] + "." + chromosomes[ci],
            model         = fit_ancestry_model.out_model,
            em            = false,
            probs         = probs,
            min_maf       = min_maf,
            min_mac       = min_mac,
            preemptible   = 1,
            wandb_api_key = wandb_api_key
        }
      }
    }
    # After the inner scatter, infer_ancestry.anc_vcf is Array[File?] of
    # length N_chroms — None at model_chr_idx, File elsewhere. Same for
    # infer_ancestry.global_anc / log.

    # Re-assemble a full per-chromosome anc-VCF array by slotting
    # fit_ancestry_model's output back in at model_chr_idx. select_first
    # picks the File when the infer_ancestry slot is set and falls back to
    # fit_ancestry_model.anc_vcf otherwise (i.e. exactly at the model
    # chromosome position).
    scatter (ci in range(length(chromosomes))) {
      File chrom_anc_vcf = select_first([infer_ancestry.anc_vcf[ci], fit_ancestry_model.anc_vcf])
    }
  }

  # =========================================================================
  # Stage D: per-chromosome cross-cluster merge.
  #
  # After the cluster scatter, chrom_anc_vcf has shape [cluster][chrom].
  # Each (cluster, chrom) anc VCF covers a disjoint sample subset at the
  # same chrom-wide variant positions, so combining clusters within a chrom
  # is a column-wise merge — `bcftools merge`, not `bcftools concat`. The
  # delivery shape matches the input (one VCF per chrom, all 535K samples).
  # =========================================================================
  Array[Array[File]] anc_vcfs_by_chrom = transpose(chrom_anc_vcf)

  if (do_merge) {
    scatter (ci in range(length(chromosomes))) {
      call merge_wf.bcftools_merge as merge_chrom_anc {
        input:
          vcfs          = anc_vcfs_by_chrom[ci],
          output_prefix = chromosomes[ci] + ".anc",
          output_type   = "z",
          write_index   = true,
          wandb_api_key = wandb_api_key
      }
    }
  }

  output {
    # Stage B outputs
    Array[File] cluster_models              = fit_ancestry_model.out_model
    Array[File] cluster_model_chr_anc_vcfs  = fit_ancestry_model.anc_vcf
    Array[File] cluster_global_anc          = fit_ancestry_model.global_anc

    # Stage C outputs (per cluster, in chromosome order, model chrom slot
    # carries the stage-B anc VCF — same files as cluster_model_chr_anc_vcfs).
    Array[Array[File]] cluster_anc_vcfs_per_chrom = chrom_anc_vcf

    # Stage D outputs: per-chrom merged anc VCFs (one per chrom, all clusters'
    # samples). None when do_merge=false.
    Array[File]? chrom_anc_vcfs = merge_chrom_anc.merged_vcf
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

