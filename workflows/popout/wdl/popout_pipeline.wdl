version 1.0

## End-to-end popout local-ancestry inference pipeline.
##
## Mirrors flare_pipeline.wdl's train-once / scatter-infer shape:
##
##   B. popout train (mode=train) on a single user-defined chromosome
##      -> .model.npz
##   C. popout infer (mode=infer) scattered over the remaining
##      chromosomes, each consuming the model from B
##      -> per-chrom .global.tsv + .tracts.tsv.gz [+ .decode.parquet]
##
## Defaults are tuned to the main_v10 production config (recursive
## seeding, block_size=64, 20 EM iters, gen_since_admix=10).

import "./popout.wdl" as popout_wf

workflow popout_pipeline {
  input {
    # ---- Per-chromosome arrays (position-aligned, length N_chroms) ----
    # PGEN triplets per chromosome.
    Array[String] chromosomes
    Array[File]   pgens
    Array[File]   pvars
    Array[File]   psams
    Array[File]   genetic_maps               # per-chrom map files (position-aligned)

    # ---- Train target ----
    # Which chromosome to train the model on. Mirrors FLARE's
    # model_chromosome. Default chr1 matches every run in data/ so
    # outputs are directly comparable to main_v10.
    String train_chromosome = "chr1"

    # ---- Pre-trained model passthrough ----
    # Skip the multi-hour train step when supplied. Useful when the
    # train task succeeded in a prior submission but a later WDL change
    # invalidated Cromwell's call-cache. Scatter will then run infer
    # over ALL chromosomes (including train_chromosome) since the
    # supplied model carries no train-time decode.
    File? pretrained_model_npz

    String output_prefix = "popout"

    # ---- Seeding (used by stage B) ----
    String  seed_method               = "recursive"
    Int?    n_ancestries
    Int     max_ancestries            = 20
    String  ancestry_detection        = "marchenko-pastur"
    Int     recursive_max_leaves      = 20
    Int     recursive_min_leaf_size   = 1000
    Int     recursive_min_cluster_size = 1000
    Int     recursive_max_depth       = 6
    Float   recursive_merge_hellinger = 0.008
    File?   exclude_seeding_samples

    # ---- EM (used by stage B) ----
    Int     n_em_iter        = 20
    Float   gen_since_admix  = 10.0
    String  em_t_policy      = "gated"
    Int?    freeze_anchors_iters
    String  held_out_init    = "soft"
    Boolean block_emissions  = true
    Int     block_size       = 64
    Boolean per_hap_T        = false

    # ---- Decode (used by stage C scatter) ----
    Boolean write_probs        = false
    Boolean write_dense_decode = false

    # ---- Shared ----
    Float?  thin_cm
    Float   maf              = 0.01
    Int     seed             = 42
    String? ancestry_names
    String  extra_args       = ""

    # ---- Observability ----
    String? wandb_key

    # ---- Runtime ----
    String  machine_type  = "a2-highgpu-1g"
    String  gpu_type      = "nvidia-tesla-a100"
    String  zones         = "us-central1-c us-central1-a"
    Int     disk_size_gb  = 500
    String  docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  # Resolve train_chromosome to a position in `chromosomes`. Used to pick
  # the right per-chrom PGEN triplet for stage B and to gate stage C.
  call find_chrom_index {
    input:
      chromosomes = chromosomes,
      target      = train_chromosome
  }
  Int train_idx = find_chrom_index.idx

  # =========================================================================
  # Stage B: train the model on a single chromosome.
  # Skipped when pretrained_model_npz is supplied; the workflow then
  # behaves as an infer-only scatter.
  # =========================================================================
  if (!defined(pretrained_model_npz)) {
  call popout_wf.popout as train {
    input:
      mode                       = "train",
      pgen                       = pgens[train_idx],
      pvar                       = pvars[train_idx],
      psam                       = psams[train_idx],
      genetic_map                = genetic_maps[train_idx],
      chromosome                 = train_chromosome,
      output_prefix              = output_prefix + "." + train_chromosome + ".train",
      seed_method                = seed_method,
      n_ancestries               = n_ancestries,
      max_ancestries             = max_ancestries,
      ancestry_detection         = ancestry_detection,
      recursive_max_leaves       = recursive_max_leaves,
      recursive_min_leaf_size    = recursive_min_leaf_size,
      recursive_min_cluster_size = recursive_min_cluster_size,
      recursive_max_depth        = recursive_max_depth,
      recursive_merge_hellinger  = recursive_merge_hellinger,
      exclude_seeding_samples    = exclude_seeding_samples,
      n_em_iter                  = n_em_iter,
      gen_since_admix            = gen_since_admix,
      em_t_policy                = em_t_policy,
      freeze_anchors_iters       = freeze_anchors_iters,
      held_out_init              = held_out_init,
      block_emissions            = block_emissions,
      block_size                 = block_size,
      per_hap_T                  = per_hap_T,
      write_probs                = write_probs,
      write_dense_decode         = write_dense_decode,
      thin_cm                    = thin_cm,
      maf                        = maf,
      seed                       = seed,
      ancestry_names             = ancestry_names,
      extra_args                 = extra_args,
      wandb_key                  = wandb_key,
      machine_type               = machine_type,
      gpu_type                   = gpu_type,
      zones                      = zones,
      disk_size_gb               = disk_size_gb,
      docker_image               = docker_image
  }
  } # close: if (!defined(pretrained_model_npz))

  # Select the effective model: caller-supplied or freshly trained.
  File effective_model_npz = select_first([pretrained_model_npz, train.model_npz_out])

  # =========================================================================
  # Stage C: infer on the other chromosomes using the (trained or supplied)
  # model. Skip the train_chromosome slot ONLY in the fresh-train case
  # (where train decoded that chrom in-task). When pretrained, infer
  # every chrom because the supplied model carries no train-time decode.
  # =========================================================================
  scatter (ci in range(length(chromosomes))) {
    if (ci != train_idx || defined(pretrained_model_npz)) {
      call popout_wf.popout as infer {
        input:
          mode                = "infer",
          pgen                = pgens[ci],
          pvar                = pvars[ci],
          psam                = psams[ci],
          genetic_map         = genetic_maps[ci],
          chromosome          = chromosomes[ci],
          output_prefix       = output_prefix + "." + chromosomes[ci] + ".infer",
          model_npz           = effective_model_npz,
          write_probs         = write_probs,
          write_dense_decode  = write_dense_decode,
          thin_cm             = thin_cm,
          maf                 = maf,
          seed                = seed,
          ancestry_names      = ancestry_names,
          extra_args          = extra_args,
          wandb_key           = wandb_key,
          machine_type        = machine_type,
          gpu_type            = gpu_type,
          zones               = zones,
          disk_size_gb        = disk_size_gb,
          docker_image        = docker_image
      }
    }
  }

  output {
    # Stage B outputs are present only when train ran (i.e. no
    # pretrained_model_npz supplied). When pretrained, these are None
    # and the train-chrom outputs live at index train_idx of the
    # per-chrom arrays below.
    File? model         = train.model
    File model_npz     = effective_model_npz
    File? train_summary = train.summary
    File? train_global_tsv = train.global_tsv
    File? train_tracts     = train.tracts

    # Stage C: per-chrom outputs from the scatter (one entry per non-train chrom).
    Array[File?] global_tsvs_per_chrom = infer.global_tsv
    Array[File?] tracts_per_chrom      = infer.tracts
    Array[Array[File]?] decode_parquet_per_chrom = infer.decode_parquet
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
    echo "ERROR: train_chromosome '$target' not found in chromosomes list: ${chroms[*]}" >&2
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
