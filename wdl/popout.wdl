version 1.0

## Run popout (GPU-accelerated local ancestry inference) on Terra.
##
## Inputs: per-chromosome PGEN files + optional genetic map.
## The task localizes all chromosomes into a single directory and invokes
## popout, which processes them together on the GPU.
##
## Default machine type is a2-highgpu-1g (1x A100 40 GB).
##
## GPU driver workaround: Cromwell's predefinedMachineType does not trigger
## GPU driver installation on its own.  Setting gpuType to match the
## built-in GPU (e.g. nvidia-tesla-a100 for A2) triggers the driver flag
## while GCP accepts the redundant accelerator declaration.

task popout_task {
  input {
    Array[File] pgens
    Array[File] pvars
    Array[File] psams
    File?        genetic_map
    String       output_prefix = "popout"

    # popout algorithm options
    Int?    n_ancestries
    String  ancestry_detection = "marchenko-pastur"
    Int     max_ancestries     = 20
    Int     n_em_iter          = 5
    Float?  thin_cm
    String  method             = "hmm"
    Float   gen_since_admix    = 20.0
    Boolean export_panel       = false
    Boolean block_emissions    = false
    Boolean write_probs        = false
    Boolean write_dense_decode = false
    String? ancestry_names      # comma list or gs:// URL to a TSV

    # Recursive seeding (--seed-method recursive)
    String  seed_method              = "gmm"
    Int     freeze_anchors_iters     = 0
    Float   recursive_merge_hellinger = 0.08
    Int     recursive_max_leaves     = 20
    Boolean stop_after_seeding       = false
    Boolean checkpoint_after_em      = true
    File?   resume_from_checkpoint
    String  extra_args               = ""

    # Weights & Biases — API key string or gs:// URL to a file containing it
    String? wandb_key

    # Runtime — machine_type and gpu_type must match (see header comment)
    String machine_type  = "a2-highgpu-1g"
    String gpu_type      = "nvidia-tesla-a100"
    String zones         = "us-central1-c us-central1-a"
    Int    disk_size_gb  = 500
    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  command <<<
    set -euo pipefail

    # ---- Weights & Biases setup ----
    WANDB_RAW="~{default="" wandb_key}"
    if [ -n "$WANDB_RAW" ]; then
      if [[ "$WANDB_RAW" == gs://* ]]; then
        export WANDB_API_KEY=$(gsutil cat "$WANDB_RAW")
      else
        export WANDB_API_KEY="$WANDB_RAW"
      fi
      echo "W&B API key configured"
    fi

    # ---- Localize PGEN files into a single directory ----
    # Terra scatters files into separate paths; popout expects them
    # co-located in one directory.
    mkdir -p pgen_dir
    pgens=(~{sep=' ' pgens})
    pvars=(~{sep=' ' pvars})
    psams=(~{sep=' ' psams})

    for i in "${!pgens[@]}"; do
      # Use basename of pgen to derive consistent naming
      base=$(basename "${pgens[$i]}" .pgen)
      ln -sf "${pgens[$i]}" "pgen_dir/${base}.pgen"
      ln -sf "${pvars[$i]}" "pgen_dir/${base}.pvar"
      ln -sf "${psams[$i]}" "pgen_dir/${base}.psam"
    done

    echo "=== Localized PGEN files ==="
    ls -lh pgen_dir/

    # ---- GPU check ----
    nvidia-smi || echo "WARNING: nvidia-smi failed"

    # Disable Triton GEMM autotuner — fails on some A100 driver combos
    # with "All configs failed during profiling / WRONG RESULTS". cuBLAS
    # fallback is equally fast for the matmul shapes popout uses.
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

    # ---- Build popout command ----
    CMD="popout --pgen pgen_dir/ --out ~{output_prefix}"
    ~{if defined(genetic_map) then 'CMD="$CMD --map ~{genetic_map}"' else ''}
    CMD="$CMD --n-em-iter ~{n_em_iter}"
    CMD="$CMD --method ~{method}"
    CMD="$CMD --gen-since-admix ~{gen_since_admix}"
    CMD="$CMD --ancestry-detection ~{ancestry_detection}"
    CMD="$CMD --max-ancestries ~{max_ancestries}"

    ~{if defined(n_ancestries) then 'CMD="$CMD --n-ancestries ~{n_ancestries}"' else ''}
    ~{if defined(thin_cm) then 'CMD="$CMD --thin-cm ~{thin_cm}"' else ''}
    ~{if export_panel then 'CMD="$CMD --export-panel"' else ''}
    ~{if block_emissions then 'CMD="$CMD --block-emissions"' else ''}
    ~{if write_probs then 'CMD="$CMD --probs"' else ''}
    ~{if write_dense_decode then 'CMD="$CMD --write-dense-decode"' else ''}
    ~{if defined(ancestry_names) then 'CMD="$CMD --ancestry-names ~{ancestry_names}"' else ''}

    CMD="$CMD --seed-method ~{seed_method}"
    CMD="$CMD --recursive-merge-hellinger ~{recursive_merge_hellinger}"
    CMD="$CMD --recursive-max-leaves ~{recursive_max_leaves}"
    if [ "~{freeze_anchors_iters}" -gt 0 ]; then
      CMD="$CMD --freeze-anchors-iters ~{freeze_anchors_iters}"
    fi
    ~{if stop_after_seeding then 'CMD="$CMD --stop-after-seeding"' else ''}
    ~{if checkpoint_after_em then 'CMD="$CMD --checkpoint-after-em"' else ''}
    ~{if defined(resume_from_checkpoint) then 'CMD="$CMD --resume-from-checkpoint ~{resume_from_checkpoint}"' else ''}

    if [ -n "$WANDB_RAW" ]; then CMD="$CMD --monitor wandb"; fi

    if [ -n "~{extra_args}" ]; then
      CMD="$CMD ~{extra_args}"
    fi

    echo "=== Running: $CMD ==="
    eval "$CMD"

    echo "=== Outputs ==="
    ls -lh ~{output_prefix}.*
  >>>

  output {
    # These are optional because --stop-after-seeding exits before producing them
    File? global_ancestry = "~{output_prefix}.global.tsv"
    File? tracts           = "~{output_prefix}.tracts.tsv.gz"
    File? model            = "~{output_prefix}.model"
    File? model_npz        = "~{output_prefix}.model.npz"
    File? summary          = "~{output_prefix}.summary.json"
    File? spectral_npz     = "~{output_prefix}.spectral.npz"
    File? stats_jsonl      = "~{output_prefix}.stats.jsonl"

    # Optional panel exports (only produced when export_panel = true)
    File? panel_haplotypes  = "~{output_prefix}.panel.haplotypes.tsv"
    File? panel_segments    = "~{output_prefix}.panel.segments.tsv.gz"
    File? panel_frequencies = "~{output_prefix}.panel.frequencies.tsv.gz"
    File? panel_proportions = "~{output_prefix}.panel.proportions.tsv"

    # Pre-merge dump (produced when seed_method = recursive)
    File? recursive_leaves     = "~{output_prefix}.recursive_pre_merge.leaves.tsv"
    File? recursive_leaf_meta  = "~{output_prefix}.recursive_pre_merge.leaf_meta.tsv"
    File? recursive_leaf_freqs = "~{output_prefix}.recursive_pre_merge.leaf_freqs.npz"

    # Checkpoint (produced when seed_method = recursive or stop_after_seeding)
    File? checkpoint      = "~{output_prefix}.checkpoint.npz"
    File? checkpoint_meta = "~{output_prefix}.checkpoint.meta.json"

    # Dense decode (produced when write_probs or write_dense_decode = true)
    Array[File] decode_parquet = glob("~{output_prefix}.chr*.decode.parquet")

    # Post-EM checkpoint (produced when checkpoint_after_em = true)
    File? em_checkpoint   = "~{output_prefix}.em_checkpoint.npz"
  }

  runtime {
    docker:               docker_image
    predefinedMachineType: machine_type
    # gpu_type must match the built-in GPU on the machine to trigger
    # Cromwell's setInstallGpuDrivers(true) without causing a conflict.
    gpuType:              gpu_type
    gpuCount:             1
    zones:                zones
    disks:                "local-disk ~{disk_size_gb} SSD"
    bootDiskSizeGb:       50
  }
}

workflow popout {
  input {
    Array[File] pgens
    Array[File] pvars
    Array[File] psams
    File?        genetic_map
    String       output_prefix = "popout"

    # Algorithm options
    Int?    n_ancestries
    String  ancestry_detection = "marchenko-pastur"
    Int     max_ancestries     = 20
    Int     n_em_iter          = 5
    Float?  thin_cm
    String  method             = "hmm"
    Float   gen_since_admix    = 20.0
    Boolean export_panel       = false
    Boolean block_emissions    = false
    Boolean write_probs        = false
    Boolean write_dense_decode = false
    String? ancestry_names

    # Recursive seeding
    String  seed_method              = "gmm"
    Int     freeze_anchors_iters     = 0
    Float   recursive_merge_hellinger = 0.08
    Int     recursive_max_leaves     = 20
    Boolean stop_after_seeding       = false
    Boolean checkpoint_after_em      = true
    File?   resume_from_checkpoint
    String  extra_args               = ""

    # Weights & Biases
    String? wandb_key

    # Runtime — machine_type and gpu_type must match (see header comment)
    String machine_type  = "a2-highgpu-1g"
    String gpu_type      = "nvidia-tesla-a100"
    String zones         = "us-central1-c us-central1-a"
    Int    disk_size_gb  = 500
    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  call popout_task {
    input:
      pgens           = pgens,
      pvars           = pvars,
      psams           = psams,
      genetic_map     = genetic_map,
      output_prefix   = output_prefix,
      n_ancestries       = n_ancestries,
      ancestry_detection = ancestry_detection,
      max_ancestries     = max_ancestries,
      n_em_iter          = n_em_iter,
      thin_cm            = thin_cm,
      method          = method,
      gen_since_admix = gen_since_admix,
      export_panel    = export_panel,
      block_emissions = block_emissions,
      write_probs        = write_probs,
      write_dense_decode = write_dense_decode,
      ancestry_names     = ancestry_names,
      seed_method               = seed_method,
      freeze_anchors_iters      = freeze_anchors_iters,
      recursive_merge_hellinger = recursive_merge_hellinger,
      recursive_max_leaves      = recursive_max_leaves,
      stop_after_seeding        = stop_after_seeding,
      checkpoint_after_em       = checkpoint_after_em,
      resume_from_checkpoint    = resume_from_checkpoint,
      extra_args                = extra_args,
      wandb_key       = wandb_key,
      machine_type    = machine_type,
      gpu_type        = gpu_type,
      zones           = zones,
      disk_size_gb    = disk_size_gb,
      docker_image    = docker_image
  }

  output {
    File? global_ancestry = popout_task.global_ancestry
    File? tracts           = popout_task.tracts
    File? model            = popout_task.model
    File? model_npz        = popout_task.model_npz
    File? summary          = popout_task.summary
    File? spectral_npz     = popout_task.spectral_npz
    File? stats_jsonl      = popout_task.stats_jsonl

    File? panel_haplotypes  = popout_task.panel_haplotypes
    File? panel_segments    = popout_task.panel_segments
    File? panel_frequencies = popout_task.panel_frequencies
    File? panel_proportions = popout_task.panel_proportions

    File? recursive_leaves     = popout_task.recursive_leaves
    File? recursive_leaf_meta  = popout_task.recursive_leaf_meta
    File? recursive_leaf_freqs = popout_task.recursive_leaf_freqs

    File? checkpoint      = popout_task.checkpoint
    File? checkpoint_meta = popout_task.checkpoint_meta

    Array[File] decode_parquet = popout_task.decode_parquet

    File? em_checkpoint   = popout_task.em_checkpoint
  }
}
