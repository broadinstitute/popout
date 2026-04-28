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
    Float   recursive_merge_hellinger = 0.012
    Int     recursive_max_leaves     = 20
    Int     recursive_min_leaf_size  = 500
    Boolean stop_after_seeding       = false
    Boolean post_em_consolidation    = true
    File?   exclude_seeding_samples    # TSV of sample_ids to exclude from recursive seeding

    # Checkpoint/resume — work dir is captured as output for future resume
    File?   resume_work_dir                        # tar.gz of a previous .work/ dir
    String? restart_stage                          # seed, em, decode, tracts, or all
    Boolean no_checkpoint            = false

    # Reproducibility mode — passed through to popout's --reproducible.
    # off: no XLA determinism (default; fastest).
    # seeding: spawn a deterministic seeding subprocess, then run EM at
    #   full speed via checkpoint resume (requires no_checkpoint=false).
    # all: force XLA determinism for the whole run (EM ~50× slower at
    #   biobank scale; only when bit-exact EM is required).
    String  reproducible             = "off"

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

    # ---- Reproducibility / determinism environment ----
    # Set BEFORE any python invocation so popout (and any subprocess it
    # spawns for --reproducible=seeding) inherit these.
    #
    # PYTHONHASHSEED is read at interpreter startup and cannot be
    # retrofitted; popout will warn if it sees a non-zero value here.
    #
    # XLA_FLAGS:
    #   --xla_gpu_enable_triton_gemm=false — Triton autotuner fails on
    #     some A100 driver combos ("All configs failed during profiling
    #     / WRONG RESULTS"); cuBLAS is equally fast for popout shapes.
    #
    # Note: --xla_gpu_deterministic_ops is NOT set globally because it
    # serializes parallel atomicAdd and slows the block-emissions E-step
    # ~50× at biobank scale. popout opts in via --reproducible=seeding
    # (subprocess-scoped) or --reproducible=all (process-wide).
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
    export PYTHONHASHSEED=0

    echo "=== Reproducibility env (must appear before 'Running: popout ...') ==="
    echo "  PYTHONHASHSEED=${PYTHONHASHSEED}"
    echo "  XLA_FLAGS=${XLA_FLAGS}"
    echo "  reproducible=~{reproducible}"
    if [ "${PYTHONHASHSEED}" != "0" ]; then
      echo "ERROR: PYTHONHASHSEED is '${PYTHONHASHSEED}', expected '0'." >&2
      echo "Refusing to launch popout: hash-based key derivation would not be reproducible." >&2
      exit 1
    fi

    # ---- Restore work directory from a previous run (for resume) ----
    RESUME_TAR="~{default="" resume_work_dir}"
    if [ -n "$RESUME_TAR" ]; then
      echo "Restoring work directory from $RESUME_TAR"
      tar xzf "$RESUME_TAR"
    fi

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
    CMD="$CMD --reproducible ~{reproducible}"
    CMD="$CMD --recursive-merge-hellinger ~{recursive_merge_hellinger}"
    CMD="$CMD --recursive-max-leaves ~{recursive_max_leaves}"
    CMD="$CMD --recursive-min-leaf-size ~{recursive_min_leaf_size}"
    if [ "~{freeze_anchors_iters}" -gt 0 ]; then
      CMD="$CMD --freeze-anchors-iters ~{freeze_anchors_iters}"
    fi
    ~{if stop_after_seeding then 'CMD="$CMD --stop-after-seeding"' else ''}
    ~{if post_em_consolidation then '' else 'CMD="$CMD --no-post-em-consolidation"'}
    ~{if defined(exclude_seeding_samples) then 'CMD="$CMD --exclude-seeding-samples ~{exclude_seeding_samples}"' else ''}
    ~{if no_checkpoint then 'CMD="$CMD --no-checkpoint"' else ''}
    ~{if defined(restart_stage) then 'CMD="$CMD --restart-stage ~{restart_stage}"' else ''}

    if [ -n "$WANDB_RAW" ]; then CMD="$CMD --monitor wandb"; fi

    if [ -n "~{extra_args}" ]; then
      CMD="$CMD ~{extra_args}"
    fi

    echo "=== Running: $CMD ==="
    eval "$CMD"

    # ---- Capture work directory for future resume ----
    if [ -d "~{output_prefix}.work" ]; then
      tar czf "~{output_prefix}.work.tar.gz" "~{output_prefix}.work/"
      echo "Work directory archived to ~{output_prefix}.work.tar.gz"
    fi

    echo "=== Outputs ==="
    ls -lh ~{output_prefix}.* || true
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

    # Work directory archive for resume (contains manifest, seed, em, decode)
    File? work_dir_tar = "~{output_prefix}.work.tar.gz"

    # Legacy checkpoints (still produced alongside work dir for compatibility)
    File? checkpoint      = "~{output_prefix}.checkpoint.npz"
    File? checkpoint_meta = "~{output_prefix}.checkpoint.meta.json"

    # Dense decode (produced when write_probs or write_dense_decode = true)
    Array[File] decode_parquet = glob("~{output_prefix}.chr*.decode.parquet")

    # Post-EM consolidation report (produced when post_em_consolidation = true)
    File? consolidation_report = "~{output_prefix}.post_em_consolidation.tsv"
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
    Float   recursive_merge_hellinger = 0.012
    Int     recursive_max_leaves     = 20
    Int     recursive_min_leaf_size  = 500
    Boolean stop_after_seeding       = false
    Boolean post_em_consolidation    = true
    File?   exclude_seeding_samples

    # Checkpoint/resume
    File?   resume_work_dir
    String? restart_stage
    Boolean no_checkpoint            = false

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
      recursive_min_leaf_size   = recursive_min_leaf_size,
      stop_after_seeding        = stop_after_seeding,
      post_em_consolidation     = post_em_consolidation,
      exclude_seeding_samples   = exclude_seeding_samples,
      resume_work_dir           = resume_work_dir,
      restart_stage             = restart_stage,
      no_checkpoint             = no_checkpoint,
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

    File? work_dir_tar    = popout_task.work_dir_tar

    File? checkpoint      = popout_task.checkpoint
    File? checkpoint_meta = popout_task.checkpoint_meta

    Array[File] decode_parquet = popout_task.decode_parquet

    File? consolidation_report = popout_task.consolidation_report
  }
}
