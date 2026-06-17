version 1.0

## Run popout on ONE chromosome in one of three modes.
##
##   mode = "seed"   -> popout seed   (writes .seed.npz)
##   mode = "train"  -> popout train  (writes .model.npz + .summary.json)
##   mode = "infer"  -> popout infer  (writes .global.tsv + .tracts.tsv.gz
##                                     + optional .decode.parquet)
##
## Inputs match the popout CLI surface for each mode; irrelevant inputs
## are ignored. Compose multi-chromosome runs at the calling layer
## (see popout_pipeline.wdl for the train-once / scatter-infer shape).

task popout_task {
  input {
    String  mode                                   # "seed" | "train" | "infer"
    File    pgen
    File    pvar
    File    psam
    File?   genetic_map
    String  chromosome
    String  output_prefix = "popout"

    # Mode "train" / "infer": optional model NPZ input (infer requires it)
    File?   model_npz                              # required for mode=infer
    File?   seed_input                             # optional for mode=train

    # Seeding (mode = seed | train)
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

    # EM (mode = train)
    Int     n_em_iter        = 20
    Float   gen_since_admix  = 10.0
    String  em_t_policy      = "gated"
    Int?    freeze_anchors_iters
    String  held_out_init    = "soft"
    Boolean block_emissions  = true
    Int     block_size       = 64
    Boolean per_hap_T        = false

    # Decode (mode = infer)
    Boolean write_probs         = false
    Boolean write_dense_decode  = false

    # Shared
    Float?  thin_cm
    Float   maf              = 0.01
    Int     seed             = 42
    String? ancestry_names
    String  extra_args       = ""

    String? wandb_key

    String machine_type  = "a2-highgpu-1g"
    String gpu_type      = "nvidia-tesla-a100"
    String zones         = "us-central1-c us-central1-a"
    Int    disk_size_gb  = 500
    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  command <<<
    set -euo pipefail

    if [ "~{mode}" != "seed" ] && [ "~{mode}" != "train" ] && [ "~{mode}" != "infer" ]; then
      echo "ERROR: mode must be one of seed|train|infer (got '~{mode}')" >&2
      exit 1
    fi
    if [ "~{mode}" = "infer" ] && [ -z "~{default="" model_npz}" ]; then
      echo "ERROR: mode=infer requires model_npz" >&2
      exit 1
    fi

    WANDB_RAW="~{default="" wandb_key}"
    if [ -n "$WANDB_RAW" ]; then
      if [[ "$WANDB_RAW" == gs://* ]]; then
        export WANDB_API_KEY=$(gsutil cat "$WANDB_RAW")
      else
        export WANDB_API_KEY="$WANDB_RAW"
      fi
    fi

    # Localize PGEN triplet into one dir so popout iter_chromosomes scans it
    mkdir -p pgen_dir
    base=$(basename "~{pgen}" .pgen)
    ln -sf "~{pgen}" "pgen_dir/${base}.pgen"
    ln -sf "~{pvar}" "pgen_dir/${base}.pvar"
    ln -sf "~{psam}" "pgen_dir/${base}.psam"

    nvidia-smi || echo "WARNING: nvidia-smi failed"
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
    export PYTHONHASHSEED=0

    CMD="popout ~{mode} --pgen pgen_dir/ --out ~{output_prefix} --chromosomes ~{chromosome}"
    CMD="$CMD --maf ~{maf} --seed ~{seed}"
    ~{if defined(genetic_map) then 'CMD="$CMD --map ~{genetic_map}"' else ''}
    ~{if defined(thin_cm) then 'CMD="$CMD --thin-cm ~{thin_cm}"' else ''}

    if [ "~{mode}" = "seed" ] || [ "~{mode}" = "train" ]; then
      CMD="$CMD --seed-method ~{seed_method}"
      CMD="$CMD --max-ancestries ~{max_ancestries}"
      CMD="$CMD --ancestry-detection ~{ancestry_detection}"
      CMD="$CMD --gen-since-admix ~{gen_since_admix}"
      CMD="$CMD --recursive-max-leaves ~{recursive_max_leaves}"
      CMD="$CMD --recursive-min-leaf-size ~{recursive_min_leaf_size}"
      CMD="$CMD --recursive-min-cluster-size ~{recursive_min_cluster_size}"
      CMD="$CMD --recursive-max-depth ~{recursive_max_depth}"
      CMD="$CMD --recursive-merge-hellinger ~{recursive_merge_hellinger}"
      ~{if defined(n_ancestries) then 'CMD="$CMD --n-ancestries ~{n_ancestries}"' else ''}
      ~{if defined(exclude_seeding_samples) then 'CMD="$CMD --exclude-seeding-samples ~{exclude_seeding_samples}"' else ''}
    fi

    if [ "~{mode}" = "train" ]; then
      CMD="$CMD --n-em-iter ~{n_em_iter} --em-t-policy ~{em_t_policy}"
      CMD="$CMD --held-out-init ~{held_out_init} --block-size ~{block_size}"
      if [ "~{block_emissions}" = "true" ]; then
        CMD="$CMD --block-emissions"
      else
        CMD="$CMD --no-block-emissions"
      fi
      if [ "~{per_hap_T}" = "true" ]; then
        CMD="$CMD --per-hap-T"
      else
        CMD="$CMD --no-per-hap-T"
      fi
      ~{if defined(freeze_anchors_iters) then 'CMD="$CMD --freeze-anchors-iters ~{freeze_anchors_iters}"' else ''}
      ~{if defined(seed_input) then 'CMD="$CMD --seed-input ~{seed_input}"' else ''}
      ~{if defined(ancestry_names) then 'CMD="$CMD --ancestry-names ~{ancestry_names}"' else ''}
    fi

    if [ "~{mode}" = "infer" ]; then
      CMD="$CMD --model ~{model_npz}"
      if [ "~{write_probs}" = "true" ]; then
        CMD="$CMD --probs"
      fi
      if [ "~{write_dense_decode}" = "true" ]; then
        CMD="$CMD --write-dense-decode"
      fi
      ~{if defined(ancestry_names) then 'CMD="$CMD --ancestry-names ~{ancestry_names}"' else ''}
    fi

    if [ -n "~{extra_args}" ]; then
      CMD="$CMD ~{extra_args}"
    fi

    echo "Running: $CMD"
    eval "$CMD"
  >>>

  output {
    # Mode = seed
    File? seed_npz       = "~{output_prefix}.seed.npz"

    # Mode = train
    File? model          = "~{output_prefix}.model"
    File? model_npz_out  = "~{output_prefix}.model.npz"

    # Mode = train and infer
    File? summary        = "~{output_prefix}.summary.json"

    # Mode = infer
    File? global_tsv     = "~{output_prefix}.global.tsv"
    File? tracts         = "~{output_prefix}.tracts.tsv.gz"
    Array[File] decode_parquet = glob("~{output_prefix}.chr*.decode.parquet")
  }

  runtime {
    docker:                docker_image
    predefinedMachineType: machine_type
    gpuType:               gpu_type
    gpuCount:              1
    zones:                 zones
    disks:                 "local-disk ~{disk_size_gb} SSD"
    bootDiskSizeGb:        50
  }
}

workflow popout {
  input {
    String mode
    File   pgen
    File   pvar
    File   psam
    File?  genetic_map
    String chromosome
    String output_prefix = "popout"

    File?  model_npz
    File?  seed_input

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

    Int     n_em_iter        = 20
    Float   gen_since_admix  = 10.0
    String  em_t_policy      = "gated"
    Int?    freeze_anchors_iters
    String  held_out_init    = "soft"
    Boolean block_emissions  = true
    Int     block_size       = 64
    Boolean per_hap_T        = false

    Boolean write_probs         = false
    Boolean write_dense_decode  = false

    Float?  thin_cm
    Float   maf              = 0.01
    Int     seed             = 42
    String? ancestry_names
    String  extra_args       = ""

    String? wandb_key
    String  machine_type  = "a2-highgpu-1g"
    String  gpu_type      = "nvidia-tesla-a100"
    String  zones         = "us-central1-c us-central1-a"
    Int     disk_size_gb  = 500
    String  docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  call popout_task {
    input:
      mode                       = mode,
      pgen                       = pgen,
      pvar                       = pvar,
      psam                       = psam,
      genetic_map                = genetic_map,
      chromosome                 = chromosome,
      output_prefix              = output_prefix,
      model_npz                  = model_npz,
      seed_input                 = seed_input,
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

  output {
    File? seed_npz     = popout_task.seed_npz
    File? model        = popout_task.model
    File? model_npz_out = popout_task.model_npz_out
    File? summary      = popout_task.summary
    File? global_tsv   = popout_task.global_tsv
    File? tracts       = popout_task.tracts
    Array[File] decode_parquet = popout_task.decode_parquet
  }
}
