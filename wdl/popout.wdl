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
    Int     n_em_iter          = 20
    Float?  thin_cm
    String  method             = "hmm"
    Float   gen_since_admix    = 20.0
    Boolean export_panel       = false
    Boolean block_emissions    = false
    Float   freq_damping       = 0.0
    String  extra_args         = ""

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

    # ---- QC: biallelic SNPs, common variants only ----
    # pgenlib (v0.94) can't read multiallelic+phased PGENs, and its count()
    # returns wrong values for plink2-filtered files.  Offload all variant
    # QC to plink2 which is proven correct at biobank scale.
    mkdir -p pgen_biallelic
    for pf in pgen_dir/*.pgen; do
      base=$(basename "$pf" .pgen)
      plink2 --pfile "pgen_dir/${base}" \
             --max-alleles 2 \
             --snps-only just-acgt \
             --maf 0.005 \
             --mac 50 \
             --make-pgen \
             --out "pgen_biallelic/${base}" \
             --threads "$(nproc)" \
             --memory "$(free -m | awk '/Mem:/{print int($7*0.8)}')"
    done
    # Use the filtered files from now on
    rm -rf pgen_dir
    mv pgen_biallelic pgen_dir
    echo "=== After biallelic filter ==="
    ls -lh pgen_dir/

    # ---- GPU check ----
    nvidia-smi || echo "WARNING: nvidia-smi failed"

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
    if awk "BEGIN{exit !(~{freq_damping} > 0)}"; then CMD="$CMD --freq-damping ~{freq_damping}"; fi
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
    File global_ancestry = "~{output_prefix}.global.tsv"
    File tracts           = "~{output_prefix}.tracts.tsv.gz"
    File model            = "~{output_prefix}.model"
    File model_npz        = "~{output_prefix}.model.npz"
    File summary          = "~{output_prefix}.summary.json"
    File? spectral_npz     = "~{output_prefix}.spectral.npz"
    File? stats_jsonl      = "~{output_prefix}.stats.jsonl"

    # Optional panel exports (only produced when export_panel = true)
    File? panel_haplotypes  = "~{output_prefix}.panel.haplotypes.tsv"
    File? panel_segments    = "~{output_prefix}.panel.segments.tsv.gz"
    File? panel_frequencies = "~{output_prefix}.panel.frequencies.tsv.gz"
    File? panel_proportions = "~{output_prefix}.panel.proportions.tsv"
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
    Int     n_em_iter          = 20
    Float?  thin_cm
    String  method             = "hmm"
    Float   gen_since_admix    = 20.0
    Boolean export_panel       = false
    Boolean block_emissions    = false
    Float   freq_damping       = 0.0
    String  extra_args         = ""

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
      freq_damping    = freq_damping,
      extra_args      = extra_args,
      wandb_key       = wandb_key,
      machine_type    = machine_type,
      gpu_type        = gpu_type,
      zones           = zones,
      disk_size_gb    = disk_size_gb,
      docker_image    = docker_image
  }

  output {
    File global_ancestry = popout_task.global_ancestry
    File tracts           = popout_task.tracts
    File model            = popout_task.model
    File model_npz        = popout_task.model_npz
    File summary          = popout_task.summary
    File? spectral_npz     = popout_task.spectral_npz
    File? stats_jsonl      = popout_task.stats_jsonl

    File? panel_haplotypes  = popout_task.panel_haplotypes
    File? panel_segments    = popout_task.panel_segments
    File? panel_frequencies = popout_task.panel_frequencies
    File? panel_proportions = popout_task.panel_proportions
  }
}
