version 1.0

## Run popout (GPU-accelerated local ancestry inference) on Terra.
##
## Inputs: per-chromosome PGEN files + genetic map.
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
    File         genetic_map
    String       output_prefix = "popout"

    # popout algorithm options
    Int?    n_ancestries
    Int     n_em_iter       = 3
    Float?  thin_cm
    String  method          = "hmm"
    Float   gen_since_admix = 20.0
    Boolean export_panel    = false
    Boolean block_emissions = false
    String  extra_args      = ""

    # Runtime — machine_type and gpu_type must match (see header comment)
    String machine_type  = "a2-highgpu-1g"
    String gpu_type      = "nvidia-tesla-a100"
    String zones         = "us-central1-c us-central1-a"
    Int    disk_size_gb  = 500
    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:0.1.0"
  }

  command <<<
    set -euo pipefail

    # ---- Localize PGEN files into a single directory ----
    # Terra scatters files into separate paths; popout expects them
    # co-located in one directory.
    mkdir -p pgen_dir
    pgens=(~{sep=' ' pgens})
    pvars=(~{sep=' ' pvars})
    psams=(~{sep=' ' psams})

    for i in "${!pgens[@]}"; do
      ln -s "${pgens[$i]}" pgen_dir/
      ln -s "${pvars[$i]}" pgen_dir/
      ln -s "${psams[$i]}" pgen_dir/
    done

    echo "=== Localized PGEN files ==="
    ls -lh pgen_dir/

    # ---- GPU check ----
    nvidia-smi || echo "WARNING: nvidia-smi failed"

    # ---- Build popout command ----
    CMD="popout --pgen pgen_dir/ --map ~{genetic_map} --out ~{output_prefix}"
    CMD="$CMD --n-em-iter ~{n_em_iter}"
    CMD="$CMD --method ~{method}"
    CMD="$CMD --gen-since-admix ~{gen_since_admix}"

    ~{if defined(n_ancestries) then 'CMD="$CMD --n-ancestries ~{n_ancestries}"' else ''}
    ~{if defined(thin_cm) then 'CMD="$CMD --thin-cm ~{thin_cm}"' else ''}
    ~{if export_panel then 'CMD="$CMD --export-panel"' else ''}
    ~{if block_emissions then 'CMD="$CMD --block-emissions"' else ''}

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
    File         genetic_map
    String       output_prefix = "popout"

    # Algorithm options
    Int?    n_ancestries
    Int     n_em_iter       = 3
    Float?  thin_cm
    String  method          = "hmm"
    Float   gen_since_admix = 20.0
    Boolean export_panel    = false
    Boolean block_emissions = false
    String  extra_args      = ""

    # Runtime — machine_type and gpu_type must match (see header comment)
    String machine_type  = "a2-highgpu-1g"
    String gpu_type      = "nvidia-tesla-a100"
    String zones         = "us-central1-c us-central1-a"
    Int    disk_size_gb  = 500
    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:0.1.0"
  }

  call popout_task {
    input:
      pgens           = pgens,
      pvars           = pvars,
      psams           = psams,
      genetic_map     = genetic_map,
      output_prefix   = output_prefix,
      n_ancestries    = n_ancestries,
      n_em_iter       = n_em_iter,
      thin_cm         = thin_cm,
      method          = method,
      gen_since_admix = gen_since_admix,
      export_panel    = export_panel,
      block_emissions = block_emissions,
      extra_args      = extra_args,
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

    File? panel_haplotypes  = popout_task.panel_haplotypes
    File? panel_segments    = popout_task.panel_segments
    File? panel_frequencies = popout_task.panel_frequencies
    File? panel_proportions = popout_task.panel_proportions
  }
}
