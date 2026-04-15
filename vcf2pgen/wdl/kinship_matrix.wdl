version 1.0

## Compute KING-robust kinship table from PGEN files.
##
## Uses --make-king-table (sparse pair table) instead of --make-king
## (full NxN matrix) because the full matrix is infeasible at biobank
## scale (500K samples × 500K = ~1 TB RAM + ~1 TB output).
##
## The table outputs only pairs above --king-table-filter threshold,
## keeping output manageable.  Default filter 0.0442 captures 3rd-degree
## relatives and closer.
##
## For generating a sample exclusion list, use kinship_prune.wdl instead.

task kinship_matrix_task {
  input {
    File   pgen
    File   pvar
    File   psam
    String output_prefix = basename(pgen, ".pgen") + ".kinship"

    # Restrict to specific chromosomes (e.g. "1" or "1-22")
    String? chromosomes

    # Only output pairs with kinship above this threshold
    Float king_table_filter = 0.0442

    String extra_args  = ""

    # Kinship is O(n^2) in sample count — needs more memory than
    # simple plink2 operations.  Default 128 GB, override for larger cohorts.
    Int?    cpu_override
    String? memory_override
    Int?    disk_size_gb_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  Float pgen_gb = size(pgen, "GB")

  # Kinship memory scales with sample count (n^2), not variant count.
  # PGEN size is a rough proxy — larger files usually mean more samples
  # or more variants.  These tiers are higher than filter_pgen because
  # of the O(n^2) memory footprint.
  Int auto_cpu = if pgen_gb > 60.0 then 96
                 else if pgen_gb > 30.0 then 64
                 else if pgen_gb > 10.0 then 32
                 else 16

  String auto_memory = if pgen_gb > 60.0 then "624 GB"
                       else if pgen_gb > 30.0 then "256 GB"
                       else if pgen_gb > 10.0 then "128 GB"
                       else "64 GB"

  Int    cpu         = select_first([cpu_override, auto_cpu])
  String memory      = select_first([memory_override, auto_memory])
  Int    disk_size_gb = select_first([disk_size_gb_override, ceil(pgen_gb * 3) + 200])

  command <<<
    set -euo pipefail

    INPUT_PREFIX="input_pfile"
    ln -sf ~{pgen} "${INPUT_PREFIX}.pgen"
    ln -sf ~{pvar} "${INPUT_PREFIX}.pvar"
    ln -sf ~{psam} "${INPUT_PREFIX}.psam"

    plink2 \
      --pfile "${INPUT_PREFIX}" \
      ~{if defined(chromosomes) then '--chr ~{chromosomes}' else ''} \
      --make-king-table \
      --king-table-filter ~{king_table_filter} \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      ~{extra_args}

    echo "=== Kinship pairs found ==="
    wc -l ~{output_prefix}.kin0
    ls -lh ~{output_prefix}.kin0
  >>>

  output {
    File kinship_table = "~{output_prefix}.kin0"
    File log           = "~{output_prefix}.log"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

workflow kinship_matrix {
  input {
    File    pgen
    File    pvar
    File    psam
    String? chromosomes
    Float   king_table_filter = 0.0442
    String  extra_args        = ""
    Int?    cpu_override
    String? memory_override
    Int?    disk_size_gb_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  call kinship_matrix_task {
    input:
      pgen                  = pgen,
      pvar                  = pvar,
      psam                  = psam,
      chromosomes           = chromosomes,
      king_table_filter     = king_table_filter,
      extra_args            = extra_args,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      docker_image          = docker_image
  }

  output {
    File kinship_table = kinship_matrix_task.kinship_table
    File log           = kinship_matrix_task.log
  }
}
