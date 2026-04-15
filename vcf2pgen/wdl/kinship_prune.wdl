version 1.0

## Compute KING-robust kinship and output a sample exclusion list.
##
## Produces a .king.cutoff.out.id file listing samples to remove.
## Pass this file as the --remove input to filter_pgen.wdl.
##
## --king-cutoff is more memory-efficient than --make-king because
## plink2 streams the computation without materializing the full
## NxN matrix.  Suitable for biobank-scale (500K+ samples).
##
## Default threshold 0.0625 removes 2nd-degree relatives or closer.
## Common thresholds:
##   0.354  — duplicate/MZ twin
##   0.177  — 1st degree (parent-child, full sibling)
##   0.0625 — 2nd degree (half-sibling, avuncular, grandparent)
##   0.0442 — 3rd degree

task kinship_prune_task {
  input {
    File   pgen
    File   pvar
    File   psam
    String output_prefix = basename(pgen, ".pgen") + ".kinship"

    # Kinship coefficient threshold — pairs above this are pruned
    Float  king_cutoff = 0.0625

    # Restrict to specific chromosomes (e.g. "1" or "1-22")
    String? chromosomes

    String extra_args  = ""

    # Higher memory tiers than filter_pgen — kinship is O(n^2)
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  Float pgen_gb = size(pgen, "GB")

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
  Int    disk_size_gb = ceil(pgen_gb * 3) + 200

  command <<<
    set -euo pipefail

    INPUT_PREFIX="input_pfile"
    ln -sf ~{pgen} "${INPUT_PREFIX}.pgen"
    ln -sf ~{pvar} "${INPUT_PREFIX}.pvar"
    ln -sf ~{psam} "${INPUT_PREFIX}.psam"

    plink2 \
      --pfile "${INPUT_PREFIX}" \
      ~{if defined(chromosomes) then '--chr ~{chromosomes}' else ''} \
      --king-cutoff ~{king_cutoff} \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      ~{extra_args}

    echo "=== Samples to remove ==="
    wc -l ~{output_prefix}.king.cutoff.out.id
    echo "=== Samples to keep ==="
    wc -l ~{output_prefix}.king.cutoff.in.id
  >>>

  output {
    File samples_to_remove = "~{output_prefix}.king.cutoff.out.id"
    File samples_to_keep   = "~{output_prefix}.king.cutoff.in.id"
    File log               = "~{output_prefix}.log"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

workflow kinship_prune {
  input {
    File    pgen
    File    pvar
    File    psam
    Float   king_cutoff  = 0.0625
    String? chromosomes
    String  extra_args   = ""
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  call kinship_prune_task {
    input:
      pgen            = pgen,
      pvar            = pvar,
      psam            = psam,
      king_cutoff     = king_cutoff,
      chromosomes     = chromosomes,
      extra_args      = extra_args,
      cpu_override    = cpu_override,
      memory_override = memory_override,
      docker_image    = docker_image
  }

  output {
    File samples_to_remove = kinship_prune_task.samples_to_remove
    File samples_to_keep   = kinship_prune_task.samples_to_keep
    File log               = kinship_prune_task.log
  }
}
