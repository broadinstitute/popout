version 1.0

## Pure VCF-to-PGEN conversion.  No filtering — use filter_pgen.wdl for that.
##
## Resources auto-scale based on VCF file size.  Override with
## cpu_override / memory_override if needed.
##
## Usage on Terra:
##   Scatter across chromosomes via data table rows, one VCF per row.

task vcf_to_pgen_task {
  input {
    File   vcf
    String output_prefix = basename(vcf, ".vcf.gz")
    String extra_args    = ""

    # Resource overrides — leave unset for auto-scaling by VCF size
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  Float vcf_gb = size(vcf, "GB")

  # Auto-scale: bigger VCFs get more CPU + memory
  #   < 10 GB  →  8 CPU,  32 GB  (chr20-22)
  #   10-30 GB → 16 CPU,  64 GB  (chr10-19)
  #   30-60 GB → 32 CPU, 128 GB  (chr3-9)
  #   > 60 GB  → 64 CPU, 256 GB  (chr1-2)
  Int auto_cpu = if vcf_gb > 60.0 then 64
                 else if vcf_gb > 30.0 then 32
                 else if vcf_gb > 10.0 then 16
                 else 8

  String auto_memory = if vcf_gb > 60.0 then "256 GB"
                       else if vcf_gb > 30.0 then "128 GB"
                       else if vcf_gb > 10.0 then "64 GB"
                       else "32 GB"

  Int    cpu         = select_first([cpu_override, auto_cpu])
  String memory      = select_first([memory_override, auto_memory])
  Int    disk_size_gb = ceil(vcf_gb * 3) + 100

  command <<<
    set -euo pipefail

    plink2 \
      --vcf ~{vcf} \
      --make-pgen \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      ~{extra_args}

    ls -lh ~{output_prefix}.{pgen,pvar,psam}
  >>>

  output {
    File pgen = "~{output_prefix}.pgen"
    File pvar = "~{output_prefix}.pvar"
    File psam = "~{output_prefix}.psam"
    File log  = "~{output_prefix}.log"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

workflow vcf_to_pgen {
  input {
    File    vcf
    String  extra_args   = ""
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  call vcf_to_pgen_task {
    input:
      vcf             = vcf,
      extra_args      = extra_args,
      cpu_override    = cpu_override,
      memory_override = memory_override,
      docker_image    = docker_image
  }

  output {
    File pgen = vcf_to_pgen_task.pgen
    File pvar = vcf_to_pgen_task.pvar
    File psam = vcf_to_pgen_task.psam
    File log  = vcf_to_pgen_task.log
  }
}
