version 1.0

## Convert phased VCF files to PGEN format for use with popout.
##
## Usage on Terra:
##   - Single chromosome: call vcf_to_pgen_task directly
##   - All chromosomes: call vcf_to_pgen workflow with Array[File] of VCFs

task vcf_to_pgen_task {
  input {
    File   vcf
    String output_prefix = basename(vcf, ".vcf.gz")

    # plink2 flags
    String extra_args = ""

    # Runtime — tuned for 500K-sample WGS chromosomes
    Int    cpu          = 8
    String memory       = "32 GB"
    Int    disk_size_gb = ceil(size(vcf, "GB") * 3) + 100
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  command <<<
    set -euo pipefail

    plink2 \
      --vcf ~{vcf} \
      --make-pgen \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      ~{extra_args}

    # Log output sizes
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
    Array[File] vcfs

    String extra_args   = ""
    Int    cpu          = 8
    String memory       = "32 GB"
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  scatter (vcf in vcfs) {
    call vcf_to_pgen_task {
      input:
        vcf          = vcf,
        extra_args   = extra_args,
        cpu          = cpu,
        memory       = memory,
        docker_image = docker_image
    }
  }

  output {
    Array[File] pgens = vcf_to_pgen_task.pgen
    Array[File] pvars = vcf_to_pgen_task.pvar
    Array[File] psams = vcf_to_pgen_task.psam
    Array[File] logs  = vcf_to_pgen_task.log
  }
}
