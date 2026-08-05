version 1.0

## Convert a phased BCF to bgzipped VCF (+ tabix index).
##
## FLARE reads its `gt=` input via HTSJDK, which handles .vcf.gz + .tbi
## but not .bcf directly for the phased target VCF path used at biobank
## scale. This task is the minimal shim: one `bcftools view -Oz` + one
## `bcftools index --tbi`, sized to fit input + output + slack on
## local disk with a generous multiplier.
##
## No magicwand instrumentation by design — this is a straightforward
## conversion that runs once per chromosome and doesn't need per-command
## W&B rows.

task bcftools_bcf_to_vcf_task {
  input {
    File   bcf
    File   bcf_index               # .csi (co-localized so the input can be indexed for consistency)
    String output_basename

    # Resource overrides
    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 2

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  # Disk sizing: input BCF localizes to /cromwell_root, output VCF.bgz is
  # typically 1.2-1.5x the BCF for phased genotype content, plus the tbi
  # (tiny) and slack for /cromwell_root overhead. 4x + 50GB is intentionally
  # generous per "err on the side of the analysis"; floored at 100GB so
  # even sub-GB inputs get a comfortable working volume.
  Float bcf_gb          = size(bcf, "GB")
  Int   sized_disk_gb   = ceil(bcf_gb * 4.0) + 50
  Int   auto_disk_gb    = if sized_disk_gb > 100 then sized_disk_gb else 100

  Int    cpu          = select_first([cpu_override, 8])
  String memory       = select_first([memory_override, "8 GB"])
  Int    disk_size_gb = select_first([disk_size_gb_override, auto_disk_gb])

  command <<<
    set -euo pipefail

    OUT_FILE="~{output_basename}.vcf.gz"

    echo "input:  ~{bcf} ($(stat -c %s ~{bcf}) bytes)"
    echo "output: $OUT_FILE"
    echo "cpu=~{cpu} memory=~{memory} disk=~{disk_size_gb}GB (~{disk_type})"

    bcftools view \
      --output-type z \
      --output "$OUT_FILE" \
      --threads ~{cpu} \
      ~{bcf}

    bcftools index --tbi --threads ~{cpu} "$OUT_FILE"

    echo "output size: $(stat -c %s "$OUT_FILE") bytes"
    ls -lh "$OUT_FILE" "$OUT_FILE.tbi"
  >>>

  output {
    File vcf_gz     = "~{output_basename}.vcf.gz"
    File vcf_gz_tbi = "~{output_basename}.vcf.gz.tbi"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow bcftools_bcf_to_vcf {
  input {
    File   bcf
    File   bcf_index
    String output_basename

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 2

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  call bcftools_bcf_to_vcf_task {
    input:
      bcf                   = bcf,
      bcf_index             = bcf_index,
      output_basename       = output_basename,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      disk_type             = disk_type,
      preemptible           = preemptible,
      docker_image          = docker_image
  }

  output {
    File vcf_gz     = bcftools_bcf_to_vcf_task.vcf_gz
    File vcf_gz_tbi = bcftools_bcf_to_vcf_task.vcf_gz_tbi
  }
}
