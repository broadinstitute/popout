version 1.0

## Pure PGEN-to-VCF export with optional sample subsetting.  Symmetric
## companion to vcf_to_pgen.wdl.  Reproduces the input VCF as faithfully
## as plink2 allows: CHROM, POS, ID, REF, ALT, GT (with phasing), and
## sample IDs are preserved.  QUAL, FILTER, and INFO are not stored in
## pgen and will appear as "." in the output.
##
## Sample subsetting uses plink2's native --keep / --remove formats.
## The simplest accepted form is a tab-delimited file with a `#IID`
## header line followed by one IID per line; plink2 will fill FID with 0.
## See https://www.cog-genomics.org/plink/2.0/filter#sample for full syntax.
##
## Resources auto-scale based on PGEN file size.  Override with
## cpu_override / memory_override if needed.
##
## Usage on Terra:
##   Scatter across chromosomes via data table rows, one PGEN triplet per row.

task pgen_to_vcf_task {
  input {
    File   pgen
    File   pvar
    File   psam
    String output_prefix = basename(pgen, ".pgen")

    # Optional sample subsetting (plink2 native sample-list formats)
    File?  keep
    File?  remove

    # Escape hatch for extra plink2 flags
    String extra_args = ""

    # Resource overrides — leave unset for auto-scaling by PGEN size
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/plink2:latest"
  }

  Float pgen_gb = size(pgen, "GB")

  # Auto-scale: bigger PGENs get more CPU + memory
  #   < 10 GB  →  8 CPU,  32 GB  (chr20-22)
  #   10-30 GB → 16 CPU,  64 GB  (chr10-19)
  #   30-60 GB → 32 CPU, 128 GB  (chr3-9)
  #   > 60 GB  → 64 CPU, 256 GB  (chr1-2)
  Int auto_cpu = if pgen_gb > 60.0 then 64
                 else if pgen_gb > 30.0 then 32
                 else if pgen_gb > 10.0 then 16
                 else 8

  String auto_memory = if pgen_gb > 60.0 then "256 GB"
                       else if pgen_gb > 30.0 then "128 GB"
                       else if pgen_gb > 10.0 then "64 GB"
                       else "32 GB"

  Int    cpu          = select_first([cpu_override, auto_cpu])
  String memory       = select_first([memory_override, auto_memory])
  # PGEN → bgzipped VCF expands ~3-5×; size disk for the worst case.
  Int    disk_size_gb = ceil(pgen_gb * 5) + 100

  command <<<
    set -euo pipefail

    # Co-locate pfile triplet (Terra may scatter them across directories)
    INPUT_PREFIX="input_pfile"
    ln -sf ~{pgen} "${INPUT_PREFIX}.pgen"
    ln -sf ~{pvar} "${INPUT_PREFIX}.pvar"
    ln -sf ~{psam} "${INPUT_PREFIX}.psam"

    ARGS=()
    ~{if defined(keep)   then 'ARGS+=(--keep ~{keep})'     else ''}
    ~{if defined(remove) then 'ARGS+=(--remove ~{remove})' else ''}

    plink2 \
      --pfile "${INPUT_PREFIX}" \
      --export vcf-4.2 bgz id-paste=iid \
      --output-chr chrM \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      "${ARGS[@]}" \
      ~{extra_args}

    ls -lh ~{output_prefix}.vcf.gz
    grep -E '(samples|variants) (loaded|remaining|written)' ~{output_prefix}.log || true
  >>>

  output {
    File vcf = "~{output_prefix}.vcf.gz"
    File log = "~{output_prefix}.log"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

workflow pgen_to_vcf {
  input {
    File   pgen
    File   pvar
    File   psam
    String output_prefix = basename(pgen, ".pgen")

    File?  keep
    File?  remove

    String extra_args = ""

    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/plink2:latest"
  }

  call pgen_to_vcf_task {
    input:
      pgen            = pgen,
      pvar            = pvar,
      psam            = psam,
      output_prefix   = output_prefix,
      keep            = keep,
      remove          = remove,
      extra_args      = extra_args,
      cpu_override    = cpu_override,
      memory_override = memory_override,
      docker_image    = docker_image
  }

  output {
    File vcf = pgen_to_vcf_task.vcf
    File log = pgen_to_vcf_task.log
  }
}
