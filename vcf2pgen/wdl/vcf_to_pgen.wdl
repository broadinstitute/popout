version 1.0

## Convert phased VCF files to PGEN format for use with popout.
##
## By default, applies strict QC filters during conversion (single plink2 pass):
##   --chr 1-22 --min-alleles 2 --max-alleles 2 --snps-only just-acgt
##   --var-filter --maf 0.01 --geno 0.01 --set-all-var-ids @:#:$r:$a
##   --rm-dup exclude-all
## Set apply_qc_filters = false for unfiltered conversion.
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

    # QC filters — defaults match popout pipeline requirements
    Boolean apply_qc_filters = true
    String  chromosomes      = "1-22"
    Int     min_alleles      = 2
    Int     max_alleles      = 2
    String  snps_only        = "just-acgt"
    Boolean var_filter       = true
    Float   maf              = 0.01
    Float   geno             = 0.01
    String  set_all_var_ids  = "@:#:$r:$a"
    String  rm_dup           = "exclude-all"

    # Optional variant extraction list
    File?   extract

    # Additional plink2 flags (e.g., --keep)
    String extra_args = ""

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

    ARGS=()

    if [ "~{apply_qc_filters}" = "true" ]; then
      ARGS+=(--chr ~{chromosomes})
      ARGS+=(--min-alleles ~{min_alleles} --max-alleles ~{max_alleles})
      if [ -n "~{snps_only}" ]; then
        ARGS+=(--snps-only '~{snps_only}')
      fi
      if [ "~{var_filter}" = "true" ]; then
        ARGS+=(--var-filter)
      fi
      ARGS+=(--maf ~{maf} --geno ~{geno})
      if [ -n "~{set_all_var_ids}" ]; then
        ARGS+=(--set-all-var-ids '~{set_all_var_ids}')
      fi
      if [ -n "~{rm_dup}" ]; then
        ARGS+=(--rm-dup ~{rm_dup})
      fi
    fi

    plink2 \
      --vcf ~{vcf} \
      --make-pgen \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      "${ARGS[@]}" \
      ~{"--extract " + extract} \
      ~{extra_args}

    # Log output sizes and filter summary
    ls -lh ~{output_prefix}.{pgen,pvar,psam}
    grep -E '(variants loaded|remaining after)' ~{output_prefix}.log || true
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
    File vcf

    # QC filters
    Boolean apply_qc_filters = true
    String  chromosomes      = "1-22"
    Int     min_alleles      = 2
    Int     max_alleles      = 2
    String  snps_only        = "just-acgt"
    Boolean var_filter       = true
    Float   maf              = 0.01
    Float   geno             = 0.01
    String  set_all_var_ids  = "@:#:$r:$a"
    String  rm_dup           = "exclude-all"

    File?   extract

    String  extra_args       = ""

    # Resource overrides
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  call vcf_to_pgen_task {
    input:
      vcf              = vcf,
      apply_qc_filters = apply_qc_filters,
      chromosomes      = chromosomes,
      min_alleles      = min_alleles,
      max_alleles      = max_alleles,
      snps_only        = snps_only,
      var_filter       = var_filter,
      maf              = maf,
      geno             = geno,
      set_all_var_ids  = set_all_var_ids,
      rm_dup           = rm_dup,
      extract          = extract,
      extra_args       = extra_args,
      cpu_override     = cpu_override,
      memory_override  = memory_override,
      docker_image     = docker_image
  }

  output {
    File pgen = vcf_to_pgen_task.pgen
    File pvar = vcf_to_pgen_task.pvar
    File psam = vcf_to_pgen_task.psam
    File log  = vcf_to_pgen_task.log
  }
}
