version 1.0

## Filter existing PGEN files (biallelic SNPs, common variants).
##
## Applies the same QC filters as vcf_to_pgen but reads from --pfile
## instead of --vcf.  Useful for filtering PGENs that were converted
## without QC, or for re-filtering with different thresholds.
##
## Resources auto-scale based on PGEN file size.

task filter_pgen_task {
  input {
    File   pgen
    File   pvar
    File   psam
    String output_prefix = basename(pgen, ".pgen") + ".filtered"

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

    # Additional plink2 flags
    String extra_args = ""

    # Resource overrides — leave unset for auto-scaling by PGEN size
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  Float pgen_gb = size(pgen, "GB")

  Int auto_cpu = if pgen_gb > 60.0 then 64
                 else if pgen_gb > 30.0 then 32
                 else if pgen_gb > 10.0 then 16
                 else 8

  String auto_memory = if pgen_gb > 60.0 then "256 GB"
                       else if pgen_gb > 30.0 then "128 GB"
                       else if pgen_gb > 10.0 then "64 GB"
                       else "32 GB"

  Int    cpu         = select_first([cpu_override, auto_cpu])
  String memory      = select_first([memory_override, auto_memory])
  Int    disk_size_gb = ceil(pgen_gb * 3) + 100

  command <<<
    set -euo pipefail

    # Co-locate pfile triplet (Terra may scatter them across directories)
    INPUT_PREFIX="input_pfile"
    ln -sf ~{pgen} "${INPUT_PREFIX}.pgen"
    ln -sf ~{pvar} "${INPUT_PREFIX}.pvar"
    ln -sf ~{psam} "${INPUT_PREFIX}.psam"

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
      --pfile "${INPUT_PREFIX}" \
      --make-pgen \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      "${ARGS[@]}" \
      ~{"--extract " + extract} \
      ~{extra_args}

    ls -lh ~{output_prefix}.{pgen,pvar,psam}
    grep -E '(variants loaded|remaining after)' ~{output_prefix}.log || true
  >>>

  output {
    File filtered_pgen = "~{output_prefix}.pgen"
    File filtered_pvar = "~{output_prefix}.pvar"
    File filtered_psam = "~{output_prefix}.psam"
    File log           = "~{output_prefix}.log"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

workflow filter_pgen {
  input {
    Array[File] pgens
    Array[File] pvars
    Array[File] psams

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

    # Resource overrides (per-task)
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  scatter (idx in range(length(pgens))) {
    call filter_pgen_task {
      input:
        pgen             = pgens[idx],
        pvar             = pvars[idx],
        psam             = psams[idx],
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
  }

  output {
    Array[File] filtered_pgens = filter_pgen_task.filtered_pgen
    Array[File] filtered_pvars = filter_pgen_task.filtered_pvar
    Array[File] filtered_psams = filter_pgen_task.filtered_psam
    Array[File] logs           = filter_pgen_task.log
  }
}
