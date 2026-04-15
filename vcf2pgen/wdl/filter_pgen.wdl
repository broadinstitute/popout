version 1.0

## Filter PGEN files with plink2.  Each filter is individually optional —
## undefined inputs are not passed to plink2.  Combine inputs to build
## different filter profiles for downstream analysis.
##
## Resources auto-scale based on PGEN file size.
##
## Usage on Terra:
##   Scatter across chromosomes via data table rows, one PGEN per row.

task filter_pgen_task {
  input {
    File   pgen
    File   pvar
    File   psam
    String output_prefix = basename(pgen, ".pgen") + ".filtered"

    # ---- Variant-type filters ----
    String?  chromosomes                        # --chr, e.g. "1-22"
    Int?     min_alleles                        # --min-alleles
    Int?     max_alleles                        # --max-alleles
    String?  snps_only                          # --snps-only, e.g. "just-acgt"
    Boolean  exclude_palindromic_snps = false   # --exclude-palindromic-snps

    # ---- Quality filters ----
    Boolean  var_filter = true                  # --var-filter (exclude FILTER!=PASS)
    Float?   maf                                # --maf
    Float?   geno                               # --geno (variant missingness)
    Float?   mind                               # --mind (sample missingness)
    Float?   hwe                                # --hwe <threshold> [modifier]
    String   hwe_modifier = "keep-fewhet"       # modifier for --hwe

    # ---- Variant ID normalization ----
    String?  set_all_var_ids                    # --set-all-var-ids, e.g. "@:#:$r:$a"
    String?  rm_dup                             # --rm-dup, e.g. "exclude-all"

    # ---- Include/exclude lists ----
    File?        extract                        # --extract <variant list>
    File?        exclude                        # --exclude <variant list>
    File?        remove                         # --remove <sample list>
    File?        keep                           # --keep <sample list>
    Array[File]  exclude_range_beds = []        # BED files concatenated into --exclude range

    # ---- Escape hatch ----
    String extra_args = ""

    # ---- Resources ----
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

    # -- Variant-type filters --
    ~{if defined(chromosomes) then 'ARGS+=(--chr ~{chromosomes})'                     else ''}
    ~{if defined(min_alleles) then 'ARGS+=(--min-alleles ~{min_alleles})'             else ''}
    ~{if defined(max_alleles) then 'ARGS+=(--max-alleles ~{max_alleles})'             else ''}
    ~{if defined(snps_only)   then "ARGS+=(--snps-only '~{snps_only}')"               else ''}
    ~{if exclude_palindromic_snps then 'ARGS+=(--exclude-palindromic-snps)'           else ''}

    # -- Quality filters --
    ~{if var_filter    then 'ARGS+=(--var-filter)'    else ''}
    ~{if defined(maf)  then 'ARGS+=(--maf ~{maf})'   else ''}
    ~{if defined(geno) then 'ARGS+=(--geno ~{geno})' else ''}
    ~{if defined(mind) then 'ARGS+=(--mind ~{mind})' else ''}

    # HWE: compound flag (threshold + optional modifier)
    if [ -n "~{default='' hwe}" ]; then
      if [ -n "~{hwe_modifier}" ]; then
        ARGS+=(--hwe ~{hwe} '~{hwe_modifier}')
      else
        ARGS+=(--hwe ~{hwe})
      fi
    fi

    # -- Variant ID normalization --
    # Single quotes protect $r/$a from bash expansion under set -u
    if [ -n '~{default="" set_all_var_ids}' ]; then
      ARGS+=(--set-all-var-ids '~{set_all_var_ids}')
    fi
    ~{if defined(rm_dup) then 'ARGS+=(--rm-dup ~{rm_dup})' else ''}

    # -- Include/exclude lists --
    ~{if defined(extract) then 'ARGS+=(--extract ~{extract})' else ''}
    ~{if defined(exclude) then 'ARGS+=(--exclude ~{exclude})' else ''}
    ~{if defined(remove)  then 'ARGS+=(--remove ~{remove})'   else ''}
    ~{if defined(keep)    then 'ARGS+=(--keep ~{keep})'        else ''}

    # Region exclusion: concatenate multiple BED files for --exclude range
    BEDS=(~{sep=' ' exclude_range_beds})
    if [ "${#BEDS[@]}" -gt 0 ]; then
      cat "${BEDS[@]}" > combined_exclude_ranges.bed
      ARGS+=(--exclude range combined_exclude_ranges.bed)
    fi

    plink2 \
      --pfile "${INPUT_PREFIX}" \
      --make-pgen \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      "${ARGS[@]}" \
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
    File   pgen
    File   pvar
    File   psam

    # Variant-type filters
    String?  chromosomes
    Int?     min_alleles
    Int?     max_alleles
    String?  snps_only
    Boolean  exclude_palindromic_snps = false

    # Quality filters
    Boolean  var_filter = true
    Float?   maf
    Float?   geno
    Float?   mind
    Float?   hwe
    String   hwe_modifier = "keep-fewhet"

    # Variant ID normalization
    String?  set_all_var_ids
    String?  rm_dup

    # Include/exclude lists
    File?        extract
    File?        exclude
    File?        remove
    File?        keep
    Array[File]  exclude_range_beds = []

    String  extra_args = ""

    # Resources
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  call filter_pgen_task {
    input:
      pgen                      = pgen,
      pvar                      = pvar,
      psam                      = psam,
      chromosomes               = chromosomes,
      min_alleles               = min_alleles,
      max_alleles               = max_alleles,
      snps_only                 = snps_only,
      exclude_palindromic_snps  = exclude_palindromic_snps,
      var_filter                = var_filter,
      maf                       = maf,
      geno                      = geno,
      mind                      = mind,
      hwe                       = hwe,
      hwe_modifier              = hwe_modifier,
      set_all_var_ids           = set_all_var_ids,
      rm_dup                    = rm_dup,
      extract                   = extract,
      exclude                   = exclude,
      remove                    = remove,
      keep                      = keep,
      exclude_range_beds        = exclude_range_beds,
      extra_args                = extra_args,
      cpu_override              = cpu_override,
      memory_override           = memory_override,
      docker_image              = docker_image
  }

  output {
    File filtered_pgen = filter_pgen_task.filtered_pgen
    File filtered_pvar = filter_pgen_task.filtered_pvar
    File filtered_psam = filter_pgen_task.filtered_psam
    File log           = filter_pgen_task.log
  }
}
