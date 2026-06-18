version 1.0

## Filter PGEN files with plink2 for downstream popout.
##
## Popout-focused, single plink2 pass: biallelic SNPs only, palindromic
## removal, FILTER!=PASS exclusion, MAF/missingness/HWE thresholds,
## centromere/blacklist BED exclusion, deterministic variant IDs and
## duplicate removal.
##
## No `--extract`, `--exclude`, `--keep`, `--remove`, or `extra_args`
## escape hatches are exposed -- those previously enabled the 3-pass
## AIM-panel-union workflow whose Pass C emitted variants without
## re-applying MAF/HWE/mind/palindromic checks, producing 40M+ -variant
## "filtered" PGENs that popout's own MAF filter then catastrophically
## reduced. This task is the single source of truth for popout-ready
## filtered PGEN triplets; AIM-panel work belongs in a separate
## workflow.
##
## Defaults are tuned for the AoU v9 cohort. Override any of them per
## inputs.json without touching the WDL.
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
    # Defaults match the standard popout filter profile: biallelic
    # ACGT SNPs only, palindromic removed (strand-ambiguous so popout's
    # phased-genotype HMM can't disambiguate them anyway).
    String?  chromosomes                         # --chr (usually unneeded per-chrom scatter)
    Int      min_alleles               = 2       # --min-alleles
    Int      max_alleles               = 2       # --max-alleles
    String   snps_only                 = "just-acgt"  # --snps-only
    Boolean  exclude_palindromic_snps  = true    # --exclude-palindromic-snps

    # ---- Quality filters ----
    # MAF 0.05 matches the main_v10 baseline. Tighten or loosen per
    # cohort. mind/geno bound sample-level and variant-level
    # missingness; HWE excludes hard violations at 1e-10 (keep-fewhet
    # only removes the heterozygote-deficit side, preserving population
    # structure signal popout uses).
    Boolean  var_filter   = true                 # --var-filter (exclude FILTER!=PASS)
    Float    maf          = 0.05                 # --maf
    Float    geno         = 0.01                 # --geno
    Float    mind         = 0.05                 # --mind
    Float    hwe          = 0.0000000001         # --hwe 1e-10 threshold
    String   hwe_modifier = "keep-fewhet"        # modifier for --hwe

    # ---- Variant ID normalization ----
    # Stable chr:pos:ref:alt IDs so cross-chrom merges and later panel
    # joins are unambiguous; exclude-all drops duplicate IDs after
    # normalization.
    String   set_all_var_ids = "@:#:$r:$a"       # --set-all-var-ids
    String   rm_dup          = "exclude-all"     # --rm-dup

    # ---- Region exclusion ----
    # Centromeres + ENCODE blacklist + high-LD regions. Empty array =
    # no region exclusion. Standard popout inputs:
    #   gs://fc-secure-1c6b9393-5e5d-4a87-b483-d1f0b019af92/beds/hg38-blacklist.v2.bed
    #   gs://fc-secure-1c6b9393-5e5d-4a87-b483-d1f0b019af92/beds/high-LD-regions-hg38-GRCh38.bed
    Array[File]  exclude_range_beds = []

    # ---- Resources ----
    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/plink2:latest"
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

  Int    cpu          = select_first([cpu_override, auto_cpu])
  String memory       = select_first([memory_override, auto_memory])
  Int    disk_size_gb = ceil(pgen_gb * 3) + 100

  command <<<
    set -euo pipefail

    # Co-locate pfile triplet (Terra may scatter them across directories)
    INPUT_PREFIX="input_pfile"
    ln -sf ~{pgen} "${INPUT_PREFIX}.pgen"
    ln -sf ~{pvar} "${INPUT_PREFIX}.pvar"
    ln -sf ~{psam} "${INPUT_PREFIX}.psam"

    ARGS=()

    # -- Variant-type --
    ~{if defined(chromosomes) then 'ARGS+=(--chr ~{chromosomes})' else ''}
    ARGS+=(--min-alleles ~{min_alleles})
    ARGS+=(--max-alleles ~{max_alleles})
    ARGS+=(--snps-only '~{snps_only}')
    ~{if exclude_palindromic_snps then 'ARGS+=(--exclude-palindromic-snps)' else ''}

    # -- Quality --
    ~{if var_filter then 'ARGS+=(--var-filter)' else ''}
    ARGS+=(--maf ~{maf})
    ARGS+=(--geno ~{geno})
    ARGS+=(--mind ~{mind})
    ARGS+=(--hwe ~{hwe} '~{hwe_modifier}')

    # -- Variant ID normalization --
    # Single quotes protect $r/$a from bash expansion under set -u
    ARGS+=(--set-all-var-ids '~{set_all_var_ids}')
    ARGS+=(--rm-dup ~{rm_dup})

    # -- Region exclusion --
    BEDS=(~{sep=' ' exclude_range_beds})
    if [ "${#BEDS[@]}" -gt 0 ]; then
      cat "${BEDS[@]}" > combined_exclude_ranges.bed
      ARGS+=(--exclude range combined_exclude_ranges.bed)
    fi

    echo "=== plink2 filter args ==="
    printf '%s\n' "${ARGS[@]}"
    echo "=========================="

    plink2 \
      --pfile "${INPUT_PREFIX}" \
      --make-pgen \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      "${ARGS[@]}"

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

    String?      chromosomes
    Int          min_alleles               = 2
    Int          max_alleles               = 2
    String       snps_only                 = "just-acgt"
    Boolean      exclude_palindromic_snps  = true

    Boolean      var_filter   = true
    Float        maf          = 0.05
    Float        geno         = 0.01
    Float        mind         = 0.05
    Float        hwe          = 0.0000000001
    String       hwe_modifier = "keep-fewhet"

    String       set_all_var_ids = "@:#:$r:$a"
    String       rm_dup          = "exclude-all"

    Array[File]  exclude_range_beds = []

    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/plink2:latest"
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
      exclude_range_beds        = exclude_range_beds,
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
