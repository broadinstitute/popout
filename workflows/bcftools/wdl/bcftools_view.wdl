version 1.0

## General-purpose `bcftools view` wrapper. Most of bcftools view's options
## are exposed as optional inputs; unset options are omitted from the
## command line. Scatter at the workflow/Terra level for per-chromosome or
## per-region chunking (one shard per region) rather than baking the
## region list into this task.
##
## Example: chunk a single merged VCF into 22 per-chrom outputs from a
## Terra `chromosome` data table by setting `regions = this.chromosome_id`
## on a "Run on selected entities" submission with 22 chromosome rows.
##
## Region queries (`-r/-R`) require a .tbi/.csi index next to the input;
## pass it via `vcf_index` so Cromwell co-localizes it. Pure sample/filter
## subsetting (no region) does not need an index.
##
## Observability: magicwand instrumentation matches the rest of the
## workflows/bcftools/ group — single optional `wandb_api_key`, project
## hardcoded to `bcftools_view`.

task bcftools_view_task {
  input {
    File   vcf
    File?  vcf_index             # required when `regions` or `regions_file` is set
    String output_basename

    # ---- Region / target selection ----
    String?  regions             # -r/--regions, e.g. "chr1" or "chr1,chr2"
    File?    regions_file        # -R/--regions-file (BED or chr:from-to)
    String?  targets             # -t/--targets
    File?    targets_file        # -T/--targets-file

    # ---- Sample selection ----
    String?  samples             # -s/--samples (comma-separated; "^" prefix excludes)
    File?    samples_file        # -S/--samples-file
    Boolean  force_samples       = false      # --force-samples
    Boolean  no_update           = false      # --no-update (skip INFO/AC,AN recompute after sample subset)

    # ---- Variant filter expressions ----
    String?  include_expr        # -i/--include
    String?  exclude_expr        # -e/--exclude

    # ---- Allele count / frequency filters ----
    Int?     min_ac
    Int?     max_ac
    Float?   min_af
    Float?   max_af
    Int?     min_alleles
    Int?     max_alleles

    # ---- Type filters ----
    String?  types               # --types (snps,indels,mnps,refs,other; comma-separated)
    String?  exclude_types

    # ---- Site-level pass-through filters ----
    String?  apply_filters       # -f/--apply-filters (e.g. "PASS,.")
    Boolean  trim_alt_alleles    = false      # -a/--trim-alt-alleles
    Boolean  drop_genotypes      = false      # -G/--drop-genotypes

    # ---- Output ----
    String   output_type   = "z"               # b/u/z/v (bgz BCF / uncompressed BCF / bgz VCF / uncompressed VCF)
    Boolean  write_index   = true

    # ---- Escape hatch ----
    String   extra_args    = ""

    # ---- Resources ----
    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 0

    # ---- Observability ----
    String?  wandb_api_key

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  Float vcf_gb = size(vcf, "GB")

  # Output file extension derived from output_type so the output stanza
  # below can declare a typed `File filtered_vcf` (not `Array[File]`).
  String out_ext = if output_type == "z" then "vcf.gz"
                   else if output_type == "b" then "bcf"
                   else if output_type == "v" then "vcf"
                   else "bcf"   # "u" — uncompressed BCF

  # bcftools view is streaming I/O; threads accelerate bgzip on output.
  # Memory needs are modest because filtering is record-by-record.
  Int auto_cpu = if vcf_gb > 100.0 then 16
                 else if vcf_gb > 30.0 then 8
                 else 4
  String auto_memory = if vcf_gb > 100.0 then "16 GB"
                       else if vcf_gb > 30.0 then "8 GB"
                       else "4 GB"

  # Disk: full localized input + output (worst case ~= input for a sample
  # subset; much smaller for a region subset). 2.5x covers all common cases.
  Int auto_disk = ceil(vcf_gb * 2.5) + 50

  Int    cpu          = select_first([cpu_override, auto_cpu])
  String memory       = select_first([memory_override, auto_memory])
  Int    disk_size_gb = select_first([disk_size_gb_override, auto_disk])

  command <<<
    set -euo pipefail

    # ---- magicwand bootstrap ----------------------------------------
    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=bcftools_view
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init
    # -----------------------------------------------------------------

    OUT_FILE="~{output_basename}.~{out_ext}"

    magicwand log \
      bcftools_view.input_bytes="$(stat -c %s ~{vcf})" \
      bcftools_view.output_type="~{output_type}" \
      bcftools_view.cpu="~{cpu}" \
      bcftools_view.disk_gb="~{disk_size_gb}"

    # Build an ARGS array so options carrying shell metacharacters
    # (include/exclude expressions especially) survive correctly.
    ARGS=()

    ~{if defined(regions)       then "ARGS+=(--regions '~{regions}')"             else ""}
    ~{if defined(regions_file)  then "ARGS+=(--regions-file ~{regions_file})"     else ""}
    ~{if defined(targets)       then "ARGS+=(--targets '~{targets}')"             else ""}
    ~{if defined(targets_file)  then "ARGS+=(--targets-file ~{targets_file})"     else ""}

    ~{if defined(samples)       then "ARGS+=(--samples '~{samples}')"             else ""}
    ~{if defined(samples_file)  then "ARGS+=(--samples-file ~{samples_file})"     else ""}
    ~{if force_samples          then "ARGS+=(--force-samples)"                    else ""}
    ~{if no_update              then "ARGS+=(--no-update)"                        else ""}

    if [ -n '~{default="" include_expr}' ]; then
      ARGS+=(--include '~{include_expr}')
    fi
    if [ -n '~{default="" exclude_expr}' ]; then
      ARGS+=(--exclude '~{exclude_expr}')
    fi

    ~{if defined(min_ac)        then "ARGS+=(--min-ac ~{min_ac})"                 else ""}
    ~{if defined(max_ac)        then "ARGS+=(--max-ac ~{max_ac})"                 else ""}
    ~{if defined(min_af)        then "ARGS+=(--min-af ~{min_af})"                 else ""}
    ~{if defined(max_af)        then "ARGS+=(--max-af ~{max_af})"                 else ""}
    ~{if defined(min_alleles)   then "ARGS+=(--min-alleles ~{min_alleles})"       else ""}
    ~{if defined(max_alleles)   then "ARGS+=(--max-alleles ~{max_alleles})"       else ""}

    ~{if defined(types)         then "ARGS+=(--types ~{types})"                   else ""}
    ~{if defined(exclude_types) then "ARGS+=(--exclude-types ~{exclude_types})"   else ""}
    ~{if defined(apply_filters) then "ARGS+=(--apply-filters ~{apply_filters})"   else ""}
    ~{if trim_alt_alleles       then "ARGS+=(--trim-alt-alleles)"                 else ""}
    ~{if drop_genotypes         then "ARGS+=(--drop-genotypes)"                   else ""}

    bcftools view \
      --output-type ~{output_type} \
      --output "$OUT_FILE" \
      --threads ~{cpu} \
      "${ARGS[@]}" \
      ~{extra_args} \
      ~{vcf}

    # Index only the bgzipped output types; v/u are uncompressed and not indexable.
    if [ "~{write_index}" = "true" ]; then
      case "~{output_type}" in
        z) bcftools index --tbi --threads ~{cpu} "$OUT_FILE" ;;
        b) bcftools index       --threads ~{cpu} "$OUT_FILE" ;;
      esac
    fi

    magicwand log \
      bcftools_view.output_bytes="$(stat -c %s "$OUT_FILE")"
  >>>

  output {
    File  filtered_vcf       = "~{output_basename}.~{out_ext}"
    # Optional: present only for z (tbi) / b (csi) output types when write_index=true.
    File? filtered_index_tbi = "~{output_basename}.vcf.gz.tbi"
    File? filtered_index_csi = "~{output_basename}.bcf.csi"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow bcftools_view {
  input {
    File   vcf
    File?  vcf_index
    String output_basename

    String?  regions
    File?    regions_file
    String?  targets
    File?    targets_file

    String?  samples
    File?    samples_file
    Boolean  force_samples       = false
    Boolean  no_update           = false

    String?  include_expr
    String?  exclude_expr

    Int?     min_ac
    Int?     max_ac
    Float?   min_af
    Float?   max_af
    Int?     min_alleles
    Int?     max_alleles

    String?  types
    String?  exclude_types

    String?  apply_filters
    Boolean  trim_alt_alleles    = false
    Boolean  drop_genotypes      = false

    String   output_type         = "z"
    Boolean  write_index         = true

    String   extra_args          = ""

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type           = "HDD"
    Int      preemptible         = 0

    String?  wandb_api_key

    String   docker_image        = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  call bcftools_view_task {
    input:
      vcf                   = vcf,
      vcf_index             = vcf_index,
      output_basename       = output_basename,
      regions               = regions,
      regions_file          = regions_file,
      targets               = targets,
      targets_file          = targets_file,
      samples               = samples,
      samples_file          = samples_file,
      force_samples         = force_samples,
      no_update             = no_update,
      include_expr          = include_expr,
      exclude_expr          = exclude_expr,
      min_ac                = min_ac,
      max_ac                = max_ac,
      min_af                = min_af,
      max_af                = max_af,
      min_alleles           = min_alleles,
      max_alleles           = max_alleles,
      types                 = types,
      exclude_types         = exclude_types,
      apply_filters         = apply_filters,
      trim_alt_alleles      = trim_alt_alleles,
      drop_genotypes        = drop_genotypes,
      output_type           = output_type,
      write_index           = write_index,
      extra_args            = extra_args,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      disk_type             = disk_type,
      preemptible           = preemptible,
      wandb_api_key         = wandb_api_key,
      docker_image          = docker_image
  }

  output {
    File  filtered_vcf   = bcftools_view_task.filtered_vcf
    File? filtered_index = if defined(bcftools_view_task.filtered_index_tbi)
                           then bcftools_view_task.filtered_index_tbi
                           else bcftools_view_task.filtered_index_csi
  }
}
