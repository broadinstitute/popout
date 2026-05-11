version 1.0

## Concatenate VCFs with `bcftools concat`. Inputs must already be in the
## final desired output order — bcftools concat does NOT reorder by contig.
##
## `naive=true` enables `bcftools concat --naive` which skips per-record
## header reconciliation: ~10x faster, but only valid when every input
## shares an identical header (same INFO/FORMAT/contig declarations and
## same sample list). The orchestrator passes naive=true because FLARE
## emits identical headers across chromosomes for a given cluster.
##
## Observability: magicwand instrumentation matches the rest of the
## workflows/ group — single optional `wandb_api_key`, project hardcoded
## to `bcftools_concat`.

task bcftools_concat_task {
  input {
    Array[File] vcfs
    String      output_prefix

    # Output behavior
    String  output_type   = "z"      # z = bgzipped VCF, b = BCF
    Boolean write_index   = true
    Boolean naive         = false

    # Resource overrides (auto-scaled by total input size by default)
    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 0

    # Observability (magicwand -> W&B). Optional API key for online tracking.
    String?  wandb_api_key

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  # Output extension derived from output_type so the output stanza can
  # declare a typed `File concat_vcf` (not `Array[File]`).
  String out_ext = if output_type == "z" then "vcf.gz"
                   else if output_type == "b" then "bcf"
                   else if output_type == "v" then "vcf"
                   else "bcf"

  # Resources: literal defaults. See bcftools_view.wdl for why size()-driven
  # auto-scaling is avoided here. Override the *_override inputs for very
  # large concat jobs (e.g. WGS-scale 22-chrom merge).
  Int    cpu          = select_first([cpu_override, 16])
  String memory       = select_first([memory_override, "16 GB"])
  Int    disk_size_gb = select_first([disk_size_gb_override, 500])

  command <<<
    set -euo pipefail

    # ---- magicwand bootstrap ----------------------------------------
    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=bcftools_concat
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init
    # -----------------------------------------------------------------

    # bcftools concat reads input paths from a file-list, one per line.
    cat > files.txt <<'EOF'
~{sep='\n' vcfs}
EOF

    NUM_INPUTS=$(wc -l < files.txt)
    TOTAL_INPUT_BYTES=$(awk '{ "stat -c %s " $0 | getline s; sum += s; close("stat -c %s " $0) } END { print sum }' files.txt)
    echo "concatenating $NUM_INPUTS files, total input $TOTAL_INPUT_BYTES bytes"

    magicwand log \
      bcftools_concat.num_input_vcfs="$NUM_INPUTS" \
      bcftools_concat.total_input_bytes="$TOTAL_INPUT_BYTES" \
      bcftools_concat.naive="~{naive}" \
      bcftools_concat.cpu="~{cpu}" \
      bcftools_concat.disk_gb="~{disk_size_gb}"

    OUT_FILE="~{output_prefix}.~{out_ext}"

    bcftools concat \
      ~{if naive then "--naive" else ""} \
      --file-list files.txt \
      --output-type ~{output_type} \
      --output "$OUT_FILE" \
      --threads ~{cpu}

    if [ "~{write_index}" = "true" ]; then
      case "~{output_type}" in
        z) bcftools index --tbi --threads ~{cpu} "$OUT_FILE" ;;
        b) bcftools index       --threads ~{cpu} "$OUT_FILE" ;;
      esac
    fi

    magicwand log \
      bcftools_concat.output_bytes="$(stat -c %s "$OUT_FILE")"
  >>>

  output {
    File  concat_vcf       = "~{output_prefix}.~{out_ext}"
    File? concat_index_tbi = "~{output_prefix}.vcf.gz.tbi"
    File? concat_index_csi = "~{output_prefix}.bcf.csi"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow bcftools_concat {
  input {
    Array[File] vcfs
    String      output_prefix

    String  output_type   = "z"
    Boolean write_index   = true
    Boolean naive         = false

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 0

    String?  wandb_api_key

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  call bcftools_concat_task {
    input:
      vcfs                  = vcfs,
      output_prefix         = output_prefix,
      output_type           = output_type,
      write_index           = write_index,
      naive                 = naive,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      disk_type             = disk_type,
      preemptible           = preemptible,
      wandb_api_key         = wandb_api_key,
      docker_image          = docker_image
  }

  output {
    File  concat_vcf   = bcftools_concat_task.concat_vcf
    File? concat_index = if defined(bcftools_concat_task.concat_index_tbi)
                         then bcftools_concat_task.concat_index_tbi
                         else bcftools_concat_task.concat_index_csi
  }
}
