version 1.0

## Column-wise merge of per-sample-partition VCFs with `bcftools merge`.
## Use case: K partitions cover disjoint sample subsets at the same chromosome
## (e.g. K cluster outputs from FLARE for one chrom). Merge stacks them
## column-wise into a single VCF containing the union of samples at the
## shared variant positions.
##
## NOT to be confused with `bcftools concat` (row-wise stacking of
## non-overlapping regions). If your inputs share samples and span different
## genomic ranges, use bcftools_concat.wdl instead.
##
## bcftools merge requires every input to be indexed. The task generates
## .tbi indices for any input that doesn't already carry one alongside
## (Cromwell localizes only files explicitly declared as inputs, so we don't
## rely on co-located indices arriving for free).
##
## Observability: magicwand instrumentation matches the rest of the
## workflows/ group — single optional `wandb_api_key`, project hardcoded
## to `bcftools_merge`.

task bcftools_merge_task {
  input {
    Array[File] vcfs
    String      output_prefix

    # Output behavior
    String  output_type   = "z"      # z = bgzipped VCF, b = BCF
    Boolean write_index   = true

    # Resource overrides (literal defaults match bcftools_concat.wdl;
    # override for very large merges, e.g. 535K samples per chrom).
    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 1

    # Observability (magicwand -> W&B). Optional API key for online tracking.
    String?  wandb_api_key

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  String out_ext = if output_type == "z" then "vcf.gz"
                   else if output_type == "b" then "bcf"
                   else if output_type == "v" then "vcf"
                   else "bcf"

  # bcftools merge is mostly I/O-bound (stream-read each input, stream-write
  # the merged output); 8 CPU saturates the merge thread pool comfortably.
  Int    cpu = select_first([cpu_override, 8])
  String memory = select_first([memory_override, "16 GB"])

  # Disk = inputs + merged output (roughly same scale) + slack. At AoU
  # scale, chr1's 16 cluster inputs total ~11 TB and the merged output is
  # comparable, so the prior 500 GB literal would have OOM'd silently.
  Float  total_input_gb = size(vcfs, "GB")
  Int    auto_disk      = ceil(total_input_gb * 2.5) + 50
  Int    disk_size_gb   = select_first([disk_size_gb_override, auto_disk])

  command <<<
    set -euo pipefail

    # ---- magicwand bootstrap ----------------------------------------
    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=bcftools_merge
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init
    # -----------------------------------------------------------------

    # bcftools merge reads input paths from a file-list, one per line. The
    # paths must point at indexed bgzipped VCFs.
    cat > files.txt <<'EOF'
~{sep='\n' vcfs}
EOF

    # Ensure every input has an adjacent .tbi. Cromwell localizes Files one
    # at a time without sibling indices, so generate them here if missing.
    while IFS= read -r vcf; do
      if [ ! -f "${vcf}.tbi" ]; then
        bcftools index --tbi --threads ~{cpu} "$vcf"
      fi
    done < files.txt

    NUM_INPUTS=$(wc -l < files.txt | tr -d ' ')
    TOTAL_INPUT_BYTES=$(xargs -d '\n' -a files.txt stat -c %s | awk '{s+=$1} END {print s+0}')
    echo "merging $NUM_INPUTS files, total input $TOTAL_INPUT_BYTES bytes"

    magicwand log \
      bcftools_merge.num_input_vcfs="$NUM_INPUTS" \
      bcftools_merge.total_input_bytes="$TOTAL_INPUT_BYTES" \
      bcftools_merge.cpu="~{cpu}" \
      bcftools_merge.disk_gb="~{disk_size_gb}"

    OUT_FILE="~{output_prefix}.~{out_ext}"

    bcftools merge \
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
      bcftools_merge.output_bytes="$(stat -c %s "$OUT_FILE")"
  >>>

  output {
    File  merged_vcf       = "~{output_prefix}.~{out_ext}"
    File? merged_index_tbi = "~{output_prefix}.vcf.gz.tbi"
    File? merged_index_csi = "~{output_prefix}.bcf.csi"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow bcftools_merge {
  input {
    Array[File] vcfs
    String      output_prefix

    String  output_type   = "z"
    Boolean write_index   = true

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 1

    String?  wandb_api_key

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  call bcftools_merge_task {
    input:
      vcfs                  = vcfs,
      output_prefix         = output_prefix,
      output_type           = output_type,
      write_index           = write_index,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      disk_type             = disk_type,
      preemptible           = preemptible,
      wandb_api_key         = wandb_api_key,
      docker_image          = docker_image
  }

  output {
    File  merged_vcf   = bcftools_merge_task.merged_vcf
    File? merged_index = if defined(bcftools_merge_task.merged_index_tbi)
                         then bcftools_merge_task.merged_index_tbi
                         else bcftools_merge_task.merged_index_csi
  }
}
