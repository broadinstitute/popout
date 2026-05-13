version 1.0

## Streaming variant of bcftools_split: reads its slice of a bgzipped+tabix
## VCF directly from gs:// (no Cromwell localization of the input), splits
## the records in a specific region into per-cluster sub-VCFs.
##
## Why streaming: at biobank scale the input is hundreds of GB per chrom;
## localizing it costs ~30 min and CPU sits idle waiting. Region-scatter
## means each task only pulls ~10 GB of records via tabix-indexed Range
## requests, and many tasks run in parallel on cheap preemptibles.
##
## Resilience: htslib's libcurl backend (PR #1987, on develop) handles
## transient HTTP 5xx, partial transfers, and stall detection with exponential
## backoff. Retries resume from the last successful byte offset via Range
## requests rather than restarting from zero, so a 9-min-in flake costs
## ~1 s of retry, not 9 min of redo. Tune via HTS_RETRY_* / HTS_LOW_SPEED_*
## env vars (set in the command preamble below).
##
## Output ordering: subset_vcfs is in cluster_ids order (deterministic, no
## pair_by_basename helper needed downstream). The task writes an
## `output_list.txt` manifest and the WDL outputs Array[File] via read_lines.

task bcftools_split_streaming_task {
  input {
    String       vcf_url                # gs:// URL of the bgzipped VCF
    String       region                 # e.g. "chr1:1-25000000"
    String       region_id              # safe partition id, used in output names
    Array[File]  sample_groups          # per-cluster sample-list files (small, localize fine)
    Array[String] cluster_ids           # parallel to sample_groups; basename(group_file, ext) == cluster_id

    Int          cpu          = 4
    String       memory       = "8 GB"
    Int          disk_size_gb = 50
    String       disk_type    = "HDD"
    Int          preemptible  = 3

    # htslib retry knobs (PR #1987). Sensible biobank-streaming defaults.
    Int          hts_low_speed_limit = 1048576    # 1 MB/s minimum
    Int          hts_low_speed_time  = 120        # 2 min below threshold = stall
    Int          hts_retry_max       = 5
    Int          hts_retry_delay     = 1000       # ms
    Int          hts_retry_max_delay = 120000     # ms

    String?      wandb_api_key

    String       docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  command <<<
    set -euo pipefail

    # ---- magicwand bootstrap ----------------------------------------
    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=bcftools_split_streaming
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init
    # -----------------------------------------------------------------

    # htslib resilience env vars — the bcftools binary in this image was
    # built against htslib develop (PR #1987 merged), which honors these.
    export HTS_LOW_SPEED_LIMIT=~{hts_low_speed_limit}
    export HTS_LOW_SPEED_TIME=~{hts_low_speed_time}
    export HTS_RETRY_MAX=~{hts_retry_max}
    export HTS_RETRY_DELAY=~{hts_retry_delay}
    export HTS_RETRY_MAX_DELAY=~{hts_retry_max_delay}

    OUT_DIR=out
    mkdir -p "$OUT_DIR"

    # 3-column groups.tsv: sample <TAB> - <TAB> output_basename.
    # The output_basename column is what makes bcftools name the per-group
    # output `<basename>.vcf.gz` rather than per-sample.
    : > groups.tsv
    for f in ~{sep=' ' sample_groups}; do
      group=$(basename "$f")
      group="${group%.*}"
      awk -v g="$group" 'NF && $1 !~ /^#/ { print $1 "\t-\t" g }' "$f" >> groups.tsv
    done

    NUM_GROUPS=~{length(sample_groups)}
    NUM_MAPPINGS=$(wc -l < groups.tsv)
    echo "groups.tsv: $NUM_MAPPINGS sample/group rows across $NUM_GROUPS groups"

    magicwand log \
      bcftools_split_streaming.region="~{region}" \
      bcftools_split_streaming.region_id="~{region_id}" \
      bcftools_split_streaming.num_groups="$NUM_GROUPS" \
      bcftools_split_streaming.num_sample_mappings="$NUM_MAPPINGS" \
      bcftools_split_streaming.cpu="~{cpu}" \
      bcftools_split_streaming.disk_gb="~{disk_size_gb}"

    # Stream the region directly from gs://. The .tbi is auto-discovered by
    # htslib via the standard <url>.tbi convention.
    bcftools +split \
      "~{vcf_url}" \
      --regions "~{region}" \
      --groups-file groups.tsv \
      --output-type z \
      --output "$OUT_DIR" \
      --hts-opts "nthreads=~{cpu}"

    ls -lh "$OUT_DIR"

    # Emit subset_vcfs in cluster_ids order. If a cluster had no records in
    # this region, bcftools may have skipped emitting its file; synthesize a
    # header-only stub so downstream gather doesn't trip on a missing File.
    : > output_list.txt
    : > index_list.txt
    for cid in ~{sep=' ' cluster_ids}; do
      out_vcf="$OUT_DIR/${cid}.vcf.gz"
      if [ ! -s "$out_vcf" ]; then
        echo "Empty region for cluster ${cid}; synthesizing header-only stub" >&2
        bcftools view -h --output-type z --output "$out_vcf" "~{vcf_url}" --regions "~{region}"
      fi
      bcftools index --tbi --threads ~{cpu} "$out_vcf"
      echo "$PWD/$out_vcf"     >> output_list.txt
      echo "$PWD/$out_vcf.tbi" >> index_list.txt
    done

    OUTPUT_COUNT=$(wc -l < output_list.txt)
    TOTAL_OUTPUT_BYTES=$(xargs -d '\n' -a output_list.txt stat -c %s | awk '{s+=$1} END {print s+0}')
    magicwand log \
      bcftools_split_streaming.output_count="$OUTPUT_COUNT" \
      bcftools_split_streaming.total_output_bytes="$TOTAL_OUTPUT_BYTES"

    # ---- Coverage QC: informative-only, never fails the task -------
    # Compare per-cluster output record counts against the total records
    # bcftools observed in the region. Mirror the bullet-proof pattern from
    # flare.wdl: subshell + set +eu + trailing `|| true`.
    QC="~{region_id}.qc.tsv"
    touch "$QC"
    (
      set +eu
      set +o pipefail

      input_records=$(bcftools view -H "~{vcf_url}" --regions "~{region}" 2>/dev/null | wc -l | tr -d ' ')
      total_out_records=0
      {
        printf 'region\t%s\n'        "~{region}"
        printf 'input_records\t%s\n' "${input_records:-unknown}"
        for cid in ~{sep=' ' cluster_ids}; do
          r=$(bcftools view -H "$OUT_DIR/${cid}.vcf.gz" 2>/dev/null | wc -l | tr -d ' ')
          printf 'out_records.%s\t%s\n' "$cid" "${r:-unknown}"
          total_out_records=$((total_out_records + ${r:-0}))
        done
        printf 'total_output_records\t%s\n' "${total_out_records}"
      } > "$QC" 2>/dev/null

      echo "===== bcftools_split_streaming QC for ~{region_id} =====" >&2
      cat "$QC" >&2 2>/dev/null

      if [ "${total_out_records:-0}" = "0" ] 2>/dev/null; then
        echo "QC WARNING: region produced 0 output records across all clusters; check region bounds and sample coverage." >&2
      fi
      if [ -n "${input_records:-}" ] && [ "${input_records}" != "0" ] && [ -n "${total_out_records:-}" ]; then
        ratio=$(awk -v a="$total_out_records" -v b="$input_records" 'BEGIN {if (b+0 > 0) printf "%.3f", (a+0)/(b+0); else print "n/a"}' 2>/dev/null)
        echo "QC INFO: total_output_records / input_records = ${ratio:-n/a}" >&2
      fi
      exit 0
    ) || true
    # -----------------------------------------------------------------
  >>>

  output {
    Array[File] subset_vcfs    = read_lines("output_list.txt")
    Array[File] subset_indices = read_lines("index_list.txt")
    File        qc_report      = "~{region_id}.qc.tsv"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow bcftools_split_streaming {
  input {
    String        vcf_url
    String        region
    String        region_id
    Array[File]   sample_groups
    Array[String] cluster_ids

    Int           cpu          = 4
    String        memory       = "8 GB"
    Int           disk_size_gb = 50
    String        disk_type    = "HDD"
    Int           preemptible  = 3

    String?       wandb_api_key

    String        docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  call bcftools_split_streaming_task {
    input:
      vcf_url       = vcf_url,
      region        = region,
      region_id     = region_id,
      sample_groups = sample_groups,
      cluster_ids   = cluster_ids,
      cpu           = cpu,
      memory        = memory,
      disk_size_gb  = disk_size_gb,
      disk_type     = disk_type,
      preemptible   = preemptible,
      wandb_api_key = wandb_api_key,
      docker_image  = docker_image
  }

  output {
    Array[File] subset_vcfs    = bcftools_split_streaming_task.subset_vcfs
    Array[File] subset_indices = bcftools_split_streaming_task.subset_indices
    File        qc_report      = bcftools_split_streaming_task.qc_report
  }
}
