version 1.0

## throughput_probe — diagnose where streaming throughput is being lost.
##
## Runs an 8-probe matrix against a real gs:// VCF inside a Terra task and
## emits results.tsv (one row per probe with start/end timestamps + bytes
## counters + rc). Probes ascend the layer stack:
##
##   1. curl_byterange           — raw HTTPS Range GET (TCP/TLS/OAuth/GCS baseline)
##   2. gcloud_cat_byterange     — same range via gcloud client
##   3. tabix_stream_region      — htslib (system, apt-installed) network+decode
##   4. bcftools_view_stream_bcf — htslib (develop) + bcftools view, streaming
##   5. bcftools_split_stream    — full production workload, streaming
##   6. gcloud_cat_to_local      — localizes the slice for the local-control probes
##   7. bcftools_view_local_bcf  — same as #4 but reading from local disk
##   8. bcftools_split_local     — same as #5 but reading from local disk
##
## Each probe is wrapped in `timeout $probe_timeout_s`; a slow probe records
## rc=124 and the run continues. See workflows/lai-tools/scripts/throughput_probe.sh
## for the script.
##
## Caller supplies a bp region (chrN:start-end). The task localizes the .tbi
## and derives BGZF-block-aligned byte_start/byte_end from the linear index
## before the probes run, so the JSON only has to know the region — not the
## byte offsets, which are file-specific.

task throughput_probe_task {
  input {
    String        vcf_url
    File          vcf_index                         # localized .tbi (small; auto-localized)
    String        region                            # e.g. chr21:25000000-25210000  (~200 MB at AoU density)
    Int           header_byte_end = 67108864        # 64 MiB; covers AoU-scale VCF header
    Array[File]   sample_groups                     # per-cluster sample lists
                                                    # (cluster ids inferred from filename basenames,
                                                    # matching bcftools_split_streaming.wdl)

    Array[String] probes          = []              # empty = run all 8
    Int           probe_timeout_s = 900             # 15 min ceiling per probe

    Boolean       enable_strace                     = false
    Boolean       enable_background_instrumentation = false
    Boolean       enable_netdev_sample              = false

    Int           cpu          = 4
    String        memory       = "8 GB"
    Int           disk_size_gb = 30
    String        disk_type    = "HDD"
    Int           preemptible  = 0                  # diagnostic; don't get yanked mid-probe

    String        docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  command <<<
    set -euo pipefail

    # ---- GCS auth refresher --------------------------------------
    # Same FIFO pattern as bcftools_split_streaming.wdl. htslib reads
    # the bearer token from HTS_AUTH_LOCATION on every request, so a
    # FIFO backed by a curl loop keeps the token fresh for long probes.
    # The throughput_probe.sh script independently uses GCS_OAUTH_TOKEN
    # (or fetches its own token) for the raw curl probe.
    if [ -z "${GCS_OAUTH_TOKEN:-}" ] && [ -z "${HTS_AUTH_LOCATION:-}" ] \
       && curl -fsS -m 2 -H 'Metadata-Flavor: Google' \
            http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
            >/dev/null 2>&1; then
      TOKEN_FIFO=/tmp/gcs_token_fifo
      mkfifo "$TOKEN_FIFO"
      (
        while true; do
          curl -fsS -H 'Metadata-Flavor: Google' \
            http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
            > "$TOKEN_FIFO" 2>/dev/null || true
        done
      ) &
      export HTS_AUTH_LOCATION="$TOKEN_FIFO"
      echo "GCS auth: HTS_AUTH_LOCATION FIFO refresher running (pid $!)"
    fi
    # --------------------------------------------------------------

    # 3-column groups.tsv: sample <TAB> - <TAB> output_basename
    # Same shape as bcftools_split_streaming.wdl. Required by `bcftools +split`
    # for probes 5 and 8.
    : > groups.tsv
    for f in ~{sep=' ' sample_groups}; do
      group=$(basename "$f")
      group="${group%.*}"
      awk -v g="$group" 'NF && $1 !~ /^#/ { print $1 "\t-\t" g }' "$f" >> groups.tsv
    done
    echo "groups.tsv: $(wc -l < groups.tsv) sample/group rows"

    # Use a bash array so an empty --probes filter doesn't pass empty args
    # and a non-empty one passes "--probes" "a,b,c" as two args.
    PROBES_ARGS=()
    if [ "~{length(probes)}" != "0" ]; then
      PROBES_ARGS=(--probes "~{sep=',' probes}")
    fi

    throughput_probe.sh \
      --vcf-url "~{vcf_url}" \
      --tbi-path "~{vcf_index}" \
      --region "~{region}" \
      --header-byte-end ~{header_byte_end} \
      --groups-file groups.tsv \
      --cpu ~{cpu} \
      --probe-timeout-s ~{probe_timeout_s} \
      ~{if enable_strace then "--enable-strace" else ""} \
      ~{if enable_background_instrumentation then "--enable-background-instrumentation" else ""} \
      ~{if enable_netdev_sample then "--enable-netdev-sample" else ""} \
      "${PROBES_ARGS[@]}" \
      --out-tsv results.tsv \
      --out-dir .

    # Surface results.tsv to stderr so the live Cromwell log shows the
    # summary without waiting for outputs.
    echo "===== throughput_probe results ====="
    cat results.tsv
    echo "===================================="
  >>>

  output {
    File          results_tsv  = "results.tsv"
    File?         vmstat_log   = "vmstat.log"
    File?         iostat_log   = "iostat.log"
    File?         netdev_log   = "netdev.log"
    Array[File]   strace_logs  = glob("strace_*.txt")
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow throughput_probe {
  input {
    String        vcf_url
    File          vcf_index
    String        region
    Int           header_byte_end = 67108864
    Array[File]   sample_groups

    Array[String] probes          = []
    Int           probe_timeout_s = 900

    Boolean       enable_strace                     = false
    Boolean       enable_background_instrumentation = false
    Boolean       enable_netdev_sample              = false

    Int           cpu          = 4
    String        memory       = "8 GB"
    Int           disk_size_gb = 30
    String        disk_type    = "HDD"
    Int           preemptible  = 0

    String        docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  call throughput_probe_task {
    input:
      vcf_url                           = vcf_url,
      vcf_index                         = vcf_index,
      region                            = region,
      header_byte_end                   = header_byte_end,
      sample_groups                     = sample_groups,
      probes                            = probes,
      probe_timeout_s                   = probe_timeout_s,
      enable_strace                     = enable_strace,
      enable_background_instrumentation = enable_background_instrumentation,
      enable_netdev_sample              = enable_netdev_sample,
      cpu                               = cpu,
      memory                            = memory,
      disk_size_gb                      = disk_size_gb,
      disk_type                         = disk_type,
      preemptible                       = preemptible,
      docker_image                      = docker_image
  }

  output {
    File          results_tsv  = throughput_probe_task.results_tsv
    File?         vmstat_log   = throughput_probe_task.vmstat_log
    File?         iostat_log   = throughput_probe_task.iostat_log
    File?         netdev_log   = throughput_probe_task.netdev_log
    Array[File]   strace_logs  = throughput_probe_task.strace_logs
  }
}
