#!/bin/bash
# throughput_probe.sh — diagnose where the ~263 KB/s streaming bottleneck lives.
#
# Runs a layered probe matrix against a real gs:// VCF and emits results.tsv
# with per-probe wall-clock + bytes counters. See the plan file for the
# 8-probe design and what each diff reveals.
#
# Each probe is wrapped in `timeout $PROBE_TIMEOUT_S`. A probe that errors out
# records its rc and bytes_out and the run continues — the failure pattern is
# the diagnostic.

set -u
set -o pipefail
# Deliberately NOT set -e: probe failures are data, not script errors.

# ------------------------------------------------------------------
# arg parsing
# ------------------------------------------------------------------
VCF_URL=""
TBI_PATH=""
REGION=""
HEADER_BYTE_END=67108864      # 64 MiB; enough to cover the VCF header at AoU scale
GROUPS_FILE=""
CPU=4
PROBES_FILTER=""
PROBE_TIMEOUT_S=900
ENABLE_STRACE=0
ENABLE_BG_INSTR=0
ENABLE_NETDEV=0
OUT_TSV=""
OUT_DIR="."

usage() {
  cat <<EOF >&2
usage: throughput_probe.sh \\
  --vcf-url gs://... --tbi-path /path/to/local.tbi --region CHR:START-END \\
  [--header-byte-end INT] \\
  --groups-file groups.tsv --cpu N \\
  --out-tsv results.tsv --out-dir . \\
  [--probes name1,name2,...] [--probe-timeout-s 900] \\
  [--enable-strace] [--enable-background-instrumentation] [--enable-netdev-sample]

byte_start/byte_end are derived from --tbi-path + --region; you don't pass them.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --vcf-url) VCF_URL="$2"; shift 2 ;;
    --tbi-path) TBI_PATH="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --header-byte-end) HEADER_BYTE_END="$2"; shift 2 ;;
    --groups-file) GROUPS_FILE="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --probes) PROBES_FILTER="$2"; shift 2 ;;
    --probe-timeout-s) PROBE_TIMEOUT_S="$2"; shift 2 ;;
    --enable-strace) ENABLE_STRACE=1; shift ;;
    --enable-background-instrumentation) ENABLE_BG_INSTR=1; shift ;;
    --enable-netdev-sample) ENABLE_NETDEV=1; shift ;;
    --out-tsv) OUT_TSV="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

for v in VCF_URL TBI_PATH REGION GROUPS_FILE OUT_TSV; do
  if [ -z "${!v}" ]; then echo "missing required arg: $v" >&2; usage; exit 2; fi
done

mkdir -p "$OUT_DIR"

# Derive byte_start, byte_end from the .tbi linear index. Errors hard if the
# region falls outside the index — caller picks a different region.
echo "[probe] resolving byte range for region $REGION via $TBI_PATH"
read -r BYTE_START BYTE_END < <(tbi_byte_range.py --tbi-path "$TBI_PATH" --region "$REGION")
if [ -z "${BYTE_START:-}" ] || [ -z "${BYTE_END:-}" ]; then
  echo "[probe] tbi_byte_range.py failed to produce byte_start/byte_end" >&2
  exit 2
fi
echo "[probe] resolved byte_start=$BYTE_START byte_end=$BYTE_END"

SCOPE_BYTES=$(( BYTE_END - BYTE_START + 1 ))
echo "[probe] scope: $SCOPE_BYTES bytes ($((SCOPE_BYTES / 1048576)) MB); region: $REGION"
echo "[probe] header_byte_end: $HEADER_BYTE_END"
echo "[probe] vcf: $VCF_URL"
echo "[probe] timeout_per_probe: ${PROBE_TIMEOUT_S}s"

# ------------------------------------------------------------------
# gs:// -> https:// for raw-curl probe
# ------------------------------------------------------------------
case "$VCF_URL" in
  gs://*)
    BUCKET_OBJECT="${VCF_URL#gs://}"
    HTTPS_URL="https://storage.googleapis.com/${BUCKET_OBJECT}"
    ;;
  *)
    echo "expected gs:// URL, got: $VCF_URL" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------
# OAuth token for the raw curl probe. Prefer GCS_OAUTH_TOKEN; fall back
# to the GCE metadata server. If neither works, probe 1 will be skipped.
# ------------------------------------------------------------------
TOK=""
if [ -n "${GCS_OAUTH_TOKEN:-}" ]; then
  TOK="$GCS_OAUTH_TOKEN"
  echo "[probe] auth: using GCS_OAUTH_TOKEN from env (len=${#TOK})"
elif curl -fsS -m 2 -H 'Metadata-Flavor: Google' \
        http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
        2>/dev/null > "$OUT_DIR/_tok.json"; then
  TOK=$(jq -r .access_token < "$OUT_DIR/_tok.json")
  rm -f "$OUT_DIR/_tok.json"
  echo "[probe] auth: fetched bearer token from metadata server (len=${#TOK})"
else
  echo "[probe] auth: no GCS_OAUTH_TOKEN and metadata server unreachable" >&2
fi

# ------------------------------------------------------------------
# Background instrumentation
# ------------------------------------------------------------------
BG_PIDS=()
if [ "$ENABLE_BG_INSTR" = 1 ]; then
  vmstat -t 5 > "$OUT_DIR/vmstat.log" 2>&1 &
  BG_PIDS+=($!)
  iostat -tx 5 > "$OUT_DIR/iostat.log" 2>&1 &
  BG_PIDS+=($!)
  echo "[probe] background vmstat+iostat: pids ${BG_PIDS[*]}"
fi
if [ "$ENABLE_NETDEV" = 1 ]; then
  (
    while true; do
      printf '----- %s\n' "$(date +%s.%N)"
      cat /proc/net/dev
      sleep 1
    done
  ) > "$OUT_DIR/netdev.log" 2>&1 &
  BG_PIDS+=($!)
  echo "[probe] background netdev sampler: pid $!"
fi
cleanup() {
  for pid in "${BG_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

# ------------------------------------------------------------------
# TSV header
# ------------------------------------------------------------------
printf 'probe_name\tstart_ts\tend_ts\telapsed_s\tbytes_in_declared\tbytes_out_measured\trate_mbps_in\trate_mbps_out\trc\tcommand\n' \
  > "$OUT_TSV"

# ------------------------------------------------------------------
# Probe machinery
# ------------------------------------------------------------------
PROBE_BYTES_OUT=0

maybe_strace() {
  local name="$1"; shift
  if [ "$ENABLE_STRACE" = 1 ]; then
    timeout "$PROBE_TIMEOUT_S" strace -f -c \
      -o "$OUT_DIR/strace_${name}.txt" \
      -e trace=read,recvfrom,sendto,write,openat \
      "$@"
  else
    timeout "$PROBE_TIMEOUT_S" "$@"
  fi
}

run_probe() {
  # usage: run_probe NAME BYTES_IN_DECLARED CMD_DESC FN_NAME
  local name="$1"
  local bytes_in="$2"
  local cmd_desc="$3"
  local fn="$4"

  if [ -n "$PROBES_FILTER" ]; then
    if ! printf ',%s,' "$PROBES_FILTER" | grep -q ",${name},"; then
      echo "[probe] SKIP $name (not in --probes)"
      return 0
    fi
  fi

  PROBE_BYTES_OUT=0
  local rc=0 start_ts end_ts elapsed_s
  start_ts=$(date +%s.%N)
  echo "==================================================================="
  echo "[probe] START $name @ $start_ts"
  echo "[probe]   cmd: $cmd_desc"

  "$fn" || rc=$?

  end_ts=$(date +%s.%N)
  elapsed_s=$(awk -v a="$end_ts" -v b="$start_ts" 'BEGIN{printf "%.3f", a-b}')

  local rate_in rate_out
  rate_in=$(awk  -v b="$bytes_in"        -v t="$elapsed_s" 'BEGIN{if (t+0>0) printf "%.3f", b/t/1048576; else print "0"}')
  rate_out=$(awk -v b="$PROBE_BYTES_OUT" -v t="$elapsed_s" 'BEGIN{if (t+0>0) printf "%.3f", b/t/1048576; else print "0"}')

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$start_ts" "$end_ts" "$elapsed_s" \
    "$bytes_in" "$PROBE_BYTES_OUT" "$rate_in" "$rate_out" \
    "$rc" "$cmd_desc" >> "$OUT_TSV"

  echo "[probe] END   $name rc=$rc elapsed=${elapsed_s}s"
  echo "[probe]   bytes_in_declared=${bytes_in} rate_in=${rate_in} MB/s"
  echo "[probe]   bytes_out_measured=${PROBE_BYTES_OUT} rate_out=${rate_out} MB/s"
}

sum_dir_bytes() {
  local d="$1"
  if [ -d "$d" ]; then
    find "$d" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
  else
    echo 0
  fi
}

# ------------------------------------------------------------------
# Probe 1 — raw HTTPS Range GET
# ------------------------------------------------------------------
probe_curl_byterange() {
  if [ -z "$TOK" ]; then
    echo "[probe]   no bearer token; skipping"
    PROBE_BYTES_OUT=0
    return 1
  fi
  local cmd_rc=0
  maybe_strace curl_byterange curl -fsS \
    -H "Authorization: Bearer $TOK" \
    -H "Range: bytes=${BYTE_START}-${BYTE_END}" \
    -o "$OUT_DIR/_p1.bin" "$HTTPS_URL" || cmd_rc=$?
  PROBE_BYTES_OUT=$(stat -c %s "$OUT_DIR/_p1.bin" 2>/dev/null || echo 0)
  rm -f "$OUT_DIR/_p1.bin"
  return "$cmd_rc"
}

# ------------------------------------------------------------------
# Probe 2 — gcloud storage cat byte range
# ------------------------------------------------------------------
probe_gcloud_cat_byterange() {
  local cmd_rc=0
  # gcloud storage cat -r START-END writes raw bytes to stdout
  maybe_strace gcloud_cat_byterange bash -c \
    "gcloud storage cat -r ${BYTE_START}-${BYTE_END} '${VCF_URL}' > '${OUT_DIR}/_p2.bin'" \
    || cmd_rc=$?
  PROBE_BYTES_OUT=$(stat -c %s "$OUT_DIR/_p2.bin" 2>/dev/null || echo 0)
  rm -f "$OUT_DIR/_p2.bin"
  return "$cmd_rc"
}

# ------------------------------------------------------------------
# Probe 3 — tabix region stream (apt-installed tabix, system htslib).
# If the system tabix lacks --enable-gcs/--enable-libcurl, this will
# fail with "Protocol not supported" — that's a useful data point.
# ------------------------------------------------------------------
probe_tabix_stream_region() {
  local cmd_rc=0
  maybe_strace tabix_stream_region bash -c \
    "tabix -h '${VCF_URL}' '${REGION}' > '${OUT_DIR}/_p3.vcf'" \
    || cmd_rc=$?
  PROBE_BYTES_OUT=$(stat -c %s "$OUT_DIR/_p3.vcf" 2>/dev/null || echo 0)
  rm -f "$OUT_DIR/_p3.vcf"
  return "$cmd_rc"
}

# ------------------------------------------------------------------
# Probe 4 — bcftools view from gs:// to local BCF. Decode + materialize
# bcf1_t records + write uncompressed BCF (minimal post-decode cost).
# ------------------------------------------------------------------
probe_bcftools_view_stream_bcf() {
  local cmd_rc=0
  maybe_strace bcftools_view_stream_bcf bcftools view \
    --regions "$REGION" --regions-overlap 0 \
    -O u -o "$OUT_DIR/_p4.bcf" \
    "$VCF_URL" || cmd_rc=$?
  PROBE_BYTES_OUT=$(stat -c %s "$OUT_DIR/_p4.bcf" 2>/dev/null || echo 0)
  rm -f "$OUT_DIR/_p4.bcf"
  return "$cmd_rc"
}

# ------------------------------------------------------------------
# Probe 5 — bcftools +split from gs:// (the production workload).
# ------------------------------------------------------------------
probe_bcftools_split_stream() {
  rm -rf "$OUT_DIR/out_stream"
  mkdir -p "$OUT_DIR/out_stream"
  local cmd_rc=0
  maybe_strace bcftools_split_stream bcftools +split \
    --groups-file "$GROUPS_FILE" \
    --regions "$REGION" --regions-overlap 0 \
    --output-type z --output "$OUT_DIR/out_stream" \
    --hts-opts "nthreads=$CPU" \
    "$VCF_URL" || cmd_rc=$?
  PROBE_BYTES_OUT=$(sum_dir_bytes "$OUT_DIR/out_stream")
  return "$cmd_rc"
}

# ------------------------------------------------------------------
# Pre-step for probes 7/8: localize the slice (this is also probe 6).
# We prepend the VCF header bytes (everything in [0, HEADER_BYTE_END))
# so bcftools can parse the local file. Any records that happen to live
# inside the header chunk get processed too — probes 7/8 see a slightly
# larger workload than the strict region, but bytes_in is reported
# accurately so the rate is meaningful.
# ------------------------------------------------------------------
probe_gcloud_cat_to_local() {
  rm -f "$OUT_DIR/_slice.bgz"
  local cmd_rc=0
  maybe_strace gcloud_cat_to_local bash -c \
    "gcloud storage cat -r ${BYTE_START}-${BYTE_END} '${VCF_URL}' > '${OUT_DIR}/_slice.bgz'" \
    || cmd_rc=$?
  PROBE_BYTES_OUT=$(stat -c %s "$OUT_DIR/_slice.bgz" 2>/dev/null || echo 0)
  return "$cmd_rc"
}

prep_local_slice() {
  echo "[prep] fetching header [0, ${HEADER_BYTE_END})..."
  rm -f "$OUT_DIR/_header.bgz" "$OUT_DIR/local_slice.bgz"
  if ! gcloud storage cat -r "0-$((HEADER_BYTE_END - 1))" "$VCF_URL" > "$OUT_DIR/_header.bgz"; then
    echo "[prep] header fetch failed; local probes (7, 8) will likely fail too" >&2
    return 1
  fi
  echo "[prep] header bytes: $(stat -c %s "$OUT_DIR/_header.bgz")"
  cat "$OUT_DIR/_header.bgz" "$OUT_DIR/_slice.bgz" > "$OUT_DIR/local_slice.bgz"
  # Append empty-BGZF EOF terminator (28 bytes) so bcftools doesn't warn
  # about truncated stream.
  printf '\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00\x42\x43\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00' \
    >> "$OUT_DIR/local_slice.bgz"
  rm -f "$OUT_DIR/_header.bgz" "$OUT_DIR/_slice.bgz"
  echo "[prep] local_slice.bgz: $(stat -c %s "$OUT_DIR/local_slice.bgz") bytes"
}

# ------------------------------------------------------------------
# Probe 7 — bcftools view from local file. Same workload as #4 but
# without network in the loop.
# ------------------------------------------------------------------
probe_bcftools_view_local_bcf() {
  if [ ! -s "$OUT_DIR/local_slice.bgz" ]; then
    echo "[probe]   local_slice.bgz missing; skipping"
    PROBE_BYTES_OUT=0
    return 1
  fi
  local cmd_rc=0
  maybe_strace bcftools_view_local_bcf bcftools view \
    -O u -o "$OUT_DIR/_p7.bcf" \
    "$OUT_DIR/local_slice.bgz" || cmd_rc=$?
  PROBE_BYTES_OUT=$(stat -c %s "$OUT_DIR/_p7.bcf" 2>/dev/null || echo 0)
  rm -f "$OUT_DIR/_p7.bcf"
  return "$cmd_rc"
}

# ------------------------------------------------------------------
# Probe 8 — bcftools +split from local file. Same workload as #5 but
# without network. No --regions because we have no .tbi for the slice;
# instead, the slice IS roughly the region's bytes.
# ------------------------------------------------------------------
probe_bcftools_split_local() {
  if [ ! -s "$OUT_DIR/local_slice.bgz" ]; then
    echo "[probe]   local_slice.bgz missing; skipping"
    PROBE_BYTES_OUT=0
    return 1
  fi
  rm -rf "$OUT_DIR/out_local"
  mkdir -p "$OUT_DIR/out_local"
  local cmd_rc=0
  maybe_strace bcftools_split_local bcftools +split \
    --groups-file "$GROUPS_FILE" \
    --output-type z --output "$OUT_DIR/out_local" \
    --hts-opts "nthreads=$CPU" \
    "$OUT_DIR/local_slice.bgz" || cmd_rc=$?
  PROBE_BYTES_OUT=$(sum_dir_bytes "$OUT_DIR/out_local")
  return "$cmd_rc"
}

# ------------------------------------------------------------------
# Run the probes in order
# ------------------------------------------------------------------
run_probe curl_byterange           "$SCOPE_BYTES" "curl Range:bytes=$BYTE_START-$BYTE_END $HTTPS_URL"           probe_curl_byterange
run_probe gcloud_cat_byterange     "$SCOPE_BYTES" "gcloud storage cat -r $BYTE_START-$BYTE_END $VCF_URL"        probe_gcloud_cat_byterange
run_probe tabix_stream_region      "$SCOPE_BYTES" "tabix -h $VCF_URL $REGION"                                    probe_tabix_stream_region
run_probe bcftools_view_stream_bcf "$SCOPE_BYTES" "bcftools view -r $REGION -O u $VCF_URL"                       probe_bcftools_view_stream_bcf
run_probe bcftools_split_stream    "$SCOPE_BYTES" "bcftools +split --regions $REGION --regions-overlap 0 -O z $VCF_URL" probe_bcftools_split_stream

run_probe gcloud_cat_to_local      "$SCOPE_BYTES" "gcloud storage cat -r $BYTE_START-$BYTE_END $VCF_URL > _slice.bgz" probe_gcloud_cat_to_local

# Prep step (untimed): prepend header so local probes can parse.
prep_local_slice || echo "[prep] continuing despite header prep failure" >&2

LOCAL_BYTES=$(stat -c %s "$OUT_DIR/local_slice.bgz" 2>/dev/null || echo 0)
run_probe bcftools_view_local_bcf  "$LOCAL_BYTES"  "bcftools view -O u local_slice.bgz"                          probe_bcftools_view_local_bcf
run_probe bcftools_split_local     "$LOCAL_BYTES"  "bcftools +split --groups-file groups.tsv -O z local_slice.bgz" probe_bcftools_split_local

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "==================================================================="
echo "[probe] results.tsv:"
column -t -s $'\t' "$OUT_TSV" || cat "$OUT_TSV"
echo "==================================================================="

# Tidy up the slice if you don't want it returned as an output. Comment
# the next line out if you want to debug-inspect local_slice.bgz.
rm -f "$OUT_DIR/local_slice.bgz"
rm -rf "$OUT_DIR/out_stream" "$OUT_DIR/out_local"

exit 0
