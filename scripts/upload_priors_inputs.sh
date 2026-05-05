#!/bin/bash
# Upload popout priors inputs to a Terra workspace bucket.
#
# Stages four files under gs://<BUCKET>/popout/:
#   * aim_panels.bed              — for vcf2pgen filter_pgen.wdl `aim_panel_bed`
#   * priors_v2.yaml              — for popout.wdl `priors_yaml`
#   * 1kg_superpop_freq.tsv.gz    — for popout.wdl `superpop_freqs`
#   * gnomad_superpop_freq.tsv.gz — alternate superpop_freqs source
#
# Run from the repo root (or anywhere — paths are resolved from the
# script location). Requires `gcloud` CLI authenticated against the
# pmi-ops configuration.
#
# Usage:
#   ./scripts/upload_priors_inputs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUCKET="gs://fc-secure-1c6b9393-5e5d-4a87-b483-d1f0b019af92"
DEST_PREFIX="${BUCKET}/popout"

LOCAL_BED="${REPO_ROOT}/popout/data/aim_panels/all_panels.bed"
LOCAL_YAML="${REPO_ROOT}/configs/priors_v2.yaml"
LOCAL_KG="${HOME}/.popout/superpop_freqs/GRCh38/1kg_superpop_freq.tsv.gz"
LOCAL_GNOMAD="${HOME}/.popout/superpop_freqs/GRCh38/gnomad_superpop_freq.tsv.gz"

# Pre-flight: every source file must exist before we touch the bucket.
missing=0
for f in "${LOCAL_BED}" "${LOCAL_YAML}" "${LOCAL_KG}" "${LOCAL_GNOMAD}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: missing source file: ${f}" >&2
    missing=1
  fi
done
if [ "${missing}" -eq 1 ]; then
  exit 1
fi

CP="gcloud storage cp --configuration=pmi-ops"

echo "=== Uploading to ${DEST_PREFIX} ==="
${CP} "${LOCAL_BED}"    "${DEST_PREFIX}/aim_panels.bed"
${CP} "${LOCAL_YAML}"   "${DEST_PREFIX}/priors_v2.yaml"
${CP} "${LOCAL_KG}"     "${DEST_PREFIX}/1kg_superpop_freq.tsv.gz"
${CP} "${LOCAL_GNOMAD}" "${DEST_PREFIX}/gnomad_superpop_freq.tsv.gz"

echo
echo "=== Done. Terra workflow inputs: ==="
echo "  aim_panel_bed  = ${DEST_PREFIX}/aim_panels.bed"
echo "  priors_yaml    = ${DEST_PREFIX}/priors_v2.yaml"
echo "  superpop_freqs = ${DEST_PREFIX}/1kg_superpop_freq.tsv.gz"
echo "                   (or ${DEST_PREFIX}/gnomad_superpop_freq.tsv.gz)"
