#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# htslib develop pinned SHA — captures PR #1987 (libcurl retry, stall detect,
# Range-resume on reconnect) which has been merged to develop but not yet
# released as a tag.
HTSLIB_SHA="${HTSLIB_SHA:-6bb8f7e7e2c1a71a488598700c9d05e08e8a3162}"

# Upstream release tag from https://github.com/samtools/bcftools/releases
# 1.23.1 (released 2026-03-18) is the first release to include PR #2503's
# synced-reader truncation fix, so no patching needed at this layer.
BCFTOOLS_VERSION="${BCFTOOLS_VERSION:-1.23.1}"

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/bcftools"

# Tag carries both version axes so callers can pin:
#   bcftools:1.23.1-htslib-6bb8f7e
TAG="${BCFTOOLS_VERSION}-htslib-${HTSLIB_SHA:0:7}"

docker buildx build \
    -t "${REPO}:${TAG}" \
    -t "${REPO}:latest" \
    --platform linux/amd64 \
    --build-arg "HTSLIB_SHA=${HTSLIB_SHA}" \
    --build-arg "BCFTOOLS_VERSION=${BCFTOOLS_VERSION}" \
    --push \
    "$SCRIPT_DIR"
