#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# htslib develop pinned SHA — captures PR #1987 (libcurl retry, stall detect,
# Range-resume on reconnect) which has been merged to develop but not yet
# released as a tag.
HTSLIB_SHA="${HTSLIB_SHA:-6bb8f7e7e2c1a71a488598700c9d05e08e8a3162}"

# bcftools is currently pinned to gileshall's split-threads fork, which adds
# `bcftools +split --threads N`. Pinned to a SHA so the image is reproducible
# without trusting the branch ref to be immutable.
#
# When upstream merges the patch, flip BCFTOOLS_REPO back to samtools/bcftools
# and pin BCFTOOLS_REF to the release tag (e.g. "1.24").
BCFTOOLS_REPO="${BCFTOOLS_REPO:-https://github.com/gileshall/bcftools.git}"
BCFTOOLS_REF="${BCFTOOLS_REF:-06471124bbff670a947abf3c3b3dcc69486d3851}"

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/bcftools"

# Tag carries both fork-pin axes:
#   bcftools:split-threads-0647112-htslib-6bb8f7e
TAG="split-threads-${BCFTOOLS_REF:0:7}-htslib-${HTSLIB_SHA:0:7}"

docker buildx build \
    -t "${REPO}:${TAG}" \
    -t "${REPO}:latest" \
    --platform linux/amd64 \
    --build-arg "HTSLIB_SHA=${HTSLIB_SHA}" \
    --build-arg "BCFTOOLS_REPO=${BCFTOOLS_REPO}" \
    --build-arg "BCFTOOLS_REF=${BCFTOOLS_REF}" \
    --push \
    "$SCRIPT_DIR"
