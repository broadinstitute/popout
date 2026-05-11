#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Upstream release tag from https://github.com/samtools/bcftools/releases
# Override per-build:  BCFTOOLS_VERSION=1.22 ./push.sh
BCFTOOLS_VERSION="${BCFTOOLS_VERSION:-1.23.1}"

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/bcftools"

docker buildx build \
    -t "${REPO}:${BCFTOOLS_VERSION}" \
    -t "${REPO}:latest" \
    --platform linux/amd64 \
    --build-arg "BCFTOOLS_VERSION=${BCFTOOLS_VERSION}" \
    --push \
    "$SCRIPT_DIR"
