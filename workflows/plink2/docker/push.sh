#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Upstream release tag from https://github.com/chrchang/plink-ng/releases
# Override per-build:  PLINK2_VERSION=v2.0.0-a.7.1 ./push.sh
PLINK2_VERSION="${PLINK2_VERSION:-v2.0.0-a.7.1}"

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/plink2"

docker buildx build \
    -t "${REPO}:${PLINK2_VERSION}" \
    -t "${REPO}:latest" \
    --platform linux/amd64 \
    --build-arg "PLINK2_VERSION=${PLINK2_VERSION}" \
    --push \
    "$SCRIPT_DIR"
