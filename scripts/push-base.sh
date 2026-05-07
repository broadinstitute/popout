#!/bin/bash
# Build and push the popout base image. Run when pyproject.toml or system
# dependencies in Dockerfile.base change.
set -euxo pipefail

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/popout-base"
TAG="${1:-latest}"

# plink2 binary is fetched at build time. Override to bump the pinned
# version without editing Dockerfile.base, e.g.:
#   PLINK2_SERIES=alpha7 PLINK2_BUILD=20260504 ./scripts/push-base.sh
PLINK2_SERIES="${PLINK2_SERIES:-alpha7}"
PLINK2_BUILD="${PLINK2_BUILD:-20260504}"

docker buildx build \
    -f Dockerfile.base \
    -t "${REPO}:${TAG}" \
    --platform linux/amd64 \
    --build-arg "PLINK2_SERIES=${PLINK2_SERIES}" \
    --build-arg "PLINK2_BUILD=${PLINK2_BUILD}" \
    --push \
    .
