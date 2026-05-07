#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION="0.1.0"
PUSH_TAG="us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:${VERSION}"

# plink2 binary is fetched at build time. Override to bump the pinned
# version without editing the Dockerfile, e.g.:
#   PLINK2_SERIES=alpha7 PLINK2_BUILD=20260504 ./push.sh
PLINK2_SERIES="${PLINK2_SERIES:-alpha7}"
PLINK2_BUILD="${PLINK2_BUILD:-20260504}"

docker buildx build \
    -t "${PUSH_TAG}" \
    --platform linux/amd64 \
    --build-arg "PLINK2_SERIES=${PLINK2_SERIES}" \
    --build-arg "PLINK2_BUILD=${PLINK2_BUILD}" \
    --push \
    "$SCRIPT_DIR"
