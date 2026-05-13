#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTEXT="$(cd "$SCRIPT_DIR/.." && pwd)"   # build context = workflows/lai-tools/

# Pin to a specific bcftools tag (rather than :latest) so lai-tools image
# rebuilds are reproducible. Override via BCFTOOLS_TAG when bumping bcftools.
BCFTOOLS_TAG="${BCFTOOLS_TAG:-latest}"
BCFTOOLS_IMAGE="us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:${BCFTOOLS_TAG}"

# Version stamp based on the git SHA of the scripts dir — bumps automatically
# when generate_partitions.py (or anything under scripts/) changes.
SCRIPTS_SHA=$(cd "$CONTEXT" && git rev-parse --short HEAD:scripts 2>/dev/null || echo "unversioned")

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools"
TAG="${SCRIPTS_SHA}-bcftools-${BCFTOOLS_TAG}"

docker buildx build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "${REPO}:${TAG}" \
    -t "${REPO}:latest" \
    --platform linux/amd64 \
    --build-arg "BCFTOOLS_IMAGE=${BCFTOOLS_IMAGE}" \
    --push \
    "$CONTEXT"
