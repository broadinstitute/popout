#!/bin/bash
# Build and push the popout base image. Run when pyproject.toml or system
# dependencies in Dockerfile.base change.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/popout-base"
TAG="${1:-latest}"

# Upstream plink2 release tag from https://github.com/chrchang/plink-ng/releases
PLINK2_VERSION="${PLINK2_VERSION:-v2.0.0-a.7.1}"

docker buildx build \
    -f "$SCRIPT_DIR/Dockerfile.base" \
    -t "${REPO}:${TAG}" \
    --platform linux/amd64 \
    --build-arg "PLINK2_VERSION=${PLINK2_VERSION}" \
    --push \
    "$REPO_ROOT"
