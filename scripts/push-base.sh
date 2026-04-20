#!/bin/bash
# Build and push the popout base image. Run when pyproject.toml or system
# dependencies in Dockerfile.base change.
set -euxo pipefail

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/popout-base"
TAG="${1:-latest}"

docker buildx build \
    -f Dockerfile.base \
    -t "${REPO}:${TAG}" \
    --platform linux/amd64 \
    --push \
    .
