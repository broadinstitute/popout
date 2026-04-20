#!/bin/bash
# Build and push the popout app image on top of the current base.
set -euxo pipefail

VERSION="${1:-latest}"
PUSH_TAG="us-docker.pkg.dev/broad-dsde-methods/popout/popout:${VERSION}"
GIT_VERSION=$(git describe --tags --always --dirty | sed 's/^v//')

docker buildx build \
    --build-arg GIT_VERSION="${GIT_VERSION}" \
    -t "${PUSH_TAG}" \
    --platform linux/amd64 \
    --push \
    .
