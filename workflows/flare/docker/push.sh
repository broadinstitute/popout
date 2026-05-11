#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Upstream FLARE tag from https://github.com/browning-lab/flare/tags
# The Dockerfile fails the build if the (unversioned) canonical jar URL
# returns a different version than this.
FLARE_VERSION="${FLARE_VERSION:-0.6.0}"

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/flare"

docker buildx build \
    -t "${REPO}:${FLARE_VERSION}" \
    -t "${REPO}:latest" \
    --platform linux/amd64 \
    --build-arg "FLARE_VERSION=${FLARE_VERSION}" \
    --push \
    "$SCRIPT_DIR"
