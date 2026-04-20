#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION="0.1.0"
PUSH_TAG="us-docker.pkg.dev/broad-dsde-methods/popout/hap-ibd:${VERSION}"

docker buildx build \
    -t "${PUSH_TAG}" \
    --platform linux/amd64 \
    --push \
    "$SCRIPT_DIR"
