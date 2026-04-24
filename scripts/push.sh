#!/bin/bash
# Build and push the popout app image on top of the current base.
#
# On main:   pushes as :latest (and :VERSION if given)
# On branch: pushes as :BRANCH (sanitized) and :VERSION if given
#
# Usage:
#   ./push.sh              # :latest on main, :branch-name on branches
#   ./push.sh 0.3.2        # also tags as :0.3.2
set -euxo pipefail

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/popout"
VERSION="${1:-}"
GIT_VERSION=$(git describe --tags --always --dirty | sed 's/^v//')
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Sanitize branch name for Docker tag (replace non-alphanum with -)
BRANCH_TAG=$(echo "${BRANCH}" | sed 's/[^a-zA-Z0-9._-]/-/g')

if [ "${BRANCH}" = "main" ] || [ "${BRANCH}" = "master" ]; then
    TAG_ARGS="-t ${REPO}:latest"
else
    TAG_ARGS="-t ${REPO}:${BRANCH_TAG}"
fi

if [ -n "${VERSION}" ]; then
    TAG_ARGS="${TAG_ARGS} -t ${REPO}:${VERSION}"
fi

echo "Branch: ${BRANCH} → Docker tags: ${TAG_ARGS}"

docker buildx build \
    --build-arg GIT_VERSION="${GIT_VERSION}" \
    ${TAG_ARGS} \
    --platform linux/amd64 \
    --push \
    .
