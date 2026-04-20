# Thin app image: overlays popout source and version stamp onto the base.
# Rebuild and push on every code change; does not re-upload the dependency layer.
ARG BASE_IMAGE=us-docker.pkg.dev/broad-dsde-methods/popout/popout-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app
COPY popout/ popout/

ARG GIT_VERSION=unknown
ENV POPOUT_VERSION=${GIT_VERSION}

ENTRYPOINT ["popout"]
