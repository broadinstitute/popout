# bcftools built from the upstream source tarball.
# https://github.com/samtools/bcftools/releases
#
# Two-stage build keeps the runtime image small and includes the +split
# plugin (and all other plugins) under /usr/local/libexec/bcftools/.

FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf automake \
        ca-certificates curl \
        zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev \
        libgsl-dev libperl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG BCFTOOLS_VERSION

RUN test -n "${BCFTOOLS_VERSION}" \
    && curl -fsSL "https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2" \
        -o /tmp/bcftools.tar.bz2 \
    && tar -xjf /tmp/bcftools.tar.bz2 -C /tmp \
    && cd "/tmp/bcftools-${BCFTOOLS_VERSION}" \
    && ./configure --enable-libgsl --enable-perl-filters \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/bcftools*


FROM ubuntu:24.04

# Runtime libs for bcftools + the bootstrap deps for magicwand
# (python3>=3.9, pip, curl) so WDL tasks can `source install.sh; magicwand init`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        zlib1g libbz2-1.0 liblzma5 libcurl4 libssl3 libgsl28 perl \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/bcftools /usr/local/bin/bcftools
COPY --from=builder /usr/local/libexec/bcftools /usr/local/libexec/bcftools

ENTRYPOINT ["bcftools"]
