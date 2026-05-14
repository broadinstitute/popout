# bcftools 1.23.x built against a pinned htslib develop SHA, so we get the
# libcurl retry / stall-detect / Range-resume changes from PR #1987 that
# released htslib tags don't have yet.
#
# Two-stage build keeps the runtime image small and ships the +split plugin
# (and all other plugins) under /usr/local/libexec/bcftools/. The runtime
# image inherits htslib's shared library from /usr/local/lib so bcftools
# picks it up cleanly via ldconfig.

FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf automake make \
        git ca-certificates curl \
        zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev \
        libgsl-dev libperl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG HTSLIB_SHA
ARG BCFTOOLS_VERSION

# 1) htslib from develop, pinned to a SHA for reproducibility. The PR #1987
#    retry/resilience changes are merged on develop but not yet in a tagged
#    release as of plan time.
#
# --enable-libcurl: link libcurl directly into libhts.so so gs:// / https://
# URLs work out of the box.
# --enable-gcs: compile in GCS-aware auth helpers — without this, htslib only
# knows GCS_OAUTH_TOKEN / GOOGLE_APPLICATION_CREDENTIALS and never queries
# the GCE metadata server, so streaming a private bucket from a Terra worker
# fails with HTTP 403 even though Cromwell-side localization works fine.
# We deliberately do NOT use --enable-plugins, which would emit these handlers
# as separately-loadable .so files — the runtime image would then need to
# also ship /usr/local/libexec/htslib/ and any missing plugin causes
# "Protocol not supported" at open time.
RUN test -n "${HTSLIB_SHA}" \
    && git clone --recurse-submodules https://github.com/samtools/htslib.git /tmp/htslib \
    && cd /tmp/htslib \
    && git checkout "${HTSLIB_SHA}" \
    && git submodule update --init --recursive \
    && autoreconf -i \
    && ./configure --enable-libcurl --enable-gcs \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig

# 2) bcftools 1.23.x (already contains PR #2503's synced-reader truncation
#    fix from 1.23.1 onwards) linked against the htslib we just installed.
RUN test -n "${BCFTOOLS_VERSION}" \
    && git clone --branch "${BCFTOOLS_VERSION}" https://github.com/samtools/bcftools.git /tmp/bcftools \
    && cd /tmp/bcftools \
    && autoreconf -i \
    && ./configure --enable-libgsl --enable-perl-filters \
    && make -j"$(nproc)" \
    && make install


FROM ubuntu:24.04

# Runtime libs for bcftools + the bootstrap deps for magicwand
# (python3, pip, curl) so WDL tasks can `source install.sh; magicwand init`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        zlib1g libbz2-1.0 liblzma5 libcurl4 libssl3 libgsl27 libgslcblas0 perl \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# bcftools binary + its plugins, and the htslib shared library it links against.
COPY --from=builder /usr/local/bin/bcftools     /usr/local/bin/bcftools
COPY --from=builder /usr/local/libexec/bcftools /usr/local/libexec/bcftools
COPY --from=builder /usr/local/lib/libhts*      /usr/local/lib/
COPY --from=builder /usr/local/include/htslib   /usr/local/include/htslib
RUN ldconfig

ENTRYPOINT ["bcftools"]
