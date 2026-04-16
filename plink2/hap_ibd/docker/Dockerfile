# Build plink2 from gileshall/plink-ng (hap-ibd branch) with AVX2 support.
# Two-stage build keeps the runtime image small (~150 MB).

# ---- stage 1: compile ----
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make git ca-certificates \
        zlib1g-dev libopenblas-dev liblapack-dev liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*

ARG PLINK_REPO=https://github.com/gileshall/plink-ng.git
ARG PLINK_BRANCH=hap-ibd

RUN git clone --depth 1 --branch "${PLINK_BRANCH}" "${PLINK_REPO}" /src/plink-ng

WORKDIR /src/plink-ng/2.0/build_dynamic

# Enable AVX2 (safe on all modern x86_64 cloud VMs) with CPU check fallback.
RUN make NO_AVX2= NO_SSE42= CPU_CHECK=1 ZSTD_O2=1 STATIC_ZSTD=1 -j"$(nproc)" \
    && strip plink2

# ---- stage 2: runtime ----
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libopenblas0 \
        liblapack3 \
        liblapacke \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/plink-ng/2.0/build_dynamic/plink2 /usr/local/bin/plink2

ENTRYPOINT ["plink2"]
