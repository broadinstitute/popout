# ---- Base image ----
# NVIDIA CUDA 12.6 on Ubuntu 24.04.  This image ships the CUDA runtime
# and cuDNN but NOT the full toolkit (saves ~4 GB).  JAX's cuda12 wheels
# bring their own CUDA libraries via pip, so the runtime image is sufficient.
#
# Compute capabilities included: sm_70+ (V100, A100, H100, L40S, etc.)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----
# - python3 / pip: runtime
# - libhts-dev / zlib1g-dev: pysam C extension build deps
# - plink2: VCF → PGEN conversion (pre-built binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libhts-dev \
        zlib1g-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install plink2 (static binary, AVX2 — safe for any modern x86_64 GPU server)
RUN curl -fsSL https://s3.amazonaws.com/plink2-assets/plink2_linux_avx2_20260311.zip \
        -o /tmp/plink2.zip \
    && python3 -c "import zipfile; zipfile.ZipFile('/tmp/plink2.zip').extractall('/usr/local/bin')" \
    && chmod +x /usr/local/bin/plink2 \
    && rm /tmp/plink2.zip

# ---- Python environment ----
# Use a venv to keep pip happy on externally-managed Ubuntu 24.04
RUN python3 -m venv /opt/popout
ENV PATH="/opt/popout/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools

# ---- Install dependencies (cached unless pyproject.toml changes) ----
# This is the expensive layer (~3 GB for JAX+CUDA wheels).  By installing
# deps before copying source code, Docker caches this layer across
# code-only changes — turning multi-minute rebuilds into seconds.
WORKDIR /app
COPY pyproject.toml .
RUN mkdir -p popout && touch popout/__init__.py \
    && pip install --no-cache-dir ".[dev,monitor]" \
    && rm -rf popout

# ---- Install popout (code-only, fast) ----
COPY popout/ popout/
RUN pip install --no-cache-dir --no-deps .

# ---- Runtime config ----
# Tell JAX to pre-allocate 90% of GPU memory (avoids fragmentation)
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Silence JAX TPU probe warning
ENV JAX_PLATFORMS=cuda,cpu

ENTRYPOINT ["popout"]
