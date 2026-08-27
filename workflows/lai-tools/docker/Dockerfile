# lai-tools — Python helpers for the LAI pipeline, layered on top of our
# bcftools image so the same container has bcftools + pysam + our scripts.
#
# Scripts live in /usr/local/bin so they're on PATH. No package install,
# no pyproject.toml — just executable Python files.

ARG BCFTOOLS_IMAGE=us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest
FROM ${BCFTOOLS_IMAGE}

# The bcftools image sets ENTRYPOINT=bcftools; we want lai-tools containers
# to behave like generic Python utility containers.
ENTRYPOINT []

# Python deps:
#   pysam      — required by generate_partitions.py + diagnostic scripts (parse VCF/BCF)
#   numpy, scipy, statsmodels, matplotlib  — required by the FLARE validation
#       diagnostic scripts (compare_to_rf, validate_regional, plot_concordance, etc.)
#   pandas     — popout transitive dep (read_tracts uses it indirectly)
#   pyarrow    — v7 extract_tract_events / collate_tract_bundle parquet I/O
# --break-system-packages is required on Ubuntu 24.04 (PEP 668 marks the
# system Python as externally managed).
RUN pip3 install --break-system-packages --no-cache-dir \
        pysam numpy scipy statsmodels matplotlib pandas pyarrow

# Diagnostic tools for throughput_probe.sh:
#   tabix     — htslib's raw region-streamer (apt-installed, links against
#               system libhts; deliberately a different htslib build than
#               bcftools so the diff in measured throughput reveals whether
#               our pinned develop htslib helps or hurts vs stock)
#   strace    — per-probe syscall counts (optional, run_probe wraps the cmd)
#   sysstat   — vmstat + iostat for background instrumentation
#   jq        — JSON parsing in shell (oauth token responses, etc.)
#   gnupg + apt-transport-https — required to add the Google Cloud apt repo
# All of these are diagnostic; production WDLs only need bcftools/python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tabix strace sysstat jq gnupg apt-transport-https \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# google-cloud-cli — for `gcloud storage cat -r` byte-range probes. Installs
# from Google's apt repo so the version stays current; ~300 MB but the image
# is diagnostic-only so the bloat is acceptable.
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Drop the helpers in /usr/local/bin so they're directly invokable.
COPY scripts/generate_partitions.py /usr/local/bin/generate_partitions.py
COPY scripts/tbi_byte_range.py      /usr/local/bin/tbi_byte_range.py
COPY scripts/bgzf_truncate.py       /usr/local/bin/bgzf_truncate.py
COPY scripts/throughput_probe.sh    /usr/local/bin/throughput_probe.sh
RUN chmod +x /usr/local/bin/generate_partitions.py \
             /usr/local/bin/tbi_byte_range.py \
             /usr/local/bin/bgzf_truncate.py \
             /usr/local/bin/throughput_probe.sh

# ── FLARE validation layer ──────────────────────────────────────────────
#
# Four named build contexts are expected (set by push.sh via
# `--build-context`):
#   default      : the lai-tools build dir (this Dockerfile + scripts/)
#   validation   : the repo's validation/ dir (schema + orchestrator + sub-scripts)
#   popout       : the popout repo root (for pip install --no-deps)
#   region_masks : a dir push.sh staged with centromere.bed, segdup.bed,
#                  hla.bed, high_ld.bed (fetched outside the docker build
#                  so this Dockerfile is offline-buildable and deterministic)
#
# Source vendoring is avoided — everything comes through build contexts.

# validation/ — orchestrator + collator + all sub-scripts. Importable
# as `validation.schema` via PYTHONPATH. ``__init__.py`` files ship in
# the repo as of Phase 6 of the label-space retrofit; no in-container
# fixup needed.
COPY --from=validation . /opt/validation/
ENV PYTHONPATH=/opt

# popout (viz loaders) — diagnostic scripts import popout.viz._loaders and
# popout.viz._style. These are pure-numpy; --no-deps avoids pulling in jax
# / CUDA. Concrete deps (numpy, scipy, pysam, matplotlib) are installed
# above.
COPY --from=popout . /tmp/popout-src/
RUN pip3 install --break-system-packages --no-cache-dir --no-deps /tmp/popout-src \
 && rm -rf /tmp/popout-src

# Bundled reference data:
#   /opt/grch38.chrom.sizes  — for validate_regional.py
#   /opt/region_masks/       — HLA, segdup, centromere, high-LD BEDs
# Both fetched/staged by push.sh outside this build.
COPY --from=validation data/grch38.chrom.sizes /opt/grch38.chrom.sizes
COPY --from=region_masks . /opt/region_masks/

# ── FLARE cleanup / finalize scripts ────────────────────────────────────
# Consumed by workflows/flare/wdl/flare_cleanup.wdl and flare_finalize.wdl
# at /opt/flare/scripts/. Same layout convention as /opt/validation.
COPY --from=flare_scripts . /opt/flare/scripts/
