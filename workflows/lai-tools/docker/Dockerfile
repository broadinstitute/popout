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

# pysam is the only Python dep — needed for parsing the .tbi binary format
# in generate_partitions.py and for building synthetic test fixtures.
# --break-system-packages is required on Ubuntu 24.04 (PEP 668 marks the
# system Python as externally managed).
RUN pip3 install --break-system-packages --no-cache-dir pysam

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
COPY scripts/throughput_probe.sh    /usr/local/bin/throughput_probe.sh
RUN chmod +x /usr/local/bin/generate_partitions.py \
             /usr/local/bin/tbi_byte_range.py \
             /usr/local/bin/throughput_probe.sh
