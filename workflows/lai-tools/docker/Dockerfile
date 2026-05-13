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

# Drop the helpers in /usr/local/bin so they're directly invokable.
COPY scripts/generate_partitions.py /usr/local/bin/generate_partitions.py
RUN chmod +x /usr/local/bin/generate_partitions.py
