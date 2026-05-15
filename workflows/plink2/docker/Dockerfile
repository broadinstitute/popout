FROM python:3.12-slim

# bcftools is needed by plink2_export_clusters.wdl for `bcftools index --tbi`
# on each per-cluster VCF output. The distro bcftools is plenty for index
# generation — we don't need the htslib-develop features of our pinned
# bcftools image since we're operating on local files.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        bcftools \
    && rm -rf /var/lib/apt/lists/*

# plink2 AVX2 build, fetched from the upstream GitHub release.
# PLINK2_VERSION is a release tag from https://github.com/chrchang/plink-ng/releases
# e.g. v2.0.0-a.7.1
ARG PLINK2_VERSION

RUN test -n "${PLINK2_VERSION}" \
    && curl -fsSL "https://github.com/chrchang/plink-ng/releases/download/${PLINK2_VERSION}/plink2_linux_avx2.zip" \
        -o /tmp/plink2.zip \
    && python3 -c "import zipfile; zipfile.ZipFile('/tmp/plink2.zip').extractall('/usr/local/bin')" \
    && chmod +x /usr/local/bin/plink2 \
    && rm /tmp/plink2.zip

ENTRYPOINT ["plink2"]
