FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        bcftools \
    && rm -rf /var/lib/apt/lists/*

# plink2 binary download.
#
# The bare https://s3.amazonaws.com/plink2-assets/plink2_linux_avx2_<date>.zip
# path is a rotating dev snapshot — older dated artifacts get pulled when a
# new build lands, breaking docker rebuilds.  The alpha7/ subdirectory
# preserves dated builds permanently per the cog-genomics convention.
# Bump PLINK2_SERIES + PLINK2_BUILD to update; both arguments together
# define the artifact URL.
ARG PLINK2_SERIES=alpha7
ARG PLINK2_BUILD=20260504
RUN curl -fsSL "https://s3.amazonaws.com/plink2-assets/${PLINK2_SERIES}/plink2_linux_avx2_${PLINK2_BUILD}.zip" \
        -o /tmp/plink2.zip \
    && python3 -c "import zipfile; zipfile.ZipFile('/tmp/plink2.zip').extractall('/usr/local/bin')" \
    && chmod +x /usr/local/bin/plink2 \
    && rm /tmp/plink2.zip \
    && /usr/local/bin/plink2 --version

# bcftools is used by extract_panel_geno.wdl to split multi-allelic
# panel positions into per-alt biallelic rows (norm -m -any) before
# the final --make-pgen, so the consolidated sidecar is biallelic-
# only (pgenlib's phased reader cannot handle multi-allelic).

ENTRYPOINT ["plink2"]
