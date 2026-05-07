FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        bcftools \
    && rm -rf /var/lib/apt/lists/*

# plink2 (AVX2 build — covers all modern x86_64 cloud VMs)
RUN curl -fsSL https://s3.amazonaws.com/plink2-assets/plink2_linux_avx2_20260311.zip \
        -o /tmp/plink2.zip \
    && python3 -c "import zipfile; zipfile.ZipFile('/tmp/plink2.zip').extractall('/usr/local/bin')" \
    && chmod +x /usr/local/bin/plink2 \
    && rm /tmp/plink2.zip

# bcftools is used by extract_panel_geno.wdl to split multi-allelic
# panel positions into per-alt biallelic rows (norm -m -any) before
# the final --make-pgen, so the consolidated sidecar is biallelic-
# only (pgenlib's phased reader cannot handle multi-allelic).

ENTRYPOINT ["plink2"]
