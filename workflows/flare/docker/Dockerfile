# FLARE local-ancestry inference (https://github.com/browning-lab/flare).
#
# Upstream ships the jar at an unversioned URL — we download it then check
# the embedded version string against FLARE_VERSION so the build fails fast
# if Browning Lab has bumped the jar without us bumping the tag.
FROM ubuntu:24.04

ARG FLARE_VERSION

# bcftools-side runtime parity: openjdk for FLARE itself, python+pip+curl
# for the magicwand bootstrap that WDL tasks run.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        openjdk-17-jre-headless \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "${FLARE_VERSION}" \
    && mkdir -p /opt/flare \
    && curl -fsSL https://faculty.washington.edu/browning/flare.jar \
        -o /opt/flare/flare.jar \
    && actual=$(java -jar /opt/flare/flare.jar 2>&1 \
                | grep -oE 'flare version [0-9.]+' | head -1 | awk '{print $3}') \
    && if [ "$actual" != "${FLARE_VERSION}" ]; then \
         echo "ERROR: upstream flare.jar reports version '$actual', expected '${FLARE_VERSION}'" >&2; \
         exit 1; \
       fi
