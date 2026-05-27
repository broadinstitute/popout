#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTEXT="$(cd "$SCRIPT_DIR/.." && pwd)"          # build context = workflows/lai-tools/
REPO_ROOT="$(cd "$CONTEXT/../.." && pwd)"        # gpulai repo root

# popout repo: defaults to the sibling checkout next to gpulai. Override
# via POPOUT_DIR for CI / non-standard layouts.
POPOUT_DIR="${POPOUT_DIR:-$REPO_ROOT/../popout}"
if [ ! -d "$POPOUT_DIR/popout" ]; then
    echo "popout repo not found at $POPOUT_DIR (override with POPOUT_DIR=)" >&2
    exit 1
fi
POPOUT_DIR="$(cd "$POPOUT_DIR" && pwd)"

VALIDATION_DIR="$REPO_ROOT/validation"
if [ ! -d "$VALIDATION_DIR" ]; then
    echo "validation dir not found at $VALIDATION_DIR" >&2
    exit 1
fi

# Pin to a specific bcftools tag (rather than :latest) so lai-tools image
# rebuilds are reproducible. Override via BCFTOOLS_TAG when bumping bcftools.
BCFTOOLS_TAG="${BCFTOOLS_TAG:-latest}"
BCFTOOLS_IMAGE="us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:${BCFTOOLS_TAG}"

# ── Stage region-mask BEDs OUTSIDE the docker build ─────────────────────
# Region masks are fetched here (host-side, with network access) and
# mounted into the docker build via --build-context. This keeps the
# Dockerfile offline-buildable and deterministic — every build sees
# byte-identical BEDs regardless of UCSC availability at build time.
REGION_MASKS_STAGED="$(mktemp -d "${TMPDIR:-/tmp}/lai-tools-region-masks.XXXXXX")"
trap 'rm -rf "$REGION_MASKS_STAGED"' EXIT
echo "Staging region masks → $REGION_MASKS_STAGED"

# Centromere positions (UCSC centromeres.txt: bin, chrom, chromStart, chromEnd, ...).
curl -fsSL https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/centromeres.txt.gz \
    | gunzip \
    | awk 'BEGIN{OFS="\t"} $2 ~ /^chr[0-9XYM]+$/ {print $2,$3,$4,"centromere"}' \
    | sort -k1,1 -k2,2n \
    > "$REGION_MASKS_STAGED/centromere.bed"

# Segmental duplications (UCSC genomicSuperDups.txt: bin, chrom, chromStart, chromEnd, ...).
curl -fsSL https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/genomicSuperDups.txt.gz \
    | gunzip \
    | awk 'BEGIN{OFS="\t"} $2 ~ /^chr[0-9XYM]+$/ {print $2,$3,$4,"segdup"}' \
    | sort -k1,1 -k2,2n \
    > "$REGION_MASKS_STAGED/segdup.bed"

# HLA region — single interval on chr6 (GENCODE-spanning HLA locus).
printf 'chr6\t28510120\t33480577\thla\n' > "$REGION_MASKS_STAGED/hla.bed"

# High-LD regions — no canonical UCSC track for GRCh38; pulled from the
# in-repo copy. Path can be overridden via HIGH_LD_BED for forks that
# vendor a different list.
HIGH_LD_BED="${HIGH_LD_BED:-$REPO_ROOT/my_notes/tarpit/high-LD-regions-hg38-GRCh38.bed}"
if [ ! -f "$HIGH_LD_BED" ]; then
    echo "high-LD BED not found at $HIGH_LD_BED (override with HIGH_LD_BED=)" >&2
    exit 1
fi
cp "$HIGH_LD_BED" "$REGION_MASKS_STAGED/high_ld.bed"

echo "Staged region masks:"
ls -l "$REGION_MASKS_STAGED"

# ── Version tag ─────────────────────────────────────────────────────────
# Tag based on the git SHA of every input that affects the image:
# the lai-tools scripts dir + the validation dir + the popout repo SHA.
SCRIPTS_SHA=$(cd "$CONTEXT" && git rev-parse --short HEAD:scripts 2>/dev/null || echo "x")
VAL_SHA=$(cd "$REPO_ROOT" && git rev-parse --short HEAD:validation 2>/dev/null || echo "x")
POPOUT_SHA=$(cd "$POPOUT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "x")

REPO="us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools"
TAG="s${SCRIPTS_SHA}-v${VAL_SHA}-p${POPOUT_SHA}-bcftools-${BCFTOOLS_TAG}"

echo "Build inputs:"
echo "  context      = $CONTEXT"
echo "  validation   = $VALIDATION_DIR"
echo "  popout       = $POPOUT_DIR"
echo "  region_masks = $REGION_MASKS_STAGED"
echo "  bcftools     = $BCFTOOLS_IMAGE"
echo "  image tag    = ${REPO}:${TAG}"

docker buildx build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "${REPO}:${TAG}" \
    -t "${REPO}:latest" \
    --platform linux/amd64 \
    --build-arg "BCFTOOLS_IMAGE=${BCFTOOLS_IMAGE}" \
    --build-context "validation=${VALIDATION_DIR}" \
    --build-context "popout=${POPOUT_DIR}" \
    --build-context "region_masks=${REGION_MASKS_STAGED}" \
    --push \
    "$CONTEXT"
