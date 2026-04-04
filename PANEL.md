# Reference Panel Generation

popout can optionally export reference panels derived from its posterior
ancestry probabilities. This gives downstream LD-aware tools (FLARE, RFMix,
LAMP-LD) a high-quality, automatically generated reference panel — no manual
curation required.

## Quick start

Add `--export-panel` to a normal popout run:

```bash
popout --vcf biobank.vcf.gz --out results/cohort --export-panel
```

This produces the standard outputs (`.global.tsv`, `.tracts.tsv.gz`, `.model`)
**plus** four panel files:

| File | Contents |
|------|----------|
| `{prefix}.panel.haplotypes.tsv` | Whole haplotypes classified as single-ancestry |
| `{prefix}.panel.segments.tsv.gz` | High-confidence single-ancestry segments from admixed haplotypes |
| `{prefix}.panel.frequencies.tsv.gz` | Per-ancestry allele frequencies at every site |
| `{prefix}.panel.proportions.tsv` | Per-haplotype global ancestry proportions |

Without `--export-panel`, no panel files are produced and the pipeline is
unchanged.

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--export-panel` | off | Enable panel export |
| `--panel-threshold` | 0.95 | Min posterior for whole-haplotype extraction |
| `--panel-segment-threshold` | 0.99 | Min per-site posterior for segment extraction |
| `--panel-min-segment-cm` | 1.0 | Min segment length in cM |
| `--panel-max-per-ancestry` | all | Cap on haplotypes per ancestry (whole-haplotype output) |

Example with tuned thresholds:

```bash
popout --pgen /data/ukb/ \
       --out results/ukb \
       --export-panel \
       --panel-threshold 0.99 \
       --panel-segment-threshold 0.999 \
       --panel-min-segment-cm 2.0 \
       --panel-max-per-ancestry 5000
```

## Output formats

### Whole-haplotype panel (`*.panel.haplotypes.tsv`)

Lists haplotypes that are single-ancestry across the entire genome.

A haplotype passes when:
1. Its minimum per-site max-posterior exceeds `--panel-threshold` across all
   chromosomes.
2. The hard-call ancestry is the same at every site (no switches).

```
sample_id    haplotype    ancestry    min_posterior    mean_posterior
NA12878      0            0           0.9970           0.9990
NA12878      1            0           0.9930           0.9980
HG00096      0            1           0.9850           0.9960
```

In a 500K biobank with ~15% recent admixture and threshold 0.95, expect
roughly 85% of haplotypes to pass — yielding ~850K single-ancestry
haplotypes, an order of magnitude larger than any curated reference panel.

### Segment panel (`*.panel.segments.tsv.gz`)

Extracts maximal contiguous segments from *all* haplotypes (including admixed
ones) where the dominant ancestry posterior exceeds `--panel-segment-threshold`
at every site.

```
sample_id    haplotype    chrom    start_bp    end_bp    start_cm    end_cm    ancestry    mean_posterior    n_sites
NA19700      0            1        1000000     5000000   1.2000      5.8000    1           0.9980            342
NA19700      0            1        5500000     12000000  6.1000      14.3000   0           0.9950            518
```

Segments are filtered by both `--panel-min-segment-cm` and a 50-site minimum.
An individual who is 60% European and 40% African contributes European *and*
African reference segments — particularly valuable for underrepresented
populations.

### Allele frequencies (`*.panel.frequencies.tsv.gz`)

Per-ancestry allele frequencies at every site, taken directly from popout's
fitted model (the M-step frequency matrix).

```
chrom    pos        site_id    ancestry_0_freq    ancestry_1_freq    ancestry_2_freq
1        10000      rs12345    0.123400           0.567800           0.034500
```

### Per-haplotype proportions (`*.panel.proportions.tsv`)

Mean posterior ancestry proportions for each haplotype individually (not
averaged across diploid pairs like `*.global.tsv`).

```
sample_id    haplotype    ancestry_0    ancestry_1    ancestry_2
NA12878      0            0.9980        0.0010        0.0010
NA12878      1            0.9975        0.0015        0.0010
```

Useful as soft labels or GWAS covariates.

## Two-stage pipeline: popout then FLARE

The primary use case for panel export is feeding a downstream LD-aware tool
for fine-scale (within-continent) ancestry inference:

```
Stage 1 — popout (continental, A ~ 4-8)
  Input:  Full biobank, no reference panel needed
  Output: Extracted single-ancestry haplotypes + allele frequencies

    |
    v

Stage 2 — FLARE (fine-scale, per continent)
  Input:  popout-derived panel + admixed haplotypes from Stage 1
  Output: Within-continent ancestry (e.g., N. vs S. European)
```

1. Run popout on the full biobank with `--export-panel`.
2. Partition extracted haplotypes by continental ancestry.
3. Run FLARE per continent using the popout-derived panel for fine-scale
   resolution.
4. Merge continental calls from popout with fine-scale calls from FLARE.

## Computational cost

Panel extraction is a post-processing pass over the final posteriors (already
in memory). It adds negligible time compared to the HMM forward-backward pass:

- Whole-haplotype scan: O(H x T x A) — one pass over gamma.
- Segment identification: O(H x T) — contiguous run detection.
- No additional large memory allocations.
