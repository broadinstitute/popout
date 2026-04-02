# popout

**GPU-accelerated local ancestry inference at biobank scale — no reference panel required.**

Feed it phased WGS from a large cohort and ancestry structure falls out of the joint distribution.

## How it works

With 500K+ samples, the data *is* the reference panel. The pipeline:

1. **SEED** — Randomized SVD on a SNP subset projects all haplotypes into PCA space. GMM assigns soft ancestry labels. Number of ancestries auto-detected from the eigenvalue gap.

2. **INIT** — Allele frequencies per ancestry computed from soft GMM assignments via weighted GEMM. Window-based refinement handles admixed haplotypes.

3. **EM ITERATE** — Forward-backward HMM with A states (not K reference haplotypes — just A ancestries). State space is tiny: 8 floats per haplotype. All haplotypes run simultaneously on GPU. M-step updates allele frequencies, ancestry proportions μ, and generations since admixture T. Converges in 2-3 iterations with large samples.

4. **DECODE** — Final forward-backward pass produces posteriors. Argmax gives hard ancestry calls.

### Key design choices

- **A-state HMM, not K-state.** FLARE showed that composite reference haplotypes reduce effective K. We go further: with self-derived allele frequencies from 500K samples, we don't need individual reference haplotypes at all. The emission model is `P(allele | ancestry, site)` = allele frequency.

- **Sequential over sites, parallel over haplotypes.** The forward algorithm must be sequential across T sites. But all H haplotypes are independent. An A100 can hold 1M+ haplotypes' forward state (32 MB) simultaneously.

- **Memory-bandwidth bound.** The A×A transition matrix product is trivially cheap. The bottleneck is reading/writing the forward state vector at each step. This means the GPU is at <1% ALU utilization — there's room for richer emission models for free.

## Usage

```bash
# From PGEN (recommended for biobank-scale data)
popout --pgen /path/to/per_chrom_pgens/ \
       --map plink.GRCh38.map \
       --out results/cohort \
       --thin-cm 0.02

# From VCF
popout --vcf cohort.phased.vcf.gz \
       --map plink.GRCh38.map \
       --out results/cohort \
       --n-ancestries 6

# Generate QC report from a completed run
popout report --stats results/cohort.summary.json --out report/
```

### Preparing PGEN input from VCF

```bash
for chr in {1..22}; do
  plink2 --vcf chr${chr}.phased.vcf.gz --make-pgen phased-list --out pgen_dir/chr${chr}
done
```

The `phased-list` flag preserves phase information. Point `--pgen` at the directory.

## Installation

```bash
# GPU (recommended)
pip install -e ".[dev]"

# CPU (slow but works)
pip install -e ".[dev]" --no-deps
pip install jax numpy pysam Pgenlib matplotlib pytest
```

### Docker

```bash
docker build -t popout .
docker run --gpus all -v /data:/data popout \
    --pgen /data/pgens/ --map /data/map.txt --out /data/results
```

## Output files

| File | Description |
|------|-------------|
| `{prefix}.global.tsv` | Per-sample global ancestry proportions |
| `{prefix}.tracts.tsv.gz` | Local ancestry tracts (BED-like intervals per haplotype) |
| `{prefix}.model` | Fitted model parameters (n_ancestries, mu, T) |
| `{prefix}.stats.jsonl` | Timestamped runtime metrics (for live monitoring) |
| `{prefix}.summary.json` | Aggregated QC stats (consumed by `popout report`) |

### Tract format

```
#chrom  start_bp   end_bp     sample    haplotype  ancestry  n_sites
chr1    100000     5200000    SAMPLE_0  0          0         45
chr1    5250000    12800000   SAMPLE_0  0          2         62
```

## Monitoring

```bash
# Live monitoring with Weights & Biases
popout --pgen data/ --map map.txt --out results --monitor wandb

# Live monitoring with TensorBoard
popout --pgen data/ --map map.txt --out results --monitor tensorboard

# Disable stats files entirely
popout --pgen data/ --map map.txt --out results --no-stats
```

## Architecture

```
pgen_io.py     PGEN reader (biobank-scale, chunked, with site thinning)
vcf_io.py      VCF/BCF reader (pysam, for smaller datasets)
gmap.py        Genetic map loading and chromosome normalization
spectral.py    Randomized SVD + GMM for seed labels
hmm.py         Forward-backward HMM in JAX (GPU workhorse)
em.py          EM loop: seed → init → iterate → decode
output.py      Tract TSV, global ancestry TSV, model file writers
stats.py       Runtime metrics collection (JSONL + W&B/TensorBoard)
report.py      QC report generation (matplotlib plots from summary JSON)
cli.py         Command-line interface
datatypes.py   Core data structures (ChromData, AncestryModel, etc.)
simulate.py    Simulated admixed data + accuracy evaluation
demo.py        Standalone demo on simulated data
```

## Performance estimates

| Samples | Device | Wall clock (22 chroms) |
|---------|--------|----------------------|
| 1K      | CPU    | ~1 minute            |
| 10K     | CPU    | ~10 minutes          |
| 10K     | A100   | ~30 seconds          |
| 100K    | A100   | ~5 minutes           |
| 500K    | A100   | ~15 minutes          |
| 500K    | 8×A100 | ~3 minutes           |

## Limitations

- No haplotype-window emission model yet (single-site only). This limits accuracy for closely related ancestries (e.g., CHB vs JPT).
- No checkpointed backward pass — stores full `(H, T, A)` posteriors. For WGS use `--thin-cm 0.02` to reduce to array-like density.
- No multi-GPU support yet. Use `--chromosomes` to manually partition across GPUs.

## What's next

- [ ] Haplotype-window emissions (8-SNP pattern histograms per ancestry)
- [ ] Checkpointed backward pass for full WGS-scale site counts
- [ ] `jax.pmap` multi-GPU parallelism
- [ ] Rare-ancestry detection (clusters with <1% of samples)
- [ ] Benchmark against FLARE on simulated data

## References

- Browning SR, Waples RK, Browning BL. *Fast, accurate local ancestry inference with FLARE.* AJHG 2023.
- Li N, Stephens M. *Modeling linkage disequilibrium.* Genetics 2003.
- Halko N, Martinsson PG, Tropp JA. *Finding structure with randomness.* SIAM Review 2011.
