# popout

**GPU-accelerated self-bootstrapping local ancestry inference.**

No reference panel required. Feed it phased WGS from a large cohort and ancestry structure falls out of the joint distribution.

## How it works

With 500K+ samples, the data *is* the reference panel. The pipeline:

1. **SEED** — Randomized SVD on a SNP subset projects all haplotypes into PCA space. K-means assigns initial (global) ancestry labels. Number of ancestries auto-detected from the eigenvalue gap.

2. **INIT** — Allele frequencies per ancestry computed from hard labels. This is a single GEMM on GPU: `(A × H) @ (H × T)`.

3. **EM ITERATE** — Forward-backward HMM with A states (not K reference haplotypes — just A ancestries). State space is tiny: 8 floats per haplotype. All haplotypes run simultaneously on GPU. M-step updates allele frequencies, ancestry proportions μ, and generations since admixture T. Converges in 2-3 iterations with large samples.

4. **DECODE** — Final forward-backward pass produces posteriors. Argmax gives hard ancestry calls.

### Key design choices

- **A-state HMM, not K-state.** FLARE showed that composite reference haplotypes reduce effective K. We go further: with self-derived allele frequencies from 500K samples, we don't need individual reference haplotypes at all. The emission model is `P(allele | ancestry, site)` = allele frequency.

- **Sequential over sites, parallel over haplotypes.** The forward algorithm must be sequential across T sites. But all H haplotypes are independent. An A100 can hold 1M+ haplotypes' forward state (32 MB) simultaneously.

- **Memory-bandwidth bound.** The A×A transition matrix product is trivially cheap. The bottleneck is reading/writing the forward state vector at each step. This means the GPU is at <1% ALU utilization — there's room for richer emission models for free.

## Usage

```bash
popout --vcf cohort.phased.vcf.gz \
       --map plink.GRCh38.map \
       --out results/cohort \
       --n-ancestries 6 \
       --n-em-iter 3

# Auto-detect number of ancestries:
popout --vcf cohort.phased.vcf.gz \
       --map plink.GRCh38.map \
       --out results/cohort
```

## Installation

```bash
# GPU (recommended)
pip install jax[cuda12] pysam
pip install -e .

# CPU (slow but works)
pip install jax pysam
pip install -e .
```

## Output files

| File | Description |
|------|-------------|
| `{prefix}.global.tsv` | Per-sample global ancestry proportions |
| `{prefix}.model` | Fitted model parameters (mu, T, mismatch rates) |

## Architecture

```
vcf_io.py      pysam VCF reading + genetic map loading
spectral.py    Randomized SVD + k-means for seed labels
hmm.py         Forward-backward HMM in JAX (GPU workhorse)
em.py          EM loop: seed → init → iterate → decode
output.py      VCF/TSV output writers
cli.py         Command-line interface
datatypes.py   Core data structures
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

## Limitations (first draft)

- No haplotype-window emission model yet (single-site only). This limits accuracy for closely related ancestries (e.g., CHB vs JPT).
- No checkpointed backward pass — stores full `(H, T, A)` posteriors. For 1M haplotypes × 500K WGS sites, this exceeds GPU memory. Works fine for array data (~20K sites) or with `--batch-size` tuning.
- No multi-GPU support yet. Use `--chromosomes` to manually partition across GPUs.
- VCF output is slow for large cohorts; global ancestry TSV is the primary output.

## What's next

- [ ] Haplotype-window emissions (8-SNP pattern histograms per ancestry)
- [ ] Checkpointed backward pass for WGS-scale site counts
- [ ] `jax.pmap` multi-GPU parallelism
- [ ] Streaming VCF reader (don't load full chromosome into RAM)
- [ ] Benchmark against FLARE on simulated data
- [ ] Rare-ancestry detection (clusters with <1% of samples)

## References

- Browning SR, Waples RK, Browning BL. *Fast, accurate local ancestry inference with FLARE.* AJHG 2023.
- Li N, Stephens M. *Modeling linkage disequilibrium.* Genetics 2003.
- Halko N, Martinsson PG, Tropp JA. *Finding structure with randomness.* SIAM Review 2011.
