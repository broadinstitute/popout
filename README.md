# popout

**GPU-accelerated local ancestry inference at biobank scale — no reference panel required.**

Feed it phased WGS from a large cohort and ancestry structure falls out of the joint distribution.

## How it works

With 500K+ samples, the data *is* the reference panel.  See [docs/THEORY.md](docs/THEORY.md)
for the full mathematical treatment.  The pipeline:

1. **SEED** — Randomized SVD on a SNP subset projects all haplotypes into PCA space. GMM assigns soft ancestry labels. Number of ancestries auto-detected via Marchenko-Pastur law (default), recursive hierarchical splitting, or eigenvalue gap heuristic.

2. **INIT** — Allele frequencies per ancestry computed from soft GMM assignments via weighted GEMM. Window-based refinement handles admixed haplotypes.

3. **REFINE** — Two refinement backends, selectable via `--method`:

   - **HMM (default)** — Forward-backward HMM with A states (not K reference haplotypes — just A ancestries). State space is tiny: 8 floats per haplotype. All haplotypes run simultaneously on GPU via gradient-checkpointed scan (O(√T) memory). EM iteration: M-step updates allele frequencies, ancestry proportions μ, and generations since admixture T (global or per-haplotype). Converges in 2-3 iterations with large samples.

   - **CNN / CNN-CRF** — 1D dilated convolutional network that processes all sites in parallel and learns multi-site LD patterns via self-supervised pseudo-label refinement. Optional CRF output layer adds learned transition modeling. See [docs/CNN.md](docs/CNN.md).

4. **DECODE** — Final posteriors from either backend. Argmax gives hard ancestry calls.

### Key design choices

- **A-state HMM, not K-state.** FLARE showed that composite reference haplotypes reduce effective K. We go further: with self-derived allele frequencies from 500K samples, we don't need individual reference haplotypes at all. The emission model is `P(allele | ancestry, site)` = allele frequency. Optionally, `--block-emissions` uses k-SNP haplotype pattern matching for LD-aware inference.

- **Sequential over sites, parallel over haplotypes.** The forward algorithm must be sequential across T sites. But all H haplotypes are independent. An A100 can hold 1M+ haplotypes' forward state (32 MB) simultaneously. Gradient checkpointing reduces stored forward states from T to √T.

- **Memory-bandwidth bound.** The A×A transition matrix product is trivially cheap. The bottleneck is reading/writing the forward state vector at each step. This means the GPU is at <1% ALU utilization — block emissions exploit this headroom for richer per-step computation with fewer total steps.

- **Pluggable refinement backend.** The M-step operates on (H, T, A) posterior tensors regardless of how they were produced. The CNN backend is a drop-in alternative to the HMM: same spectral seed, same M-step, same decode and output — only the inference engine changes. This lets you choose the right tool: HMM for interpretability and small cohorts, CNN for LD-aware inference and incremental sample addition.

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

# With per-haplotype admixture time and block emissions
popout --pgen /path/to/pgens/ \
       --map plink.GRCh38.map \
       --out results/cohort \
       --thin-cm 0.02 \
       --per-hap-T \
       --block-emissions

# CNN backend (LD-aware, fully parallel over sites)
popout --pgen /path/to/pgens/ \
       --map plink.GRCh38.map \
       --out results/cohort \
       --thin-cm 0.02 \
       --method cnn

# CNN-CRF backend (adds learned transition smoothing)
popout --pgen /path/to/pgens/ \
       --map plink.GRCh38.map \
       --out results/cohort \
       --thin-cm 0.02 \
       --method cnn-crf

# Export a reference panel for downstream tools (FLARE, RFMix)
popout --pgen /path/to/pgens/ \
       --map plink.GRCh38.map \
       --out results/cohort \
       --export-panel \
       --panel-threshold 0.99

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

The image is split in two:

- A **base image** (`popout-base`) with CUDA, plink2, and all Python dependencies. Rebuild only when `pyproject.toml` or `Dockerfile.base` changes.
- An **app image** (`popout`) that layers the current source and a git-derived version stamp on top of the base. Rebuild on every code change; only the source layer (a few MB) is uploaded.

```bash
# One-time (and whenever deps change):
./scripts/push-base.sh

# On every code change:
./scripts/push.sh
```

The installed version is available at runtime via `popout.__version__` and the
`POPOUT_VERSION` environment variable, both populated from `git describe` at
build time.

To run locally:

```bash
docker run --gpus all -v /data:/data \
    us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest \
    --pgen /data/pgens/ --map /data/map.txt --out /data/results
```

## Output files

| File | Description |
|------|-------------|
| `{prefix}.global.tsv` | Per-sample global ancestry proportions |
| `{prefix}.tracts.tsv.gz` | Local ancestry tracts (BED-like intervals per haplotype) |
| `{prefix}.model` | Fitted model parameters (n_ancestries, mu, T) |
| `{prefix}.model.npz` | Full-precision model arrays (allele_freq, mu, ancestry_names) |
| `{prefix}.chr{N}.decode.npz` | Per-chromosome dense calls + max_post (with `--probs` or `--write-dense-decode`) |
| `{prefix}.stats.jsonl` | Timestamped runtime metrics (for live monitoring) |
| `{prefix}.summary.json` | Aggregated QC stats (consumed by `popout report`) |
| `{prefix}.panel.haplotypes.tsv` | Single-ancestry whole haplotypes (with `--export-panel`) |
| `{prefix}.panel.segments.tsv.gz` | High-confidence single-ancestry segments (with `--export-panel`) |
| `{prefix}.panel.frequencies.tsv.gz` | Per-ancestry allele frequencies (with `--export-panel`) |
| `{prefix}.panel.proportions.tsv` | Per-haplotype ancestry proportions (with `--export-panel`) |

See [docs/PANEL.md](docs/PANEL.md) for output formats, thresholds, and a
worked example of the two-stage popout → FLARE pipeline.

### Converting to FLARE-compatible VCF

Use `popout convert` to transform native outputs into a FLARE-compatible `.anc.vcf.gz`:

```bash
# Run with --probs to generate per-chromosome decode files
popout --pgen data/ --map map.txt --out results/cohort --probs

# Convert to ancestry VCF
popout convert --to vcf \
    --popout-prefix results/cohort \
    --input-vcf data/cohort.phased.vcf.gz \
    --out results/cohort.anc.vcf.gz
```

This produces `{out}.anc.vcf.gz` (per-site AN1/AN2 calls, optional ANP1/ANP2 posteriors) and a companion `.global.anc.gz` with named ancestry columns. See [docs/CONVERT.md](docs/CONVERT.md) for full details, including the ANP1/ANP2 posterior approximation.

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
spectral.py    Randomized SVD + GMM + hierarchical ancestry detection
hmm.py         Forward-backward HMM in JAX with gradient checkpointing
em.py          EM loop: seed → init → iterate → decode
blocks.py      Block-level haplotype pattern encoding for LD-aware emissions
cnn/           CNN-CRF refinement backend (--method cnn or cnn-crf)
  model.py       Dilated residual 1D CNN architecture (pure JAX)
  crf.py         Linear-chain CRF output layer
  train.py       Adam optimizer, KL loss, self-training loop
  features.py    Input feature construction (allele + freq + distance channels)
  refine.py      Pipeline entry points (run_cnn, run_cnn_genome)
output.py      Tract TSV, global ancestry TSV, model file writers, panel export
stats.py       Runtime metrics collection (JSONL + W&B/TensorBoard)
report.py      QC report generation (matplotlib plots from summary JSON)
cli.py         Command-line interface
datatypes.py   Core data structures (ChromData, AncestryModel, etc.)
simulate.py    Simulated admixed data + accuracy evaluation
demo.py        Standalone demo on simulated data
```

## Accuracy

Measured on simulated 4-ancestry admixed data (F_ST 0.05–0.15, T = 20 generations,
seed = 42).  Oracle accuracy is the Bayes-optimal ceiling — what the HMM achieves
with perfect allele frequencies, mu, and T.  See `python -m popout.demo --sweep`
to reproduce and [docs/SWEEP_RESULTS.md](docs/SWEEP_RESULTS.md) for full output.

### Biobank-like cohorts (30% pure-ancestry haplotypes)

Real biobanks contain single-continental-origin individuals that provide dense
corners in PCA space for spectral initialization.

| Samples | Sites | Oracle | EM | Gap |
|---------|-------|--------|----|-----|
| 500     | 2K    | 93.0%  | 92.5% | 0.5 pp |
| 5K      | 10K   | 97.3%  | **97.1%** | **0.2 pp** |
| 500     | 10K   | 97.3%  | 96.9% | 0.4 pp |

### Fully-admixed stress test (0% pure-ancestry)

All haplotypes are mosaics — no dense PCA corners for GMM.  This regime does
not correspond to any known human biobank but exposes the spectral init
limitation.

| Samples | Sites | Oracle | EM | Gap |
|---------|-------|--------|----|-----|
| 500     | 2K    | 90.4%  | 62.2% | 28.2 pp |
| 50K     | 2K    | 90.4%  | 69.7% | 20.8 pp |
| 5K      | 10K   | 96.2%  | 42.2% | 54.0 pp |

## Performance estimates

| Samples | Device | Wall clock (22 chroms) |
|---------|--------|----------------------|
| 1K      | CPU    | ~1 minute            |
| 10K     | CPU    | ~10 minutes          |
| 10K     | A100   | ~30 seconds          |
| 100K    | A100   | ~5 minutes           |
| 500K    | A100   | ~15 minutes          |
| 500K    | 8×A100 | ~3 minutes           |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `hmm` | Refinement backend: `hmm`, `cnn`, or `cnn-crf` |
| `--ancestry-detection` | `marchenko-pastur` | Auto-detection method: `marchenko-pastur`, `recursive` (hierarchical BIC splitting), or `eigenvalue-gap` |
| `--per-hap-T` | off | Estimate per-haplotype admixture time (disabled by default; pass `--per-hap-T` to enable) |
| `--n-T-buckets` | 20 | Number of transition-matrix buckets for per-haplotype T |
| `--block-emissions` | off | Use k-SNP haplotype pattern matching instead of single-site Bernoulli |
| `--block-size` | 8 | SNPs per block when using block emissions |
| `--thin-cm` | none | Minimum cM spacing for site thinning (recommended: 0.02 for WGS) |
| `--export-panel` | off | Export reference panel files for downstream tools |
| `--panel-threshold` | 0.95 | Min posterior for whole-haplotype extraction |
| `--panel-segment-threshold` | 0.99 | Min per-site posterior for segment extraction |
| `--panel-min-segment-cm` | 1.0 | Min segment length in cM |
| `--panel-max-per-ancestry` | all | Cap on haplotypes per ancestry (whole-haplotype output) |
| `--cnn-layers` | 12 | Number of dilated conv layers (`--method cnn`/`cnn-crf`) |
| `--cnn-channels` | 64 | Hidden channel dimension (`--method cnn`/`cnn-crf`) |
| `--cnn-epochs` | 5 | Training epochs per pseudo-label round |
| `--cnn-pseudo-rounds` | 2 | Number of self-training rounds |
| `--cnn-lr` | 1e-3 | CNN learning rate |
| `--cnn-batch-size` | 512 | Haplotypes per CNN training/inference batch |
| `--seed-method` | `gmm` | Seeding strategy: `gmm` (single-pass PCA+GMM) or `recursive` (K=2 EM splitting) |
| `--recursive-min-cluster-size` | 1000 | Minimum cluster size for further splitting |
| `--recursive-max-depth` | 6 | Maximum recursion depth for K=2 splitting |
| `--recursive-max-leaves` | 20 | Maximum leaf populations from recursive splitting |
| `--recursive-bic-per-sample` | 0.05 | Per-sample BIC improvement floor for splits (scales with N) |
| `--recursive-em-iter` | 3 | EM iterations per K=2 split |
| `--recursive-merge-hellinger` | 0.04 | Merge leaves with Hellinger distance below this (0 = disable) |
| `--freeze-anchors-iters` | 0 | Freeze seed responsibilities for first N EM iterations |

## Recursive seeding

With `--seed-method recursive`, popout replaces the single-pass PCA + GMM
seed with a recursive K=2 EM splitting strategy.  At each level, BIC
compares a 1-component vs 2-component GMM on the cluster's own sub-PCA.
If the split is justified, K=2 popout EM runs on the cluster's haplotypes,
and argmax of per-haplotype mean posteriors partitions them into two children.
Recursion continues until no more splits pass BIC, clusters are too small,
or `--recursive-max-leaves` is reached.

This is useful when small populations (low cohort fraction) or closely
related populations (low pairwise F_ST) get absorbed by the flat GMM seed.
Recursive splitting attacks the easiest split at each level and re-computes
PCA within each subset, so within-subset structure is not washed out by
global axes.

```bash
# Recursive seeding — discovers K automatically
popout --pgen /path/to/pgens/ \
       --map plink.GRCh38.map \
       --out results/cohort \
       --seed-method recursive

# With anchor freezing to protect small leaves
popout --pgen /path/to/pgens/ \
       --map plink.GRCh38.map \
       --out results/cohort \
       --seed-method recursive \
       --freeze-anchors-iters 5
```

Note: `--seed-method recursive` and `--n-ancestries` are incompatible —
the recursion discovers K from the data.

## CNN refinement backend

With `--method cnn`, popout replaces the HMM forward-backward with a 1D
dilated CNN that processes all sites in parallel and learns multi-site LD
patterns via self-supervised pseudo-label refinement.  Add `--method cnn-crf`
for an additional CRF output layer that learns transition penalties for
smoother tract boundaries.  The CNN is particularly useful for fine-scale
ancestry (closely related populations) and for incremental inference on new
samples without retraining.

See [docs/CNN.md](docs/CNN.md) for architecture details, training procedure,
and guidance on when to use CNN vs HMM.

## Reference panel export

With `--export-panel`, popout extracts single-ancestry haplotypes and segments
from its posteriors, producing a ready-made reference panel for downstream
LD-aware tools like FLARE or RFMix. No manual curation is required — in a 500K
biobank, this typically yields ~850K single-ancestry haplotypes, an order of
magnitude larger than any curated panel.

See [docs/PANEL.md](docs/PANEL.md) for full details and the two-stage
popout → FLARE pipeline.

## Limitations

- No multi-GPU support yet. Use `--chromosomes` to manually partition across GPUs.
- LD-aware block boundaries (using r² decay or recombination maps) are not yet implemented; blocks are fixed-width.
- Rare-ancestry detection (clusters with <1% of samples) needs further work.

## What's next

- [ ] LD-aware variable-width blocks (r² decay threshold or external block maps)
- [ ] `jax.pmap` multi-GPU parallelism
- [ ] Rare-ancestry detection (clusters with <1% of samples)
- [ ] Benchmark CNN vs HMM on simulated data across F_ST regimes
- [ ] Benchmark against FLARE on simulated data (two-stage pipeline; see [docs/PANEL.md](docs/PANEL.md))

## References

- Browning SR, Waples RK, Browning BL. *Fast, accurate local ancestry inference with FLARE.* AJHG 2023.
- Li N, Stephens M. *Modeling linkage disequilibrium.* Genetics 2003.
- Halko N, Martinsson PG, Tropp JA. *Finding structure with randomness.* SIAM Review 2011.
