# CNN-CRF Refinement Backend

popout offers a 1D dilated convolutional network (CNN) as an alternative
refinement backend to the default HMM forward-backward algorithm.  The CNN
processes all sites in parallel, naturally captures multi-site linkage
disequilibrium patterns, and can be reused for incremental inference on new
samples without retraining.

## Quick start

Add `--method cnn` to a normal popout run:

```bash
popout --vcf biobank.vcf.gz --out results/cohort --method cnn
```

For closely related ancestries, add the CRF output layer:

```bash
popout --vcf biobank.vcf.gz --out results/cohort --method cnn-crf
```

All standard outputs (`.global.tsv`, `.tracts.tsv.gz`, `.model`) and optional
panel exports (`--export-panel`) work identically — the CNN produces the same
posterior format as the HMM.

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `hmm` | Refinement backend: `hmm`, `cnn`, or `cnn-crf` |
| `--cnn-layers` | 12 | Number of dilated convolutional layers |
| `--cnn-channels` | 64 | Hidden channel dimension per layer |
| `--cnn-epochs` | 5 | Training epochs per pseudo-label round |
| `--cnn-pseudo-rounds` | 2 | Number of self-training rounds |
| `--cnn-lr` | 1e-3 | Learning rate (Adam with cosine decay) |
| `--cnn-batch-size` | 512 | Haplotypes per CNN training/inference batch |

Example with tuned hyperparameters:

```bash
popout --pgen /data/ukb/ \
       --out results/ukb \
       --method cnn-crf \
       --cnn-layers 10 \
       --cnn-channels 32 \
       --cnn-epochs 3 \
       --cnn-pseudo-rounds 3 \
       --cnn-batch-size 1024
```

## When to use CNN vs HMM

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Continental-scale (A = 4–6, high F_ST) | Either — similar accuracy | Single-site frequencies are enough to distinguish distant populations |
| Fine-scale (A = 8–12, low F_ST) | CNN or CNN-CRF | CNN learns multi-site LD patterns that single-site HMM emissions miss |
| Incremental cohort (new samples added) | CNN | Trained CNN runs inference on new samples without retraining |
| Small cohort (< 1K samples) | HMM | CNN self-training needs enough data to learn robust features |
| Interpretability required | HMM | HMM parameters have direct biological interpretation |

---

## How it works

### Pipeline overview

The CNN backend shares the spectral initialization and M-step with the HMM
path.  Only the refinement step differs:

```
SEED  →  INIT  →  REFINE  →  DECODE
                     │
              ┌──────┴──────┐
              │             │
           HMM (EM)    CNN (self-training)
```

Both backends consume the same spectral initialization and produce the same
output format: per-site ancestry posteriors of shape (H, T, A).  Everything
downstream of the posteriors — M-step parameter updates, decode, output
writing, panel export — is shared.

### Self-training loop

The CNN has no external labels.  Training uses a self-supervised pseudo-label
refinement strategy:

```
Spectral init → model₀
    ↓
HMM forward-backward (one pass) → pseudo-labels₀
    ↓
Train CNN on pseudo-labels₀ → CNN predictions₁
    ↓
M-step update → model₁, predictions₁ become pseudo-labels₁
    ↓
Train CNN on pseudo-labels₁ → CNN predictions₂ (final)
    ↓
M-step update → model₂ → decode
```

Each round, the pseudo-labels improve because:

- The CNN sees multi-site patterns (LD information) that the single-site
  spectral init could not use.
- The CNN smooths predictions spatially via its convolutional receptive field,
  reducing noise at individual sites.
- The M-step re-estimates allele frequencies from the improved posteriors,
  feeding better input features back to the CNN.

**Confidence weighting** prevents error reinforcement.  The loss at each site
is weighted by `max(pseudo_label)` — high-confidence sites dominate training,
while uncertain sites contribute little.

### Multi-chromosome strategy

1. **Seed chromosome:** Full self-training (spectral init + HMM bootstrap +
   N pseudo-label rounds).
2. **Subsequent chromosomes:** Reuse the trained CNN weights.  Recompute
   chromosome-specific allele frequencies via spectral init, run one
   fine-tuning epoch at reduced learning rate, then inference.

The CNN generalizes across chromosomes because it learns the *relationship*
between individual alleles and population frequencies — not the specific
frequency values.  The frequency channels provide context, but the learned
convolutional filters operate on patterns.

---

## Architecture

### Input representation

Each haplotype is a 1D tensor of shape (T, C_in), where T is the number of
sites and C_in is the number of input channels:

| Channel | Content | Shape per site |
|---------|---------|----------------|
| 0 | Allele value (0 or 1) | scalar |
| 1..A | Population allele frequencies `freq[a, t]` | A values |
| A+1 | Genetic distance to next site (Morgans) | scalar |

With A ancestries, C_in = A + 2.  For A = 8: C_in = 10.

### Backbone: dilated residual CNN

A stack of 1D dilated convolutions with exponentially increasing dilation
rates — the WaveNet / Temporal Convolutional Network (TCN) design:

```
Stem:     conv1d(C_in → 64,  kernel=1)              receptive field: 1
Layer 0:  conv1d(64   → 64,  kernel=3, dilation=1)   receptive field: 3
Layer 1:  conv1d(64   → 64,  kernel=3, dilation=2)   receptive field: 7
Layer 2:  conv1d(64   → 64,  kernel=3, dilation=4)   receptive field: 15
Layer 3:  conv1d(64   → 64,  kernel=3, dilation=8)   receptive field: 31
...
Layer 11: conv1d(64   → 64,  kernel=3, dilation=2048) receptive field: 4095
Head:     conv1d(64   → A,   kernel=1)
```

With 12 layers, the receptive field covers ~4000 sites.  At typical array
SNP density after thinning (~3K sites per chromosome), this spans a
substantial fraction of the chromosome — capturing long-range ancestry tract
structure.

Each residual block applies:

1. Layer normalization over the channel dimension
2. Dilated 1D convolution with SAME padding
3. GELU activation
4. Residual (skip) connection: add the block input to the output

The output head is a pointwise (kernel=1) convolution projecting from 64
channels to A ancestry logits.  Softmax converts logits to posteriors.

**Parameter count:** With 12 layers and 64 channels, the CNN has ~150K
parameters (~600 KB).  This is negligible relative to the data — the model is
intentionally compact to prevent overfitting on pseudo-labels.

### Analogy to semantic segmentation

LAI on a haplotype is **1D semantic segmentation**.  The haplotype is the
"image" (1D, T pixels).  Ancestry labels are the "class labels."  Tract
boundaries are the "edges" between segments.  The dilated CNN architecture
that solved 2D semantic segmentation (DeepLab, U-Net) applies directly to
the 1D case.

---

## CRF output layer

The optional linear-chain Conditional Random Field adds explicit transition
modeling on top of the CNN logits.  The CRF score for a label sequence z
given CNN logits φ is:

```
score(z | φ) = Σ_t φ[t, z_t]  +  Σ_t W[z_{t-1}, z_t]
```

where W is a learnable (A × A) transition matrix.  The partition function is
computed via the forward algorithm — the same `jax.lax.scan` + `logsumexp`
structure as the HMM forward pass, but over A states with learned (not
model-derived) transitions.

**Marginals** for the final posteriors are computed via forward-backward on
the CRF potentials, producing the same (H, T, A) posterior shape as the
plain CNN.

**When to use the CRF:** For cohorts where tract boundaries are subtle
(closely related populations, old admixture with short tracts).  For
well-separated continental ancestries, the CNN alone produces clean tracts and
the CRF adds little.

The CRF adds negligible overhead.  Its forward-backward runs over A states
(not H × A), and A is small (typically 4–12).

---

## Training details

### Loss function

Confidence-weighted cross-entropy (equivalent to KL divergence up to a
constant):

```
L = -Σ_{h,t,a}  w[h,t] · pseudo_label[h,t,a] · log softmax(logits[h,t,a])
```

where w[h,t] = max_a pseudo_label[h,t,a].  High-confidence sites contribute
more; uncertain sites are downweighted to prevent the CNN from memorizing
noise.

When the CRF is enabled, a soft transition penalty is added to encourage
smooth tracts.

### Optimizer

Adam with cosine learning rate decay and linear warmup (100 steps).  Default
base learning rate: 1e-3.

### Gradient checkpointing

For memory efficiency during training, each residual block's activations are
recomputed during backpropagation rather than stored.  This reduces activation
memory from O(n_layers × batch × T × channels) to O(batch × T × channels)
at the cost of one additional forward pass.

---

## GPU execution model

### Full parallelism over both dimensions

Unlike the HMM, the CNN forward pass has **no sequential dependency** across
sites.  All sites and all haplotypes are processed in parallel via standard
batched 1D convolution:

- **Haplotype dimension:** Batch dimension — standard GPU data parallelism.
- **Site dimension:** Spatial dimension of the convolution — all sites
  processed simultaneously per layer.

### Memory analysis

| Component | Size | Notes |
|-----------|------|-------|
| Parameters | ~600 KB | Negligible |
| Activations (inference) | batch × T × 64 × 4 bytes | 512 × 3K × 64 × 4 ≈ 400 MB |
| Activations (training, checkpointed) | batch × T × 64 × 4 bytes | Same as inference (blocks recomputed) |
| Adam state | 2 × 600 KB ≈ 1.2 MB | Negligible |

### Comparison to HMM

| Metric | HMM | CNN |
|--------|-----|-----|
| Site processing | Sequential (scan) | Parallel (conv) |
| Haplotype processing | Parallel (batch) | Parallel (batch) |
| LD awareness | None (single-site) or block patterns | Native (receptive field) |
| Inference reuse | Must re-run forward-backward | Single forward pass on new samples |
| Training cost | None (EM) | Upfront: ~5 epochs over cohort |

---

## Incremental inference

A significant advantage over the HMM: once trained, the CNN processes new
samples without retraining.  If a biobank adds 1000 new samples:

- **HMM path:** Must re-run forward-backward (at minimum on new haplotypes,
  ideally on all for M-step re-estimation).
- **CNN path:** A single forward pass on the new haplotypes.  No retraining
  needed unless the population composition has changed substantially.

---

## Implementation

The CNN backend is implemented as a `popout/cnn/` subpackage:

| File | Contents |
|------|----------|
| `cnn/model.py` | Dilated CNN architecture (`CNNConfig`, `CNNParams`, `conv1d`, `layer_norm`, `cnn_forward`) |
| `cnn/features.py` | Input feature construction (`build_cnn_features`) |
| `cnn/train.py` | Adam optimizer, KL loss, training loop (`train_cnn`) |
| `cnn/crf.py` | Linear-chain CRF (`crf_marginals`, `crf_log_partition`, `crf_log_likelihood`) |
| `cnn/refine.py` | Pipeline entry points (`run_cnn`, `run_cnn_genome`) |

All code is pure JAX — no additional dependencies beyond what popout already
requires.  The CNN subpackage is lazily imported only when `--method cnn` or
`--method cnn-crf` is specified.
