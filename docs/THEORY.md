# Theory of Operation

This document describes the core algorithm behind popout — what it computes,
why the design choices work, and the math that holds it together.  It is not a
usage guide.  If you want to run the tool, see [README.md](README.md).

---

## 1. The problem

**Local ancestry inference (LAI)** asks: for each position along a person's
genome, which ancestral population did that segment come from?

A person whose recent ancestors came from different continental populations
carries a mosaic genome — long tracts of African ancestry interspersed with
European tracts, say.  LAI recovers this mosaic.  It matters for
association studies (admixture mapping), pharmacogenomics, clinical variant
interpretation, and population genetics research.

### The reference panel bottleneck

Classical LAI methods (RFMix, LAMP-LD, FLARE) require a **reference panel**:
a curated set of individuals with known, single-ancestry genomes.  The
algorithm asks "which reference haplotype does this stretch of DNA look most
like?" and infers ancestry from the answer.

This creates a bottleneck:

- Reference panels are expensive to curate and limited in diversity.
- Populations not well-represented in panels get poor inference.
- At biobank scale (500K+ samples) the panel becomes a tiny fraction of
  the data — most of the information is thrown away.

### The insight: the data *is* the reference

With 500K phased genomes, you don't need an external reference.  The cohort
contains enough representatives of each ancestry to estimate population-level
allele frequencies directly.  Those frequencies are all the HMM emission model
needs — individual reference haplotypes are unnecessary.

This is the core idea behind popout: **self-bootstrapping LAI**.

---

## 2. The core reduction: A states, not K states

Most HMM-based LAI methods (including FLARE) define one hidden state per
reference haplotype.  If your reference panel has K = 10,000 haplotypes, the
HMM has 10,000 states.  The forward algorithm is O(K²) per site per query
haplotype.  FLARE mitigates this with composite reference haplotypes, but the
fundamental scaling is still tied to reference panel size.

Popout takes a different approach.  The hidden states are **ancestral
populations**, not individual haplotypes.  If there are A = 6 ancestries, the
HMM has 6 states.  The emission probability at each site is simply the allele
frequency in that population:

```
P(allele = 1 | ancestry = a, site = t) = freq[a, t]
```

This is valid because, with hundreds of thousands of samples, the allele
frequency estimate *is* the sufficient statistic.  An individual reference
haplotype that carries allele 1 at a site where population A has frequency
0.60 tells you less than the frequency itself — it's one draw from the
distribution.  With enough draws, you don't need to store them individually.

**The reduction is from O(K) to O(A).**  In practice K ~ 10,000 and A ~ 4–12,
so this is a 1000× collapse in state space.  It's what makes GPU-parallel
execution over all haplotypes simultaneously feasible.

---

## 3. Algorithm overview

The pipeline has four stages:

```
SEED  →  INIT  →  EM ITERATE  →  DECODE
```

### Stage 0: SEED — spectral initialization

**Goal:** discover how many ancestries exist and assign initial soft labels.

1. **Sub-sample SNPs.**  Randomly select up to 10K sites (full WGS has
   millions; PCA on all of them is unnecessary and expensive).

2. **Patterson normalization.**  Center each site by its mean allele
   frequency, then scale by `1 / sqrt(p(1-p))`.  This is the standard
   preprocessing for population-genetic PCA (Patterson et al. 2006).
   Without it, high-frequency variants dominate the principal components
   and structure is harder to resolve.

3. **Randomized SVD** (Halko, Martinsson & Tropp 2011).  Compute the top
   ~20 principal components of the normalized H × S matrix in O(H · S · k)
   time via random projection + power iteration + QR + small SVD.  This
   avoids the O(min(H,S)²) cost of full SVD, which is intractable at
   biobank scale.

4. **Auto-detect the number of ancestries** via the Marchenko-Pastur law
   (default), recursive hierarchical splitting, or eigenvalue gap heuristic.
   The Marchenko-Pastur method counts singular values exceeding the
   theoretical bulk edge of a random matrix, giving the number of
   significant PCs + 1 ancestries (Patterson, Price & Reich 2006).
   The recursive method starts with all haplotypes in one cluster and
   repeatedly tests for substructure by comparing BIC of a 1-component
   vs 2-component GMM on the cluster's own top PCs.  Clusters that show
   significant bimodality are split; the process continues until no
   cluster can be split or the maximum A (default 20) is reached.
   The eigenvalue gap heuristic is retained as a
   `--ancestry-detection eigenvalue-gap` fallback.

5. **Gaussian Mixture Model** with diagonal covariance, fitted on the top
   (A−1) or 2 PCs (whichever is larger).  Multiple random restarts
   (default 5), each seeded with k-means++ initialization.  The GMM
   returns **soft responsibilities**: an (H, A) matrix where entry (h, a)
   is the probability that haplotype h belongs to ancestry a.

**Why GMM and not k-means?**  K-means assigns hard labels and implicitly
assumes equal cluster sizes.  In admixed cohorts, ancestry proportions are
often highly imbalanced (e.g., 70% European, 15% African, 10% Native
American, 5% East Asian in a Latin American cohort).  GMM's soft assignments
and explicit mixture weights handle this naturally.

### Alternative: recursive K=2 splitting (`--seed-method recursive`)

The flat GMM seed has basin-of-attraction problems.  Small populations
(low cohort fraction or low pairwise F_ST to a neighbour) get absorbed
into nearby larger clusters because the GMM objective rewards explaining
variance, and small clusters contribute little.

Recursive K=2 splitting solves this by attacking the *easiest* split at
each level and re-computing PCA within each subset:

1. **Start** with all H haplotypes in one cluster.

2. **For each cluster** in the BFS queue, decide whether to split:
   - If the cluster is smaller than 2 × `min_cluster_size`, it is a leaf.
   - Otherwise, compute a Patterson-normalised sub-PCA (top 2 PCs) on the
     cluster's own genotypes.  Run a BIC comparison: fit a 1-component and
     a 2-component diagonal GMM on the sub-PCA projection.  If BIC(1) −
     BIC(2) > c × N (where c = `bic_per_sample`, default 0.05, and N is
     the cluster size), the cluster has substructure.  The per-sample
     scaling ensures the threshold grows with N — a constant threshold
     would be under-calibrated at biobank scale because BIC improvements
     scale with cluster size.

3. **Split** by running K=2 popout EM on the cluster's haplotypes (spectral
   init + a few EM iterations using the full HMM forward-backward).
   Collapse the per-site posteriors to per-haplotype mean responsibilities
   and take argmax to get hard 0/1 labels.  These define two children.

4. **Enqueue** both children and continue until the queue is empty,
   `max_depth` is reached, or the total number of leaves plus queued
   clusters reaches `max_leaves`.

5. **Assign** flat integer labels 0..(n_leaves − 1) to leaves in
   finalisation order.  Convert to one-hot (H, K) responsibilities and
   pass to `run_em` as `seed_responsibilities`.  The rest of the
   pipeline is unchanged — `init_model_soft` builds allele frequencies
   from the leaf responsibilities, and EM refines them.

The BFS order ensures that the most informative splits (at the top of the
tree, where BIC delta is largest) happen before the `max_leaves` cap
cuts off further recursion.

**Post-hoc Hellinger merge** (`--recursive-merge-hellinger`, default 0.04):
after recursion finishes, pairwise Hellinger distances are computed between
leaves' allele-frequency profiles.  Leaf pairs with distance below the
threshold are iteratively merged (closest pair first).  This catches
noise splits that even a well-calibrated BIC misses — within-ancestry
sub-structure (LD, relatedness) can produce BIC deltas comparable to
real inter-ancestry splits, but the resulting leaves will have nearly
identical allele frequencies and thus low Hellinger distance.  Continental
F_ST is typically 0.05-0.15 Hellinger; sub-continental 0.02-0.05; noise
splits < 0.01.  The default threshold of 0.04 is conservative enough to
preserve real sub-continental structure.

**Anchor freezing** (`--freeze-anchors-iters N`): for the first N EM
iterations, the M-step overrides the E-step-derived allele frequencies
with frequencies computed from the frozen seed responsibilities.  This
gives small clusters time to establish their identity before competing
with larger neighbours.  Default is 0 (no freezing).

### Stage 1: INIT — build the initial model

Convert soft GMM responsibilities into a working `AncestryModel`:

1. **Allele frequencies via weighted GEMM.**  For each ancestry a and site t:

   ```
   freq[a, t] = (Σ_h resp[h,a] · geno[h,t] + 0.5) / (Σ_h resp[h,a] + 1.0)
   ```

   The pseudocount (0.5 / 1.0, a Beta(0.5, 0.5) prior) prevents zero
   frequencies that would make log-likelihoods degenerate.  The entire
   computation is a single matrix multiply: `resp.T @ geno → (A, T)`.

2. **Window-based refinement.**  The GMM assigns one ancestry label per
   haplotype *globally*.  But admixed haplotypes have different ancestry
   at different positions.  Window refinement fixes this: for each block
   of 50 SNPs, compute per-haplotype log-likelihood under each ancestry
   using the global frequencies, softmax to get local responsibilities,
   then recompute frequencies from local assignments.  After this step,
   allele frequencies reflect the local ancestry structure, not just the
   global average.

3. **Initialize mu and T.**  Global ancestry proportions mu are the mean
   of the soft responsibilities.  Generations since admixture T is
   initialized to 20 (a reasonable default for recent admixture).

### Stages 2–3: EM ITERATE

Expectation-Maximization alternates between:

- **E-step:** Run the forward-backward HMM (details in §4) to compute
  posterior ancestry probabilities γ[h, t, a] at every haplotype and site.

- **M-step:** Update model parameters from the posteriors (details in §5).

Convergence is checked by the maximum absolute change in allele frequencies.
With 500K+ samples, the sufficient statistics are so well-determined that
**2–3 iterations typically suffice** and max(Δfreq) drops below 1e-4.

T (generations since admixture) is held fixed during the first iteration to
let frequencies stabilize before the switch-rate estimator kicks in.

### Stage 4: DECODE

One final forward-backward pass with the converged model.  Hard ancestry
calls are the argmax of the posteriors.  Per-sample global ancestry
proportions are the mean of the posteriors across all sites.

---

## 4. The HMM

### States and emissions

The hidden state at each site is one of A ancestral populations.  Two
emission models are available:

**Single-site Bernoulli (default).**  Each site is treated independently:

```
P(g[h,t] | z[h,t] = a) = freq[a,t]^g[h,t] · (1 - freq[a,t])^(1 - g[h,t])
```

where g[h,t] ∈ {0, 1} is the observed allele and z[h,t] is the hidden
ancestry.  This is computed for all H haplotypes, T sites, and A ancestries
simultaneously: an (H, T, A) tensor of log-emissions.

**Haplotype-window block emissions (`--block-emissions`).**  Sites are
grouped into blocks of k SNPs (default k = 8).  Within each block, the k
binary alleles are packed into a pattern index.  For each block b and
ancestry a, a categorical frequency table records how often each observed
pattern appears:

```
P(pattern p | z = a, block b) = pattern_freq[b, p, a]
```

Pattern frequencies are initialized from per-site Bernoulli probabilities
(product over sites in the block) and refined during the M-step by
accumulating posterior weight per pattern.  The HMM scan iterates over
blocks instead of sites, with block-level genetic distances for
transitions.  This captures linkage disequilibrium information that the
single-site model discards, substantially improving resolution for closely
related ancestries (e.g., Northern vs Southern European, Han Chinese vs
Japanese) where F_ST is small but multi-site haplotype patterns differ.
Because the algorithm is memory-bandwidth bound (§8), the richer emissions
add negligible compute cost while reducing the number of scan steps by a
factor of k.

### Transitions

The transition model follows Li & Stephens (2003), parameterized by genetic
distance and admixture time:

```
P(z[t] = j | z[t−1] = i) = (1 − p) · δ(i,j) + p · μ[j]
```

where:

- **p = 1 − exp(−d · T)** is the probability that at least one recombination
  event occurred in the interval of genetic distance d (in Morgans), given T
  generations since admixture.  This is a Poisson process: in T meioses, the
  expected number of crossovers in an interval of d Morgans is d·T.
- **μ[j]** is the global proportion of ancestry j.  When a recombination
  happens, the new ancestry is drawn from the population-level mixture.
- **δ(i,j)** is the Kronecker delta — probability 1 of staying in the same
  state if no recombination occurs.

In matrix form, the transition matrix at each interval is:

```
Trans = (1 − p) · I  +  p · 1·μᵀ
```

where I is the identity and 1·μᵀ is the rank-1 matrix with μ in every row.
In log-space, the diagonal requires `logaddexp` to combine the two terms;
off-diagonal entries are simply `log(p) + log(μ[j])`.

Key property: the transition matrix is **site-dependent** (through the genetic
distance d).  In the default mode it is also **haplotype-independent** — every
haplotype shares the same (T−1) transition matrices, computed once and
broadcast.

**Per-haplotype admixture time (`--per-hap-T`).**  Real biobank cohorts
contain individuals with admixture at different historical depths.  A Latin
American cohort includes individuals with 20-generation-old admixture
alongside individuals with a grandparent from a different population (T ≈ 2).
A single T produces a compromise that misestimates tract lengths for most
individuals.

When enabled, the M-step estimates a per-haplotype T_h from each haplotype's
own **density-invariant expected switch count** — the sum of P(z_t ≠ z_{t+1} |
data, h) at every interval, derived from xi posteriors (see §5 below):

```
T_h = E[switches_h] / (genetic_distance · (1 − Σ μ²))
```

T_h is regularized by blending with the global estimate: `T_final = (1−λ)·T_h
+ λ·T_global`, where λ = 1/(1 + switches/5) so that haplotypes with few
switches (low confidence) are pulled toward the global mean.

To recover GPU efficiency, T_h values are quantized into B = 20
geometrically spaced buckets (from T = 1 to T = 1000).  B transition
matrices are precomputed (one per bucket), and haplotypes are partitioned
by bucket for independent forward-backward passes.  The memory overhead is
B × (T−1) × A × A × 4 bytes — negligible.  Emissions are shared across
buckets, computed once.

Per-hap-T is supported under both single-site Bernoulli emissions and
`--block-emissions`. Under block emissions the same xi-diagonal trick is
applied at block boundaries; the M-step input is density-invariant.

**Caveat: seed-chromosome freezing.** The per-haplotype T estimate is
computed once on the seed chromosome and frozen for the rest of the
genome. Quality is therefore bounded by the seed chromosome's switch-count
statistics; using a small chromosome (e.g., chr22) as the seed gives noisy
per-hap T that propagates genome-wide. For best results, pick a large
chromosome (chr1 or chr2) as the seed. The per-haplotype T histogram
(`popout viz per_hap_t`) shows the seed-chromosome estimate.

### Forward algorithm

The forward variable α[h, t, a] = P(g[h,1..t], z[h,t] = a) is computed
sequentially across sites, in log-space for numerical stability:

```
log α[h, t, j] = log_emit[h, t, j] + logsumexp_i( log α[h, t−1, i] + log Trans[t−1, i, j] )
```

The inner logsumexp is a batched log-space matrix-vector product.  At each
step, the live data is the (H, A) forward state — for 1M haplotypes and 8
ancestries, that's 32 MB of float32.  The entire forward pass is a
`jax.lax.scan` loop over T steps.

### Backward algorithm

Same structure in reverse:

```
log β[h, t, i] = logsumexp_j( log Trans[t, i, j] + log_emit[h, t+1, j] + log β[h, t+1, j] )
```

initialized with log β[h, T, :] = 0 (i.e., β = 1).

### Posteriors

```
γ[h, t, a] = exp( log α[h,t,a] + log β[h,t,a] − log Z[h,t] )
```

where log Z is the per-(haplotype, site) normalizer computed via logsumexp
over ancestries.  γ[h, t, a] is the posterior probability that haplotype h
is ancestry a at site t, given all observed data on that haplotype.

---

## 5. The M-step

### Allele frequencies

Weighted average of observed alleles under the posteriors:

```
freq[a, t] = ( Σ_h γ[h,t,a] · g[h,t] + 0.5 ) / ( Σ_h γ[h,t,a] + 1.0 )
```

This is a single `einsum('hta,ht->at', gamma, geno)` — a batched GEMM
contracting over the H (haplotype) dimension.  The pseudocount provides
Beta-Bernoulli smoothing to prevent degenerate frequencies.

### Ancestry proportions

```
μ[a] = mean over h,t of γ[h,t,a]
```

then normalized to sum to 1.

### Generations since admixture

Estimated from the hard-call switch rate.  Hard calls are the argmax of γ.
A "switch" is any pair of adjacent sites where the hard call changes
ancestry.  Under the model:

```
E[switches at interval] = P(recombination) · P(new ancestry ≠ old ancestry)
                        = (1 − exp(−d·T)) · (1 − Σ_a μ[a]²)
```

The correction factor (1 − Σ μ²) accounts for the possibility that a
recombination event re-samples the *same* ancestry and produces no visible
switch.  Summing over all intervals and solving for T:

```
T_est = total_switches / ( total_genetic_distance · (1 − Σ μ²) )
```

This is regularized by blending with the previous estimate (70% new, 30%
old) and clamped to [1, 1000] generations.  Using hard calls rather than
soft overlaps for switch counting is more robust when posteriors are diffuse.

When `--per-hap-T` is enabled, the per-haplotype variant uses xi-based
**soft** switch counts (expected number of transitions per haplotype)
rather than hard-call switches; see §4 Transitions above. This is what
makes the estimator density-invariant: doubling the SNP density at fixed
genetic length leaves the expected switch count per haplotype roughly
unchanged.

---

## 6. Spectral initialization: the math

### Patterson normalization

Given an H × S genotype matrix X with entries in {0, 1}:

```
p[s] = mean(X[:, s])                  # allele frequency at site s
X_norm[:, s] = (X[:, s] − p[s]) / sqrt(p[s] · (1 − p[s]))
```

This standardization ensures that the covariance matrix of X_norm is
proportional to the genetic relatedness matrix (GRM), where each site
contributes equally regardless of allele frequency.  Without it, common
variants overwhelm the PCA.

Reference: Patterson N, Price AL, Reich D. *Population structure and
eigenanalysis.* PLoS Genetics 2006.

### Randomized SVD

Computing the full SVD of an H × S matrix (e.g., 1M × 10K) costs
O(min(H,S)² · max(H,S)).  The randomized algorithm (Halko et al. 2011)
reduces this to O(H · S · k) where k ≪ min(H, S):

1. **Random projection:** Draw Ω ~ N(0,1) of shape (S, k) where k =
   n_components + n_oversamples.  Compute Y = X · Ω.  This projects S
   dimensions down to k, preserving the dominant singular subspace.

2. **Power iteration** (2 rounds default): Y ← X · (Xᵀ · Y).  This
   amplifies the gap between signal and noise singular values, improving
   the approximation quality.  Each round costs 2 matrix multiplies.

3. **QR factorization:** Q, R = QR(Y).  Q is an orthonormal basis for
   the column space of Y (and approximately for the dominant left
   singular subspace of X).

4. **Project and decompose:** B = Qᵀ · X is a small (k × S) matrix.
   Compute its exact SVD: B = Û · S · Vᵀ.  Then U = Q · Û gives the
   approximate left singular vectors of X.

The top n_components singular vectors and values are returned.  With k = 30
(20 components + 10 oversamples), this is fast even for millions of
haplotypes.

### GMM fitting

The diagonal-covariance GMM is fitted on the PCA projection (H points in
n_pc dimensions) via standard EM:

**E-step:** For each haplotype h and cluster c:

```
log P(h ∈ c) = log π[c] − ½ Σ_d [ log σ²[c,d] + (x[h,d] − μ[c,d])² / σ²[c,d] ] + const
resp[h, c] = softmax over c
```

**M-step:**

```
N[c] = Σ_h resp[h,c]
π[c] = N[c] / H
μ[c] = (resp[:, c]ᵀ · X) / N[c]
σ²[c, d] = Σ_h resp[h,c] · (x[h,d] − μ[c,d])² / N[c]
```

Diagonal covariance (σ² per dimension rather than a full d×d matrix) keeps
parameters at O(k·d) instead of O(k·d²) and is justified because PCA
coordinates are uncorrelated by construction.

Multiple restarts with k-means++ seeding ensure the GMM doesn't settle in a
poor local optimum.  The run with the highest log-likelihood wins.

---

## 7. Why it works at biobank scale

### Allele frequency convergence

The key statistical fact: with H haplotypes, the standard error of an allele
frequency estimate is:

```
SE(freq) = sqrt( p(1−p) / H )
```

For H = 1,000,000 (500K diploid samples) and a common variant (p = 0.3):

```
SE ≈ sqrt(0.21 / 1e6) ≈ 0.00046
```

This means allele frequencies are determined to **four decimal places** from
the data alone.  The emission model — which is just `P(allele | freq)` — is
essentially exact.  There is no meaningful loss from using population
frequencies instead of individual reference haplotypes.

At smaller sample sizes (e.g., 1K samples), the frequencies are noisier
(SE ≈ 0.014), but still sufficient for distinguishing continental-level
ancestries where allele frequency differences (F_ST ~ 0.05–0.15) far exceed
the estimation noise.

### EM convergence speed

With well-estimated frequencies, the E-step posteriors are sharp (low
entropy) from the first iteration.  The M-step re-estimates are therefore
close to the truth immediately.  This is why 2–3 iterations suffice at
biobank scale — the signal-to-noise ratio is so high that the algorithm
barely needs to iterate.

At smaller sample sizes, more iterations may be needed (the default is 5),
and the convergence threshold (max Δfreq < 1e-4) acts as a safety net.

### Self-consistency

The model is self-consistent: the allele frequencies that define the emission
model are derived from the same data that the HMM is run on.  This circular
dependency is resolved by EM — each iteration uses the *current* model to
compute posteriors, then re-estimates the model from those posteriors.  The
fixed point is the maximum-likelihood solution.

At biobank scale, the initial spectral seed is good enough that EM converges
to the global optimum.  At smaller scales, the spectral initialization
quality matters more, which is why the GMM uses multiple restarts and the
init stage includes window-based refinement.

---

## 8. GPU execution model

### Parallel over haplotypes, sequential over sites

The forward algorithm has a **data dependency across sites** (each step
depends on the previous forward state).  It is inherently sequential in the
site dimension.  But **haplotypes are independent** — each haplotype runs its
own HMM with its own observations, sharing only the transition matrices and
allele frequencies.

This is a perfect fit for GPU parallelism: the site loop is the outer
sequential loop, and at each step, all H haplotypes are processed in a single
batched operation.

### Memory analysis

The live state at each forward step is the (H, A) forward vector:

```
1M haplotypes × 8 ancestries × 4 bytes = 32 MB
```

This fits comfortably in an A100's L2 cache (40 MB).  The transition matrix
is A × A = 64 floats — negligible.  The emission at the current site is
(H, A) = another 32 MB.  Total working set per step: ~96 MB.

The full forward state (H, T, A) stored for the backward pass is larger:

```
1M × 100K sites × 8 ancestries × 4 bytes ≈ 3.2 TB
```

This clearly doesn't fit.  In practice, `--thin-cm 0.02` reduces T to a few
thousand sites (array-like density), bringing storage to:

```
1M × 3K × 8 × 4 bytes ≈ 96 GB
```

This does **not** fit on any single GPU (96 GB exceeds even the 80 GB A100).
However, the HMM is independent across haplotypes, so the forward-backward
pass is batched over H.  With 50K haplotypes per batch, each batch is ~5 GB.

**Streaming M-step.**  The EM loop's M-step consumers are all reductions over
the haplotype dimension:

- `update_allele_freq`: einsum over H → (A, T)
- `update_mu`: mean over H and T → (A,)
- `update_generations`: argmax + switch count → scalar

None require the full (H, T, A) posterior simultaneously.  The implementation
accumulates these sufficient statistics inside the batching loop — gamma is
computed per batch, reduced, and freed.  Peak GPU memory during the E-step
is therefore O(batch\_size · T · A), independent of total H:

```
batch_size=50K,  T=3K, A=8:   50K × 3K × 8 × 4 ≈  4.8 GB per batch
batch_size=25K,  T=14K, A=6:  25K × 14K × 6 × 4 ≈  8.4 GB per batch
```

The final decode pass uses the same streaming pattern, computing hard calls
(argmax), per-site max posteriors, and per-haplotype global ancestry sums
per batch — writing results to CPU numpy arrays.

**Gradient checkpointing** reduces the *within-batch* forward-state storage
from O(batch\_size · T · A) to O(batch\_size · √T · A).  Instead of storing
every forward state for the backward pass, the implementation stores only √T
checkpoint alphas — one at every C-th site, where C = ⌈√T⌉.  During the
backward pass, forward states within each segment are recomputed from the
nearest checkpoint on the fly, fused with the backward computation and
posterior normalization.

The algorithm uses a two-level `jax.lax.scan`:

1. **Checkpointed forward scan.**  An outer scan over S = ⌈T/C⌉ segments,
   each containing an inner scan of C forward steps.  The inner scan discards
   intermediate states — only the segment-end alpha is carried as output.
   Sites are padded to a multiple of C with neutral (zero) emissions and
   identity transitions.

2. **Reverse recompute-backward-posterior scan.**  The S segments are
   processed in reverse order.  For each segment, a C−1-step inner forward
   scan recomputes alphas from the checkpoint, a C-step inner backward scan
   computes betas from the right boundary (carried from the previous
   segment), and posteriors are computed elementwise.

The result is numerically equivalent to the full forward-backward.  For
T = 3000 thinned sites, C ≈ 55, and checkpoint storage per batch is:

```
50K × 55 checkpoints × 8 ancestries × 4 bytes ≈ 88 MB
```

The cost is ~2× forward compute (one full pass + one segment recompute),
which is acceptable since the forward pass is memory-bandwidth bound —
recomputed segments often hit L2 cache rather than HBM.

Checkpointing is enabled automatically for T > 64 and can be forced on/off
via the `use_checkpointing` parameter.

**Peak GPU memory budget** (1M haplotypes, 3K thinned sites, A=8,
batch\_size=50K on A100 40 GB):

| Tensor                        | Size    | Location          |
|-------------------------------|---------|-------------------|
| geno (H, T) uint8            | 3 GB    | GPU               |
| gamma per batch (50K, T, A)  | 4.8 GB  | GPU (freed/batch) |
| M-step accumulators (A, T)×3 | ~1 MB   | GPU               |
| calls (H, T) int8            | 3 GB    | CPU               |
| max\_post (H, T) float32     | 12 GB   | CPU (optional)    |

Peak GPU ≈ 3 GB (geno) + 5 GB (one batch) ≈ **8 GB**. Fits comfortably.

### Arithmetic intensity

The dominant operation per step is the log-space matrix-vector product:

```
result[h, j] = logsumexp_i( α[h,i] + Trans[i,j] )
```

This is A additions and a logsumexp (involving exp, max, log) over A values,
for each of H haplotypes and A output states.  Total arithmetic per step:
O(H · A²).  Total memory traffic per step: O(H · A) reads + O(H · A)
writes.

Arithmetic intensity = A² / A = A operations per byte moved.  For A = 8:
**8 flops per byte**.  An A100 has ~2 TB/s bandwidth and ~20 TFLOPS (FP32),
so the compute-to-bandwidth ratio the hardware wants is ~10 flops/byte.
We're at 8 — **memory-bandwidth bound**, with ALU utilization under 1% in
practice because the logsumexp operations are not pure multiply-add.

This is actually good news: it means the emission model could be made
substantially richer (e.g., haplotype-window pattern matching, multi-site
emissions) without hitting a compute wall.  The bottleneck is moving the
forward state through memory, not computing on it.

### JAX implementation

The forward and backward passes are implemented as `jax.lax.scan` loops —
JAX's primitive for sequential computation with carried state.  At each step,
the scan body performs:

1. Broadcast-add of (H, A, 1) forward state with (1, A, A) transition matrix
   → (H, A, A)
2. `logsumexp` along the "from" axis → (H, A) predicted state
3. Element-wise add of (H, A) emission → (H, A) updated state

JAX traces this computation graph once, then executes it as a fused XLA
kernel.  The haplotype dimension is the natural batch dimension — no explicit
CUDA kernels are written.

---

## 9. Simulation model

The simulator (`simulate.py`) generates ground-truth admixed data using the
same generative model the inference algorithm assumes:

1. **Population-specific allele frequencies** are drawn from a Balding-Nichols
   model: start with a global frequency from Beta(0.5, 0.5) (the U-shaped
   spectrum typical of real allele frequencies), then drift each population
   using a Beta distribution parameterized by F_ST.

2. **Ancestry tracts** are generated by the Markov chain with transition
   probability `1 − exp(−d·T)` and mixture weights μ, drawn from a symmetric
   Dirichlet.

3. **Alleles** are emitted independently from the ancestry-specific frequency
   at each site.

### Cohort composition: pure vs. admixed haplotypes

Real biobank cohorts contain a mix of ancestrally pure individuals
(first-generation immigrants, single-continental-origin samples) and
admixed individuals.  The pure individuals form dense corners in PCA space
that give the GMM spectral initialization reliable purchase on the true
population structure.

The simulator's `pure_fraction` parameter (default 0.3) controls this mix.
Pure haplotypes are assigned a single ancestry across all sites — no Markov
transitions.  They are distributed across ancestries proportionally to μ.
The remaining haplotypes get the standard admixed treatment.

**Two simulation regimes for validation:**

- **Biobank-like (`pure_fraction=0.3`):** Models cohorts like AoU where
  >20% of samples have recent single-continental-origin ancestry.  The
  algorithm reaches near-oracle accuracy (gap < 1pp at 500 samples).

- **Fully-admixed stress test (`pure_fraction=0.0`):** Every haplotype is
  a mosaic.  The PCA projection has no dense corners, and GMM initialization
  fails to recover the true population structure.  Accuracy is limited by
  spectral init quality (gap 20–50pp).  This regime does not correspond to
  any known human biobank but is useful for identifying algorithm
  limitations.  Closing the gap in this regime requires a corner-finding
  init (NMF, archetypal analysis, or SPA) instead of GMM.

### Oracle benchmark

The demo (`python -m popout.demo`) reports both inferred and oracle
accuracy.  The oracle constructs an `AncestryModel` from the true
generative parameters and decodes with `forward_backward_decode`.
Oracle accuracy is the Bayes-optimal ceiling for the given F_ST and
tract length — the best any method could achieve with perfect parameters.

This allows closed-loop validation: run the inference pipeline on simulated
data and compare inferred calls to ground truth.  Because inferred ancestry
labels may be permuted relative to truth, the evaluator tries all
permutations of A labels and reports accuracy under the best match.

---

## 10. Multi-chromosome strategy

Genome-wide parameters (μ, T) are shared across chromosomes, but allele
frequencies are chromosome-specific (different genes, different population
differentiation patterns).  The pipeline exploits this:

1. **Seed chromosome:** Run full EM (spectral init + 5 iterations) on the
   first chromosome.  This estimates μ, T, and the number of ancestries.

2. **Remaining chromosomes:** Warm-start with the fitted μ and T.  Compute
   chromosome-specific allele frequencies from a quick spectral init + soft
   model initialization, then run **one** EM iteration to refine.  This is
   sufficient because μ and T are already known — only the per-site
   frequencies need local adaptation.

This amortizes the expensive spectral + multi-iteration EM cost across
all chromosomes.

---

## 11. What makes this different

| | FLARE | RFMix | popout |
|---|---|---|---|
| **Reference panel** | Required | Required | None (self-bootstrapped) |
| **HMM states** | K composite ref haplotypes | N/A (random forest) | A ancestral populations |
| **State space** | O(K), K ~ 1000s | — | O(A), A ~ 4–12 |
| **Emission model** | Haplotype matching | Window features | Single-site or block-level pattern matching |
| **GPU acceleration** | No | No | Native (JAX), with gradient checkpointing |
| **Scales to 500K+** | With effort | No | Yes (designed for it) |
| **Ancestry count** | User-specified | User-specified | Auto-detected (Marchenko-Pastur, recursive, or eigenvalue gap) |
| **Admixture time** | Estimated (global) | Not modeled | Estimated (global or per-haplotype) |

### The A-state tradeoff

Using A states instead of K has a real cost: the emission model is less
powerful.  A K-state HMM can distinguish ancestry based on multi-site
haplotype *patterns* (linkage disequilibrium), while an A-state model with
single-site emissions treats each site independently.  This limits resolution
for closely related ancestries (e.g., Han Chinese vs. Japanese, where F_ST is
very small and the distinctive signal is in LD patterns, not single-site
frequencies).

The mitigation is **haplotype-window block emissions** (`--block-emissions`):
instead of emitting one allele at a time, sites are grouped into blocks of k
SNPs (default 8) and the emission is a categorical distribution over observed
haplotype patterns within each block.  This recovers LD information while
keeping the A-state structure.  The HMM scan iterates over blocks (T/k steps)
instead of individual sites (T steps), making it both richer and faster.
Because the algorithm is memory-bandwidth bound (not compute bound), the
richer per-step emission computation adds negligible cost.  See §4 for
details.

---

## 12. Helpful reading

### Foundational

- **Li N, Stephens M.** *Modeling linkage disequilibrium and identifying
  recombination hotspots using single-nucleotide polymorphism data.*
  Genetics, 2003.
  — The Li-Stephens model: the HMM framework that underlies most modern LAI
  methods.  Popout's transition model is a direct descendant.

- **Patterson N, Price AL, Reich D.** *Population structure and
  eigenanalysis.* PLoS Genetics, 2006.
  — Introduces the normalization `(x − p) / sqrt(p(1−p))` for
  population-genetic PCA.  This is what makes the spectral initialization
  work.

### LAI methods

- **Browning SR, Waples RK, Browning BL.** *Fast, accurate local ancestry
  inference with FLARE.* American Journal of Human Genetics, 2023.
  — FLARE showed that composite reference haplotypes reduce effective K
  without losing much accuracy.  Popout takes this further: if composite
  haplotypes converge to allele frequencies as K grows, just use the
  frequencies directly.

- **Maples BK, Gravel S, Kenny EE, Bustamante CD.** *RFMix: a
  discriminative modeling approach for rapid and robust local ancestry
  inference.* American Journal of Human Genetics, 2013.
  — The random-forest approach to LAI.  Complementary to HMM methods;
  RFMix uses conditional random fields for smoothing.

- **Pasaniuc B, Sankararaman S, Kimmel G, Halperin E.** *Inference of
  locus-specific ancestry in closely related populations.* Bioinformatics,
  2009. (LAMP-LD)
  — Window-based LAI using a probabilistic model.  Popout's window-based
  refinement during initialization echoes this idea.

### Numerical methods

- **Halko N, Martinsson PG, Tropp JA.** *Finding structure with randomness:
  probabilistic algorithms for constructing approximate matrix
  decompositions.* SIAM Review, 2011.
  — The randomized SVD algorithm used for spectral initialization.
  Sections 1–5 cover the theory; Algorithm 4.4 is what's implemented.

- **Rabiner LR.** *A tutorial on hidden Markov models and selected
  applications in speech recognition.* Proceedings of the IEEE, 1989.
  — The classic forward-backward tutorial.  Popout's HMM follows this
  exactly, in log-space.

### Population genetics background

- **Balding DJ, Nichols RA.** *A method for quantifying differentiation
  between populations at multi-allelic loci and its implications for
  investigating identity and paternity.* Genetica, 1995.
  — The Balding-Nichols model used by the simulator to generate
  population-specific allele frequencies with controlled F_ST.

- **Price AL, Patterson NJ, Plenge RM, Weinblatt ME, Shadick NA, Reich D.**
  *Principal components analysis corrects for stratification in genome-wide
  association studies.* Nature Genetics, 2006.
  — Practical application of PCA to population structure.  Relevant context
  for why spectral methods work as initialization for ancestry inference.

### GPU / systems

- **Bradbury J, Frostig R, Hawkins P, et al.** *JAX: composable
  transformations of Python+NumPy programs.* 2018.
  — The framework.  `jax.lax.scan` is the key primitive for the sequential
  forward-backward loop.  JAX's XLA backend fuses the per-step operations
  into efficient GPU kernels without hand-written CUDA.
