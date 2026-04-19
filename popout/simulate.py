"""Simulate admixed haplotype data for testing.

Generates synthetic data with known ancestry labels so you can
verify the pipeline recovers them.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

from .datatypes import ChromData

log = logging.getLogger(__name__)


def simulate_admixed(
    n_samples: int = 1000,
    n_sites: int = 5000,
    n_ancestries: int = 4,
    gen_since_admix: int = 20,
    chrom_length_cm: float = 100.0,
    fst_range: tuple[float, float] = (0.05, 0.15),
    pure_fraction: float = 0.3,
    rng_seed: int = 42,
) -> tuple[ChromData, np.ndarray, dict]:
    """Generate simulated admixed haplotype data.

    Parameters
    ----------
    n_samples : number of diploid samples (will produce 2× haplotypes)
    n_sites : number of SNP sites
    n_ancestries : A
    gen_since_admix : T
    chrom_length_cm : total genetic length in cM
    fst_range : range of Fst between populations (controls divergence)
    pure_fraction : fraction of haplotypes that are single-ancestry
        (no recombination). Models real cohorts where some individuals
        have recent single-continental-origin ancestry. Set to 0 for
        a fully-admixed stress test.
    rng_seed : random seed

    Returns
    -------
    chrom_data : ChromData with simulated genotypes
    true_ancestry : array (n_haps, n_sites) — ground truth ancestry labels
    true_params : dict with keys 'pop_freq' (A, n_sites), 'mu' (A,),
        'gen_since_admix' (int) — generative parameters for oracle comparison
    """
    rng = np.random.default_rng(rng_seed)
    A = n_ancestries
    T = gen_since_admix
    n_haps = 2 * n_samples
    H = n_haps

    log.info("Simulating %d samples (%d haplotypes), %d sites, %d ancestries, "
             "%.0f%% pure",
             n_samples, n_haps, n_sites, A, 100 * pure_fraction)

    # --- Ancestry-specific allele frequencies ---
    # Start with a global frequency, then drift each population
    global_freq = rng.beta(0.5, 0.5, size=n_sites)  # U-shaped MAF spectrum
    # Per-ancestry frequencies: add population-specific drift
    pop_freq = np.zeros((A, n_sites))
    for a in range(A):
        fst = rng.uniform(*fst_range)
        # Beta-binomial drift: parameterize beta by freq and fst
        alpha = global_freq * (1 - fst) / fst
        beta = (1 - global_freq) * (1 - fst) / fst
        alpha = np.maximum(alpha, 0.01)
        beta = np.maximum(beta, 0.01)
        pop_freq[a] = rng.beta(alpha, beta)

    # --- Ancestry proportions ---
    mu = rng.dirichlet(np.ones(A) * 2)
    log.info("  True mu: %s", mu.round(3))
    log.info("  True T: %d generations", T)

    # --- Genetic positions ---
    pos_cm = np.linspace(0, chrom_length_cm, n_sites)
    pos_bp = (pos_cm * 1e6 / chrom_length_cm * 1e8 / 100).astype(np.int64)
    d_morgan = np.diff(pos_cm) / 100.0

    # --- Partition haplotypes into pure and admixed ---
    n_pure = int(H * pure_fraction)
    n_admixed = H - n_pure

    # Assign pure haplotypes to ancestries proportionally to mu
    pure_counts = np.zeros(A, dtype=int)
    if n_pure > 0:
        pure_counts = rng.multinomial(n_pure, mu)
        # Ensure at least 1 per ancestry if n_pure >= A
        if n_pure >= A:
            deficit = (pure_counts == 0)
            while deficit.any():
                donor = pure_counts.argmax()
                recip = np.where(deficit)[0][0]
                pure_counts[donor] -= 1
                pure_counts[recip] += 1
                deficit = (pure_counts == 0)

    # --- Simulate ancestry tracts and genotypes ---
    geno = np.zeros((H, n_sites), dtype=np.uint8)
    true_ancestry = np.zeros((H, n_sites), dtype=np.int8)

    h = 0
    # Pure-ancestry haplotypes: single ancestry, no transitions
    for a in range(A):
        for _ in range(pure_counts[a]):
            true_ancestry[h, :] = a
            geno[h] = (rng.random(n_sites) < pop_freq[a]).astype(np.uint8)
            h += 1

    # Admixed haplotypes: Markov chain with recombination
    for _ in range(n_admixed):
        anc = rng.choice(A, p=mu)
        for t in range(n_sites):
            if t > 0:
                p_switch = 1.0 - np.exp(-d_morgan[t - 1] * T)
                if rng.random() < p_switch:
                    anc = rng.choice(A, p=mu)
            true_ancestry[h, t] = anc
            geno[h, t] = rng.random() < pop_freq[anc, t]
        h += 1

    # Shuffle so pure and admixed haplotypes are interleaved
    if n_pure > 0:
        perm = rng.permutation(H)
        geno = geno[perm]
        true_ancestry = true_ancestry[perm]

    chrom_data = ChromData(
        geno=geno,
        pos_bp=pos_bp,
        pos_cm=pos_cm,
        chrom="sim",
    )

    # Report true ancestry proportions
    for a in range(A):
        prop = (true_ancestry == a).mean()
        log.info("  True ancestry %d: %.1f%%", a, 100 * prop)
    if n_pure > 0:
        log.info("  Pure haplotypes: %s (total %d)", dict(enumerate(pure_counts.tolist())), n_pure)

    true_params = {
        "pop_freq": pop_freq,
        "mu": mu,
        "gen_since_admix": T,
    }
    return chrom_data, true_ancestry, true_params


def evaluate_accuracy(
    calls: np.ndarray,
    true_ancestry: np.ndarray,
    n_ancestries: int,
) -> dict:
    """Evaluate ancestry inference accuracy.

    Since inferred labels may be permuted relative to true labels,
    finds the best permutation match.  Handles the case where
    n_ancestries (inferred K) differs from the true K.

    Parameters
    ----------
    calls : (H, T) inferred ancestry labels
    true_ancestry : (H, T) true ancestry labels
    n_ancestries : A — number of inferred ancestries

    Returns
    -------
    dict with accuracy metrics
    """
    from itertools import permutations

    A_inferred = n_ancestries
    A_true = int(true_ancestry.max()) + 1

    # Build confusion matrix: (A_inferred, A_true)
    confusion = np.zeros((A_inferred, A_true), dtype=np.int64)
    for i in range(A_inferred):
        mask = calls == i
        if mask.any():
            for j in range(A_true):
                confusion[i, j] = int(((calls == i) & (true_ancestry == j)).sum())

    best_acc = 0.0
    best_perm = None

    if A_inferred == A_true and A_inferred <= 8:
        # Exact: try all permutations
        for perm in permutations(range(A_inferred)):
            remapped = np.zeros_like(calls)
            for i, j in enumerate(perm):
                remapped[calls == i] = j
            acc = (remapped == true_ancestry).mean()
            if acc > best_acc:
                best_acc = acc
                best_perm = perm
    else:
        # Greedy assignment: map each inferred cluster to its best true ancestry
        # This handles both A_inferred != A_true and large A
        assigned_true = set()
        mapping = {}
        # Sort inferred clusters by size (largest first) for greedy stability
        cluster_sizes = [(confusion[i].sum(), i) for i in range(A_inferred)]
        cluster_sizes.sort(reverse=True)

        for _, i in cluster_sizes:
            # Find the true ancestry with highest overlap not yet assigned
            scores = confusion[i].copy().astype(float)
            # Prefer unassigned, but allow duplicates if needed
            for j in assigned_true:
                scores[j] *= 0.5  # penalise but don't prohibit
            best_j = int(scores.argmax())
            mapping[i] = best_j
            assigned_true.add(best_j)

        remapped = np.zeros_like(calls)
        for i, j in mapping.items():
            remapped[calls == i] = j
        best_acc = float((remapped == true_ancestry).mean())
        best_perm = tuple(mapping.get(i, 0) for i in range(A_inferred))

    # Per-ancestry accuracy under best mapping
    per_anc = {}
    remapped = np.zeros_like(calls)
    for i, j in enumerate(best_perm):
        remapped[calls == i] = j

    for a in range(A_true):
        mask = true_ancestry == a
        if mask.sum() > 0:
            per_anc[a] = float((remapped[mask] == a).mean())
        else:
            per_anc[a] = float("nan")

    return {
        "overall_accuracy": best_acc,
        "best_permutation": best_perm,
        "per_ancestry_accuracy": per_anc,
    }
