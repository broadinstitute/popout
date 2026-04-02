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
    rng_seed: int = 42,
) -> tuple[ChromData, np.ndarray]:
    """Generate simulated admixed haplotype data.

    Parameters
    ----------
    n_samples : number of diploid samples (will produce 2× haplotypes)
    n_sites : number of SNP sites
    n_ancestries : A
    gen_since_admix : T
    chrom_length_cm : total genetic length in cM
    fst_range : range of Fst between populations (controls divergence)
    rng_seed : random seed

    Returns
    -------
    chrom_data : ChromData with simulated genotypes
    true_ancestry : array (n_haps, n_sites) — ground truth ancestry labels
    """
    rng = np.random.default_rng(rng_seed)
    A = n_ancestries
    T = gen_since_admix
    n_haps = 2 * n_samples
    H = n_haps

    log.info("Simulating %d samples (%d haplotypes), %d sites, %d ancestries",
             n_samples, n_haps, n_sites, A)

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

    # --- Simulate ancestry tracts and genotypes ---
    geno = np.zeros((H, n_sites), dtype=np.uint8)
    true_ancestry = np.zeros((H, n_sites), dtype=np.int8)

    for h in range(H):
        # Start with random ancestry
        anc = rng.choice(A, p=mu)
        for t in range(n_sites):
            if t > 0:
                # Recombination: switch ancestry?
                p_switch = 1.0 - np.exp(-d_morgan[t - 1] * T)
                if rng.random() < p_switch:
                    anc = rng.choice(A, p=mu)

            true_ancestry[h, t] = anc
            # Emit allele
            geno[h, t] = rng.random() < pop_freq[anc, t]

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

    return chrom_data, true_ancestry


def evaluate_accuracy(
    calls: np.ndarray,
    true_ancestry: np.ndarray,
    n_ancestries: int,
) -> dict:
    """Evaluate ancestry inference accuracy.

    Since inferred labels may be permuted relative to true labels,
    finds the best permutation match.

    Parameters
    ----------
    calls : (H, T) inferred ancestry labels
    true_ancestry : (H, T) true ancestry labels
    n_ancestries : A

    Returns
    -------
    dict with accuracy metrics
    """
    from itertools import permutations

    A = n_ancestries
    best_acc = 0.0
    best_perm = None

    # For small A, try all permutations
    if A <= 8:
        for perm in permutations(range(A)):
            remapped = np.zeros_like(calls)
            for i, j in enumerate(perm):
                remapped[calls == i] = j
            acc = (remapped == true_ancestry).mean()
            if acc > best_acc:
                best_acc = acc
                best_perm = perm

    # Per-ancestry accuracy under best permutation
    per_anc = {}
    remapped = np.zeros_like(calls)
    for i, j in enumerate(best_perm):
        remapped[calls == i] = j

    for a in range(A):
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
