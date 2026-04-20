"""Block-level haplotype pattern encoding for LD-aware emissions.

Sites are grouped into fixed-width blocks of k SNPs. Within each block,
the k binary alleles are packed into an integer pattern index. The HMM
then operates over blocks instead of individual sites, with per-ancestry
pattern frequency tables as emissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np


@dataclass
class BlockData:
    """Block-level representation of haplotype data.

    Attributes
    ----------
    pattern_indices : np.ndarray, (H, n_blocks), uint8/uint16
        Dense pattern index per haplotype per block.
    block_starts : np.ndarray, (n_blocks,) int
        Start site index of each block.
    block_ends : np.ndarray, (n_blocks,) int
        End site index (exclusive) of each block.
    block_distances : np.ndarray, (n_blocks - 1,) float
        Genetic distance between consecutive blocks in Morgans.
    pattern_counts : np.ndarray, (n_blocks,) int
        Number of distinct patterns per block.
    max_patterns : int
        Maximum pattern count across blocks (for dense array padding).
    block_size : int
        Number of SNPs per block (k).
    """

    pattern_indices: np.ndarray
    block_starts: np.ndarray
    block_ends: np.ndarray
    block_distances: np.ndarray
    pattern_counts: np.ndarray
    max_patterns: int
    block_size: int

    @property
    def n_blocks(self) -> int:
        return len(self.block_starts)


def pack_blocks(
    geno: np.ndarray,
    block_size: int = 8,
    pos_cm: Optional[np.ndarray] = None,
) -> BlockData:
    """Pack site-level genotypes into block-level pattern indices.

    Parameters
    ----------
    geno : (H, T) uint8 — binary allele matrix
    block_size : k — sites per block
    pos_cm : (T,) — genetic positions (for block distances)

    Returns
    -------
    BlockData
    """
    H, T = geno.shape
    k = block_size
    n_blocks = (T + k - 1) // k

    pattern_indices = np.zeros((H, n_blocks), dtype=np.uint16)
    block_starts = np.zeros(n_blocks, dtype=np.int64)
    block_ends = np.zeros(n_blocks, dtype=np.int64)
    pattern_counts = np.zeros(n_blocks, dtype=np.int64)

    for b in range(n_blocks):
        s = b * k
        e = min(s + k, T)
        block_starts[b] = s
        block_ends[b] = e
        w = e - s

        # Pack alleles into integer: bit i = allele at position s+i
        block_geno = geno[:, s:e].astype(np.uint16)  # (H, w)
        packed = np.zeros(H, dtype=np.uint16)
        for i in range(w):
            packed |= block_geno[:, i] << i

        # Map to dense indices
        unique_patterns, inverse = np.unique(packed, return_inverse=True)
        pattern_indices[:, b] = inverse
        pattern_counts[b] = len(unique_patterns)

    max_patterns = int(pattern_counts.max())

    # Block distances
    if pos_cm is not None and n_blocks > 1:
        # Distance between consecutive block midpoints in Morgans
        midpoints = np.array([(pos_cm[block_starts[b]] + pos_cm[min(block_ends[b] - 1, T - 1)]) / 2
                              for b in range(n_blocks)])
        block_distances = np.diff(midpoints) / 100.0  # cM → Morgans
        block_distances = np.maximum(block_distances, 1e-10)
    else:
        block_distances = np.full(max(n_blocks - 1, 0), 0.001)

    return BlockData(
        pattern_indices=pattern_indices,
        block_starts=block_starts,
        block_ends=block_ends,
        block_distances=block_distances,
        pattern_counts=pattern_counts,
        max_patterns=max_patterns,
        block_size=block_size,
    )


def init_pattern_freq(
    allele_freq: jnp.ndarray,
    block_data: BlockData,
    geno: np.ndarray,
    pseudocount: float = 0.01,
) -> jnp.ndarray:
    """Initialize pattern frequency tables from per-site allele frequencies.

    For each block and pattern, compute the probability under each ancestry
    as the product of per-site Bernoulli probabilities.

    Parameters
    ----------
    allele_freq : (A, T) — per-site allele frequencies
    block_data : BlockData
    geno : (H, T) — used to enumerate patterns

    Returns
    -------
    pattern_freq : (n_blocks, max_patterns, A)
    """
    import logging
    _ipf_log = logging.getLogger("popout.blocks.init_pattern_freq")

    A = allele_freq.shape[0]
    n_blocks = block_data.n_blocks
    max_p = block_data.max_patterns

    _ipf_log.info("init_pattern_freq: entering (n_blocks=%d, max_p=%d, A=%d, H=%d, T=%d)",
                  n_blocks, max_p, A, geno.shape[0], geno.shape[1])

    freq = np.array(allele_freq)
    freq = np.clip(freq, 1e-6, 1.0 - 1e-6)

    pf = np.full((n_blocks, max_p, A), pseudocount, dtype=np.float32)

    for b in range(n_blocks):
        if b % 10 == 0:
            _ipf_log.info("init_pattern_freq: block %d/%d (patterns_this_block=%d)",
                          b, n_blocks, block_data.pattern_counts[b])
        s = block_data.block_starts[b]
        e = block_data.block_ends[b]
        n_p = block_data.pattern_counts[b]

        # Get the unique patterns in this block
        block_geno = geno[:, s:e]  # (H, w)
        pat_idx = block_data.pattern_indices[:, b]  # (H,)

        # For each unique pattern, compute P(pattern | ancestry a)
        for p in range(n_p):
            # Find a haplotype with this pattern
            exemplar = np.where(pat_idx == p)[0][0]
            bits = block_geno[exemplar]  # (w,) uint8

            for a in range(A):
                log_prob = 0.0
                for i, bit in enumerate(bits):
                    f = freq[a, s + i]
                    log_prob += np.log(f) if bit == 1 else np.log(1.0 - f)
                pf[b, p, a] = np.exp(log_prob)

    _ipf_log.info("init_pattern_freq: main loop done, starting normalization")

    # Normalize per block per ancestry
    for b in range(n_blocks):
        n_p = block_data.pattern_counts[b]
        for a in range(A):
            total = pf[b, :n_p, a].sum()
            if total > 0:
                pf[b, :n_p, a] /= total

    _ipf_log.info("init_pattern_freq: normalization done, converting to jnp")
    result = jnp.array(pf)
    _ipf_log.info("init_pattern_freq: returning (pf.shape=%s)", result.shape)
    return result


def update_pattern_freq(
    block_data: BlockData,
    gamma_block: jnp.ndarray,
    pseudocount: float = 0.01,
) -> jnp.ndarray:
    """M-step: update pattern frequency tables from block posteriors.

    Parameters
    ----------
    block_data : BlockData
    gamma_block : (H, n_blocks, A) — posteriors at block level

    Returns
    -------
    pattern_freq : (n_blocks, max_patterns, A)
    """
    H = gamma_block.shape[0]
    A = gamma_block.shape[2]
    n_blocks = block_data.n_blocks
    max_p = block_data.max_patterns

    pat_idx = jnp.array(block_data.pattern_indices)  # (H, n_blocks)

    # Per-block scatter-add: for each block b, accumulate posterior weight
    # by pattern index
    pf = jnp.full((n_blocks, max_p, A), pseudocount)

    for b in range(n_blocks):
        n_p = block_data.pattern_counts[b]
        idx = pat_idx[:, b]  # (H,)
        weights = gamma_block[:, b, :]  # (H, A)

        # Scatter-add: for each pattern p, sum weights of matching haplotypes
        counts = jnp.zeros((max_p, A))
        counts = counts.at[idx].add(weights)

        total = counts[:n_p].sum(axis=0, keepdims=True)  # (1, A)
        pf = pf.at[b, :n_p, :].set(
            (counts[:n_p] + pseudocount) / (total + pseudocount * n_p)
        )

    return pf


def expand_block_posteriors(
    gamma_block: jnp.ndarray,
    block_data: BlockData,
    n_sites: int,
) -> jnp.ndarray:
    """Expand block-level posteriors to site-level.

    Each site inherits the posterior of its containing block.

    Parameters
    ----------
    gamma_block : (H, n_blocks, A)
    block_data : BlockData
    n_sites : T — original number of sites

    Returns
    -------
    gamma_site : (H, T, A)
    """
    H, _, A = gamma_block.shape

    # Build site-to-block mapping
    site_to_block = np.zeros(n_sites, dtype=np.int32)
    for b in range(block_data.n_blocks):
        s = block_data.block_starts[b]
        e = block_data.block_ends[b]
        site_to_block[s:e] = b

    return gamma_block[:, site_to_block, :]
