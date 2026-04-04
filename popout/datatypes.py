"""Core data structures shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ChromData:
    """Haplotype data for a single chromosome.

    Attributes
    ----------
    geno : np.ndarray, shape (n_haps, n_sites), dtype uint8
        Binary allele matrix (0/1).  Rows are haplotypes (2 per diploid
        sample), columns are variant sites after MAF filtering.
    pos_bp : np.ndarray, shape (n_sites,), dtype int64
        Physical positions in base pairs.
    pos_cm : np.ndarray, shape (n_sites,), dtype float64
        Genetic positions in centiMorgans (from recombination map).
    chrom : str
        Chromosome name (e.g. "chr1", "1").
    site_ids : Optional[np.ndarray]
        Variant IDs if available.
    """

    geno: np.ndarray
    pos_bp: np.ndarray
    pos_cm: np.ndarray
    chrom: str
    site_ids: Optional[np.ndarray] = None

    @property
    def n_haps(self) -> int:
        return self.geno.shape[0]

    @property
    def n_sites(self) -> int:
        return self.geno.shape[1]

    @property
    def genetic_distances(self) -> np.ndarray:
        """Inter-site genetic distances in Morgans."""
        return np.diff(self.pos_cm) / 100.0


@dataclass
class GeneticMap:
    """HapMap-format recombination map for one chromosome.

    Columns: chrom, pos_bp, rate_cM_per_Mb, pos_cM
    """

    pos_bp: np.ndarray
    pos_cm: np.ndarray

    def interpolate(self, query_bp: np.ndarray) -> np.ndarray:
        """Linearly interpolate genetic positions for query base-pair positions."""
        return np.interp(query_bp, self.pos_bp, self.pos_cm)


@dataclass
class AncestryModel:
    """Parameters of the ancestry model, updated during EM.

    Attributes
    ----------
    n_ancestries : int
        Number of inferred ancestral populations (A).
    mu : jnp.ndarray, shape (A,)
        Global ancestry proportions.
    gen_since_admix : float
        Estimated generations since admixture (T).
    allele_freq : jnp.ndarray, shape (A, n_sites)
        Per-ancestry allele frequencies at each site.
    mismatch : jnp.ndarray, shape (A,)
        Per-ancestry mismatch/error rate.
    """

    n_ancestries: int
    mu: jnp.ndarray
    gen_since_admix: float
    allele_freq: jnp.ndarray
    mismatch: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    # Per-haplotype T (optional — None means scalar T for all haplotypes)
    gen_per_hap: Optional[jnp.ndarray] = None       # (H,)
    bucket_centers: Optional[jnp.ndarray] = None     # (B,)
    bucket_assignments: Optional[jnp.ndarray] = None # (H,) int32
    # Block emission model (optional — None means single-site Bernoulli)
    pattern_freq: Optional[jnp.ndarray] = None       # (n_blocks, max_patterns, A)
    block_data: object = None                         # BlockData (avoid circular import)

    def log_transition_matrix(self, d_morgan: jnp.ndarray) -> jnp.ndarray:
        """Build (n_intervals, A, A) log transition matrices.

        Parameters
        ----------
        d_morgan : array, shape (n_intervals,)
            Genetic distance between consecutive sites in Morgans.

        Returns
        -------
        log_trans : array, shape (n_intervals, A, A)
        """
        T = self.gen_since_admix
        A = self.n_ancestries
        # Probability of at least one recombination in interval
        p_switch = 1.0 - jnp.exp(-d_morgan * T)  # (n_intervals,)
        # Off-diagonal: switch to ancestry j with prob mu[j] * p_switch
        # Diagonal: stay with prob (1 - p_switch) + p_switch * mu[i]
        log_mu = jnp.log(self.mu)  # (A,)
        log_p = jnp.log(p_switch + 1e-30)[:, None, None]       # (I, 1, 1)
        log_1mp = jnp.log(1.0 - p_switch + 1e-30)[:, None, None]

        # trans[i,j] = (1-p)*I(i==j) + p*mu[j]
        # In log-space, use logsumexp for diagonal
        eye = jnp.eye(A)
        # Off-diagonal terms: log(p * mu[j])
        log_off = log_p + log_mu[None, None, :]  # (I, 1, A)
        # Diagonal terms: log((1-p) + p*mu[i])
        log_diag = jnp.logaddexp(log_1mp, log_p + log_mu[None, None, :])

        log_trans = jnp.where(
            eye[None, :, :],
            jnp.broadcast_to(log_diag, (d_morgan.shape[0], A, A)),
            jnp.broadcast_to(log_off, (d_morgan.shape[0], A, A)),
        )
        return log_trans

    def log_transition_matrices_bucketed(
        self, d_morgan: jnp.ndarray
    ) -> jnp.ndarray:
        """Build transition matrices for each T-bucket.

        Returns
        -------
        log_trans : (B, n_intervals, A, A)
        """
        assert self.bucket_centers is not None
        A = self.n_ancestries
        mu = self.mu
        log_mu = jnp.log(mu)
        eye = jnp.eye(A)

        def _single_bucket(T_val):
            p_switch = 1.0 - jnp.exp(-d_morgan * T_val)
            log_p = jnp.log(p_switch + 1e-30)[:, None, None]
            log_1mp = jnp.log(1.0 - p_switch + 1e-30)[:, None, None]
            log_off = log_p + log_mu[None, None, :]
            log_diag = jnp.logaddexp(log_1mp, log_p + log_mu[None, None, :])
            return jnp.where(
                eye[None, :, :],
                jnp.broadcast_to(log_diag, (d_morgan.shape[0], A, A)),
                jnp.broadcast_to(log_off, (d_morgan.shape[0], A, A)),
            )

        return jax.vmap(_single_bucket)(self.bucket_centers)  # (B, I, A, A)

    def log_emission_block(self, block_data) -> jnp.ndarray:
        """Compute block-level log emissions from pattern frequency tables.

        Parameters
        ----------
        block_data : BlockData with pattern_indices (H, n_blocks)

        Returns
        -------
        log_emit : (H, n_blocks, A)
        """
        assert self.pattern_freq is not None
        log_pf = jnp.log(jnp.clip(self.pattern_freq, 1e-10, 1.0))
        # log_pf: (n_blocks, max_patterns, A)
        pat_idx = jnp.array(block_data.pattern_indices)  # (H, n_blocks)
        n_blocks = block_data.n_blocks
        # Gather: log_pf[b, pat_idx[h, b], :] for each (h, b)
        block_range = jnp.arange(n_blocks)[None, :]  # (1, n_blocks)
        return log_pf[block_range, pat_idx, :]  # (H, n_blocks, A)

    def log_emission(self, geno: jnp.ndarray) -> jnp.ndarray:
        """Compute log emission probabilities.

        Parameters
        ----------
        geno : array, shape (n_haps, n_sites), dtype uint8
            Binary allele matrix.

        Returns
        -------
        log_emit : array, shape (n_haps, n_sites, A)
            Log P(observed allele | ancestry a) at each site.
        """
        freq = self.allele_freq  # (A, n_sites)
        # Clip to avoid log(0)
        freq = jnp.clip(freq, 1e-6, 1.0 - 1e-6)
        log_f1 = jnp.log(freq)      # log P(allele=1 | ancestry)
        log_f0 = jnp.log(1.0 - freq)  # log P(allele=0 | ancestry)
        # geno is (H, T), freq is (A, T)
        # result should be (H, T, A)
        g = geno[:, :, None].astype(jnp.float32)  # (H, T, 1)
        log_emit = g * log_f1.T[None, :, :] + (1.0 - g) * log_f0.T[None, :, :]
        return log_emit


@dataclass
class AncestryResult:
    """Output of the LAI pipeline for one chromosome.

    Attributes
    ----------
    posteriors : jnp.ndarray, shape (n_haps, n_sites, A)
        Posterior ancestry probabilities.
    calls : np.ndarray, shape (n_haps, n_sites), dtype int8
        Hard ancestry calls (argmax of posteriors).
    model : AncestryModel
        Final fitted model parameters.
    chrom : str
        Chromosome name.
    """

    posteriors: jnp.ndarray
    calls: np.ndarray
    model: AncestryModel
    chrom: str
