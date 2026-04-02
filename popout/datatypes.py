"""Core data structures shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
