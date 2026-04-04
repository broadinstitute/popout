"""Construct CNN input features from genotype data and model parameters."""

from __future__ import annotations

import jax.numpy as jnp


def build_cnn_features(
    geno: jnp.ndarray,
    allele_freq: jnp.ndarray,
    d_morgan: jnp.ndarray,
) -> jnp.ndarray:
    """Build per-site feature tensor for the CNN.

    Channels
    --------
    0       : allele value (0 or 1), cast to float32
    1..A    : allele_freq[a, t] for each ancestry a (broadcast to all haps)
    A+1     : genetic distance to next site (0 for last site)

    Parameters
    ----------
    geno : (H, T) uint8 — binary allele matrix
    allele_freq : (A, T) — per-ancestry allele frequencies
    d_morgan : (T-1,) — inter-site genetic distances in Morgans

    Returns
    -------
    features : (H, T, A+2) float32
    """
    H, T = geno.shape
    A = allele_freq.shape[0]

    # Channel 0: allele value
    allele_ch = geno[:, :, None].astype(jnp.float32)  # (H, T, 1)

    # Channels 1..A: population allele frequencies (same for all haplotypes)
    freq_ch = jnp.broadcast_to(
        allele_freq.T[None, :, :],  # (1, T, A)
        (H, T, A),
    )

    # Channel A+1: genetic distance (zero-padded at last site)
    d_padded = jnp.concatenate([d_morgan, jnp.zeros(1)])  # (T,)
    dist_ch = jnp.broadcast_to(
        d_padded[None, :, None],  # (1, T, 1)
        (H, T, 1),
    )

    return jnp.concatenate([allele_ch, freq_ch, dist_ch], axis=-1)
