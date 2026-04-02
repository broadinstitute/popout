"""Spectral initialization for ancestry labels.

Strategy:
  1. Randomized SVD on a SNP subset → low-dimensional projection
  2. Gaussian Mixture Model (not k-means) → soft cluster assignments
     GMM naturally handles unequal cluster sizes.
  3. Soft assignments → initial allele frequencies via weighted GEMM
  4. Window-based refinement: for each local window, re-assign ancestry
     using the allele frequencies from step 3.  This handles admixed
     haplotypes that have *different* ancestry at different positions.
"""

from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def seed_ancestry(
    geno: np.ndarray,
    n_ancestries: Optional[int] = None,
    max_snps: int = 10_000,
    n_components: int = 20,
    max_ancestries: int = 12,
    gmm_restarts: int = 5,
    rng_seed: int = 42,
) -> tuple[jnp.ndarray, int]:
    """Compute initial hard ancestry assignments.

    Returns
    -------
    labels : jnp.ndarray (n_haps,) int32
    n_ancestries : int
    """
    labels, _resp, n_anc = seed_ancestry_soft(
        geno, n_ancestries, max_snps, n_components,
        max_ancestries, gmm_restarts, rng_seed,
    )
    return labels, n_anc


def seed_ancestry_soft(
    geno: np.ndarray,
    n_ancestries: Optional[int] = None,
    max_snps: int = 10_000,
    n_components: int = 20,
    max_ancestries: int = 12,
    gmm_restarts: int = 5,
    rng_seed: int = 42,
    stats=None,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Compute soft ancestry assignments via PCA + GMM.

    Returns
    -------
    labels : jnp.ndarray (n_haps,) int32  — hard labels
    responsibilities : jnp.ndarray (n_haps, A) — soft assignments
    n_ancestries : int
    """
    key = jax.random.PRNGKey(rng_seed)
    n_haps, n_sites = geno.shape

    if n_sites > max_snps:
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, n_sites, shape=(max_snps,), replace=False)
        idx = jnp.sort(idx)
        X = jnp.array(geno[:, np.array(idx)], dtype=jnp.float32)
    else:
        X = jnp.array(geno, dtype=jnp.float32)

    log.info("Spectral init: %d haplotypes × %d SNPs", n_haps, X.shape[1])

    # --- Patterson-style normalization ---
    mean = X.mean(axis=0)
    X = X - mean
    p = jnp.clip(mean, 0.01, 0.99)
    scale = jnp.sqrt(p * (1.0 - p))
    X = X / scale

    # --- Randomized SVD ---
    key, subkey = jax.random.split(key)
    U, S, Vt = _randomized_svd(X, n_components, subkey)
    log.info("Top singular values: %s", np.array(S[:min(10, len(S))]).round(1))
    if stats is not None:
        sv = np.array(S[:min(20, len(S))])
        stats.emit("spectral/singular_values", sv)
        if len(sv) > 1:
            ratios = (sv[:-1] / (sv[1:] + 1e-10)).tolist()
            stats.emit("spectral/gap_ratios", ratios)

    # --- Auto-detect A ---
    if n_ancestries is None:
        n_ancestries = _detect_n_ancestries(S, max_ancestries)
        log.info("Auto-detected %d ancestries", n_ancestries)
    if stats is not None:
        stats.emit("spectral/n_ancestries", n_ancestries)

    # --- Project ---
    n_pc = max(n_ancestries - 1, 2)
    proj = U[:, :n_pc] * S[:n_pc]

    # --- GMM ---
    key, subkey = jax.random.split(key)
    labels, resp = _gmm(proj, n_ancestries, gmm_restarts, subkey, stats=stats)

    for a in range(n_ancestries):
        eff = float(resp[:, a].sum())
        log.info("  Ancestry %d: %.0f effective haplotypes (%.1f%%)",
                 a, eff, 100 * eff / n_haps)

    return labels, resp, n_ancestries


# ---------------------------------------------------------------------------
# Window-based local initialization
# ---------------------------------------------------------------------------

def window_init_allele_freq(
    geno: jnp.ndarray,
    global_freq: jnp.ndarray,
    n_ancestries: int,
    window_size: int = 50,
) -> jnp.ndarray:
    """Refine allele frequencies using local window assignments.

    For each window of SNPs, compute each haplotype's likelihood under
    each ancestry using global_freq, do local soft assignment, then
    recompute allele frequencies from local assignments.

    This handles admixed haplotypes properly: a haplotype that's ancestry A
    in one window and ancestry B in another contributes its alleles to the
    correct population in each region.
    """
    H, T = geno.shape
    A = n_ancestries
    geno_f = geno.astype(jnp.float32)
    freq = jnp.clip(global_freq, 1e-4, 1.0 - 1e-4)

    weighted_counts = jnp.zeros((A, T))
    weighted_totals = jnp.zeros((A, T))

    n_windows = (T + window_size - 1) // window_size

    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, T)
        w_len = end - start
        w_geno = geno_f[:, start:end]   # (H, W)
        w_freq = freq[:, start:end]      # (A, W)

        # Log-likelihood per haplotype per ancestry in this window
        log_f1 = jnp.log(w_freq)         # (A, W)
        log_f0 = jnp.log(1.0 - w_freq)   # (A, W)
        # ll[h, a] = sum over sites in window
        ll = jnp.einsum("ht,at->ha", w_geno, log_f1) + \
             jnp.einsum("ht,at->ha", 1.0 - w_geno, log_f0)  # (H, A)

        resp = jax.nn.softmax(ll, axis=1)  # (H, A)

        w_counts = resp.T @ w_geno                          # (A, W)
        w_totals = resp.sum(axis=0)[:, None] * jnp.ones((1, w_len))

        weighted_counts = weighted_counts.at[:, start:end].set(w_counts)
        weighted_totals = weighted_totals.at[:, start:end].set(w_totals)

    refined_freq = (weighted_counts + 0.5) / (weighted_totals + 1.0)
    return refined_freq


# ---------------------------------------------------------------------------
# Gaussian Mixture Model (diagonal covariance)
# ---------------------------------------------------------------------------

def _gmm(
    X: jnp.ndarray,
    k: int,
    n_restarts: int,
    key: jax.Array,
    max_iter: int = 100,
    tol: float = 1e-5,
    stats=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """GMM with multiple restarts.  Returns (labels, responsibilities)."""
    best_labels = None
    best_resp = None
    best_ll = -jnp.inf

    for r in range(n_restarts):
        key, subkey = jax.random.split(key)
        labels, resp, ll = _gmm_single(X, k, subkey, max_iter, tol)
        ll_f = float(ll)
        log.debug("  GMM restart %d: ll=%.2f", r, ll_f)
        if stats is not None:
            stats.emit("spectral/gmm_ll", ll_f, tags={"restart": r})
        if ll > best_ll:
            best_ll = ll
            best_labels = labels
            best_resp = resp

    if stats is not None:
        stats.emit("spectral/gmm_best_ll", float(best_ll))

    return best_labels, best_resp


def _gmm_single(
    X: jnp.ndarray,
    k: int,
    key: jax.Array,
    max_iter: int,
    tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Single EM run for diagonal-covariance GMM."""
    n, d = X.shape

    key, subkey = jax.random.split(key)
    means = _kmeans_pp_init(X, k, subkey)
    covs = jnp.ones((k, d)) * jnp.var(X, axis=0)
    weights = jnp.ones(k) / k

    prev_ll = -jnp.inf

    for iteration in range(max_iter):
        # E-step
        log_probs = _gmm_log_prob(X, means, covs, weights)
        log_resp = log_probs - jax.nn.logsumexp(log_probs, axis=1, keepdims=True)
        resp = jnp.exp(log_resp)

        ll = float(jax.nn.logsumexp(log_probs, axis=1).mean())
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # M-step
        Nk = jnp.maximum(resp.sum(axis=0), 1e-6)
        weights = Nk / n
        means = (resp.T @ X) / Nk[:, None]

        # Update diagonal covariances
        new_covs = jnp.zeros_like(covs)
        for c in range(k):
            diff = X - means[c]
            new_covs = new_covs.at[c].set(
                (resp[:, c:c + 1] * diff ** 2).sum(axis=0) / Nk[c]
            )
        covs = jnp.maximum(new_covs, 1e-6)

    labels = jnp.argmax(resp, axis=1).astype(jnp.int32)
    return labels, resp, ll


def _gmm_log_prob(
    X: jnp.ndarray,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Log P(x | component c) + log weight[c].  Returns (n, k)."""
    n, d = X.shape
    k = means.shape[0]
    log_2pi = jnp.log(2 * jnp.pi)
    log_w = jnp.log(weights + 1e-30)

    log_probs = jnp.zeros((n, k))
    for c in range(k):
        diff = X - means[c]
        log_det = jnp.sum(jnp.log(covs[c]))
        mahal = jnp.sum(diff ** 2 / covs[c], axis=1)
        log_probs = log_probs.at[:, c].set(
            -0.5 * (d * log_2pi + log_det + mahal) + log_w[c]
        )
    return log_probs


# ---------------------------------------------------------------------------
# Randomized SVD
# ---------------------------------------------------------------------------

def _randomized_svd(
    X: jnp.ndarray,
    n_components: int,
    key: jax.Array,
    n_oversamples: int = 10,
    n_power_iter: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Halko-Martinsson-Tropp randomized SVD."""
    n, m = X.shape
    k = n_components + n_oversamples

    key, subkey = jax.random.split(key)
    Omega = jax.random.normal(subkey, (m, k))

    Y = X @ Omega
    for _ in range(n_power_iter):
        Y = X @ (X.T @ Y)

    Q, _ = jnp.linalg.qr(Y)
    B = Q.T @ X
    U_hat, S, Vt = jnp.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    return U[:, :n_components], S[:n_components], Vt[:n_components, :]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_n_ancestries(S: jnp.ndarray, max_a: int) -> int:
    """Detect number of ancestries from eigenvalue gap."""
    S = np.array(S)
    if len(S) < 3:
        return 2

    ratios = S[:-1] / (S[1:] + 1e-10)

    for i in range(1, min(len(ratios), max_a)):
        tail = ratios[i + 1: i + 5]
        tail_median = np.median(tail) if len(tail) > 0 else 1.0
        if ratios[i] > 1.3 and ratios[i] > 1.2 * tail_median:
            return int(min(i + 1, max_a))

    return int(min(max(2, int((S > S[0] * 0.1).sum())), max_a))


def _kmeans_pp_init(X: jnp.ndarray, k: int, key: jax.Array) -> jnp.ndarray:
    """K-means++ init (used to seed GMM centers)."""
    n, d = X.shape
    key, subkey = jax.random.split(key)
    first = jax.random.randint(subkey, (), 0, n)
    centers = [X[first]]

    for _ in range(1, k):
        C = jnp.stack(centers)
        dists = _pairwise_dist_sq(X, C)
        min_dists = jnp.min(dists, axis=1)
        probs = min_dists / (min_dists.sum() + 1e-30)
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, n, p=probs)
        centers.append(X[idx])

    return jnp.stack(centers)


def _pairwise_dist_sq(X: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    return (
        (X ** 2).sum(axis=1, keepdims=True)
        - 2 * X @ C.T
        + (C ** 2).sum(axis=1, keepdims=True).T
    )
