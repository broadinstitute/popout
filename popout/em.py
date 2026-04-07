"""Expectation-Maximization loop for self-bootstrapping ancestry inference.

The model parameters (allele frequencies, ancestry proportions, admixture
time) are iteratively refined.  With 500K+ samples, sufficient statistics
converge fast — typically 2-3 EM iterations suffice.
"""

from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .datatypes import AncestryModel, ChromData, AncestryResult, EMStats, DecodeResult
from .hmm import (
    forward_backward_batched,
    forward_backward_bucketed,
    forward_backward_blocks,
    forward_backward_em,
    forward_backward_bucketed_em,
    forward_backward_decode,
    forward_backward_bucketed_decode,
)
from .spectral import seed_ancestry, seed_ancestry_soft, window_init_allele_freq

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# M-step: update model parameters from posteriors
# ---------------------------------------------------------------------------

def update_allele_freq(
    geno: jnp.ndarray,
    gamma: jnp.ndarray,
    pseudocount: float = 0.5,
) -> jnp.ndarray:
    """Compute allele frequencies per ancestry from posteriors.

    This is the key GEMM: gamma.T @ geno  →  (A, T) weighted allele counts.

    Parameters
    ----------
    geno : (H, T) uint8 — allele matrix
    gamma : (H, T, A) — posterior ancestry probabilities

    Returns
    -------
    freq : (A, T) — allele frequencies per ancestry per site
    """
    H, T, A = gamma.shape
    geno_f = geno.astype(jnp.float32)  # (H, T)

    # Weighted allele count per ancestry: sum over haplotypes
    # For each ancestry a, site t: sum_h gamma[h,t,a] * geno[h,t]
    # = einsum('hta,ht->at', gamma, geno_f)
    weighted_counts = jnp.einsum("hta,ht->at", gamma, geno_f)  # (A, T)

    # Total weight per ancestry per site: sum_h gamma[h,t,a]
    total_weights = gamma.sum(axis=0).T  # (A, T)

    # Frequency with pseudocount smoothing
    freq = (weighted_counts + pseudocount) / (total_weights + 2 * pseudocount)
    return freq


def smooth_rare_frequencies(
    freq: jnp.ndarray,
    pos_cm: jnp.ndarray,
    bandwidth_cm: float = 0.05,
    maf_threshold: float = 0.05,
) -> jnp.ndarray:
    """Gaussian-kernel smooth allele frequencies at rare variants.

    Rare variants (MAF < threshold) have their per-ancestry frequencies
    smoothed along the genomic coordinate, borrowing strength from
    nearby sites.  Common variants are left unchanged.

    Parameters
    ----------
    freq : (A, T) — per-ancestry allele frequencies
    pos_cm : (T,) — genetic positions in centiMorgans
    bandwidth_cm : Gaussian kernel bandwidth in cM (0 = disabled)
    maf_threshold : variants with MAF below this are smoothed

    Returns
    -------
    freq_smoothed : (A, T)
    """
    A, T = freq.shape
    if bandwidth_cm <= 0 or T <= 1:
        return freq

    # Identify rare variants by overall MAF (average across ancestries)
    overall_freq = freq.mean(axis=0)  # (T,)
    maf = jnp.minimum(overall_freq, 1.0 - overall_freq)
    is_rare = maf < maf_threshold  # (T,)

    # Pairwise genetic distance matrix and Gaussian kernel
    # NOTE: O(T^2) memory — fine for thinned data (T ~ 3K–10K).
    # For very large T without thinning, a windowed implementation
    # would be needed but is not required in practice.
    d = jnp.abs(pos_cm[:, None] - pos_cm[None, :])  # (T, T)
    K = jnp.exp(-0.5 * (d / bandwidth_cm) ** 2)      # (T, T)
    K_norm = K / K.sum(axis=1, keepdims=True)          # (T, T) rows sum to 1

    # Smooth: freq_smooth[a, t] = Σ_s K_norm[t, s] * freq[a, s]
    freq_smooth = freq @ K_norm.T  # (A, T)

    # Blend: keep original at common variants, use smoothed at rare ones
    result = jnp.where(is_rare[None, :], freq_smooth, freq)
    return jnp.clip(result, 1e-6, 1.0 - 1e-6)


def update_mu(gamma: jnp.ndarray) -> jnp.ndarray:
    """Update global ancestry proportions.

    Simply the mean posterior across all haplotypes and all sites.

    Parameters
    ----------
    gamma : (H, T, A)

    Returns
    -------
    mu : (A,) — ancestry proportions summing to 1
    """
    mu = gamma.mean(axis=(0, 1))  # (A,)
    return mu / mu.sum()


def update_generations(
    gamma: jnp.ndarray,
    d_morgan: jnp.ndarray,
    current_T: float,
    mu: jnp.ndarray,
) -> float:
    """Estimate generations since admixture from hard-call switch rate.

    Uses hard ancestry calls (argmax of posteriors) to count switches,
    which is more robust than soft overlap when posteriors are diffuse.

    From the model: P(switch at interval) = 1 - exp(-d * T)
    and the new ancestry is drawn from mu, so the expected fraction
    of switches that *change* ancestry is (1 - sum(mu^2)).
    Therefore: observed_switch_rate ≈ (1 - exp(-d*T)) * (1 - sum(mu^2))

    Parameters
    ----------
    gamma : (H, T_sites, A)
    d_morgan : (T_sites - 1,)
    current_T : current estimate (used as regularization anchor)
    mu : (A,) ancestry proportions

    Returns
    -------
    T_new : float — updated generations since admixture
    """
    # Hard calls
    calls = jnp.argmax(gamma, axis=2)  # (H, T_sites)

    # Count switches between adjacent sites
    switches = (calls[:, 1:] != calls[:, :-1]).astype(jnp.float32)  # (H, T-1)

    # Per-interval switch rate averaged over haplotypes
    switch_rate = switches.mean(axis=0)  # (T-1,)

    # Correction for resampling same ancestry after switch
    p_diff_ancestry = 1.0 - (mu ** 2).sum()
    p_diff_ancestry = float(jnp.maximum(p_diff_ancestry, 0.1))

    # Total observed switches / total genetic distance
    d = jnp.maximum(d_morgan, 1e-10)
    total_switches = switch_rate.sum()
    total_distance = d.sum()

    # Solve: switch_rate ≈ (1 - exp(-d*T)) * p_diff
    # For moderate d*T: observed / (total_d * p_diff) ≈ T
    T_est = float(total_switches / (total_distance * p_diff_ancestry + 1e-10))

    # Regularize: blend with current estimate (0.7 new, 0.3 old)
    T_new = 0.7 * T_est + 0.3 * current_T
    return max(1.0, min(T_new, 1000.0))


# ---------------------------------------------------------------------------
# Per-haplotype T with bucketed transitions
# ---------------------------------------------------------------------------

def compute_bucket_centers(
    n_buckets: int = 20,
    T_min: float = 1.0,
    T_max: float = 1000.0,
) -> jnp.ndarray:
    """Geometrically spaced T-bucket centers."""
    return jnp.geomspace(T_min, T_max, n_buckets)


def assign_buckets(
    T_per_hap: jnp.ndarray,
    bucket_centers: jnp.ndarray,
) -> jnp.ndarray:
    """Assign each haplotype to the nearest bucket (in log-space)."""
    log_T = jnp.log(T_per_hap)[:, None]            # (H, 1)
    log_c = jnp.log(bucket_centers)[None, :]        # (1, B)
    return jnp.argmin(jnp.abs(log_T - log_c), axis=1).astype(jnp.int32)


def update_generations_per_hap(
    gamma: jnp.ndarray,
    d_morgan: jnp.ndarray,
    current_T_global: float,
    mu: jnp.ndarray,
    bucket_centers: jnp.ndarray,
    min_switches_for_confidence: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Estimate per-haplotype T from individual switch rates.

    Returns
    -------
    T_per_hap : (H,) — regularized per-haplotype T
    bucket_assignments : (H,) int32
    T_global : float — updated global T
    """
    calls = jnp.argmax(gamma, axis=2)  # (H, T_sites)
    switches_per_hap = (calls[:, 1:] != calls[:, :-1]).astype(jnp.float32).sum(axis=1)

    p_diff = 1.0 - (mu ** 2).sum()
    p_diff = jnp.maximum(p_diff, 0.1)
    total_d = d_morgan.sum()

    T_raw = switches_per_hap / (total_d * p_diff + 1e-10)

    # Regularize: lambda ~ 1/(1 + switches/min_switches)
    lam = 1.0 / (1.0 + switches_per_hap / min_switches_for_confidence)
    T_reg = (1.0 - lam) * T_raw + lam * current_T_global
    T_reg = jnp.clip(T_reg, 1.0, 1000.0)

    bucket_assignments = assign_buckets(T_reg, bucket_centers)
    T_global = float(jnp.mean(T_reg))

    return T_reg, bucket_assignments, T_global


# ---------------------------------------------------------------------------
# Streaming M-step: operate on EMStats instead of full gamma
# ---------------------------------------------------------------------------

def update_allele_freq_from_stats(
    stats: EMStats,
    pseudocount: float = 0.5,
) -> jnp.ndarray:
    """Update allele frequencies from pre-accumulated sufficient statistics.

    Equivalent to update_allele_freq(geno, gamma) but without needing
    the full (H, T, A) gamma tensor.
    """
    return (stats.weighted_counts + pseudocount) / (stats.total_weights + 2 * pseudocount)


def update_mu_from_stats(stats: EMStats) -> jnp.ndarray:
    """Update global ancestry proportions from pre-accumulated stats.

    Equivalent to update_mu(gamma).
    """
    mu = stats.mu_sum / (stats.n_haps * stats.n_sites)
    return mu / mu.sum()


def update_generations_from_stats(
    stats: EMStats,
    d_morgan: jnp.ndarray,
    current_T: float,
    mu: jnp.ndarray,
) -> float:
    """Estimate generations since admixture from pre-accumulated switch stats.

    Equivalent to update_generations(gamma, d_morgan, current_T, mu).
    """
    # Per-interval switch rate averaged over haplotypes
    switch_rate = stats.switch_sum / stats.n_haps  # (T-1,)

    p_diff_ancestry = 1.0 - (mu ** 2).sum()
    p_diff_ancestry = float(jnp.maximum(p_diff_ancestry, 0.1))

    d = jnp.maximum(d_morgan, 1e-10)
    total_switches = switch_rate.sum()
    total_distance = d.sum()

    T_est = float(total_switches / (total_distance * p_diff_ancestry + 1e-10))

    T_new = 0.7 * T_est + 0.3 * current_T
    return max(1.0, min(T_new, 1000.0))


def update_generations_per_hap_from_stats(
    stats: EMStats,
    d_morgan: jnp.ndarray,
    current_T_global: float,
    mu: jnp.ndarray,
    bucket_centers: jnp.ndarray,
    min_switches_for_confidence: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Per-haplotype T estimation from pre-accumulated switch counts.

    Equivalent to update_generations_per_hap(gamma, ...).
    """
    switches_per_hap = jnp.array(stats.switches_per_hap, dtype=jnp.float32)

    p_diff = 1.0 - (mu ** 2).sum()
    p_diff = jnp.maximum(p_diff, 0.1)
    total_d = d_morgan.sum()

    T_raw = switches_per_hap / (total_d * p_diff + 1e-10)

    lam = 1.0 / (1.0 + switches_per_hap / min_switches_for_confidence)
    T_reg = (1.0 - lam) * T_raw + lam * current_T_global
    T_reg = jnp.clip(T_reg, 1.0, 1000.0)

    bucket_assignments = assign_buckets(T_reg, bucket_centers)
    T_global = float(jnp.mean(T_reg))

    return T_reg, bucket_assignments, T_global


# ---------------------------------------------------------------------------
# Initialization from hard labels
# ---------------------------------------------------------------------------

def init_model_from_labels(
    geno: jnp.ndarray,
    labels: jnp.ndarray,
    n_ancestries: int,
    gen_since_admix: float = 20.0,
) -> AncestryModel:
    """Build initial AncestryModel from hard ancestry assignments.

    Parameters
    ----------
    geno : (H, T) — allele matrix
    labels : (H,) — integer ancestry labels
    n_ancestries : int
    gen_since_admix : initial guess for T

    Returns
    -------
    AncestryModel with initial parameter estimates
    """
    H, T = geno.shape
    A = n_ancestries
    geno_f = geno.astype(jnp.float32)

    freq = jnp.zeros((A, T))
    mu_counts = jnp.zeros(A)
    for a in range(A):
        mask = (labels == a)
        count = mask.sum()
        mu_counts = mu_counts.at[a].set(count)
        if count > 0:
            freq = freq.at[a].set(
                (geno_f * mask[:, None]).sum(axis=0) / count
            )
        else:
            freq = freq.at[a].set(geno_f.mean(axis=0))

    freq = jnp.clip(freq, 1e-4, 1.0 - 1e-4)
    mu = mu_counts / mu_counts.sum()
    mismatch = jnp.full(A, 0.001)

    return AncestryModel(
        n_ancestries=A,
        mu=mu,
        gen_since_admix=gen_since_admix,
        allele_freq=freq,
        mismatch=mismatch,
    )


def init_model_soft(
    geno: jnp.ndarray,
    responsibilities: jnp.ndarray,
    n_ancestries: int,
    gen_since_admix: float = 20.0,
    window_refine: bool = True,
    window_size: int = 50,
) -> AncestryModel:
    """Build initial AncestryModel from soft GMM assignments.

    Uses GMM responsibilities (per-haplotype soft ancestry weights)
    to compute initial allele frequencies via a weighted GEMM,
    then optionally refines with window-based local reassignment.

    Parameters
    ----------
    geno : (H, T) — allele matrix
    responsibilities : (H, A) — soft ancestry assignments from GMM
    n_ancestries : A
    gen_since_admix : initial guess for T
    window_refine : if True, refine freq with window-based local init
    window_size : SNPs per window for local refinement

    Returns
    -------
    AncestryModel
    """
    H, T = geno.shape
    A = n_ancestries
    geno_f = geno.astype(jnp.float32)
    resp = responsibilities  # (H, A)

    # Global allele frequencies from soft assignments
    # resp.T @ geno → (A, T) weighted allele counts
    weighted_counts = resp.T @ geno_f        # (A, T)
    totals = resp.sum(axis=0)[:, None]       # (A, 1)
    freq = (weighted_counts + 0.5) / (totals + 1.0)
    freq = jnp.clip(freq, 1e-4, 1.0 - 1e-4)

    # Window-based refinement: re-assign ancestry locally using freq,
    # then recompute freq.  Handles admixed haplotypes properly.
    if window_refine:
        log.info("  Window-based refinement (window=%d SNPs)", window_size)
        freq = window_init_allele_freq(geno, freq, A, window_size)
        freq = jnp.clip(freq, 1e-4, 1.0 - 1e-4)

    mu = resp.mean(axis=0)
    mu = mu / mu.sum()
    mismatch = jnp.full(A, 0.001)

    return AncestryModel(
        n_ancestries=A,
        mu=mu,
        gen_since_admix=gen_since_admix,
        allele_freq=freq,
        mismatch=mismatch,
    )


# ---------------------------------------------------------------------------
# Main EM loop
# ---------------------------------------------------------------------------

def run_em(
    chrom_data: ChromData,
    n_ancestries: Optional[int] = None,
    n_em_iter: int = 3,
    gen_since_admix: float = 20.0,
    batch_size: int = 50_000,
    rng_seed: int = 42,
    stats=None,
    bandwidth_cm: float = 0.05,
    maf_threshold: float = 0.05,
    per_hap_T: bool = False,
    n_T_buckets: int = 20,
    use_block_emissions: bool = False,
    block_size: int = 8,
) -> AncestryResult:
    """Self-bootstrapping EM for one chromosome.

    Pipeline:
        1. Spectral seed → hard ancestry labels
        2. Init model from hard labels
        3. EM iterations: forward-backward → update params
        4. Final posterior decode

    Parameters
    ----------
    chrom_data : ChromData
    n_ancestries : int or None (auto-detect)
    n_em_iter : number of EM iterations
    gen_since_admix : initial guess for T
    batch_size : haplotypes per forward-backward batch
    rng_seed : random seed
    bandwidth_cm : Gaussian kernel bandwidth for rare-variant frequency smoothing.
                   Set to 0 to disable.
    maf_threshold : MAF threshold below which frequencies are smoothed.

    Returns
    -------
    AncestryResult with posteriors, calls, and fitted model
    """
    geno_np = chrom_data.geno
    d_morgan = chrom_data.genetic_distances.astype(np.float64)

    log.info("=== EM on chromosome %s: %d haps × %d sites ===",
             chrom_data.chrom, chrom_data.n_haps, chrom_data.n_sites)

    # --- Stage 0: Spectral seed (soft GMM) ---
    log.info("Stage 0: Spectral initialization (GMM)")
    if stats is not None:
        stats.timer_start("spectral")
    labels, responsibilities, n_anc = seed_ancestry_soft(
        geno_np, n_ancestries=n_ancestries, rng_seed=rng_seed, stats=stats,
    )
    if stats is not None:
        stats.timer_stop("spectral", chrom=chrom_data.chrom)

    # Transfer to device
    geno = jnp.array(geno_np)
    d_morgan_j = jnp.array(d_morgan)
    pos_cm_j = jnp.array(chrom_data.pos_cm.astype(np.float32))

    # --- Stage 1: Init model from soft assignments + window refinement ---
    log.info("Stage 1: Initializing model from soft assignments")
    model = init_model_soft(geno, responsibilities, n_anc, gen_since_admix)
    log.info("  mu = %s", np.array(model.mu).round(3))
    log.info("  T = %.1f generations", model.gen_since_admix)

    # --- Optional: block emission setup ---
    bd = None
    if use_block_emissions:
        from .blocks import pack_blocks, init_pattern_freq, update_pattern_freq, expand_block_posteriors
        bd = pack_blocks(geno_np, block_size=block_size, pos_cm=chrom_data.pos_cm)
        log.info("  Block emissions: %d blocks of %d SNPs, %d max patterns",
                 bd.n_blocks, block_size, bd.max_patterns)
        pf = init_pattern_freq(model.allele_freq, bd, geno_np)
        model = AncestryModel(
            n_ancestries=model.n_ancestries, mu=model.mu,
            gen_since_admix=model.gen_since_admix, allele_freq=model.allele_freq,
            mismatch=model.mismatch, pattern_freq=pf, block_data=bd,
        )

    # --- Stage 2-3: EM iterations ---
    bucket_centers = compute_bucket_centers(n_T_buckets) if per_hap_T else None
    prev_freq = model.allele_freq
    for iteration in range(n_em_iter):
        log.info("--- EM iteration %d/%d ---", iteration + 1, n_em_iter)

        # E-step: forward-backward (streaming — no full gamma materialised)
        log.info("  E-step: forward-backward on %d haplotypes", chrom_data.n_haps)
        if stats is not None:
            stats.timer_start("e_step")

        gamma_block = None  # only set for block-emissions path
        if bd is not None and model.pattern_freq is not None:
            # Block emissions: small (H, n_blocks, A) — no streaming needed
            gamma_block = forward_backward_blocks(model, bd)
            gamma = expand_block_posteriors(gamma_block, bd, chrom_data.n_sites)
            # Build EMStats from full gamma for this (small) path
            geno_f = geno.astype(jnp.float32)
            calls_tmp = jnp.argmax(gamma, axis=2)
            switches_tmp = (calls_tmp[:, 1:] != calls_tmp[:, :-1]) if gamma.shape[1] > 1 else jnp.zeros((gamma.shape[0], 0), dtype=bool)
            em_stats = EMStats(
                weighted_counts=jnp.einsum('hta,ht->at', gamma, geno_f),
                total_weights=gamma.sum(axis=0).T,
                mu_sum=gamma.sum(axis=(0, 1)),
                switch_sum=switches_tmp.sum(axis=0).astype(jnp.float32),
                switches_per_hap=np.array(switches_tmp.sum(axis=1), dtype=np.int32),
                n_haps=gamma.shape[0],
                n_sites=gamma.shape[1],
            )
            del gamma  # free the expanded block gamma
        elif model.bucket_assignments is not None:
            em_stats = forward_backward_bucketed_em(geno, model, d_morgan_j, batch_size)
        else:
            em_stats = forward_backward_em(geno, model, d_morgan_j, batch_size)

        if stats is not None:
            stats.timer_stop("e_step", chrom=chrom_data.chrom, iteration=iteration)

        # M-step: update parameters from streaming stats
        log.info("  M-step: updating parameters")
        if stats is not None:
            stats.timer_start("m_step")
        new_freq = update_allele_freq_from_stats(em_stats)
        if bandwidth_cm > 0:
            new_freq = smooth_rare_frequencies(
                new_freq, pos_cm_j, bandwidth_cm, maf_threshold,
            )
        new_mu = update_mu_from_stats(em_stats)

        # Hold T fixed for first iteration to let frequencies stabilize
        T_per_hap = None
        bucket_assignments = None
        if iteration == 0:
            new_T = model.gen_since_admix
            log.info("  (holding T fixed for first iteration)")
        elif per_hap_T and bucket_centers is not None:
            T_per_hap, bucket_assignments, new_T = update_generations_per_hap_from_stats(
                em_stats, d_morgan_j, model.gen_since_admix, new_mu, bucket_centers,
            )
            log.info("  Per-hap T: mean=%.1f, std=%.1f",
                     float(jnp.mean(T_per_hap)), float(jnp.std(T_per_hap)))
        else:
            new_T = update_generations_from_stats(em_stats, d_morgan_j, model.gen_since_admix, model.mu)

        if stats is not None:
            stats.timer_stop("m_step", chrom=chrom_data.chrom, iteration=iteration)

        # Update pattern frequencies if using block emissions
        new_pf = None
        if bd is not None and use_block_emissions:
            new_pf = update_pattern_freq(bd, gamma_block)

        model = AncestryModel(
            n_ancestries=n_anc,
            mu=new_mu,
            gen_since_admix=new_T,
            allele_freq=new_freq,
            mismatch=model.mismatch,
            gen_per_hap=T_per_hap,
            bucket_centers=bucket_centers,
            bucket_assignments=bucket_assignments,
            pattern_freq=new_pf,
            block_data=bd,
        )

        log.info("  mu = %s", np.array(model.mu).round(3))
        log.info("  T = %.1f generations", model.gen_since_admix)

        # Check convergence
        max_delta = float(jnp.abs(new_freq - prev_freq).max())
        mean_delta = float(jnp.abs(new_freq - prev_freq).mean())
        log.info("  max Δ(freq) = %.6f, mean Δ(freq) = %.6f", max_delta, mean_delta)

        if stats is not None:
            stats.emit("em/max_delta_freq", max_delta,
                       chrom=chrom_data.chrom, iteration=iteration)
            stats.emit("em/mean_delta_freq", mean_delta,
                       chrom=chrom_data.chrom, iteration=iteration)
            stats.emit("em/mu", np.array(new_mu).tolist(),
                       chrom=chrom_data.chrom, iteration=iteration)
            stats.emit("em/T", float(new_T),
                       chrom=chrom_data.chrom, iteration=iteration)
        if iteration > 0 and max_delta < 1e-4:
            log.info("  Converged.")
            break
        prev_freq = new_freq

    # --- Final decode (streaming — no full gamma materialised) ---
    log.info("Final forward-backward pass")
    if bd is not None and model.pattern_freq is not None:
        # Block emissions: small tensor, use legacy path
        gamma_block = forward_backward_blocks(model, bd)
        gamma = expand_block_posteriors(gamma_block, bd, chrom_data.n_sites)
        calls = np.array(jnp.argmax(gamma, axis=2), dtype=np.int8)
        decode = DecodeResult(
            calls=calls,
            max_post=np.array(gamma.max(axis=2)),
            global_sums=np.array(gamma.sum(axis=1)),
        )
        result = AncestryResult(
            calls=calls, model=model, chrom=chrom_data.chrom,
            decode=decode, posteriors=gamma,
        )
    else:
        if model.bucket_assignments is not None:
            decode = forward_backward_bucketed_decode(
                geno, model, d_morgan_j, batch_size,
            )
        else:
            decode = forward_backward_decode(
                geno, model, d_morgan_j, batch_size,
            )
        result = AncestryResult(
            calls=decode.calls, model=model, chrom=chrom_data.chrom,
            decode=decode,
        )

    # Summary stats
    for a in range(n_anc):
        prop = float((result.calls == a).mean())
        log.info("  Ancestry %d: %.1f%% of genome", a, 100 * prop)
        if stats is not None:
            stats.emit("em/ancestry_proportion", prop,
                       chrom=chrom_data.chrom, tags={"ancestry": a})

    return result


# ---------------------------------------------------------------------------
# Multi-chromosome wrapper
# ---------------------------------------------------------------------------

def run_em_genome(
    chrom_iter,
    n_ancestries: Optional[int] = None,
    n_em_iter: int = 3,
    gen_since_admix: float = 20.0,
    batch_size: int = 50_000,
    rng_seed: int = 42,
    seed_chrom: Optional[str] = None,
    stats=None,
    bandwidth_cm: float = 0.05,
    maf_threshold: float = 0.05,
    per_hap_T: bool = False,
    n_T_buckets: int = 20,
    use_block_emissions: bool = False,
    block_size: int = 8,
) -> list[AncestryResult]:
    """Run self-bootstrapping LAI across all chromosomes.

    Strategy (following FLARE):
        1. Run full EM on one chromosome to estimate parameters.
        2. Use those parameters as initialization for remaining chromosomes,
           with optional 1-iteration refinement.

    Parameters
    ----------
    chrom_iter : iterator yielding ChromData
    seed_chrom : which chromosome to use for initial parameter estimation.
                 If None, uses the first one.

    Returns
    -------
    List of AncestryResult, one per chromosome.
    """
    results = []
    fitted_model = None

    for chrom_data in chrom_iter:
        if stats is not None:
            stats.timer_start(f"chrom/{chrom_data.chrom}")

        if fitted_model is None:
            # First chromosome: full EM
            log.info("=== Seed chromosome: %s (full EM) ===", chrom_data.chrom)
            result = run_em(
                chrom_data,
                n_ancestries=n_ancestries,
                n_em_iter=n_em_iter,
                gen_since_admix=gen_since_admix,
                batch_size=batch_size,
                rng_seed=rng_seed,
                stats=stats,
                bandwidth_cm=bandwidth_cm,
                maf_threshold=maf_threshold,
                per_hap_T=per_hap_T,
                n_T_buckets=n_T_buckets,
                use_block_emissions=use_block_emissions,
                block_size=block_size,
            )
            fitted_model = result.model
        else:
            # Subsequent chromosomes: use fitted params, 1 iteration to adapt
            log.info("=== Chromosome %s (warm-started, 1 iter) ===", chrom_data.chrom)

            geno = jnp.array(chrom_data.geno)
            d_morgan_j = jnp.array(chrom_data.genetic_distances)

            # Quick soft init for this chromosome's allele frequencies
            _labels, resp, n_anc = seed_ancestry_soft(
                chrom_data.geno,
                n_ancestries=fitted_model.n_ancestries,
                rng_seed=rng_seed,
            )
            model = init_model_soft(
                geno, resp, fitted_model.n_ancestries,
                fitted_model.gen_since_admix,
            )
            # Override mu and T from the fitted model (including per-hap T)
            model = AncestryModel(
                n_ancestries=fitted_model.n_ancestries,
                mu=fitted_model.mu,
                gen_since_admix=fitted_model.gen_since_admix,
                allele_freq=model.allele_freq,
                mismatch=fitted_model.mismatch,
                gen_per_hap=fitted_model.gen_per_hap,
                bucket_centers=fitted_model.bucket_centers,
                bucket_assignments=fitted_model.bucket_assignments,
            )

            # One EM iteration (streaming — no full gamma)
            if model.bucket_assignments is not None:
                em_stats = forward_backward_bucketed_em(geno, model, d_morgan_j, batch_size)
            else:
                em_stats = forward_backward_em(geno, model, d_morgan_j, batch_size)
            new_freq = update_allele_freq_from_stats(em_stats)
            if bandwidth_cm > 0:
                pos_cm_j = jnp.array(chrom_data.pos_cm.astype(np.float32))
                new_freq = smooth_rare_frequencies(
                    new_freq, pos_cm_j, bandwidth_cm, maf_threshold,
                )
            model = AncestryModel(
                n_ancestries=model.n_ancestries,
                mu=model.mu,
                gen_since_admix=model.gen_since_admix,
                allele_freq=new_freq,
                mismatch=model.mismatch,
                gen_per_hap=model.gen_per_hap,
                bucket_centers=model.bucket_centers,
                bucket_assignments=model.bucket_assignments,
            )

            # Final decode (streaming)
            if model.bucket_assignments is not None:
                decode = forward_backward_bucketed_decode(
                    geno, model, d_morgan_j, batch_size,
                )
            else:
                decode = forward_backward_decode(
                    geno, model, d_morgan_j, batch_size,
                )
            result = AncestryResult(
                calls=decode.calls, model=model, chrom=chrom_data.chrom,
                decode=decode,
            )

        if stats is not None:
            elapsed = stats.timer_stop(f"chrom/{chrom_data.chrom}", chrom=chrom_data.chrom)
            throughput = chrom_data.n_haps * chrom_data.n_sites / max(elapsed, 1e-6)
            stats.emit("runtime/throughput", round(throughput), chrom=chrom_data.chrom)

        results.append(result)

    return results
