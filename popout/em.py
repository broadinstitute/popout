"""Expectation-Maximization loop for self-bootstrapping ancestry inference.

The model parameters (allele frequencies, ancestry proportions, admixture
time) are iteratively refined.  With 500K+ samples, sufficient statistics
converge fast — typically 2-3 EM iterations suffice.
"""

from __future__ import annotations

import gc
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
    forward_backward_blocks_batched,
    forward_backward_em,
    forward_backward_bucketed_em,
    forward_backward_decode,
    forward_backward_bucketed_decode,
)
from .spectral import seed_ancestry, seed_ancestry_soft, window_init_allele_freq

log = logging.getLogger(__name__)


def _auto_batch_size(
    T: int, A: int, user_batch_size: int | None,
    H: int = 0, bucketed: bool = False,
) -> int:
    """Pick batch_size to fit GPU memory.

    Peak GPU for one forward-backward batch is ~3 × B × T × A × 4 bytes
    (three simultaneous (B, T, A) float32 tensors during log_emission).
    The full genotype array (H × T uint8) also lives on GPU throughout,
    so we subtract that from the available budget.

    When bucketed=True (per-haplotype T), apply 0.6× factor to account for
    BFC fragmentation from variable-size allocations across 20 buckets.
    """
    if user_batch_size is not None:
        return user_batch_size

    # Detect GPU memory; fall back to a conservative 16 GB estimate
    try:
        gpu_mem = jax.devices()[0].memory_stats()["bytes_limit"]
    except Exception:
        gpu_mem = 16 * 1024**3

    # Reserve memory for the full genotype array on GPU + overhead
    geno_bytes = H * T  # uint8
    budget = int(gpu_mem * 0.5) - geno_bytes  # 50% safety + geno reservation
    budget = max(budget, 512 * 1024 * 1024)   # floor at 512 MB
    batch_size = budget // (3 * T * A * 4)
    batch_size = max(256, min(batch_size, 50_000))
    if bucketed:
        batch_size = max(256, int(batch_size * 0.6))
    return batch_size


def _auto_batch_size_blocks(
    n_blocks: int, A: int, H: int,
    target_bytes: int = 1 * 1024**3,
) -> int:
    """Pick batch size for block-emissions forward-backward.

    Peak memory per haplotype is approximately n_blocks × A × 4 bytes
    for the emission tensor. Working memory during forward-backward is
    2-3× the output, so target 1 GB output → 2-4 GB working peak.
    """
    bytes_per_hap = n_blocks * A * 4
    batch = max(1000, target_bytes // max(bytes_per_hap, 1))
    return int(min(batch, H))


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
    mu = stats.mu_sum / (float(stats.n_haps) * float(stats.n_sites))
    return mu / mu.sum()


def update_generations_from_stats(
    stats: EMStats,
    d_morgan: jnp.ndarray,
    current_T: float,
    mu: jnp.ndarray,
) -> float:
    """Estimate generations since admixture from pre-accumulated stats.

    Uses xi-based soft switches (density-invariant) when available,
    falling back to hard-call switches otherwise.
    """
    p_diff_ancestry = 1.0 - (mu ** 2).sum()
    p_diff_ancestry = float(jnp.maximum(p_diff_ancestry, 0.1))

    d = jnp.maximum(d_morgan, 1e-10)
    total_distance = float(d.sum())

    # Use soft switches (density-invariant) from xi posteriors
    mean_soft_sw = float(jnp.mean(jnp.array(stats.soft_switches_per_hap)))
    T_est = mean_soft_sw / (total_distance * p_diff_ancestry + 1e-10)
    T_est = max(1.0, min(T_est, 1000.0))

    # Log-space blend: T_new = T_old^(1-α) · T_est^α
    # Conservative α keeps T from moving too fast in a single update.
    alpha = 0.3
    log_T = (1 - alpha) * np.log(max(current_T, 1.0)) + alpha * np.log(max(T_est, 1.0))
    return max(1.0, min(float(np.exp(log_T)), 1000.0))


def update_generations_per_hap_from_stats(
    stats: EMStats,
    d_morgan: jnp.ndarray,
    current_T_global: float,
    mu: jnp.ndarray,
    bucket_centers: jnp.ndarray,
    min_switches_for_confidence: float = 3.0,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Per-haplotype T estimation from xi-based soft switch counts.

    Uses density-invariant expected transition counts instead of
    hard-call argmax switches.
    """
    soft_sw = jnp.array(stats.soft_switches_per_hap, dtype=jnp.float32)

    p_diff = 1.0 - (mu ** 2).sum()
    p_diff = jnp.maximum(p_diff, 0.1)
    total_d = d_morgan.sum()

    T_raw = soft_sw / (total_d * p_diff + 1e-10)
    T_raw = jnp.clip(T_raw, 1.0, 1000.0)

    # Regularization: shrink toward global mean when few expected switches
    lam = 1.0 / (1.0 + soft_sw / min_switches_for_confidence)
    T_shrunk = (1.0 - lam) * T_raw + lam * current_T_global

    # Log-space blend toward previous global T — conservative step size
    alpha = 0.3
    log_current = jnp.log(jnp.maximum(current_T_global, 1.0))
    log_shrunk = jnp.log(jnp.maximum(T_shrunk, 1.0))
    T_reg = jnp.exp((1 - alpha) * log_current + alpha * log_shrunk)
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

    return AncestryModel(
        n_ancestries=A,
        mu=mu,
        gen_since_admix=gen_since_admix,
        allele_freq=freq,
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
    resp = responsibilities  # (H, A)

    # Global allele frequencies from soft assignments
    # resp.T @ geno → (A, T) weighted allele counts
    # Batched to avoid float32 copy of full geno (H×T×4 bytes)
    _INIT_BATCH = 50_000
    weighted_counts = jnp.zeros((A, T))
    for start in range(0, H, _INIT_BATCH):
        end = min(start + _INIT_BATCH, H)
        batch_geno = geno[start:end].astype(jnp.float32)
        weighted_counts += resp[start:end].T @ batch_geno
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

    return AncestryModel(
        n_ancestries=A,
        mu=mu,
        gen_since_admix=gen_since_admix,
        allele_freq=freq,
    )


# ---------------------------------------------------------------------------
# Main EM loop
# ---------------------------------------------------------------------------

def run_em(
    chrom_data: ChromData,
    n_ancestries: Optional[int] = None,
    n_em_iter: int = 5,
    gen_since_admix: float = 20.0,
    batch_size: int | None = None,
    rng_seed: int = 42,
    stats=None,
    per_hap_T: bool = False,
    n_T_buckets: int = 20,
    use_block_emissions: bool = False,
    block_size: int = 8,
    detection_method: str = "marchenko-pastur",
    max_ancestries: int = 20,
    seed_responsibilities: Optional[jnp.ndarray] = None,
    freeze_anchors_iters: int = 0,
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
    batch_size : haplotypes per forward-backward batch (None = auto-tune)
    rng_seed : random seed
    detection_method : ancestry auto-detection method
    max_ancestries : upper bound for auto-detected ancestry count

    Returns
    -------
    AncestryResult with posteriors, calls, and fitted model
    """
    geno_np = chrom_data.geno
    d_morgan = chrom_data.genetic_distances.astype(np.float64)

    log.info("=== EM on chromosome %s: %d haps × %d sites ===",
             chrom_data.chrom, chrom_data.n_haps, chrom_data.n_sites)

    # --- Stage 0: Seed ---
    if seed_responsibilities is not None:
        log.info("Stage 0: Using pre-computed seed responsibilities")
        responsibilities = seed_responsibilities
        n_anc = responsibilities.shape[1]
        labels = jnp.argmax(responsibilities, axis=1).astype(jnp.int32)
        pca_proj = None
    else:
        log.info("Stage 0: Spectral initialization (GMM)")
        if stats is not None:
            stats.timer_start("spectral")
        labels, responsibilities, n_anc, pca_proj = seed_ancestry_soft(
            geno_np, n_ancestries=n_ancestries, rng_seed=rng_seed, stats=stats,
            detection_method=detection_method, max_ancestries=max_ancestries,
        )
        if stats is not None:
            stats.timer_stop("spectral", chrom=chrom_data.chrom)

    # Auto-tune batch_size now that we know A
    batch_size = _auto_batch_size(
        chrom_data.n_sites, n_anc, batch_size, H=chrom_data.n_haps,
    )
    log.info("  batch_size = %d (T=%d, A=%d, H=%d)",
             batch_size, chrom_data.n_sites, n_anc, chrom_data.n_haps)

    # Transfer to device
    geno = jnp.array(geno_np)
    d_morgan_j = jnp.array(d_morgan)

    # --- Stage 1: Init model from soft assignments + window refinement ---
    log.info("Stage 1: Initializing model from soft assignments")
    model = init_model_soft(geno, responsibilities, n_anc, gen_since_admix)
    log.info("  mu = %s", np.array(model.mu).round(3))
    log.info("  T = %.1f generations", model.gen_since_admix)

    # --- Optional: block emission setup ---
    bd = None
    if use_block_emissions:
        from .blocks import pack_blocks, init_pattern_freq, update_pattern_freq, expand_block_posteriors
        log.info("  [diag] calling pack_blocks (H=%d, T=%d, block_size=%d)",
                 geno_np.shape[0], geno_np.shape[1], block_size)
        bd = pack_blocks(geno_np, block_size=block_size, pos_cm=chrom_data.pos_cm)
        log.info("  [diag] pack_blocks returned")
        log.info("  Block emissions: %d blocks of %d SNPs, %d max patterns",
                 bd.n_blocks, block_size, bd.max_patterns)
        log.info("  [diag] calling init_pattern_freq")
        pf = init_pattern_freq(model.allele_freq, bd, geno_np)
        log.info("  [diag] init_pattern_freq returned, building AncestryModel")
        model = AncestryModel(
            n_ancestries=model.n_ancestries, mu=model.mu,
            gen_since_admix=model.gen_since_admix, allele_freq=model.allele_freq,
            pattern_freq=pf, block_data=bd,
        )
        log.info("  [diag] AncestryModel with block_data built")

    # --- Stage 2-3: EM iterations ---
    bucket_centers = compute_bucket_centers(n_T_buckets) if per_hap_T else None
    prev_freq = model.allele_freq
    prev_T = model.gen_since_admix

    for iteration in range(n_em_iter):
        log.info("--- EM iteration %d/%d ---", iteration + 1, n_em_iter)

        # E-step: forward-backward (streaming — no full gamma materialised)
        log.info("  E-step: forward-backward on %d haplotypes", chrom_data.n_haps)
        if stats is not None:
            stats.timer_start("e_step")

        gamma_block = None  # only set for block-emissions path
        _pf_counts = None   # accumulated pattern-freq counts for M-step
        if bd is not None and model.pattern_freq is not None:
            # Block emissions: chunked E-step consuming block-level gamma
            # directly. No per-site expansion — all M-step reductions are
            # computed from the (chunk, n_blocks, A) tensor.
            from .blocks import BlockData
            block_batch = _auto_batch_size_blocks(
                bd.n_blocks, n_anc, chrom_data.n_haps,
            )
            log.info("  Block E-step: batch_size=%d (n_blocks=%d, A=%d, H=%d)",
                     block_batch, bd.n_blocks, n_anc, chrom_data.n_haps)
            T_sites = chrom_data.n_sites
            H_total = chrom_data.n_haps

            # Pre-compute block geometry (tiny arrays, computed once)
            b_starts = np.array(bd.block_starts, dtype=np.int64)
            b_ends = np.array(bd.block_ends, dtype=np.int64)
            block_widths_np = b_ends - b_starts
            block_widths_j = jnp.array(block_widths_np, dtype=jnp.float32)

            weighted_counts = jnp.zeros((n_anc, T_sites), dtype=jnp.float32)
            total_weights = jnp.zeros((n_anc, T_sites), dtype=jnp.float32)
            mu_sum = jnp.zeros(n_anc, dtype=jnp.float32)
            switch_sum = jnp.zeros(n_anc, dtype=jnp.float32)
            all_switches_per_hap: list[np.ndarray] = []
            _pf_counts = jnp.zeros((bd.n_blocks, bd.max_patterns, n_anc), dtype=jnp.float32)

            for bs in range(0, H_total, block_batch):
                be = min(bs + block_batch, H_total)
                bd_chunk = BlockData(
                    pattern_indices=bd.pattern_indices[bs:be],
                    block_starts=bd.block_starts,
                    block_ends=bd.block_ends,
                    block_distances=bd.block_distances,
                    pattern_counts=bd.pattern_counts,
                    max_patterns=bd.max_patterns,
                    block_size=bd.block_size,
                )
                # E-step: block-level forward-backward → (chunk, n_blocks, A)
                gb_chunk = forward_backward_blocks(model, bd_chunk)
                geno_chunk = geno[bs:be].astype(jnp.float32)

                # --- M-step accumulators from block gamma (no per-site expansion) ---

                # weighted_counts[a, t]: for sites t in block b,
                # gamma[h,t,a] = gamma_block[h,b,a], so:
                # weighted_counts[a, s:e] += gamma_block[:,b,:].T @ geno[:,s:e]
                for b_idx in range(bd.n_blocks):
                    s, e = int(b_starts[b_idx]), int(b_ends[b_idx])
                    # (A, chunk) @ (chunk, w) → (A, w)
                    weighted_counts = weighted_counts.at[:, s:e].add(
                        gb_chunk[:, b_idx, :].T @ geno_chunk[:, s:e]
                    )

                # total_weights[a, t]: for sites in block b, each site gets
                # the same per-ancestry sum
                for b_idx in range(bd.n_blocks):
                    s, e = int(b_starts[b_idx]), int(b_ends[b_idx])
                    per_anc = gb_chunk[:, b_idx, :].sum(axis=0)  # (A,)
                    total_weights = total_weights.at[:, s:e].add(
                        jnp.broadcast_to(per_anc[:, None], (n_anc, e - s))
                    )

                # mu_sum[a] = sum_h sum_b gamma_block[h,b,a] * width[b]
                mu_sum = mu_sum + jnp.einsum('hba,b->a', gb_chunk, block_widths_j)

                # Switches: only at block boundaries (within-block gamma is constant)
                calls_block = jnp.argmax(gb_chunk, axis=2)  # (chunk, n_blocks)
                if bd.n_blocks > 1:
                    sw_block = (calls_block[:, 1:] != calls_block[:, :-1])  # (chunk, n_blocks-1)
                    all_switches_per_hap.append(
                        np.array(sw_block.sum(axis=1), dtype=np.int32)
                    )
                    # Attribute switches to pre-switch ancestry
                    pre_calls = calls_block[:, :-1]
                    one_hot = jax.nn.one_hot(pre_calls, n_anc, dtype=jnp.float32)
                    switch_sum = switch_sum + (
                        one_hot * sw_block[..., None].astype(jnp.float32)
                    ).sum(axis=(0, 1))
                else:
                    all_switches_per_hap.append(np.zeros(be - bs, dtype=np.int32))

                # Accumulate pattern-freq scatter-add counts
                pat_idx_chunk = jnp.array(bd_chunk.pattern_indices)
                for b_idx in range(bd.n_blocks):
                    _pf_counts = _pf_counts.at[b_idx].add(
                        jnp.zeros((bd.max_patterns, n_anc)).at[pat_idx_chunk[:, b_idx]].add(
                            gb_chunk[:, b_idx, :]
                        )
                    )

                weighted_counts.block_until_ready()
                del gb_chunk, geno_chunk, calls_block, pat_idx_chunk

            switches_per_hap = np.concatenate(all_switches_per_hap)
            em_stats = EMStats(
                weighted_counts=weighted_counts,
                total_weights=total_weights,
                mu_sum=mu_sum,
                switch_sum=switch_sum,
                switches_per_hap=switches_per_hap,
                soft_switches_per_hap=switches_per_hap.astype(np.float32),
                n_haps=H_total,
                n_sites=T_sites,
            )
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
        new_mu = update_mu_from_stats(em_stats)

        # Anchor freezing: override frequencies with seed-derived values
        if (seed_responsibilities is not None
                and freeze_anchors_iters > 0
                and iteration < freeze_anchors_iters):
            log.info("  Anchor freeze: overriding frequencies (iter %d/%d)",
                     iteration + 1, freeze_anchors_iters)
            _FREEZE_BATCH = 50_000
            H_f, T_f = geno.shape
            A_f = seed_responsibilities.shape[1]
            frozen_wc = jnp.zeros((A_f, T_f))
            for fs in range(0, H_f, _FREEZE_BATCH):
                fe = min(fs + _FREEZE_BATCH, H_f)
                frozen_wc += seed_responsibilities[fs:fe].T @ geno[fs:fe].astype(jnp.float32)
            frozen_totals = seed_responsibilities.sum(axis=0)[:, None]
            new_freq = (frozen_wc + 0.5) / (frozen_totals + 1.0)

        # T is held fixed during iteration 0 so frequencies can stabilize from
        # the spectral init before the switch-rate estimator kicks in. From
        # iteration 1 onward, T is updated every iteration alongside mu and freq.
        T_per_hap = None
        bucket_assignments = None
        should_update_T = iteration > 0

        if not should_update_T:
            new_T = model.gen_since_admix
        elif per_hap_T and bucket_centers is not None:
            T_per_hap, bucket_assignments, new_T = update_generations_per_hap_from_stats(
                em_stats, d_morgan_j, model.gen_since_admix, new_mu, bucket_centers,
            )
            log.info("  T (per-hap): mean=%.1f, std=%.1f",
                     float(jnp.mean(T_per_hap)), float(jnp.std(T_per_hap)))
        else:
            new_T = update_generations_from_stats(em_stats, d_morgan_j, model.gen_since_admix, model.mu)
            log.info("  T: %.1f → %.1f", model.gen_since_admix, new_T)

        if stats is not None:
            stats.timer_stop("m_step", chrom=chrom_data.chrom, iteration=iteration)

        # Update pattern frequencies if using block emissions
        new_pf = None
        if bd is not None and use_block_emissions:
            if _pf_counts is not None:
                # Normalize pre-accumulated scatter-add counts
                pseudocount_pf = 0.01
                new_pf = jnp.full_like(_pf_counts, pseudocount_pf)
                for b_idx in range(bd.n_blocks):
                    n_p = int(bd.pattern_counts[b_idx])
                    total = _pf_counts[b_idx, :n_p, :].sum(axis=0, keepdims=True)
                    new_pf = new_pf.at[b_idx, :n_p, :].set(
                        (_pf_counts[b_idx, :n_p, :] + pseudocount_pf) / (total + pseudocount_pf * n_p)
                    )
            else:
                new_pf = update_pattern_freq(bd, gamma_block)

        model = AncestryModel(
            n_ancestries=n_anc,
            mu=new_mu,
            gen_since_admix=new_T,
            allele_freq=new_freq,
            gen_per_hap=T_per_hap,
            bucket_centers=bucket_centers,
            bucket_assignments=bucket_assignments,
            pattern_freq=new_pf,
            block_data=bd,
        )

        # Re-tune batch_size when entering bucketed path for first time
        if bucket_assignments is not None and iteration == 1:
            batch_size = _auto_batch_size(
                chrom_data.n_sites, n_anc, None,
                H=chrom_data.n_haps, bucketed=True,
            )
            log.info("  Re-tuned batch_size for bucketed path: %d", batch_size)

        # Free M-step intermediates; force cyclic GC so BFC can coalesce
        del em_stats
        gc.collect()

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
        T_delta = abs(float(new_T) - float(prev_T)) / max(float(prev_T), 1.0)
        freq_converged = mean_delta < 1e-4
        T_converged = T_delta < 0.01  # <1% relative change
        if iteration > 0 and freq_converged and T_converged:
            log.info("  Converged (mean Δfreq=%.2e, ΔT=%.1f%%)", mean_delta, T_delta * 100)
            break
        prev_freq = new_freq
        prev_T = new_T

    # --- Final decode (streaming — no full gamma materialised) ---
    log.info("Final forward-backward pass")
    if bd is not None and model.pattern_freq is not None:
        # Block emissions: batched decode
        from .blocks import BlockData
        block_batch = _auto_batch_size_blocks(
            bd.n_blocks, n_anc, chrom_data.n_haps,
        )
        all_calls = []
        all_max_post = []
        all_global_sums = []
        for bs in range(0, chrom_data.n_haps, block_batch):
            be = min(bs + block_batch, chrom_data.n_haps)
            bd_chunk = BlockData(
                pattern_indices=bd.pattern_indices[bs:be],
                block_starts=bd.block_starts,
                block_ends=bd.block_ends,
                block_distances=bd.block_distances,
                pattern_counts=bd.pattern_counts,
                max_patterns=bd.max_patterns,
                block_size=bd.block_size,
            )
            gb_chunk = forward_backward_blocks(model, bd_chunk)
            gamma_chunk = expand_block_posteriors(gb_chunk, bd, chrom_data.n_sites)
            all_calls.append(np.array(jnp.argmax(gamma_chunk, axis=2), dtype=np.int8))
            all_max_post.append(np.array(gamma_chunk.max(axis=2)))
            all_global_sums.append(np.array(gamma_chunk.sum(axis=1)))
            del gamma_chunk, gb_chunk
        calls = np.concatenate(all_calls, axis=0)
        decode = DecodeResult(
            calls=calls,
            max_post=np.concatenate(all_max_post, axis=0),
            global_sums=np.concatenate(all_global_sums, axis=0),
        )
        result = AncestryResult(
            calls=calls, model=model, chrom=chrom_data.chrom,
            decode=decode,
            spectral={"pca_proj": pca_proj, "gmm_labels": np.array(labels)} if pca_proj is not None else None,
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
            spectral={"pca_proj": pca_proj, "gmm_labels": np.array(labels)},
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
    n_em_iter: int = 5,
    gen_since_admix: float = 20.0,
    batch_size: int | None = None,
    rng_seed: int = 42,
    seed_chrom: Optional[str] = None,
    stats=None,
    per_hap_T: bool = False,
    n_T_buckets: int = 20,
    use_block_emissions: bool = False,
    block_size: int = 8,
    detection_method: str = "marchenko-pastur",
    max_ancestries: int = 20,
    seed_method: str = "gmm",
    recursive_kwargs: Optional[dict] = None,
    freeze_anchors_iters: int = 0,
    out_prefix: Optional[str] = None,
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

            # Recursive seeding: run before run_em, pass responsibilities in
            seed_resp = None
            em_n_ancestries = n_ancestries
            if seed_method == "recursive":
                from .recursive_seed import recursive_split_seed
                rkw = recursive_kwargs or {}
                if out_prefix is not None and "dump_pre_merge_path" not in rkw:
                    rkw["dump_pre_merge_path"] = f"{out_prefix}.recursive_pre_merge"
                leaf_labels, leaf_info = recursive_split_seed(
                    chrom_data.geno,
                    chrom_data=chrom_data,
                    gen_since_admix=gen_since_admix,
                    rng_seed=rng_seed,
                    stats=stats,
                    **rkw,
                )
                n_leaves = len(leaf_info)
                # Convert to one-hot responsibilities
                seed_resp = jnp.zeros((chrom_data.n_haps, n_leaves), dtype=jnp.float32)
                seed_resp = seed_resp.at[
                    jnp.arange(chrom_data.n_haps), jnp.array(leaf_labels)
                ].set(1.0)
                em_n_ancestries = n_leaves

            result = run_em(
                chrom_data,
                n_ancestries=em_n_ancestries,
                n_em_iter=n_em_iter,
                gen_since_admix=gen_since_admix,
                batch_size=batch_size,
                rng_seed=rng_seed,
                stats=stats,
                per_hap_T=per_hap_T,
                n_T_buckets=n_T_buckets,
                use_block_emissions=use_block_emissions,
                block_size=block_size,
                detection_method=detection_method,
                max_ancestries=max_ancestries,
                seed_responsibilities=seed_resp,
                freeze_anchors_iters=freeze_anchors_iters,
            )
            fitted_model = result.model
        else:
            # Subsequent chromosomes: use fitted params, 1 iteration to adapt
            log.info("=== Chromosome %s (warm-started, 1 iter) ===", chrom_data.chrom)

            geno = jnp.array(chrom_data.geno)
            d_morgan_j = jnp.array(chrom_data.genetic_distances)

            # Quick soft init for this chromosome's allele frequencies
            _labels, resp, n_anc, _proj = seed_ancestry_soft(
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
                gen_per_hap=fitted_model.gen_per_hap,
                bucket_centers=fitted_model.bucket_centers,
                bucket_assignments=fitted_model.bucket_assignments,
            )

            # Auto-tune batch_size for this chromosome's site count
            chrom_batch = _auto_batch_size(
                chrom_data.n_sites, fitted_model.n_ancestries, batch_size,
                H=chrom_data.n_haps,
            )

            # One EM iteration (streaming — no full gamma)
            if model.bucket_assignments is not None:
                em_stats = forward_backward_bucketed_em(geno, model, d_morgan_j, chrom_batch)
            else:
                em_stats = forward_backward_em(geno, model, d_morgan_j, chrom_batch)
            new_freq = update_allele_freq_from_stats(em_stats)
            model = AncestryModel(
                n_ancestries=model.n_ancestries,
                mu=model.mu,
                gen_since_admix=model.gen_since_admix,
                allele_freq=new_freq,
                gen_per_hap=model.gen_per_hap,
                bucket_centers=model.bucket_centers,
                bucket_assignments=model.bucket_assignments,
            )

            # Final decode (streaming)
            if model.bucket_assignments is not None:
                decode = forward_backward_bucketed_decode(
                    geno, model, d_morgan_j, chrom_batch,
                )
            else:
                decode = forward_backward_decode(
                    geno, model, d_morgan_j, chrom_batch,
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
