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


def init_model_soft(
    geno,
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

    Notes
    -----
    When ``window_refine=True``, frequencies are refined via per-window
    local ancestry reassignment, streaming over haplotypes in batches
    (default 50k).  Safe at biobank scale without full geno on device.
    """
    H, T = geno.shape
    A = n_ancestries
    resp = responsibilities  # (H, A)

    # Global allele frequencies from soft assignments
    # resp.T @ geno → (A, T) weighted allele counts
    # Batched to avoid float32 copy of full geno (H×T×4 bytes)
    _INIT_BATCH = 20_000
    weighted_counts = jnp.zeros((A, T))
    for start in range(0, H, _INIT_BATCH):
        end = min(start + _INIT_BATCH, H)
        batch_geno = jnp.asarray(geno[start:end]).astype(jnp.float32)
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
    checkpoint_after_em: Optional[str] = None,
    ancestry_names: Optional[list[str]] = None,
    write_dense_decode: bool = False,
    decode_parquet_path: Optional[str] = None,
    skip_decode: bool = False,
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
    from ._device import fits_on_device
    if fits_on_device(geno_np.nbytes):
        geno = jnp.array(geno_np)
        log.info("  geno %.1f GB → device-resident", geno_np.nbytes / 1e9)
    else:
        geno = geno_np
        log.info(
            "  geno %.1f GB > device budget → host-resident, batched transfers",
            geno_np.nbytes / 1e9,
        )
    d_morgan_j = jnp.array(d_morgan)

    # --- Stage 1: Init model from soft assignments + window refinement ---
    log.info("Stage 1: Initializing model from soft assignments")
    model = init_model_soft(geno, responsibilities, n_anc, gen_since_admix)
    log.info("  mu = %s", np.array(model.mu).round(3))
    log.info("  T = %.1f generations", model.gen_since_admix)

    # --- Optional: block emission setup ---
    bd = None
    if use_block_emissions:
        from .blocks import pack_blocks, init_pattern_freq, update_pattern_freq
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
            soft_switches_per_hap = np.zeros(H_total, dtype=np.float32)
            hard_switches_per_hap = np.zeros(H_total, dtype=np.int32)
            _pf_counts = jnp.zeros((bd.n_blocks, bd.max_patterns, n_anc), dtype=jnp.float32)

            # Pre-compute block geometry for vectorized M-step:
            # Pad site dimension to exact multiple of block_size so we can
            # reshape (chunk, T_padded) → (chunk, n_blocks, block_size)
            # and replace Python loops with a single einsum + broadcast.
            T_padded = bd.n_blocks * bd.block_size
            pad_right = T_padded - T_sites

            # Pattern-freq scatter helper (vmapped over blocks)
            _max_p, _n_anc = bd.max_patterns, n_anc
            def _scatter_block(pat_col, gb_col):
                return jnp.zeros((_max_p, _n_anc)).at[pat_col].add(gb_col)

            # Bucket dispatch: when bucket_assignments is set, run one
            # forward-backward per bucket against a per-bucket model
            # carrying that bucket's center as gen_since_admix. Otherwise
            # run once over the full cohort with the parent model.
            if model.bucket_assignments is not None:
                bucket_centers_np = np.array(model.bucket_centers)
                bucket_np = np.array(model.bucket_assignments)
                B = len(bucket_centers_np)
                assert bucket_np.shape == (H_total,), (
                    f"bucket_assignments shape {bucket_np.shape} != H={H_total}"
                )
                ba_min, ba_max = int(bucket_np.min()), int(bucket_np.max())
                assert 0 <= ba_min and ba_max < B, (
                    f"bucket_assignments range [{ba_min}, {ba_max}] outside [0, {B})"
                )
                bucket_iter = list(range(B))
            else:
                bucket_centers_np = None
                bucket_np = None
                bucket_iter = [None]

            for b in bucket_iter:
                if b is None:
                    b_model = model
                    bucket_hap_idx = None  # signal: contiguous full range
                    n_b = H_total
                else:
                    mask = bucket_np == b
                    n_b = int(mask.sum())
                    if n_b == 0:
                        continue
                    bucket_hap_idx = np.where(mask)[0]
                    b_model = AncestryModel(
                        n_ancestries=model.n_ancestries,
                        mu=model.mu,
                        gen_since_admix=float(bucket_centers_np[b]),
                        allele_freq=model.allele_freq,
                        pattern_freq=model.pattern_freq,
                        block_data=model.block_data,
                    )

                for bs in range(0, n_b, block_batch):
                    be = min(bs + block_batch, n_b)
                    if bucket_hap_idx is None:
                        batch_idx = np.arange(bs, be)
                        pat_idx_np = bd.pattern_indices[bs:be]
                        geno_np = geno[bs:be]
                    else:
                        batch_idx = bucket_hap_idx[bs:be]
                        pat_idx_np = bd.pattern_indices[batch_idx]
                        geno_np = geno[batch_idx]

                    bd_chunk = BlockData(
                        pattern_indices=pat_idx_np,
                        block_starts=bd.block_starts,
                        block_ends=bd.block_ends,
                        block_distances=bd.block_distances,
                        pattern_counts=bd.pattern_counts,
                        max_patterns=bd.max_patterns,
                        block_size=bd.block_size,
                    )
                    # E-step: block-level forward-backward → (chunk, n_blocks, A)
                    # plus density-invariant soft switches (H,) for per-hap-T.
                    gb_chunk, sw_chunk = forward_backward_blocks(
                        b_model, bd_chunk, compute_soft_switches=True,
                    )
                    geno_chunk = jnp.asarray(geno_np, dtype=jnp.float32)

                    # --- M-step accumulators (vectorized, no Python loops) ---

                    # weighted_counts: pad geno to (chunk, n_blocks, block_size),
                    # then einsum with block gamma (chunk, n_blocks, A).
                    if pad_right > 0:
                        geno_blocked = jnp.pad(
                            geno_chunk, ((0, 0), (0, pad_right)),
                        ).reshape(-1, bd.n_blocks, bd.block_size)
                    else:
                        geno_blocked = geno_chunk.reshape(-1, bd.n_blocks, bd.block_size)
                    # (A, n_blocks, block_size) = einsum('hba,hbs->abs', gamma, geno)
                    wc_blocked = jnp.einsum('hba,hbs->abs', gb_chunk, geno_blocked)
                    weighted_counts = weighted_counts + wc_blocked.reshape(n_anc, T_padded)[:, :T_sites]

                    # total_weights: per-ancestry sum per block, broadcast to sites
                    per_anc_all = gb_chunk.sum(axis=0)  # (n_blocks, A)
                    tw_blocked = jnp.broadcast_to(
                        per_anc_all.T[:, :, None],  # (A, n_blocks, 1)
                        (n_anc, bd.n_blocks, bd.block_size),
                    )
                    total_weights = total_weights + tw_blocked.reshape(n_anc, T_padded)[:, :T_sites]

                    # mu_sum[a] = sum_h sum_b gamma_block[h,b,a] * width[b]
                    mu_sum = mu_sum + jnp.einsum('hba,b->a', gb_chunk, block_widths_j)

                    # Soft switches: density-invariant per-haplotype counts
                    # consumed by update_generations_per_hap_from_stats.
                    soft_switches_per_hap[batch_idx] = np.asarray(sw_chunk)

                    # Hard switches: only at block boundaries (within-block
                    # gamma is constant). Used for switch_sum ancestry
                    # attribution and the legacy switches_per_hap field.
                    calls_block = jnp.argmax(gb_chunk, axis=2)  # (chunk, n_blocks)
                    if bd.n_blocks > 1:
                        sw_block = (calls_block[:, 1:] != calls_block[:, :-1])  # (chunk, n_blocks-1)
                        hard_switches_per_hap[batch_idx] = np.array(
                            sw_block.sum(axis=1), dtype=np.int32
                        )
                        # Attribute switches to pre-switch ancestry
                        pre_calls = calls_block[:, :-1]
                        one_hot = jax.nn.one_hot(pre_calls, n_anc, dtype=jnp.float32)
                        switch_sum = switch_sum + (
                            one_hot * sw_block[..., None].astype(jnp.float32)
                        ).sum(axis=(0, 1))

                    # Pattern-freq scatter: vmap over blocks instead of Python loop
                    pat_idx_chunk = jnp.array(bd_chunk.pattern_indices)
                    _pf_counts = _pf_counts + jax.vmap(_scatter_block)(
                        pat_idx_chunk.T,                    # (n_blocks, chunk)
                        jnp.moveaxis(gb_chunk, 1, 0),       # (n_blocks, chunk, A)
                    )

                    weighted_counts.block_until_ready()
                    del gb_chunk, geno_chunk, geno_blocked, calls_block, pat_idx_chunk, sw_chunk

            em_stats = EMStats(
                weighted_counts=weighted_counts,
                total_weights=total_weights,
                mu_sum=mu_sum,
                switch_sum=switch_sum,
                switches_per_hap=hard_switches_per_hap,
                soft_switches_per_hap=soft_switches_per_hap,
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
                # Normalize pre-accumulated scatter-add counts (vectorized)
                pseudocount_pf = 0.01
                pattern_counts_j = jnp.array(bd.pattern_counts)
                p_range = jnp.arange(bd.max_patterns)
                valid_mask = p_range[None, :] < pattern_counts_j[:, None]  # (n_blocks, max_p)
                masked_counts = _pf_counts * valid_mask[:, :, None]
                totals = masked_counts.sum(axis=1, keepdims=True)  # (n_blocks, 1, A)
                n_p_j = pattern_counts_j[:, None, None].astype(jnp.float32)
                normalized = (masked_counts + pseudocount_pf) / (totals + pseudocount_pf * n_p_j)
                new_pf = jnp.where(valid_mask[:, :, None], normalized, pseudocount_pf)
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

    # --- Skip decode if no EM iterations or skip_decode requested ---
    if n_em_iter == 0 or skip_decode:
        log.info("Skipping final decode%s",
                 " (n_em_iter=0)" if n_em_iter == 0 else " (skip_decode)")
        calls = np.zeros((chrom_data.n_haps, chrom_data.n_sites), dtype=np.int8)
        result = AncestryResult(
            calls=calls, model=model, chrom=chrom_data.chrom,
            spectral={"pca_proj": pca_proj, "gmm_labels": np.array(labels)} if pca_proj is not None else None,
        )
        return result

    # --- Post-EM checkpoint (before decode, which may OOM) ---
    if checkpoint_after_em is not None:
        _save_em_checkpoint(checkpoint_after_em, model, chrom_data)

    # --- Final decode via decode_chromosome() ---
    decode = decode_chromosome(
        chrom_data, model,
        batch_size=batch_size,
        write_dense_decode=write_dense_decode,
        decode_parquet_path=decode_parquet_path,
        stats=stats,
    )
    spectral = (
        {"pca_proj": pca_proj, "gmm_labels": np.array(labels)}
        if pca_proj is not None else None
    )
    result = AncestryResult(
        calls=decode.calls, model=model, chrom=chrom_data.chrom,
        decode=decode, spectral=spectral,
    )
    return result


# ---------------------------------------------------------------------------
# Standalone decode (extracted from run_em for checkpoint-stage separation)
# ---------------------------------------------------------------------------

def decode_chromosome(
    chrom_data: ChromData,
    model: AncestryModel,
    *,
    batch_size: int | None = None,
    write_dense_decode: bool = False,
    decode_parquet_path: str | None = None,
    stats=None,
) -> DecodeResult:
    """Final forward-backward decode for one chromosome.

    Handles three decode paths: block emissions, bucketed per-hap T,
    and standard.  Returns a DecodeResult with hard calls, optional
    max_post, and global_sums.

    Parameters
    ----------
    chrom_data : ChromData
        Genotype data and genetic map for the chromosome.
    model : AncestryModel
        Converged model from EM.
    batch_size : int or None
        Haplotypes per batch (None = auto-tune).
    write_dense_decode : bool
        If True, compute and store max_post (posteriors).
    decode_parquet_path : str or None
        If set, stream decode output to this parquet path.
    stats : StatsCollector or None
    """
    n_anc = model.n_ancestries
    bd = model.block_data

    # Auto-tune batch size
    batch_size = _auto_batch_size(
        chrom_data.n_sites, n_anc, batch_size, H=chrom_data.n_haps,
    )

    log.info("Final forward-backward pass")

    if bd is not None and model.pattern_freq is not None:
        # Block emissions: block-level decode
        from .blocks import BlockData
        block_batch = _auto_batch_size_blocks(
            bd.n_blocks, n_anc, chrom_data.n_haps,
        )

        site_to_block = np.empty(chrom_data.n_sites, dtype=np.int32)
        for b_idx in range(bd.n_blocks):
            site_to_block[bd.block_starts[b_idx]:bd.block_ends[b_idx]] = b_idx
        block_widths_j = jnp.array(
            [bd.block_ends[b] - bd.block_starts[b] for b in range(bd.n_blocks)],
            dtype=jnp.float32,
        )

        H_total = chrom_data.n_haps
        global_sums = np.zeros((H_total, n_anc), dtype=np.float64)

        stream_to_parquet = (write_dense_decode
                             and decode_parquet_path is not None)
        if stream_to_parquet:
            from pathlib import Path as _Path
            _Path(decode_parquet_path).parent.mkdir(parents=True, exist_ok=True)
            calls_mmap_path = str(
                _Path(decode_parquet_path).with_suffix(".calls.dat")
            )
            calls = np.memmap(
                calls_mmap_path, dtype=np.int8, mode='w+',
                shape=(H_total, chrom_data.n_sites),
            )
            log.info("  calls backed by memmap: %s (%.1f GB virtual)",
                     calls_mmap_path,
                     H_total * chrom_data.n_sites / 1e9)
            max_post = None
            log.info("  Streaming decode to %s (no full max_post alloc)",
                     decode_parquet_path)
        else:
            calls = np.empty((H_total, chrom_data.n_sites), dtype=np.int8)
            max_post = (
                np.empty((H_total, chrom_data.n_sites), dtype=np.float16)
                if write_dense_decode else None
            )

        # Bucket dispatch: when bucket_assignments is set, run the decode
        # chunk loop once per bucket against a per-bucket model carrying
        # that bucket's center as gen_since_admix. When streaming, each
        # bucket writes to a per-bucket parquet; after the bucket loop we
        # merge them in hap order via _merge_bucket_parquets.
        from .output import DecodeParquetWriter
        if model.bucket_assignments is not None:
            bucket_centers_np = np.array(model.bucket_centers)
            bucket_np = np.array(model.bucket_assignments)
            B = len(bucket_centers_np)
            assert bucket_np.shape == (H_total,), (
                f"bucket_assignments shape {bucket_np.shape} != H={H_total}"
            )
            ba_min, ba_max = int(bucket_np.min()), int(bucket_np.max())
            assert 0 <= ba_min and ba_max < B, (
                f"bucket_assignments range [{ba_min}, {ba_max}] outside [0, {B})"
            )
            bucket_iter = list(range(B))
        else:
            bucket_centers_np = None
            bucket_np = None
            bucket_iter = [None]

        bucket_writer_paths: list = []
        bucket_hap_indices: list = []
        decode_writer = None  # used in single-writer path (no buckets, streaming)

        for b in bucket_iter:
            if b is None:
                b_model = model
                bucket_hap_idx = None
                n_b = H_total
            else:
                mask = bucket_np == b
                n_b = int(mask.sum())
                if n_b == 0:
                    continue
                bucket_hap_idx = np.where(mask)[0]
                b_model = AncestryModel(
                    n_ancestries=model.n_ancestries,
                    mu=model.mu,
                    gen_since_admix=float(bucket_centers_np[b]),
                    allele_freq=model.allele_freq,
                    pattern_freq=model.pattern_freq,
                    block_data=model.block_data,
                )

            if stream_to_parquet:
                if b is None:
                    decode_writer = DecodeParquetWriter(
                        decode_parquet_path,
                        T=chrom_data.n_sites, K=n_anc,
                        chrom=chrom_data.chrom, pos_bp=chrom_data.pos_bp,
                        include_max_post=True,
                    )
                else:
                    from pathlib import Path as _Path
                    bucket_path = _Path(decode_parquet_path).with_suffix(
                        f".bucket{b}.parquet"
                    )
                    decode_writer = DecodeParquetWriter(
                        str(bucket_path),
                        T=chrom_data.n_sites, K=n_anc,
                        chrom=chrom_data.chrom, pos_bp=chrom_data.pos_bp,
                        include_max_post=True,
                    )
                    bucket_writer_paths.append(bucket_path)
                    bucket_hap_indices.append(bucket_hap_idx)
            else:
                decode_writer = None

            for bs in range(0, n_b, block_batch):
                be = min(bs + block_batch, n_b)
                if bucket_hap_idx is None:
                    batch_idx = np.arange(bs, be)
                    pat_idx_np = bd.pattern_indices[bs:be]
                else:
                    batch_idx = bucket_hap_idx[bs:be]
                    pat_idx_np = bd.pattern_indices[batch_idx]

                bd_chunk = BlockData(
                    pattern_indices=pat_idx_np,
                    block_starts=bd.block_starts,
                    block_ends=bd.block_ends,
                    block_distances=bd.block_distances,
                    pattern_counts=bd.pattern_counts,
                    max_patterns=bd.max_patterns,
                    block_size=bd.block_size,
                )
                gb_chunk = forward_backward_blocks(b_model, bd_chunk)
                gb_chunk.block_until_ready()
                calls_block = np.array(
                    jnp.argmax(gb_chunk, axis=2), dtype=np.int8,
                )
                global_sums[batch_idx] = np.array(
                    jnp.einsum('hba,b->ha', gb_chunk, block_widths_j),
                    dtype=np.float64,
                )
                calls_chunk = calls_block[:, site_to_block]
                calls[batch_idx] = calls_chunk

                if decode_writer is not None:
                    max_post_block = np.array(
                        jnp.max(gb_chunk, axis=2), dtype=np.float16,
                    )
                    mp_chunk = max_post_block[:, site_to_block]
                    decode_writer.write_batch(calls_chunk, mp_chunk)
                    del mp_chunk, max_post_block
                elif max_post is not None:
                    max_post_block = np.array(
                        jnp.max(gb_chunk, axis=2), dtype=np.float16,
                    )
                    max_post[batch_idx] = max_post_block[:, site_to_block]

                del gb_chunk
                log.info("  decode bucket %s chunk %d–%d / %d",
                         "all" if b is None else str(b), bs, be, n_b)

            if decode_writer is not None:
                decode_writer.close()
                decode_writer = None

        # Merge per-bucket parquets into the final hap-ordered file
        if stream_to_parquet and bucket_writer_paths:
            from .output import _merge_bucket_parquets
            _merge_bucket_parquets(
                bucket_writer_paths,
                bucket_hap_indices,
                decode_parquet_path,
                chrom=chrom_data.chrom,
                pos_bp=chrom_data.pos_bp,
                T=chrom_data.n_sites,
                K=n_anc,
                include_max_post=True,
            )
            for p in bucket_writer_paths:
                p.unlink()

        if max_post is not None:
            assert max_post.dtype == np.float16, (
                f"max_post must be float16 for memory; got {max_post.dtype}"
            )
        decode = DecodeResult(
            calls=calls, max_post=max_post, global_sums=global_sums,
            parquet_path=decode_parquet_path if stream_to_parquet else None,
        )
    else:
        # Standard or bucketed decode
        from ._device import fits_on_device
        from .output import DecodeParquetWriter
        geno_np = chrom_data.geno
        if fits_on_device(geno_np.nbytes):
            geno = jnp.array(geno_np)
        else:
            geno = geno_np
        d_morgan_j = jnp.array(
            chrom_data.genetic_distances.astype(np.float64),
        )

        H_total = chrom_data.n_haps
        T_sites = chrom_data.n_sites
        stream_to_parquet = (write_dense_decode
                             and decode_parquet_path is not None)

        if stream_to_parquet:
            from pathlib import Path as _Path
            _Path(decode_parquet_path).parent.mkdir(parents=True, exist_ok=True)
            calls_mmap_path = str(
                _Path(decode_parquet_path).with_suffix(".calls.dat")
            )
            calls = np.memmap(
                calls_mmap_path, dtype=np.int8, mode='w+',
                shape=(H_total, T_sites),
            )
            log.info("  calls backed by memmap: %s (%.1f GB virtual)",
                     calls_mmap_path,
                     H_total * T_sites / 1e9)
            log.info("  Streaming decode to %s (no full max_post alloc)",
                     decode_parquet_path)

            if model.bucket_assignments is not None:
                # Per-bucket decode + per-bucket parquet writers + merge
                bucket_centers_np = np.array(model.bucket_centers)
                bucket_np = np.array(model.bucket_assignments)
                B = len(bucket_centers_np)
                assert bucket_np.shape == (H_total,), (
                    f"bucket_assignments shape {bucket_np.shape} != H={H_total}"
                )
                ba_min, ba_max = int(bucket_np.min()), int(bucket_np.max())
                assert 0 <= ba_min and ba_max < B, (
                    f"bucket_assignments range [{ba_min}, {ba_max}] outside [0, {B})"
                )

                bucket_writer_paths: list = []
                bucket_hap_indices: list = []
                global_sums = np.zeros((H_total, n_anc), dtype=np.float64)

                from .hmm import forward_backward_decode as _fb_decode
                for b in range(B):
                    mask = bucket_np == b
                    n_b = int(mask.sum())
                    if n_b == 0:
                        continue
                    hap_idx = np.where(mask)[0]
                    bucket_path = _Path(decode_parquet_path).with_suffix(
                        f".bucket{b}.parquet"
                    )
                    bucket_writer = DecodeParquetWriter(
                        str(bucket_path),
                        T=T_sites, K=n_anc,
                        chrom=chrom_data.chrom, pos_bp=chrom_data.pos_bp,
                        include_max_post=True,
                    )
                    bucket_writer_paths.append(bucket_path)
                    bucket_hap_indices.append(hap_idx)

                    def _mp_writer(target_idx, mp_chunk,
                                   _bw=bucket_writer, _calls=calls):
                        _bw.write_batch(
                            np.asarray(_calls[target_idx]), mp_chunk,
                        )

                    b_model = AncestryModel(
                        n_ancestries=n_anc, mu=model.mu,
                        gen_since_admix=float(bucket_centers_np[b]),
                        allele_freq=model.allele_freq,
                    )
                    bucket_decode = _fb_decode(
                        geno[hap_idx], b_model, d_morgan_j, batch_size,
                        compute_max_post=True,
                        calls_out=calls,
                        max_post_writer=_mp_writer,
                        hap_idx_map=hap_idx,
                    )
                    global_sums[hap_idx] = bucket_decode.global_sums
                    bucket_writer.close()

                from .output import _merge_bucket_parquets
                _merge_bucket_parquets(
                    bucket_writer_paths, bucket_hap_indices,
                    decode_parquet_path,
                    chrom=chrom_data.chrom, pos_bp=chrom_data.pos_bp,
                    T=T_sites, K=n_anc, include_max_post=True,
                )
                for p in bucket_writer_paths:
                    p.unlink()

                decode = DecodeResult(
                    calls=calls, max_post=None, global_sums=global_sums,
                    parquet_path=decode_parquet_path,
                )
            else:
                # Single decode_writer, no buckets
                decode_writer = DecodeParquetWriter(
                    decode_parquet_path,
                    T=T_sites, K=n_anc,
                    chrom=chrom_data.chrom, pos_bp=chrom_data.pos_bp,
                    include_max_post=True,
                )

                def _mp_writer(target_idx, mp_chunk,
                               _dw=decode_writer, _calls=calls):
                    _dw.write_batch(
                        np.asarray(_calls[target_idx]), mp_chunk,
                    )

                decode = forward_backward_decode(
                    geno, model, d_morgan_j, batch_size,
                    compute_max_post=True,
                    calls_out=calls,
                    max_post_writer=_mp_writer,
                )
                decode_writer.close()
                decode = DecodeResult(
                    calls=decode.calls, max_post=None,
                    global_sums=decode.global_sums,
                    parquet_path=decode_parquet_path,
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

    # Summary stats — chunked bincount
    H = decode.calls.shape[0]
    bincount = np.zeros(n_anc, dtype=np.int64)
    BINCOUNT_CHUNK = 50_000
    for start in range(0, H, BINCOUNT_CHUNK):
        end = min(start + BINCOUNT_CHUNK, H)
        bincount += np.bincount(
            decode.calls[start:end].ravel(),
            minlength=n_anc,
        )
    total = decode.calls.size
    for a in range(n_anc):
        prop = float(bincount[a]) / total
        log.info("  Ancestry %d: %.1f%% of genome", a, 100 * prop)
        if stats is not None:
            stats.emit("em/ancestry_proportion", prop,
                       chrom=chrom_data.chrom, tags={"ancestry": a})

    return decode


# ---------------------------------------------------------------------------
# Multi-chromosome wrapper
# ---------------------------------------------------------------------------

def _save_em_checkpoint(path: str, model: AncestryModel, chrom_data: 'ChromData') -> None:
    """Save converged model state after EM, before decode."""
    save_dict = dict(
        mu=np.array(model.mu),
        gen_since_admix=np.float64(model.gen_since_admix),
        allele_freq=np.array(model.allele_freq),
        n_ancestries=np.int32(model.n_ancestries),
        n_sites=np.int64(chrom_data.n_sites),
        n_haps=np.int64(chrom_data.n_haps),
        chrom=np.array(str(chrom_data.chrom)),
    )
    if model.pattern_freq is not None:
        save_dict["pattern_freq"] = np.array(model.pattern_freq)
    bd = model.block_data
    if bd is not None:
        save_dict["pattern_indices"] = np.array(bd.pattern_indices)
        save_dict["block_starts"] = np.array(bd.block_starts)
        save_dict["block_ends"] = np.array(bd.block_ends)
        save_dict["block_distances"] = np.array(bd.block_distances)
        save_dict["pattern_counts"] = np.array(bd.pattern_counts)
        save_dict["max_patterns"] = np.int32(bd.max_patterns)
        save_dict["block_size"] = np.int32(bd.block_size)
    if not path.endswith(".npz"):
        path = f"{path}.em_checkpoint.npz"
    np.savez_compressed(path, **save_dict)
    log.info("Post-EM checkpoint written to %s (A=%d)", path, model.n_ancestries)


def _load_em_checkpoint(path: str, chrom_data: 'ChromData') -> AncestryModel:
    """Load converged model from a post-EM checkpoint."""
    from .blocks import BlockData
    if not path.endswith(".npz"):
        path = f"{path}.em_checkpoint.npz"
    data = np.load(path, allow_pickle=True)
    n_anc = int(data["n_ancestries"])
    assert int(data["n_haps"]) == chrom_data.n_haps, (
        f"Checkpoint H={data['n_haps']} != input H={chrom_data.n_haps}"
    )
    assert int(data["n_sites"]) == chrom_data.n_sites, (
        f"Checkpoint T={data['n_sites']} != input T={chrom_data.n_sites}"
    )
    bd = None
    pf = None
    if "block_starts" in data:
        bd = BlockData(
            pattern_indices=data["pattern_indices"],
            block_starts=data["block_starts"],
            block_ends=data["block_ends"],
            block_distances=data["block_distances"],
            pattern_counts=data["pattern_counts"],
            max_patterns=int(data["max_patterns"]),
            block_size=int(data["block_size"]),
        )
    if "pattern_freq" in data:
        pf = jnp.array(data["pattern_freq"])
    model = AncestryModel(
        n_ancestries=n_anc,
        mu=jnp.array(data["mu"]),
        gen_since_admix=float(data["gen_since_admix"]),
        allele_freq=jnp.array(data["allele_freq"]),
        pattern_freq=pf,
        block_data=bd,
    )
    log.info("Loaded post-EM checkpoint: A=%d, H=%d, T=%d", n_anc, chrom_data.n_haps, chrom_data.n_sites)
    return model


def _save_checkpoint(
    out_prefix: str,
    model: AncestryModel,
    leaf_labels: np.ndarray,
    leaf_info: list,
    chrom_data: 'ChromData',
) -> None:
    """Save post-seeding state for resume."""
    import json, datetime
    bd = model.block_data
    save_dict = dict(
        leaf_labels=np.array(leaf_labels, dtype=np.int32),
        leaf_paths=np.array([li.path for li in leaf_info]),
        mu=np.array(model.mu),
        gen_since_admix=np.float64(model.gen_since_admix),
        allele_freq=np.array(model.allele_freq),
        n_ancestries=np.int32(model.n_ancestries),
        chrom=np.array(str(chrom_data.chrom)),
        n_sites=np.int64(chrom_data.n_sites),
        n_haps=np.int64(chrom_data.n_haps),
    )
    if model.pattern_freq is not None:
        save_dict["pattern_freq"] = np.array(model.pattern_freq)
    if bd is not None:
        save_dict["pattern_indices"] = np.array(bd.pattern_indices)
        save_dict["block_starts"] = np.array(bd.block_starts)
        save_dict["block_ends"] = np.array(bd.block_ends)
        save_dict["block_distances"] = np.array(bd.block_distances)
        save_dict["pattern_counts"] = np.array(bd.pattern_counts)
        save_dict["max_patterns"] = np.int32(bd.max_patterns)
        save_dict["block_size"] = np.int32(bd.block_size)
    np.savez_compressed(f"{out_prefix}.checkpoint.npz", **save_dict)
    meta = {
        "n_ancestries": int(model.n_ancestries),
        "n_haps": int(chrom_data.n_haps),
        "n_sites": int(chrom_data.n_sites),
        "n_leaves": len(leaf_info),
        "chrom": str(chrom_data.chrom),
        "date": datetime.datetime.now().isoformat(),
    }
    with open(f"{out_prefix}.checkpoint.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Checkpoint written to %s.checkpoint.npz (%d leaves, A=%d)",
             out_prefix, len(leaf_info), model.n_ancestries)


def _load_checkpoint(
    path_prefix: str,
    chrom_data: 'ChromData',
) -> tuple[AncestryModel, np.ndarray]:
    """Load post-seeding checkpoint."""
    from .blocks import BlockData
    if path_prefix.endswith(".checkpoint.npz"):
        path_prefix = path_prefix.removesuffix(".checkpoint.npz")
    data = np.load(f"{path_prefix}.checkpoint.npz", allow_pickle=True)

    n_anc = int(data["n_ancestries"])
    assert int(data["n_haps"]) == chrom_data.n_haps, (
        f"Checkpoint H={data['n_haps']} != input H={chrom_data.n_haps}"
    )
    assert int(data["n_sites"]) == chrom_data.n_sites, (
        f"Checkpoint T={data['n_sites']} != input T={chrom_data.n_sites}"
    )

    bd = None
    pf = None
    if "block_starts" in data:
        bd = BlockData(
            pattern_indices=data["pattern_indices"],
            block_starts=data["block_starts"],
            block_ends=data["block_ends"],
            block_distances=data["block_distances"],
            pattern_counts=data["pattern_counts"],
            max_patterns=int(data["max_patterns"]),
            block_size=int(data["block_size"]),
        )
    if "pattern_freq" in data:
        pf = jnp.array(data["pattern_freq"])

    model = AncestryModel(
        n_ancestries=n_anc,
        mu=jnp.array(data["mu"]),
        gen_since_admix=float(data["gen_since_admix"]),
        allele_freq=jnp.array(data["allele_freq"]),
        pattern_freq=pf,
        block_data=bd,
    )
    leaf_labels = data["leaf_labels"]
    log.info("Loaded checkpoint: A=%d, H=%d, T=%d",
             n_anc, chrom_data.n_haps, chrom_data.n_sites)
    return model, leaf_labels


def _save_seed_workdir(wd, model, leaf_labels, leaf_info, chrom_data):
    """Save seed-stage checkpoint into the work directory."""
    bd = model.block_data
    save_dict = dict(
        leaf_labels=np.array(leaf_labels, dtype=np.int32),
        leaf_paths=np.array([li.path for li in leaf_info]),
        mu=np.array(model.mu),
        gen_since_admix=np.float64(model.gen_since_admix),
        allele_freq=np.array(model.allele_freq),
        n_ancestries=np.int32(model.n_ancestries),
        chrom=np.array(str(chrom_data.chrom)),
        n_sites=np.int64(chrom_data.n_sites),
        n_haps=np.int64(chrom_data.n_haps),
    )
    if model.pattern_freq is not None:
        save_dict["pattern_freq"] = np.array(model.pattern_freq)
    if bd is not None:
        save_dict["pattern_indices"] = np.array(bd.pattern_indices)
        save_dict["block_starts"] = np.array(bd.block_starts)
        save_dict["block_ends"] = np.array(bd.block_ends)
        save_dict["block_distances"] = np.array(bd.block_distances)
        save_dict["pattern_counts"] = np.array(bd.pattern_counts)
        save_dict["max_patterns"] = np.int32(bd.max_patterns)
        save_dict["block_size"] = np.int32(bd.block_size)
    wd.atomic_write_npz(wd.stage_path("seed"), save_dict)
    log.info("Seed checkpoint written to %s (A=%d)",
             wd.stage_path("seed"), model.n_ancestries)


def _load_seed_workdir(wd, chrom_data):
    """Load seed-stage checkpoint from work directory.

    Returns (model, leaf_labels, leaf_info).
    """
    from .blocks import BlockData
    from .recursive_seed import LeafInfo

    data = np.load(str(wd.stage_path("seed")), allow_pickle=True)
    n_anc = int(data["n_ancestries"])
    assert int(data["n_haps"]) == chrom_data.n_haps, (
        f"Seed checkpoint H={data['n_haps']} != input H={chrom_data.n_haps}"
    )
    assert int(data["n_sites"]) == chrom_data.n_sites, (
        f"Seed checkpoint T={data['n_sites']} != input T={chrom_data.n_sites}"
    )
    bd = None
    pf = None
    if "block_starts" in data:
        bd = BlockData(
            pattern_indices=data["pattern_indices"],
            block_starts=data["block_starts"],
            block_ends=data["block_ends"],
            block_distances=data["block_distances"],
            pattern_counts=data["pattern_counts"],
            max_patterns=int(data["max_patterns"]),
            block_size=int(data["block_size"]),
        )
    if "pattern_freq" in data:
        pf = jnp.array(data["pattern_freq"])
    model = AncestryModel(
        n_ancestries=n_anc,
        mu=jnp.array(data["mu"]),
        gen_since_admix=float(data["gen_since_admix"]),
        allele_freq=jnp.array(data["allele_freq"]),
        pattern_freq=pf,
        block_data=bd,
    )
    leaf_labels = data["leaf_labels"]
    leaf_paths = data["leaf_paths"] if "leaf_paths" in data else None
    # Reconstruct leaf_info from saved paths
    if leaf_paths is not None:
        leaf_info = [
            LeafInfo(label=i, n_haps=int((leaf_labels == i).sum()),
                     depth=0, path=str(p), bic_score=0.0)
            for i, p in enumerate(leaf_paths)
        ]
    else:
        leaf_info = [
            LeafInfo(label=i, n_haps=int((leaf_labels == i).sum()),
                     depth=0, path=f"L{i}", bic_score=0.0)
            for i in range(n_anc)
        ]
    log.info("Loaded seed checkpoint: A=%d, H=%d, T=%d",
             n_anc, chrom_data.n_haps, chrom_data.n_sites)
    return model, leaf_labels, leaf_info


def _save_em_workdir(wd, model, chrom_data):
    """Save EM-stage checkpoint (converged model) into work directory."""
    bd = model.block_data
    save_dict = dict(
        mu=np.array(model.mu),
        gen_since_admix=np.float64(model.gen_since_admix),
        allele_freq=np.array(model.allele_freq),
        n_ancestries=np.int32(model.n_ancestries),
        n_sites=np.int64(chrom_data.n_sites),
        n_haps=np.int64(chrom_data.n_haps),
        chrom=np.array(str(chrom_data.chrom)),
    )
    if model.pattern_freq is not None:
        save_dict["pattern_freq"] = np.array(model.pattern_freq)
    if bd is not None:
        save_dict["pattern_indices"] = np.array(bd.pattern_indices)
        save_dict["block_starts"] = np.array(bd.block_starts)
        save_dict["block_ends"] = np.array(bd.block_ends)
        save_dict["block_distances"] = np.array(bd.block_distances)
        save_dict["pattern_counts"] = np.array(bd.pattern_counts)
        save_dict["max_patterns"] = np.int32(bd.max_patterns)
        save_dict["block_size"] = np.int32(bd.block_size)
    if model.gen_per_hap is not None:
        save_dict["gen_per_hap"] = np.array(model.gen_per_hap)
    if model.bucket_centers is not None:
        save_dict["bucket_centers"] = np.array(model.bucket_centers)
    if model.bucket_assignments is not None:
        save_dict["bucket_assignments"] = np.array(model.bucket_assignments)
    wd.atomic_write_npz(wd.stage_path("em"), save_dict)
    log.info("EM checkpoint written to %s (A=%d)",
             wd.stage_path("em"), model.n_ancestries)


def _load_em_workdir(wd, chrom_data):
    """Load EM-stage checkpoint (converged model) from work directory."""
    from .blocks import BlockData

    data = np.load(str(wd.stage_path("em")), allow_pickle=True)
    n_anc = int(data["n_ancestries"])
    assert int(data["n_haps"]) == chrom_data.n_haps, (
        f"EM checkpoint H={data['n_haps']} != input H={chrom_data.n_haps}"
    )
    assert int(data["n_sites"]) == chrom_data.n_sites, (
        f"EM checkpoint T={data['n_sites']} != input T={chrom_data.n_sites}"
    )
    bd = None
    pf = None
    if "block_starts" in data:
        bd = BlockData(
            pattern_indices=data["pattern_indices"],
            block_starts=data["block_starts"],
            block_ends=data["block_ends"],
            block_distances=data["block_distances"],
            pattern_counts=data["pattern_counts"],
            max_patterns=int(data["max_patterns"]),
            block_size=int(data["block_size"]),
        )
    if "pattern_freq" in data:
        pf = jnp.array(data["pattern_freq"])
    gen_per_hap = jnp.array(data["gen_per_hap"]) if "gen_per_hap" in data else None
    bucket_centers = jnp.array(data["bucket_centers"]) if "bucket_centers" in data else None
    bucket_assignments = jnp.array(data["bucket_assignments"]) if "bucket_assignments" in data else None
    model = AncestryModel(
        n_ancestries=n_anc,
        mu=jnp.array(data["mu"]),
        gen_since_admix=float(data["gen_since_admix"]),
        allele_freq=jnp.array(data["allele_freq"]),
        pattern_freq=pf,
        block_data=bd,
        gen_per_hap=gen_per_hap,
        bucket_centers=bucket_centers,
        bucket_assignments=bucket_assignments,
    )
    log.info("Loaded EM checkpoint: A=%d, H=%d, T=%d",
             n_anc, chrom_data.n_haps, chrom_data.n_sites)
    return model


def _save_decode_workdir(wd, decode_result, chrom):
    """Save decode global_sums companion file alongside the parquet."""
    if decode_result.global_sums is not None:
        gs_path = wd.stage_path("decode", chrom=chrom).with_suffix(
            ".global_sums.npy",
        )
        wd.atomic_write_npy(gs_path, decode_result.global_sums)


def _load_decode_workdir(wd, chrom, chrom_data):
    """Load decode result from work directory parquet + global_sums.

    Returns a DecodeResult with calls loaded from parquet and
    global_sums loaded from companion file.
    """
    from .output import read_decode_parquet

    pq_path = str(wd.stage_path("decode", chrom=chrom))
    pq_data = read_decode_parquet(pq_path)
    calls = pq_data["calls"].astype(np.int8)

    gs_path = wd.stage_path("decode", chrom=chrom).with_suffix(
        ".global_sums.npy",
    )
    global_sums = np.load(str(gs_path)) if gs_path.exists() else None

    max_post = pq_data.get("max_post")

    return DecodeResult(
        calls=calls,
        max_post=max_post,
        global_sums=global_sums,
        parquet_path=pq_path,
    )


def _labels_to_resp(labels, n_haps, n_anc, seeding_mask=None):
    """Convert hard leaf labels to soft responsibilities matrix."""
    if seeding_mask is not None:
        kept_idx = np.where(seeding_mask)[0]
        # labels may be (H_kept,) or (H_total,) with -1 for excluded
        if len(labels) == n_haps:
            # Full-size labels with -1 for excluded
            seed_resp_np = np.full(
                (n_haps, n_anc), 1.0 / n_anc, dtype=np.float32,
            )
            mask = labels >= 0
            seed_resp_np[mask] = 0.0
            seed_resp_np[mask, labels[mask]] = 1.0
        else:
            # Compact labels (H_kept,)
            seed_resp_np = np.full(
                (n_haps, n_anc), 1.0 / n_anc, dtype=np.float32,
            )
            seed_resp_np[kept_idx] = 0.0
            seed_resp_np[kept_idx, labels] = 1.0
        return jnp.array(seed_resp_np)
    else:
        resp = jnp.zeros((n_haps, n_anc), dtype=jnp.float32)
        return resp.at[jnp.arange(n_haps), jnp.array(labels)].set(1.0)


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
    stop_after_seeding: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    checkpoint_after_em: bool = False,
    ancestry_names: Optional[list[str]] = None,
    write_dense_decode: bool = False,
    seeding_mask: np.ndarray | None = None,
    work_dir=None,
) -> list[AncestryResult] | None:
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
    work_dir : WorkDir or None
        If provided, enables automatic checkpoint/resume via the work
        directory.  Replaces the legacy ``resume_from_checkpoint`` and
        ``checkpoint_after_em`` flags.

    Returns
    -------
    List of AncestryResult, one per chromosome.
    """
    import time as _time

    wd = work_dir  # shorthand; may be None

    results = []
    fitted_model = None

    for chrom_data in chrom_iter:
        if stats is not None:
            stats.timer_start(f"chrom/{chrom_data.chrom}")

        if fitted_model is None:
            # ==========================================================
            # First chromosome: seed → EM → decode
            # ==========================================================
            log.info("=== Seed chromosome: %s (full EM) ===", chrom_data.chrom)

            seed_resp = None
            em_n_ancestries = n_ancestries
            leaf_labels_for_ckpt = None
            leaf_info_for_ckpt = None

            # ----------------------------------------------------------
            # SEED STAGE
            # ----------------------------------------------------------
            seed_loaded = False
            if wd is not None and wd.stage_done("seed"):
                # Resume from work dir seed checkpoint
                log.info("stage seed: loading from %s", wd.stage_path("seed"))
                t0_stage = _time.perf_counter()
                ckpt_model, ckpt_labels, leaf_info_for_ckpt = \
                    _load_seed_workdir(wd, chrom_data)
                n_leaves = ckpt_model.n_ancestries
                seed_resp = _labels_to_resp(
                    ckpt_labels, chrom_data.n_haps, n_leaves,
                    seeding_mask=seeding_mask,
                )
                em_n_ancestries = n_leaves
                leaf_labels_for_ckpt = ckpt_labels
                seed_loaded = True
            elif resume_from_checkpoint is not None:
                # Legacy --resume-from-checkpoint
                log.info("Resuming from legacy checkpoint: %s",
                         resume_from_checkpoint)
                ckpt_model, ckpt_labels = _load_checkpoint(
                    resume_from_checkpoint, chrom_data,
                )
                n_leaves = ckpt_model.n_ancestries
                seed_resp = _labels_to_resp(
                    ckpt_labels, chrom_data.n_haps, n_leaves,
                )
                em_n_ancestries = n_leaves
                leaf_labels_for_ckpt = ckpt_labels
                seed_loaded = True
            else:
                # Run seeding
                t0_stage = _time.perf_counter()
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
                        seeding_mask=seeding_mask,
                        **rkw,
                    )
                    n_leaves = len(leaf_info)
                    H_total = chrom_data.n_haps

                    if seeding_mask is not None:
                        kept_idx = np.where(seeding_mask)[0]
                        seed_resp_np = np.full(
                            (H_total, n_leaves), 1.0 / n_leaves,
                            dtype=np.float32,
                        )
                        seed_resp_np[kept_idx] = 0.0
                        seed_resp_np[kept_idx, leaf_labels] = 1.0
                        seed_resp = jnp.array(seed_resp_np)
                        full_leaf_labels = np.full(H_total, -1, dtype=np.int32)
                        full_leaf_labels[kept_idx] = leaf_labels
                        leaf_labels_for_ckpt = full_leaf_labels
                    else:
                        seed_resp = jnp.zeros(
                            (H_total, n_leaves), dtype=jnp.float32,
                        )
                        seed_resp = seed_resp.at[
                            jnp.arange(H_total), jnp.array(leaf_labels)
                        ].set(1.0)
                        leaf_labels_for_ckpt = leaf_labels

                    em_n_ancestries = n_leaves
                    leaf_info_for_ckpt = leaf_info

                # Save seed to work dir
                if wd is not None and leaf_labels_for_ckpt is not None:
                    # For GMM path: derive labels/info from seed_resp
                    if leaf_info_for_ckpt is None and seed_resp is not None:
                        leaf_labels_for_ckpt = np.array(
                            jnp.argmax(seed_resp, axis=1)
                        )
                    # We need a model to save — run init-only to get one
                    # (mark_done happens after the full seed stage completes
                    # below)

            # --stop-after-seeding: seeding done, save and exit
            if stop_after_seeding:
                result = run_em(
                    chrom_data,
                    n_ancestries=em_n_ancestries,
                    n_em_iter=0,
                    gen_since_admix=gen_since_admix,
                    batch_size=batch_size,
                    rng_seed=rng_seed,
                    seed_responsibilities=seed_resp,
                    use_block_emissions=use_block_emissions,
                    block_size=block_size,
                    detection_method=detection_method,
                    max_ancestries=max_ancestries,
                )
                # Save to work dir
                if wd is not None and not seed_loaded:
                    if leaf_labels_for_ckpt is None:
                        leaf_labels_for_ckpt = np.array(
                            jnp.argmax(seed_resp, axis=1) if seed_resp is not None
                            else jnp.zeros(chrom_data.n_haps, dtype=jnp.int32)
                        )
                    if leaf_info_for_ckpt is None:
                        from .recursive_seed import LeafInfo
                        leaf_info_for_ckpt = [
                            LeafInfo(label=i, n_haps=int((leaf_labels_for_ckpt==i).sum()),
                                     depth=0, path=f"L{i}", bic_score=0.0)
                            for i in range(result.model.n_ancestries)
                        ]
                    _save_seed_workdir(
                        wd, result.model, leaf_labels_for_ckpt,
                        leaf_info_for_ckpt, chrom_data,
                    )
                    wd.mark_done("seed",
                                 wall_s=_time.perf_counter() - t0_stage)
                # Also save legacy checkpoint for compatibility
                if out_prefix is not None:
                    if leaf_labels_for_ckpt is None:
                        leaf_labels_for_ckpt = np.array(
                            jnp.argmax(seed_resp, axis=1) if seed_resp is not None
                            else jnp.zeros(chrom_data.n_haps, dtype=jnp.int32)
                        )
                    if leaf_info_for_ckpt is None:
                        from .recursive_seed import LeafInfo
                        leaf_info_for_ckpt = [
                            LeafInfo(label=i, n_haps=int((leaf_labels_for_ckpt==i).sum()),
                                     depth=0, path=f"L{i}", bic_score=0.0)
                            for i in range(result.model.n_ancestries)
                        ]
                    _save_checkpoint(
                        out_prefix, result.model,
                        leaf_labels_for_ckpt, leaf_info_for_ckpt, chrom_data,
                    )
                log.info("=== Stopped after seeding (--stop-after-seeding) ===")
                return None

            # ----------------------------------------------------------
            # EM STAGE
            # ----------------------------------------------------------
            em_loaded = False
            if wd is not None and wd.stage_done("em"):
                log.info("stage em: loading from %s", wd.stage_path("em"))
                fitted_model = _load_em_workdir(wd, chrom_data)
                em_loaded = True
            else:
                t0_em = _time.perf_counter()
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
                    skip_decode=True,
                )
                fitted_model = result.model

                # Save seed checkpoint (if we ran seeding fresh)
                if wd is not None and not seed_loaded:
                    if leaf_labels_for_ckpt is not None and leaf_info_for_ckpt is not None:
                        _save_seed_workdir(
                            wd, result.model, leaf_labels_for_ckpt,
                            leaf_info_for_ckpt, chrom_data,
                        )
                        wd.mark_done("seed",
                                     wall_s=_time.perf_counter() - t0_stage)

                # Save EM checkpoint
                if wd is not None:
                    _save_em_workdir(wd, fitted_model, chrom_data)
                    wd.mark_done("em",
                                 wall_s=_time.perf_counter() - t0_em)

                # Also save legacy checkpoints for compatibility
                if out_prefix is not None and leaf_labels_for_ckpt is not None:
                    _save_checkpoint(
                        out_prefix, result.model,
                        leaf_labels_for_ckpt, leaf_info_for_ckpt, chrom_data,
                    )

            # Attach leaf paths for post-EM consolidation
            if leaf_info_for_ckpt is not None:
                # Store on a temporary attribute for later
                _leaf_paths = [li.path for li in leaf_info_for_ckpt]
            else:
                _leaf_paths = None

            # ----------------------------------------------------------
            # DECODE STAGE (seed chromosome)
            # ----------------------------------------------------------
            if wd is not None and wd.stage_done("decode", chrom=chrom_data.chrom):
                log.info("stage decode chr%s: loading from %s",
                         chrom_data.chrom,
                         wd.stage_path("decode", chrom=chrom_data.chrom))
                decode = _load_decode_workdir(wd, chrom_data.chrom, chrom_data)
                result = AncestryResult(
                    calls=decode.calls, model=fitted_model,
                    chrom=chrom_data.chrom, decode=decode,
                )
            else:
                t0_decode = _time.perf_counter()
                _decode_pq = None
                if write_dense_decode and out_prefix:
                    if wd is not None:
                        _decode_pq = str(wd.stage_path(
                            "decode", chrom=chrom_data.chrom,
                        ))
                    else:
                        _decode_pq = (
                            f"{out_prefix}.chr{chrom_data.chrom}.decode.parquet"
                        )
                decode = decode_chromosome(
                    chrom_data, fitted_model,
                    batch_size=batch_size,
                    write_dense_decode=write_dense_decode,
                    decode_parquet_path=_decode_pq,
                    stats=stats,
                )
                result = AncestryResult(
                    calls=decode.calls, model=fitted_model,
                    chrom=chrom_data.chrom, decode=decode,
                )
                if wd is not None:
                    _save_decode_workdir(wd, decode, chrom_data.chrom)
                    wd.mark_done("decode", chrom=chrom_data.chrom,
                                 wall_s=_time.perf_counter() - t0_decode)

            # Attach leaf paths for post-EM consolidation
            if _leaf_paths is not None:
                if result.spectral is None:
                    result.spectral = {}
                result.spectral["leaf_paths"] = _leaf_paths

        else:
            # ==========================================================
            # Subsequent chromosomes: warm-start, 1 iteration
            # ==========================================================
            log.info("=== Chromosome %s (warm-started, 1 iter) ===",
                     chrom_data.chrom)

            # Check decode checkpoint
            if wd is not None and wd.stage_done("decode", chrom=chrom_data.chrom):
                log.info("stage decode chr%s: loading from %s",
                         chrom_data.chrom,
                         wd.stage_path("decode", chrom=chrom_data.chrom))
                decode = _load_decode_workdir(
                    wd, chrom_data.chrom, chrom_data,
                )
                result = AncestryResult(
                    calls=decode.calls, model=fitted_model,
                    chrom=chrom_data.chrom, decode=decode,
                )
            else:
                t0_decode = _time.perf_counter()
                from ._device import fits_on_device
                if fits_on_device(chrom_data.geno.nbytes):
                    geno = jnp.array(chrom_data.geno)
                    log.info("  geno %.1f GB → device-resident",
                             chrom_data.geno.nbytes / 1e9)
                else:
                    geno = chrom_data.geno
                    log.info("  geno %.1f GB > device budget → host-resident",
                             chrom_data.geno.nbytes / 1e9)
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
                # Override mu and T from the fitted model
                model = AncestryModel(
                    n_ancestries=fitted_model.n_ancestries,
                    mu=fitted_model.mu,
                    gen_since_admix=fitted_model.gen_since_admix,
                    allele_freq=model.allele_freq,
                    gen_per_hap=fitted_model.gen_per_hap,
                    bucket_centers=fitted_model.bucket_centers,
                    bucket_assignments=fitted_model.bucket_assignments,
                )

                # Auto-tune batch_size
                chrom_batch = _auto_batch_size(
                    chrom_data.n_sites, fitted_model.n_ancestries, batch_size,
                    H=chrom_data.n_haps,
                )

                # One EM iteration
                if model.bucket_assignments is not None:
                    em_stats = forward_backward_bucketed_em(
                        geno, model, d_morgan_j, chrom_batch,
                    )
                else:
                    em_stats = forward_backward_em(
                        geno, model, d_morgan_j, chrom_batch,
                    )
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

                # Decode
                _decode_pq = None
                if write_dense_decode and out_prefix:
                    if wd is not None:
                        _decode_pq = str(wd.stage_path(
                            "decode", chrom=chrom_data.chrom,
                        ))
                    else:
                        _decode_pq = (
                            f"{out_prefix}.chr{chrom_data.chrom}.decode.parquet"
                        )
                decode = decode_chromosome(
                    chrom_data, model,
                    batch_size=chrom_batch,
                    write_dense_decode=write_dense_decode,
                    decode_parquet_path=_decode_pq,
                    stats=stats,
                )
                result = AncestryResult(
                    calls=decode.calls, model=model,
                    chrom=chrom_data.chrom, decode=decode,
                )
                if wd is not None:
                    _save_decode_workdir(wd, decode, chrom_data.chrom)
                    wd.mark_done("decode", chrom=chrom_data.chrom,
                                 wall_s=_time.perf_counter() - t0_decode)

        if stats is not None:
            elapsed = stats.timer_stop(f"chrom/{chrom_data.chrom}",
                                       chrom=chrom_data.chrom)
            throughput = (chrom_data.n_haps * chrom_data.n_sites
                          / max(elapsed, 1e-6))
            stats.emit("runtime/throughput", round(throughput),
                       chrom=chrom_data.chrom)

        results.append(result)

    return results
