"""Expectation-Maximization loop for self-bootstrapping ancestry inference.

The model parameters (allele frequencies, ancestry proportions, admixture
time) are iteratively refined.  With 500K+ samples, sufficient statistics
converge fast — typically 2-3 EM iterations suffice.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .datatypes import AncestryModel, ChromData, AncestryResult, EMStats, DecodeResult
from .hmm import (
    forward_backward_batched,
    forward_backward_blocks,
    forward_backward_blocks_batched,
    forward_backward_blocks_em,
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


def build_component_states(
    allele_freq: jnp.ndarray,
    mu: jnp.ndarray,
    chrom_data,
) -> list:
    """Build one :class:`popout.identity.ComponentState` per ancestry
    from the most recent M-step allele frequencies.

    Identity scoring is per-chromosome in the current EM loop — each
    state carries this chromosome's ``pos_bp`` and ``chrom`` only.
    Aggregating across chromosomes is an additive future change (see
    ``docs/PRIORS.md``); the per-chromosome view is acceptable because
    a component's ancestry is visible from any one chromosome's freqs.
    """
    from .identity import ComponentState

    af = np.asarray(allele_freq)
    mu_np = np.asarray(mu)
    pos_bp = np.asarray(chrom_data.pos_bp)
    chrom = str(chrom_data.chrom)
    return [
        ComponentState(freq=af[k], mu=float(mu_np[k]), pos_bp=pos_bp, chrom=chrom)
        for k in range(af.shape[0])
    ]


def write_priors_assignment_dump(
    path: str,
    assignment: np.ndarray,
    priors,
    model,
    chrom_data,
) -> None:
    """Write the (P, K) prior→component assignment matrix as TSV.

    Header rows
    -----------
    1. ``# nearest_1KG`` line — per-component nearest 1KG superpop label
       (or ``-`` if the 1KG reference cache is not populated).
    2. Column header: ``prior\\tcomp_0\\tcomp_1\\t...``.

    Each subsequent row is one prior's name plus its softmax weights.

    The annotation row is the load-bearing diagnostic: at a glance you
    see whether each prior latched onto a component whose nearest-1KG
    label matches the prior's name. priors_v1 (where AFR bound to a
    EUR-bearing component) would have shown its mismatch instantly.
    """
    P, K = assignment.shape
    annotations = _annotate_components_with_1kg(model, chrom_data)
    annot_strs = [
        annotations.get(k, ("-", float("nan"))) for k in range(K)
    ]

    lines = []
    lines.append(
        "# nearest_1KG\t"
        + "\t".join(
            f"{name}(r={r:.3f})" if np.isfinite(r) else name
            for name, r in annot_strs
        )
    )
    lines.append("prior\t" + "\t".join(f"comp_{k}" for k in range(K)))
    for p, prior in enumerate(priors.priors):
        row = [prior.name] + [f"{assignment[p, k]:.6f}" for k in range(K)]
        lines.append("\t".join(row))

    Path(path).write_text("\n".join(lines) + "\n")
    log.info("Wrote priors assignment dump (%d priors x %d components) to %s",
             P, K, path)


def _annotate_components_with_1kg(
    model, chrom_data,
) -> dict[int, tuple[str, float]]:
    """Map component_idx → (nearest 1KG superpop name, correlation).

    Uses the same site-aligned Pearson correlation as ``popout label``.
    Returns ``("-", NaN)`` per component if the 1KG cache is missing or
    no positions overlap.
    """
    try:
        from .fetch_ref import load_ref_frequencies, resolve_ref_path
        from .label import _correlation_matrix
    except Exception:
        return {k: ("-", float("nan")) for k in range(int(model.n_ancestries))}

    K = int(model.n_ancestries)
    try:
        ref_path = resolve_ref_path("GRCh38")
    except FileNotFoundError:
        return {k: ("-", float("nan")) for k in range(K)}

    try:
        ref_freq, ref_pos, ref_names = load_ref_frequencies(
            ref_path, chrom=str(chrom_data.chrom),
        )
    except Exception:
        return {k: ("-", float("nan")) for k in range(K)}

    pos_bp = np.asarray(chrom_data.pos_bp)
    _common, model_idx, ref_idx = np.intersect1d(
        pos_bp, ref_pos, return_indices=True,
    )
    if len(_common) < 10:
        return {k: ("-", float("nan")) for k in range(K)}

    af = np.asarray(model.allele_freq)[:, model_idx]   # (K, n_overlap)
    rf = ref_freq[:, ref_idx]                          # (P_ref, n_overlap)
    corr = _correlation_matrix(af, rf)                 # (K, P_ref)

    out: dict[int, tuple[str, float]] = {}
    for k in range(K):
        j = int(np.argmax(corr[k]))
        out[k] = (str(ref_names[j]), float(corr[k, j]))
    return out


def log_priors_assignment(assignment, priors, component_states) -> None:
    """Log each prior's dominant component and weight."""
    A = assignment.shape[1] if assignment.size else 0
    for p, prior in enumerate(priors.priors):
        if A == 0:
            continue
        k_dom = int(np.argmax(assignment[p]))
        w_dom = float(assignment[p, k_dom])
        log.info(
            "  prior %s → comp %d (w=%.3f, gen_mean=%.1f)",
            prior.name, k_dom, w_dom, prior.gen_mean,
        )


def update_generations_with_priors(
    stats: EMStats,
    current_T_per_comp: jnp.ndarray,
    mu: jnp.ndarray,
    priors,        # popout.prior_spec.Priors (avoid import cycle in hint)
    assignment: np.ndarray,  # (P, K) soft prior→component weights
) -> jnp.ndarray:
    """Per-component MAP T-update under similarity-weighted Beta priors.

    For each component ``k``, the effective Beta(α_eff, β_eff) is
    constructed by accumulating each prior's pseudocount excess
    (α_p - 1, β_p - 1) weighted by the soft assignment::

        α_eff[k] = 1 + Σ_p assignment[p, k] * (α_p - 1)
        β_eff[k] = 1 + Σ_p assignment[p, k] * (β_p - 1)

    The standard Beta-Bernoulli MAP then gives::

        r_MAP[k] = (successes_eff[k] + α_eff[k] - 1)
                 / (trials_eff[k] + α_eff[k] + β_eff[k] - 2)
        T_MAP[k] = -log(1 - r_MAP[k]) / morgans_per_step

    The mu-correction (``successes_eff = switches_per_comp / (1 - mu)``)
    and log-space blend toward ``current_T_per_comp`` are unchanged
    from the previous M-step.

    Limits — both verified by unit test:

    * If ``Σ_p assignment[p, k] == 0`` for some k, α_eff=β_eff=1
      (Beta(1,1) is uniform), so the MAP collapses to the bare MLE.
    * If ``assignment[p, k] = 1`` for one p and 0 for others,
      α_eff=α_p, β_eff=β_p — the standard MAP shift for that prior.

    Intermediate ``assignment[p, k] = 0.5`` is *not* a half-strength
    prior — it is the full prior weighted at 0.5 effective pseudocount
    strength.

    Parameters
    ----------
    stats
        Must have ``switches_per_comp`` and ``d_weighted_occupancy``
        populated (raises if either is None — caller is responsible for
        running the xi-with-transitions branch of forward-backward).
    current_T_per_comp : (A,)
        Previous per-component T estimate, used for log-space blending.
    mu : (A,)
        Current global ancestry proportions.
    priors
        :class:`popout.prior_spec.Priors`.
    assignment : (P, K) ndarray
        Soft prior→component weights from
        :func:`popout.identity_assignment.assign_priors_to_components`.

    Returns
    -------
    (A,) array of updated per-component T values.
    """
    if stats.switches_per_comp is None or stats.d_weighted_occupancy is None:
        raise ValueError(
            "EMStats lacks per-component switch stats; the xi-with-transitions "
            "branch of forward-backward must populate switches_per_comp and "
            "d_weighted_occupancy when priors are in use."
        )

    sw = np.asarray(stats.switches_per_comp, dtype=np.float64)   # (A,)
    occ = np.asarray(stats.d_weighted_occupancy, dtype=np.float64)  # (A,)
    mu_np = np.asarray(mu, dtype=np.float64)
    cur_T = np.asarray(current_T_per_comp, dtype=np.float64)
    assignment_np = np.asarray(assignment, dtype=np.float64)  # (P, K)

    A = sw.shape[0]
    P = len(priors.priors)
    if assignment_np.shape != (P, A):
        raise ValueError(
            f"assignment shape {assignment_np.shape} does not match "
            f"(n_priors={P}, n_components={A})"
        )

    mps = float(priors.morgans_per_step)
    one_minus_mu = np.maximum(1.0 - mu_np, 1e-3)
    successes_eff = sw / one_minus_mu
    trials_eff = occ / max(mps, 1e-12)

    # Per-prior pseudocount excesses (alpha - 1, beta - 1).
    alpha_excess = np.array(
        [p.alpha - 1.0 for p in priors.priors], dtype=np.float64,
    )                                                            # (P,)
    beta_excess = np.array(
        [p.beta - 1.0 for p in priors.priors], dtype=np.float64,
    )                                                            # (P,)
    # alpha_eff[k] = 1 + sum_p assignment[p, k] * alpha_excess[p]
    alpha_eff = 1.0 + assignment_np.T @ alpha_excess              # (A,)
    beta_eff = 1.0 + assignment_np.T @ beta_excess                # (A,)

    new_T = np.empty(A, dtype=np.float64)
    for k in range(A):
        num = max(successes_eff[k] + alpha_eff[k] - 1.0, 1e-9)
        den = max(trials_eff[k] + alpha_eff[k] + beta_eff[k] - 2.0, 1e-9)
        r_k = float(np.clip(num / den, 1e-12, 1.0 - 1e-12))

        T_est = -np.log1p(-r_k) / mps
        T_est = max(1.0, min(T_est, 1000.0))

        blend_alpha = 0.3
        log_T = (
            (1.0 - blend_alpha) * np.log(max(cur_T[k], 1.0))
            + blend_alpha * np.log(T_est)
        )
        new_T[k] = max(1.0, min(float(np.exp(log_T)), 1000.0))

    return jnp.asarray(new_T, dtype=jnp.float32)


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
    ancestry_names: Optional[list[str]] = None,
    write_dense_decode: bool = False,
    decode_parquet_path: Optional[str] = None,
    skip_decode: bool = False,
    priors=None,  # popout.prior_spec.Priors | None
    priors_dump_path: Optional[str] = None,
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
    last_priors_assignment = None  # (P, K), captured for the end-of-run dump

    for iteration in range(n_em_iter):
        log.info("--- EM iteration %d/%d ---", iteration + 1, n_em_iter)

        # E-step: forward-backward (streaming — no full gamma materialised)
        log.info("  E-step: forward-backward on %d haplotypes", chrom_data.n_haps)
        if stats is not None:
            stats.timer_start("e_step")

        _pf_counts = None   # accumulated pattern-freq counts for M-step
        if bd is not None and model.pattern_freq is not None:
            block_batch = _auto_batch_size_blocks(
                bd.n_blocks, n_anc, chrom_data.n_haps,
            )
            log.info("  Block E-step: batch_size=%d (n_blocks=%d, A=%d, H=%d)",
                     block_batch, bd.n_blocks, n_anc, chrom_data.n_haps)
            em_stats, _pf_counts = forward_backward_blocks_em(
                geno, model, bd, batch_size=block_batch,
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
        new_gen_per_comp = model.gen_per_comp  # carry forward unless updated
        should_update_T = iteration > 0

        if not should_update_T:
            new_T = model.gen_since_admix
        elif per_hap_T and bucket_centers is not None:
            T_per_hap, bucket_assignments, new_T = update_generations_per_hap_from_stats(
                em_stats, d_morgan_j, model.gen_since_admix, new_mu, bucket_centers,
            )
            log.info("  T (per-hap): mean=%.1f, std=%.1f",
                     float(jnp.mean(T_per_hap)), float(jnp.std(T_per_hap)))
        elif priors is not None:
            from .identity_assignment import assign_priors_to_components

            cur_per = (
                model.gen_per_comp
                if model.gen_per_comp is not None
                else jnp.full((n_anc,), model.gen_since_admix, dtype=jnp.float32)
            )
            component_states = build_component_states(
                new_freq, new_mu, chrom_data,
            )
            assignment = assign_priors_to_components(
                priors, component_states, iteration,
            )
            log_priors_assignment(assignment, priors, component_states)
            last_priors_assignment = np.array(assignment, copy=True)
            new_gen_per_comp = update_generations_with_priors(
                em_stats, cur_per, new_mu, priors, assignment,
            )
            new_T = float(jnp.mean(new_gen_per_comp))   # logging-only summary
            log.info(
                "  T (per-comp): %s → %s",
                np.array(cur_per).round(2).tolist(),
                np.array(new_gen_per_comp).round(2).tolist(),
            )
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
            gen_per_comp=new_gen_per_comp,
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

    # --- Optional priors-assignment dump (audit artifact) ---
    if priors_dump_path is not None and last_priors_assignment is not None:
        write_priors_assignment_dump(
            priors_dump_path, last_priors_assignment, priors, model,
            chrom_data,
        )

    # --- Skip decode if no EM iterations or skip_decode requested ---
    if n_em_iter == 0 or skip_decode:
        log.info("Skipping final decode%s",
                 " (n_em_iter=0)" if n_em_iter == 0 else " (skip_decode)")
        result = AncestryResult(
            calls=np.zeros((0, 0), dtype=np.int8),
            model=model, chrom=chrom_data.chrom,
            spectral={"pca_proj": pca_proj, "gmm_labels": np.array(labels)} if pca_proj is not None else None,
        )
        return result

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

def _open_streaming_decode_outputs(
    decode_parquet_path: str | None,
    write_dense_decode: bool,
    H_total: int,
    T_sites: int,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Allocate ``calls`` and ``max_post`` for decode_chromosome.

    When ``write_dense_decode`` and ``decode_parquet_path`` are both set,
    backs ``calls`` with ``np.memmap`` (sibling of the parquet) and
    returns ``max_post=None`` (the streaming writer handles posteriors).
    Otherwise allocates in-memory ``np.empty`` for both.

    Returns ``(calls, max_post, stream_to_parquet)``.
    """
    stream_to_parquet = write_dense_decode and decode_parquet_path is not None
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
                 calls_mmap_path, H_total * T_sites / 1e9)
        log.info("  Streaming decode to %s (no full max_post alloc)",
                 decode_parquet_path)
        max_post = None
    else:
        calls = np.empty((H_total, T_sites), dtype=np.int8)
        max_post = (
            np.empty((H_total, T_sites), dtype=np.float16)
            if write_dense_decode else None
        )
    return calls, max_post, stream_to_parquet


def _close_and_merge_bucket_writers(
    bucket_writer_paths: list,
    bucket_hap_indices: list,
    decode_parquet_path: str,
    chrom_data: ChromData,
    n_anc: int,
) -> None:
    """Close per-bucket parquets are written by the bucket loop; this
    helper just merges them into a single hap-ordered file and removes
    the per-bucket files.

    No-op when ``bucket_writer_paths`` is empty.
    """
    if not bucket_writer_paths:
        return
    from .output import _merge_bucket_parquets
    _merge_bucket_parquets(
        bucket_writer_paths, bucket_hap_indices,
        decode_parquet_path,
        chrom=chrom_data.chrom, pos_bp=chrom_data.pos_bp,
        T=chrom_data.n_sites, K=n_anc, include_max_post=True,
    )
    for p in bucket_writer_paths:
        p.unlink()


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

        calls, max_post, stream_to_parquet = _open_streaming_decode_outputs(
            decode_parquet_path, write_dense_decode,
            H_total, chrom_data.n_sites,
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
        if stream_to_parquet:
            _close_and_merge_bucket_writers(
                bucket_writer_paths, bucket_hap_indices,
                decode_parquet_path, chrom_data, n_anc,
            )

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
            calls, _, _ = _open_streaming_decode_outputs(
                decode_parquet_path, write_dense_decode, H_total, T_sites,
            )

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

                _close_and_merge_bucket_writers(
                    bucket_writer_paths, bucket_hap_indices,
                    decode_parquet_path, chrom_data, n_anc,
                )

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
    ancestry_names: Optional[list[str]] = None,
    write_dense_decode: bool = False,
    seeding_mask: np.ndarray | None = None,
    work_dir=None,
    priors=None,  # popout.prior_spec.Priors | None
    priors_dump_path: Optional[str] = None,
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
        directory.

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
                    wd.load_seed(chrom_data)
                n_leaves = ckpt_model.n_ancestries
                seed_resp = _labels_to_resp(
                    ckpt_labels, chrom_data.n_haps, n_leaves,
                    seeding_mask=seeding_mask,
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
                    wd.save_seed(
                        result.model, leaf_labels_for_ckpt,
                        leaf_info_for_ckpt, chrom_data,
                    )
                    wd.mark_done("seed",
                                 wall_s=_time.perf_counter() - t0_stage)
                log.info("=== Stopped after seeding (--stop-after-seeding) ===")
                return None

            # ----------------------------------------------------------
            # EM STAGE
            # ----------------------------------------------------------
            em_loaded = False
            if wd is not None and wd.stage_done("em"):
                log.info("stage em: loading from %s", wd.stage_path("em"))
                fitted_model = wd.load_em(chrom_data)
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
                    priors=priors,
                    priors_dump_path=priors_dump_path,
                )
                fitted_model = result.model

                if per_hap_T and chrom_data.n_sites < 5000:
                    log.warning(
                        "Per-hap-T was estimated on a small seed chromosome "
                        "(%d sites). Bucket assignments are frozen across "
                        "the genome; quality is bounded by switch-count "
                        "statistics on this chromosome.",
                        chrom_data.n_sites,
                    )

                # Save seed checkpoint (if we ran seeding fresh)
                if wd is not None and not seed_loaded:
                    if leaf_labels_for_ckpt is not None and leaf_info_for_ckpt is not None:
                        wd.save_seed(
                            result.model, leaf_labels_for_ckpt,
                            leaf_info_for_ckpt, chrom_data,
                        )
                        wd.mark_done("seed",
                                     wall_s=_time.perf_counter() - t0_stage)

                # Save EM checkpoint
                if wd is not None:
                    wd.save_em(fitted_model, chrom_data)
                    wd.mark_done("em",
                                 wall_s=_time.perf_counter() - t0_em)

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
                decode = wd.load_decode(chrom_data.chrom, chrom_data)
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
                    wd.save_decode(decode, chrom_data.chrom)
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
                decode = wd.load_decode(chrom_data.chrom, chrom_data)
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
                    gen_per_comp=fitted_model.gen_per_comp,
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
                    wd.save_decode(decode, chrom_data.chrom)
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
