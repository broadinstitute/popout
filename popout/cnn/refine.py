"""Self-training CNN refinement pipeline for local ancestry inference.

This module provides ``run_cnn`` (single chromosome) and ``run_cnn_genome``
(multi-chromosome) as drop-in alternatives to the HMM-based ``run_em`` /
``run_em_genome`` in :mod:`popout.em`.  Both produce :class:`AncestryResult`
objects that are consumed by the same downstream output and panel-export code.
"""

from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..datatypes import AncestryModel, AncestryResult, ChromData, DecodeResult
from ..em import (
    init_model_soft,
    smooth_rare_frequencies,
    update_allele_freq,
    update_generations,
    update_mu,
)
from ..hmm import forward_backward_batched, forward_backward_em
from ..spectral import seed_ancestry_soft
from .features import build_cnn_features
from .model import CNNConfig, CNNParams, cnn_forward, init_cnn_params
from .train import kl_loss, train_cnn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batched CNN inference
# ---------------------------------------------------------------------------

def cnn_inference_batched(
    params: CNNParams,
    config: CNNConfig,
    features: jnp.ndarray,
    crf_params=None,
    batch_size: int = 512,
) -> jnp.ndarray:
    """Run CNN inference in batches over haplotypes.

    Parameters
    ----------
    params : CNNParams
    config : CNNConfig
    features : (H, T, C_in)
    crf_params : optional CRFParams
    batch_size : haplotypes per batch

    Returns
    -------
    gamma : (H, T, A) — posterior ancestry probabilities
    """
    H = features.shape[0]

    def _infer_batch(feat_batch):
        logits = cnn_forward(params, config, feat_batch)
        if crf_params is not None:
            from .crf import crf_marginals
            return crf_marginals(logits, crf_params.W)
        return jax.nn.softmax(logits, axis=-1)

    if H <= batch_size:
        return _infer_batch(features)

    gammas = []
    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        gamma_batch = _infer_batch(features[start:end])
        gammas.append(gamma_batch)
    return jnp.concatenate(gammas, axis=0)


def cnn_inference_decode(
    params: CNNParams,
    config: CNNConfig,
    features: jnp.ndarray,
    crf_params=None,
    batch_size: int = 512,
    compute_max_post: bool = True,
) -> DecodeResult:
    """Batched CNN inference returning DecodeResult (no full gamma).

    Same as cnn_inference_batched but avoids materialising the full
    (H, T, A) posterior on GPU.
    """
    H = features.shape[0]
    T = features.shape[1]
    A = config.n_ancestries

    def _infer_batch(feat_batch):
        logits = cnn_forward(params, config, feat_batch)
        if crf_params is not None:
            from .crf import crf_marginals
            return crf_marginals(logits, crf_params.W)
        return jax.nn.softmax(logits, axis=-1)

    calls = np.zeros((H, T), dtype=np.int8)
    max_post = np.zeros((H, T), dtype=np.float32) if compute_max_post else None
    global_sums = np.zeros((H, A), dtype=np.float64) if compute_max_post else None

    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        gamma = _infer_batch(features[start:end])
        calls[start:end] = np.array(jnp.argmax(gamma, axis=2), dtype=np.int8)
        if compute_max_post:
            max_post[start:end] = np.array(gamma.max(axis=2))
            global_sums[start:end] = np.array(gamma.sum(axis=1))

    return DecodeResult(calls=calls, max_post=max_post, global_sums=global_sums)


# ---------------------------------------------------------------------------
# Single-chromosome CNN refinement
# ---------------------------------------------------------------------------

def run_cnn(
    chrom_data: ChromData,
    n_ancestries: Optional[int] = None,
    gen_since_admix: float = 20.0,
    hmm_batch_size: int = 50_000,
    rng_seed: int = 42,
    stats=None,
    bandwidth_cm: float = 0.05,
    maf_threshold: float = 0.05,
    # CNN-specific
    n_layers: int = 12,
    hidden_dim: int = 64,
    n_epochs: int = 5,
    n_pseudo_rounds: int = 2,
    cnn_lr: float = 1e-3,
    cnn_batch_size: int = 512,
    use_crf: bool = False,
) -> tuple[AncestryResult, CNNParams, object]:
    """Self-bootstrapping CNN refinement for one chromosome.

    Pipeline
    --------
    1. Spectral seed → soft responsibilities
    2. Init model from soft assignments
    3. Bootstrap pseudo-labels via one HMM forward-backward pass
    4. For each pseudo-label round:
       a. Build CNN input features from current model
       b. Train CNN on pseudo-labels
       c. CNN inference → new posteriors
       d. M-step: update allele_freq, mu, T
       e. New posteriors become next round's pseudo-labels
    5. Final CNN inference → posteriors → hard calls

    Returns
    -------
    (AncestryResult, trained_params, crf_params) — result and trained CNN
    weights for reuse on subsequent chromosomes.
    """
    geno_np = chrom_data.geno
    d_morgan = chrom_data.genetic_distances.astype(np.float64)

    log.info("=== CNN refinement on chromosome %s: %d haps × %d sites ===",
             chrom_data.chrom, chrom_data.n_haps, chrom_data.n_sites)

    # --- Stage 0: Spectral seed ---
    log.info("Stage 0: Spectral initialization")
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

    # --- Stage 1: Init model ---
    log.info("Stage 1: Initializing model from soft assignments")
    model = init_model_soft(geno, responsibilities, n_anc, gen_since_admix)
    log.info("  mu = %s", np.array(model.mu).round(3))
    log.info("  T = %.1f generations", model.gen_since_admix)

    # --- Stage 2: Bootstrap pseudo-labels via HMM ---
    log.info("Stage 2: Bootstrap pseudo-labels (one HMM pass)")
    if stats is not None:
        stats.timer_start("cnn/bootstrap")
    gamma = forward_backward_batched(geno, model, d_morgan_j, hmm_batch_size)
    if stats is not None:
        stats.timer_stop("cnn/bootstrap", chrom=chrom_data.chrom)

    # --- Stage 3: CNN setup ---
    c_in = n_anc + 2
    config = CNNConfig(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        kernel_size=3,
        n_ancestries=n_anc,
        c_in=c_in,
    )
    key = jax.random.PRNGKey(rng_seed)
    key, subkey = jax.random.split(key)
    params = init_cnn_params(config, subkey)

    crf_params = None
    crf_loss_fn = None
    if use_crf:
        from .crf import CRFParams, init_crf_params, crf_soft_loss
        crf_params = init_crf_params(n_anc)
        crf_loss_fn = crf_soft_loss

    # --- Stage 4: Self-training rounds ---
    for round_idx in range(n_pseudo_rounds):
        log.info("--- Pseudo-label round %d/%d ---", round_idx + 1, n_pseudo_rounds)

        # Build features from current model
        if stats is not None:
            stats.timer_start("cnn/features")
        features = build_cnn_features(geno, model.allele_freq, d_morgan_j)
        if stats is not None:
            stats.timer_stop("cnn/features", chrom=chrom_data.chrom, round=round_idx)

        # Train CNN on pseudo-labels
        log.info("  Training CNN (%d epochs, batch_size=%d)", n_epochs, cnn_batch_size)
        if stats is not None:
            stats.timer_start("cnn/train")
        key, subkey = jax.random.split(key)
        params, crf_params = train_cnn(
            params, config, features, gamma,
            n_epochs=n_epochs,
            batch_size=cnn_batch_size,
            lr=cnn_lr,
            key=subkey,
            crf_params=crf_params,
            crf_loss_fn=crf_loss_fn,
            stats=stats,
        )
        if stats is not None:
            stats.timer_stop("cnn/train", chrom=chrom_data.chrom, round=round_idx)

        # Inference: get new posteriors
        log.info("  CNN inference on %d haplotypes", chrom_data.n_haps)
        if stats is not None:
            stats.timer_start("cnn/inference")
        gamma = cnn_inference_batched(
            params, config, features,
            crf_params=crf_params,
            batch_size=cnn_batch_size,
        )
        if stats is not None:
            stats.timer_stop("cnn/inference", chrom=chrom_data.chrom, round=round_idx)

        # M-step: update model from CNN posteriors
        log.info("  M-step: updating parameters")
        new_freq = update_allele_freq(geno, gamma)
        if bandwidth_cm > 0:
            new_freq = smooth_rare_frequencies(
                new_freq, pos_cm_j, bandwidth_cm, maf_threshold,
            )
        new_mu = update_mu(gamma)
        new_T = update_generations(gamma, d_morgan_j, model.gen_since_admix, model.mu)

        model = AncestryModel(
            n_ancestries=n_anc,
            mu=new_mu,
            gen_since_admix=new_T,
            allele_freq=new_freq,
            mismatch=model.mismatch,
        )
        log.info("  mu = %s", np.array(model.mu).round(3))
        log.info("  T = %.1f generations", model.gen_since_admix)

    # --- Stage 5: Final inference (streaming decode) ---
    log.info("Final CNN inference")
    features = build_cnn_features(geno, model.allele_freq, d_morgan_j)
    decode = cnn_inference_decode(
        params, config, features,
        crf_params=crf_params,
        batch_size=cnn_batch_size,
    )

    result = AncestryResult(
        calls=decode.calls,
        model=model,
        chrom=chrom_data.chrom,
        decode=decode,
    )

    for a in range(n_anc):
        prop = float((result.calls == a).mean())
        log.info("  Ancestry %d: %.1f%% of genome", a, 100 * prop)

    return result, params, crf_params


# ---------------------------------------------------------------------------
# Multi-chromosome wrapper
# ---------------------------------------------------------------------------

def run_cnn_genome(
    chrom_iter,
    n_ancestries: Optional[int] = None,
    gen_since_admix: float = 20.0,
    hmm_batch_size: int = 50_000,
    rng_seed: int = 42,
    stats=None,
    bandwidth_cm: float = 0.05,
    maf_threshold: float = 0.05,
    # CNN-specific
    n_layers: int = 12,
    hidden_dim: int = 64,
    n_epochs: int = 5,
    n_pseudo_rounds: int = 2,
    cnn_lr: float = 1e-3,
    cnn_batch_size: int = 512,
    use_crf: bool = False,
) -> list[AncestryResult]:
    """Run CNN refinement across all chromosomes.

    Strategy
    --------
    1. Run full CNN self-training on the first (seed) chromosome.
    2. For subsequent chromosomes: re-init allele frequencies via spectral,
       reuse trained CNN weights, run 1 fine-tuning epoch + inference.

    Returns
    -------
    List of AncestryResult, one per chromosome.
    """
    results = []
    fitted_model = None
    trained_params = None
    crf_params_fitted = None
    config = None

    for chrom_data in chrom_iter:
        if stats is not None:
            stats.timer_start(f"chrom/{chrom_data.chrom}")

        if fitted_model is None:
            # Seed chromosome: full self-training
            log.info("=== Seed chromosome: %s (full CNN self-training) ===",
                     chrom_data.chrom)
            result, trained_params, crf_params_fitted = run_cnn(
                chrom_data,
                n_ancestries=n_ancestries,
                gen_since_admix=gen_since_admix,
                hmm_batch_size=hmm_batch_size,
                rng_seed=rng_seed,
                stats=stats,
                bandwidth_cm=bandwidth_cm,
                maf_threshold=maf_threshold,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                n_epochs=n_epochs,
                n_pseudo_rounds=n_pseudo_rounds,
                cnn_lr=cnn_lr,
                cnn_batch_size=cnn_batch_size,
                use_crf=use_crf,
            )
            fitted_model = result.model
            config = CNNConfig(
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                kernel_size=3,
                n_ancestries=fitted_model.n_ancestries,
                c_in=fitted_model.n_ancestries + 2,
            )
        else:
            # Subsequent chromosomes: reuse CNN, warm-start allele freqs
            log.info("=== Chromosome %s (warm-started CNN, 1 fine-tune epoch) ===",
                     chrom_data.chrom)

            geno = jnp.array(chrom_data.geno)
            d_morgan_j = jnp.array(chrom_data.genetic_distances)
            pos_cm_j = jnp.array(chrom_data.pos_cm.astype(np.float32))

            # Quick spectral init for chromosome-specific allele frequencies
            _labels, resp, n_anc = seed_ancestry_soft(
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
                mismatch=fitted_model.mismatch,
            )

            # Build features and run one fine-tuning epoch
            features = build_cnn_features(geno, model.allele_freq, d_morgan_j)

            # Bootstrap pseudo-labels from current CNN (no HMM pass needed)
            gamma = cnn_inference_batched(
                trained_params, config, features,
                crf_params=crf_params_fitted,
                batch_size=cnn_batch_size,
            )

            # Fine-tune CNN for 1 epoch
            key = jax.random.PRNGKey(rng_seed + hash(chrom_data.chrom) % 2**31)
            crf_loss_fn = None
            if use_crf and crf_params_fitted is not None:
                from .crf import crf_soft_loss
                crf_loss_fn = crf_soft_loss

            trained_params, crf_params_fitted = train_cnn(
                trained_params, config, features, gamma,
                n_epochs=1,
                batch_size=cnn_batch_size,
                lr=cnn_lr * 0.1,  # reduced LR for fine-tuning
                key=key,
                crf_params=crf_params_fitted,
                crf_loss_fn=crf_loss_fn,
            )

            # Final inference
            gamma = cnn_inference_batched(
                trained_params, config, features,
                crf_params=crf_params_fitted,
                batch_size=cnn_batch_size,
            )

            # Update allele frequencies from CNN posteriors
            new_freq = update_allele_freq(geno, gamma)
            if bandwidth_cm > 0:
                new_freq = smooth_rare_frequencies(
                    new_freq, pos_cm_j, bandwidth_cm, maf_threshold,
                )
            model = AncestryModel(
                n_ancestries=model.n_ancestries,
                mu=model.mu,
                gen_since_admix=model.gen_since_admix,
                allele_freq=new_freq,
                mismatch=model.mismatch,
            )

            # Build decode result from the final gamma without keeping full tensor
            decode = DecodeResult(
                calls=np.array(jnp.argmax(gamma, axis=2), dtype=np.int8),
                max_post=np.array(gamma.max(axis=2)),
                global_sums=np.array(gamma.sum(axis=1)),
            )
            del gamma
            result = AncestryResult(
                calls=decode.calls,
                model=model,
                chrom=chrom_data.chrom,
                decode=decode,
            )

        if stats is not None:
            elapsed = stats.timer_stop(f"chrom/{chrom_data.chrom}", chrom=chrom_data.chrom)
            throughput = chrom_data.n_haps * chrom_data.n_sites / max(elapsed, 1e-6)
            stats.emit("runtime/throughput", round(throughput), chrom=chrom_data.chrom)

        results.append(result)

    return results
