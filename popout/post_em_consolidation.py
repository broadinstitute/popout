"""Post-EM ancestry consolidation.

After EM converges, flag ancestries that lack statistical support (tiny
mu, few high-posterior sites, or near-duplicate of a recursion sibling)
and merge them into their nearest neighbour.  This avoids passing ghost
ancestries to downstream output that would confuse interpretation.

The consolidation is post-hoc: it modifies the model and relabels calls
without rerunning EM.  The converged posteriors serve as evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .datatypes import AncestryModel, AncestryResult, DecodeResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F_ST between two ancestry frequency vectors
# ---------------------------------------------------------------------------

def _pairwise_fst(
    freq_i: np.ndarray,
    freq_j: np.ndarray,
) -> float:
    """Hudson F_ST between two ancestry allele-frequency vectors.

    Uses the per-site ratio estimator averaged across sites:
        F_ST(t) = (p_i - p_j)^2 / (p_bar * (1 - p_bar))
    where p_bar = (p_i + p_j) / 2  (unweighted — ancestries are
    conceptual clusters, not populations with known sizes).
    """
    p_bar = (freq_i + freq_j) / 2.0
    denom = p_bar * (1.0 - p_bar)
    valid = denom > 1e-8
    if valid.sum() == 0:
        return 0.0
    numer = (freq_i - freq_j) ** 2
    return float(np.mean(numer[valid] / denom[valid]))


# ---------------------------------------------------------------------------
# Consolidation logic
# ---------------------------------------------------------------------------

@dataclass
class _ConsolidationAction:
    """Record of one ancestry merge for the audit TSV."""
    source_idx: int
    target_idx: int
    reason: str
    mu_source: float
    mu_target: float
    fst: float
    n_high_post: int
    leaf_path_source: str
    leaf_path_target: str


def consolidate(
    results: list[AncestryResult],
    out_prefix: Optional[str] = None,
    leaf_paths: Optional[list[str]] = None,
    min_high_post: int = 1000,
    min_mu: float = 0.005,
    sibling_fst_threshold: float = 0.008,
) -> list[AncestryResult]:
    """Post-EM consolidation: merge unsupported ancestries.

    Parameters
    ----------
    results : list of AncestryResult, one per chromosome
    out_prefix : write audit TSV to {out_prefix}.post_em_consolidation.tsv
    leaf_paths : list of str, one per ancestry, from recursive seeding.
        Used to determine sibling relationships.  None if GMM seeding.
    min_high_post : flag ancestry if fewer haplotype-sites exceed 0.8 posterior
    min_mu : flag ancestry if global proportion below this
    sibling_fst_threshold : flag if F_ST to nearest ancestry < this AND
        they are recursion siblings

    Returns
    -------
    Possibly modified list of AncestryResult with reduced K.
    """
    if not results:
        return results

    # Use first chromosome's model as reference (allele_freq, mu)
    ref = results[0]
    model = ref.model
    A = model.n_ancestries
    freq = np.array(model.allele_freq)   # (A, T)
    mu = np.array(model.mu)              # (A,)

    if A <= 1:
        log.info("Post-EM consolidation: K=1, nothing to consolidate")
        return results

    # --- Compute per-ancestry diagnostics ---

    # Count high-posterior haplotype-sites per ancestry across all chroms
    high_post_counts = np.zeros(A, dtype=np.int64)
    _CHUNK = 50_000
    for res in results:
        if res.decode is not None and res.decode.max_post is not None:
            mp = res.decode.max_post
            calls = res.calls
            H_res = calls.shape[0]
            for start in range(0, H_res, _CHUNK):
                end = min(start + _CHUNK, H_res)
                mask_hp = mp[start:end] > 0.8
                bc = np.bincount(
                    calls[start:end][mask_hp], minlength=A,
                )
                high_post_counts[:len(bc)] += bc[:A]
                del mask_hp
        elif res.decode is not None and res.decode.parquet_path is not None:
            # Stream from parquet to avoid 54 GB allocation
            from .output import _iter_max_post_groups
            calls = res.calls
            for rg_start, rg_end, mp_chunk in _iter_max_post_groups(
                res.decode.parquet_path,
            ):
                calls_chunk = calls[rg_start:rg_end]
                mask_hp = mp_chunk > 0.8
                bc = np.bincount(
                    calls_chunk[mask_hp], minlength=A,
                )
                high_post_counts[:len(bc)] += bc[:A]
                del mask_hp
        else:
            # No posterior info — use calls only (assume all high-conf)
            calls = res.calls
            H_res = calls.shape[0]
            for start in range(0, H_res, _CHUNK):
                end = min(start + _CHUNK, H_res)
                bc = np.bincount(calls[start:end].ravel(), minlength=A)
                high_post_counts[:len(bc)] += bc[:A]

    # Pairwise F_ST matrix
    fst_matrix = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(i + 1, A):
            f = _pairwise_fst(freq[i], freq[j])
            fst_matrix[i, j] = f
            fst_matrix[j, i] = f

    # Nearest ancestry and F_ST for each
    np.fill_diagonal(fst_matrix, np.inf)
    nearest_idx = np.argmin(fst_matrix, axis=1)
    nearest_fst = fst_matrix[np.arange(A), nearest_idx]

    # --- Log diagnostics ---
    log.info("Post-EM consolidation diagnostics (K=%d):", A)
    for a in range(A):
        path = leaf_paths[a] if leaf_paths and a < len(leaf_paths) else "?"
        log.info("  anc %d (path=%s): mu=%.4f, n_high_post=%d, "
                 "nearest=%d (F_ST=%.4f)",
                 a, path, mu[a], high_post_counts[a],
                 nearest_idx[a], nearest_fst[a])

    # --- Flag candidates ---
    actions: list[_ConsolidationAction] = []
    flagged = set()

    for a in range(A):
        if a in flagged:
            continue

        reasons = []
        if high_post_counts[a] < min_high_post:
            reasons.append(f"n_high_post={high_post_counts[a]}<{min_high_post}")
        if mu[a] < min_mu:
            reasons.append(f"mu={mu[a]:.4f}<{min_mu}")

        # Sibling F_ST check
        ni = int(nearest_idx[a])
        is_sibling = False
        if leaf_paths and a < len(leaf_paths) and ni < len(leaf_paths):
            pa, pn = leaf_paths[a], leaf_paths[ni]
            is_sibling = (len(pa) == len(pn) and len(pa) >= 2
                          and pa[:-1] == pn[:-1])
        if nearest_fst[a] < sibling_fst_threshold and is_sibling:
            reasons.append(
                f"sibling_fst={nearest_fst[a]:.4f}<{sibling_fst_threshold}"
            )

        if not reasons:
            continue

        target = ni
        # Don't merge into something that's also flagged
        if target in flagged:
            continue

        src_path = leaf_paths[a] if leaf_paths and a < len(leaf_paths) else "?"
        tgt_path = leaf_paths[target] if leaf_paths and target < len(leaf_paths) else "?"

        actions.append(_ConsolidationAction(
            source_idx=a,
            target_idx=target,
            reason="; ".join(reasons),
            mu_source=float(mu[a]),
            mu_target=float(mu[target]),
            fst=float(nearest_fst[a]),
            n_high_post=int(high_post_counts[a]),
            leaf_path_source=src_path,
            leaf_path_target=tgt_path,
        ))
        flagged.add(a)

    if not actions:
        log.info("Post-EM consolidation: no ancestries flagged (K=%d)", A)
        if out_prefix:
            _write_report(out_prefix, actions, A)
        return results

    # --- Execute merges ---
    log.info("Post-EM consolidation: merging %d ancestries:", len(actions))
    for act in actions:
        log.info("  anc %d (path=%s, mu=%.4f) → anc %d (path=%s): %s",
                 act.source_idx, act.leaf_path_source, act.mu_source,
                 act.target_idx, act.leaf_path_target, act.reason)

    # Build merge map: old ancestry → new ancestry
    merge_map = np.arange(A, dtype=np.int32)
    for act in actions:
        merge_map[act.source_idx] = act.target_idx

    # Resolve transitive merges (A→B→C should map A→C)
    for _ in range(A):
        changed = False
        for i in range(A):
            if merge_map[merge_map[i]] != merge_map[i]:
                merge_map[i] = merge_map[merge_map[i]]
                changed = True
        if not changed:
            break

    # Surviving ancestries
    surviving = sorted(set(merge_map))
    new_A = len(surviving)
    # Build old → new contiguous label map
    old_to_new = np.full(A, -1, dtype=np.int32)
    for new_idx, old_idx in enumerate(surviving):
        old_to_new[old_idx] = new_idx
    # Map merged sources through to their target's new index
    remap = np.array([old_to_new[merge_map[a]] for a in range(A)], dtype=np.int32)

    assert A <= 127, (
        f"Cannot encode A={A} ancestries in int8. "
        "Widen calls dtype throughout the pipeline first."
    )
    assert new_A >= 1, f"Consolidation removed all ancestries: new_A={new_A}"
    # Cast to int8 before indexing into (H, T) int8 calls — fancy-indexing
    # returns the indexer's dtype, so int32 remap would produce 108 GB int32
    # output at biobank scale.
    remap_i8 = remap.astype(np.int8)

    log.info("Post-EM consolidation: K=%d → K=%d", A, new_A)

    # --- Update all results ---
    new_results = []
    for res in results:
        old_model = res.model
        old_freq = np.array(old_model.allele_freq)  # (A, T)
        old_mu = np.array(old_model.mu)              # (A,)

        # Merge allele frequencies weighted by mu
        new_freq = np.zeros((new_A, old_freq.shape[1]), dtype=np.float64)
        new_mu = np.zeros(new_A, dtype=np.float64)
        for old_a in range(A):
            new_a = int(remap[old_a])
            new_mu[new_a] += old_mu[old_a]
            new_freq[new_a] += old_freq[old_a] * old_mu[old_a]

        # Normalize frequencies by combined mu
        for new_a in range(new_A):
            if new_mu[new_a] > 0:
                new_freq[new_a] /= new_mu[new_a]
        # Renormalize mu
        new_mu /= new_mu.sum()

        import jax.numpy as jnp
        new_model = AncestryModel(
            n_ancestries=new_A,
            mu=jnp.array(new_mu, dtype=jnp.float32),
            gen_since_admix=old_model.gen_since_admix,
            allele_freq=jnp.array(new_freq, dtype=jnp.float32),
            gen_per_hap=old_model.gen_per_hap,
            bucket_centers=old_model.bucket_centers,
            bucket_assignments=old_model.bucket_assignments,
        )

        # Remap calls (int8 indexer → int8 output, avoids 4× bloat)
        new_calls = remap_i8[res.calls]

        # Remap global_sums if available
        new_decode = None
        if res.decode is not None:
            new_gs = None
            if res.decode.global_sums is not None:
                old_gs = res.decode.global_sums  # (H, A)
                new_gs = np.zeros((old_gs.shape[0], new_A), dtype=old_gs.dtype)
                for old_a in range(A):
                    new_gs[:, remap[old_a]] += old_gs[:, old_a]
            new_decode = DecodeResult(
                calls=new_calls,
                max_post=res.decode.max_post,
                global_sums=new_gs,
                parquet_path=res.decode.parquet_path,
            )

        new_results.append(AncestryResult(
            calls=new_calls,
            model=new_model,
            chrom=res.chrom,
            decode=new_decode,
            posteriors=res.posteriors,
            spectral=res.spectral,
        ))

    # --- Write audit report ---
    if out_prefix:
        _write_report(out_prefix, actions, A)

    return new_results


def _write_report(
    out_prefix: str,
    actions: list[_ConsolidationAction],
    original_K: int,
) -> None:
    """Write consolidation audit TSV."""
    path = f"{out_prefix}.post_em_consolidation.tsv"
    with open(path, "w") as f:
        f.write("source_idx\ttarget_idx\treason\tmu_source\tmu_target\t"
                "fst_to_target\tn_high_post\tleaf_path_source\t"
                "leaf_path_target\n")
        for act in actions:
            f.write(f"{act.source_idx}\t{act.target_idx}\t{act.reason}\t"
                    f"{act.mu_source:.6f}\t{act.mu_target:.6f}\t"
                    f"{act.fst:.6f}\t{act.n_high_post}\t"
                    f"{act.leaf_path_source}\t{act.leaf_path_target}\n")
    log.info("Wrote consolidation report to %s (original K=%d, %d merges)",
             path, original_K, len(actions))
