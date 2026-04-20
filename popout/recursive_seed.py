"""Recursive K=2 seeding for population structure discovery.

Discovers population structure by recursively splitting haplotypes via
K=2 EM, then returns flat leaf assignments for use as seed responsibilities
in the main EM pipeline.

The algorithm:
  1. Start with all haplotypes in one cluster.
  2. For each cluster, test whether it has substructure (BIC on sub-PCA).
  3. If yes, run K=2 popout EM on the cluster's haplotypes, split by
     argmax of per-haplotype mean posteriors.
  4. Recurse on each child until no more splits are justified.

The result is a set of leaf populations, each assigned a flat integer label.
These labels are converted to a one-hot responsibility matrix and passed
to the main flat-K EM as seed_responsibilities.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .datatypes import ChromData
from .spectral import _bic_split_test, _genotypes_to_pca_projection

log = logging.getLogger(__name__)


@dataclass
class LeafInfo:
    label: int
    n_haps: int
    depth: int
    path: str       # e.g. "L01" — sequence of 0/1 bits from root to leaf
    bic_score: float  # BIC delta of the split that created this leaf's parent


# ---------------------------------------------------------------------------
# Sub-PCA on raw genotypes
# ---------------------------------------------------------------------------

def _geno_sub_pca(
    geno: np.ndarray,
    n_components: int = 2,
    max_snps: int = 10_000,
    max_haps_svd: int = 100_000,
    projection_batch: int = 50_000,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """Patterson-normalised PCA on a genotype subset.

    Delegates to the shared _genotypes_to_pca_projection helper in
    spectral.py, which handles biobank-scale inputs by subsampling
    haplotypes for SVD and projecting all haplotypes in batches.

    Returns
    -------
    proj : (H_sub, n_components) — PCA projection
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    proj, _S, _key = _genotypes_to_pca_projection(
        geno, n_components, key,
        max_snps=max_snps,
        max_haps_svd=max_haps_svd,
        projection_batch=projection_batch,
    )
    return jnp.array(proj)


# ---------------------------------------------------------------------------
# Node structure for the recursion tree (internal bookkeeping)
# ---------------------------------------------------------------------------

@dataclass
class _SplitNode:
    """Internal node in the recursion queue."""
    indices: np.ndarray   # integer indices into the full haplotype array
    depth: int
    path: str             # bit-path from root, e.g. "L", "L0", "L01"
    bic_delta: float = 0.0  # BIC improvement at this node's creation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recursive_split_seed(
    geno: np.ndarray,
    *,
    min_cluster_size: int = 1000,
    bic_per_sample: float = 0.01,
    max_depth: int = 6,
    max_leaves: int = 20,
    em_iter_per_split: int = 3,
    rng_seed: int = 42,
    stats=None,
    chrom_data: ChromData | None = None,
    gen_since_admix: float = 20.0,
    merge_hellinger_threshold: float = 0.08,
    split_restarts: int = 5,
    balance_bic_tolerance: float = 0.10,
    max_haps_svd: int = 100_000,
    projection_batch: int = 50_000,
    dump_pre_merge_path: Optional[str] = None,
) -> tuple[np.ndarray, list[LeafInfo]]:
    """Recursively split haplotypes via K=2 EM.

    Parameters
    ----------
    geno : (H, T) uint8 genotype matrix
    min_cluster_size : don't split clusters smaller than 2× this
    bic_per_sample : per-sample BIC improvement required to split (scales with N)
    max_depth : maximum recursion depth
    max_leaves : maximum number of leaf populations
    em_iter_per_split : EM iterations per K=2 split
    rng_seed : random seed
    stats : optional StatsCollector
    chrom_data : ChromData for K=2 EM (uses pos_bp, pos_cm, chrom)
    gen_since_admix : initial T for K=2 EM splits
    merge_hellinger_threshold : merge leaves whose allele-frequency Hellinger
        distance is below this threshold (0 = disable merging)

    Returns
    -------
    leaf_labels : (H,) int32 — flat labels 0..(n_leaves - 1) per haplotype
    leaf_info : list of LeafInfo — metadata per leaf, in label order
    """
    H, T = geno.shape
    key = jax.random.PRNGKey(rng_seed)

    log.info("Recursive seed (%d haps, max_depth=%d, max_leaves=%d):",
             H, max_depth, max_leaves)

    # Build ChromData if not provided (for K=2 EM splits)
    if chrom_data is None:
        chrom_data = ChromData(
            geno=geno,
            pos_bp=np.arange(T, dtype=np.int64),
            pos_cm=np.linspace(0, 100.0, T),
            chrom="recursive",
        )

    # BFS queue
    queue: deque[_SplitNode] = deque()
    queue.append(_SplitNode(
        indices=np.arange(H, dtype=np.int64),
        depth=0,
        path="L",
    ))

    leaves: list[_SplitNode] = []
    split_log: list[dict] = []  # for tree visualisation

    while queue:
        # Check if we've hit the leaf limit
        if len(leaves) + len(queue) >= max_leaves:
            # Remaining queue items become leaves
            while queue:
                node = queue.popleft()
                leaves.append(node)
                split_log.append({
                    "path": node.path, "n": len(node.indices),
                    "action": "LEAF (max_leaves)", "bic": node.bic_delta,
                    "depth": node.depth,
                })
            break

        node = queue.popleft()
        n = len(node.indices)
        log.info("  [%d leaves, %d queued] Testing %s (%d haps, depth=%d)",
                 len(leaves), len(queue), node.path, n, node.depth)

        # Too small to split?
        if n < 2 * min_cluster_size:
            leaves.append(node)
            split_log.append({
                "path": node.path, "n": n,
                "action": "LEAF (too small)", "bic": node.bic_delta,
                "depth": node.depth,
            })
            continue

        # Too deep?
        if node.depth >= max_depth:
            leaves.append(node)
            split_log.append({
                "path": node.path, "n": n,
                "action": "LEAF (max_depth)", "bic": node.bic_delta,
                "depth": node.depth,
            })
            continue

        # Sub-PCA on this cluster's raw genotypes
        key, subkey = jax.random.split(key)
        sub_geno = geno[node.indices]
        sub_proj = _geno_sub_pca(
            sub_geno, n_components=2, key=subkey,
            max_haps_svd=max_haps_svd, projection_batch=projection_batch,
        )

        # BIC split test (gates whether we attempt a split at all)
        key, subkey = jax.random.split(key)
        should_split, _ = _bic_split_test(sub_proj, subkey, bic_per_sample)

        if not should_split:
            leaves.append(node)
            split_log.append({
                "path": node.path, "n": n,
                "action": "LEAF (BIC no split)", "bic": node.bic_delta,
                "depth": node.depth,
            })
            continue

        # Balance-preferring K=2 split: multiple GMM restarts, pick most
        # balanced among candidates within BIC tolerance of the best.
        key, subkey = jax.random.split(key)
        selected_labels, bic_delta, sel_balance = _select_balanced_split(
            sub_proj, subkey, n_restarts=split_restarts,
            bic_tolerance=balance_bic_tolerance, node_path=node.path,
        )

        if selected_labels is None:
            leaves.append(node)
            split_log.append({
                "path": node.path, "n": n,
                "action": "LEAF (no valid split)", "bic": node.bic_delta,
                "depth": node.depth,
            })
            continue

        # Use selected GMM labels as seed for K=2 EM
        sel_labels_np = np.array(selected_labels)
        # Build seed responsibilities from GMM labels
        seed_resp_k2 = jnp.zeros((n, 2), dtype=jnp.float32)
        seed_resp_k2 = seed_resp_k2.at[jnp.arange(n), selected_labels].set(1.0)

        key, subkey = jax.random.split(key)
        child_labels = _run_k2_em_split(
            geno, node.indices, chrom_data,
            n_iter=em_iter_per_split,
            gen_since_admix=gen_since_admix,
            rng_seed=int(subkey[0]) & 0x7FFFFFFF,
            seed_resp=seed_resp_k2,
        )

        idx_0 = node.indices[child_labels == 0]
        idx_1 = node.indices[child_labels == 1]

        # Accept the split if both children are large enough
        min_accept = max(min_cluster_size // 4, 50)
        if len(idx_0) < min_accept or len(idx_1) < min_accept:
            leaves.append(node)
            split_log.append({
                "path": node.path, "n": n,
                "action": "LEAF (child too small)", "bic": node.bic_delta,
                "depth": node.depth,
            })
            continue

        split_log.append({
            "path": node.path, "n": n,
            "action": f"split → {len(idx_0)}+{len(idx_1)}",
            "bic": bic_delta,
            "depth": node.depth,
            "balance": sel_balance,
        })

        queue.append(_SplitNode(idx_0, node.depth + 1, node.path + "0", bic_delta))
        queue.append(_SplitNode(idx_1, node.depth + 1, node.path + "1", bic_delta))

    # Assign flat labels to leaves in finalisation order
    leaf_labels = np.zeros(H, dtype=np.int32)
    leaf_info: list[LeafInfo] = []

    for i, leaf in enumerate(leaves):
        leaf_labels[leaf.indices] = i
        leaf_info.append(LeafInfo(
            label=i,
            n_haps=len(leaf.indices),
            depth=leaf.depth,
            path=leaf.path,
            bic_score=leaf.bic_delta,
        ))

    # Log tree, balance summary, and leaf summary
    _log_tree(split_log, H)
    splits_with_balance = [e for e in split_log if "balance" in e]
    if splits_with_balance:
        balances = [e["balance"] for e in splits_with_balance]
        log.info("Split balance summary: mean=%.2f across %d splits",
                 np.mean(balances), len(balances))
    _log_leaf_summary(leaf_info, H)

    # Pre-merge dump (always write if path is set — cheap insurance)
    if dump_pre_merge_path is not None:
        try:
            _dump_pre_merge(geno, leaf_labels, leaf_info, dump_pre_merge_path)
            log.info("Pre-merge dump written to %s.{leaves,leaf_meta,leaf_freqs}",
                     dump_pre_merge_path)
        except Exception as e:
            log.warning("Failed to write pre-merge dump: %s", e)

    # Post-hoc merge of similar leaves
    if merge_hellinger_threshold > 0 and len(leaf_info) > 1:
        n_before = len(leaf_info)
        leaf_labels, leaf_info = _merge_close_leaves(
            geno, leaf_labels, leaf_info,
            hellinger_threshold=merge_hellinger_threshold,
        )
        if len(leaf_info) < n_before:
            log.info("Post-hoc merge: %d → %d leaves (Hellinger < %.3f)",
                     n_before, len(leaf_info), merge_hellinger_threshold)
            _log_leaf_summary(leaf_info, H)

    # Emit stats
    if stats is not None:
        stats.emit("recursive_seed/n_leaves", len(leaf_info))
        max_depth_reached = max(li.depth for li in leaf_info)
        stats.emit("recursive_seed/max_depth_reached", max_depth_reached)
        stats.emit("recursive_seed/leaf_sizes",
                    [li.n_haps for li in leaf_info])
        bic_scores = [e["bic"] for e in split_log if "split" in e["action"]]
        stats.emit("recursive_seed/bic_scores", bic_scores)

    return leaf_labels, leaf_info


# ---------------------------------------------------------------------------
# Post-hoc leaf merge by Hellinger distance
# ---------------------------------------------------------------------------

def _are_merge_candidates(path_a: str, path_b: str) -> bool:
    """Check if two tree paths are candidates for Hellinger merging.

    Two leaves are candidates if they share a parent — i.e. their paths
    are the same length and differ only in the last character. This
    restricts merging to siblings that were produced by the same K=2
    split, preventing cascading merges across distant branches.
    """
    if len(path_a) < 2 or len(path_b) < 2:
        return True  # root children are always candidates
    return len(path_a) == len(path_b) and path_a[:-1] == path_b[:-1]


def _merge_close_leaves(
    geno: np.ndarray,
    leaf_labels: np.ndarray,
    leaf_info: list[LeafInfo],
    hellinger_threshold: float = 0.04,
    pseudocount: float = 0.5,
) -> tuple[np.ndarray, list[LeafInfo]]:
    """Merge sibling leaf pairs whose allele-frequency Hellinger distance is below threshold.

    Only considers sibling pairs (same parent in the recursion tree).
    Frequencies are computed once and updated incrementally on each merge
    to avoid re-scanning the full genotype array.
    """
    labels = leaf_labels.copy()
    info = list(leaf_info)
    T = geno.shape[1]

    # Compute frequencies once — O(T) memory per leaf, not O(n_masked × T).
    # numpy.sum(dtype=float64) accumulates without materializing a cast copy.
    n_leaves = len(info)
    freq = np.zeros((n_leaves, T), dtype=np.float64)
    counts = np.zeros(n_leaves, dtype=np.int64)
    for i, li in enumerate(info):
        mask = labels == li.label
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            freq[i] = (geno[mask].sum(axis=0, dtype=np.float64) + pseudocount) / (counts[i] + 2 * pseudocount)

    while len(info) > 1:
        # Find the closest sibling pair
        sqrt_p = np.sqrt(np.clip(freq, 0, 1))
        sqrt_1mp = np.sqrt(np.clip(1 - freq, 0, 1))
        best_dist = float('inf')
        best_i, best_j = -1, -1
        for i in range(len(info)):
            for j in range(i + 1, len(info)):
                if not _are_merge_candidates(info[i].path, info[j].path):
                    continue
                d2 = 0.5 * (
                    np.sum((sqrt_p[i] - sqrt_p[j]) ** 2) +
                    np.sum((sqrt_1mp[i] - sqrt_1mp[j]) ** 2)
                )
                h = np.sqrt(max(d2 / T, 0))
                if h < best_dist:
                    best_dist = h
                    best_i, best_j = i, j

        if best_dist >= hellinger_threshold:
            break

        li_i = info[best_i]
        li_j = info[best_j]
        log.info("  Merging leaf %d (path=%s, %d haps) into leaf %d (path=%s, %d haps): "
                 "Hellinger = %.4f",
                 li_j.label, li_j.path, li_j.n_haps,
                 li_i.label, li_i.path, li_i.n_haps, best_dist)

        labels[labels == li_j.label] = li_i.label

        # Update frequencies incrementally: un-normalize, add raw counts, re-normalize.
        new_count = counts[best_i] + counts[best_j]
        raw_i = freq[best_i] * (counts[best_i] + 2 * pseudocount) - pseudocount
        raw_j = freq[best_j] * (counts[best_j] + 2 * pseudocount) - pseudocount
        freq[best_i] = (raw_i + raw_j + pseudocount) / (new_count + 2 * pseudocount)
        counts[best_i] = new_count

        # Drop row best_j from freq/counts arrays
        freq = np.delete(freq, best_j, axis=0)
        counts = np.delete(counts, best_j, axis=0)

        # Update info: merged path is common prefix
        shared = 0
        for k in range(min(len(li_i.path), len(li_j.path))):
            if li_i.path[k] == li_j.path[k]:
                shared = k + 1
            else:
                break
        merged_path = li_i.path[:shared]
        info[best_i] = LeafInfo(
            label=li_i.label,
            n_haps=li_i.n_haps + li_j.n_haps,
            depth=min(li_i.depth, li_j.depth),
            path=merged_path,
            bic_score=li_i.bic_score,
        )
        info.pop(best_j)

    # Remap labels to contiguous 0..n_new-1
    old_labels = sorted(set(li.label for li in info))
    remap = {old: new for new, old in enumerate(old_labels)}
    new_labels = np.array([remap[l] for l in labels], dtype=np.int32)
    new_info = []
    for new_label, li in enumerate(info):
        new_info.append(LeafInfo(
            label=new_label,
            n_haps=li.n_haps,
            depth=li.depth,
            path=li.path,
            bic_score=li.bic_score,
        ))

    return new_labels, new_info


# ---------------------------------------------------------------------------
# K=2 EM on a haplotype subset (Path B: stripped-down helper)
# ---------------------------------------------------------------------------

def _select_balanced_split(
    X: jnp.ndarray,
    key: jax.Array,
    n_restarts: int = 5,
    bic_tolerance: float = 0.10,
    node_path: str = "",
) -> tuple[jnp.ndarray | None, float, float]:
    """Run multiple K=2 GMM restarts and pick the most balanced among top-BIC.

    Returns (labels, bic_delta, balance_score) or (None, 0, 0) if no valid split.
    """
    from .spectral import _gmm_single

    n, d = X.shape

    # 1-component BIC (closed-form)
    mean1 = X.mean(axis=0)
    var1 = jnp.maximum(jnp.var(X, axis=0), 1e-6)
    log_det1 = jnp.sum(jnp.log(var1))
    mahal1 = jnp.sum((X - mean1) ** 2 / var1, axis=1)
    ll1 = float(jnp.sum(-0.5 * (d * jnp.log(2 * jnp.pi) + log_det1 + mahal1)))
    k1_params = 2 * d
    bic1 = -2 * ll1 + k1_params * np.log(n)

    # Multiple K=2 GMM restarts
    candidates = []  # (labels, bic_improvement, balance)
    for r in range(n_restarts):
        key, subkey = jax.random.split(key)
        labels_r, _resp_r, ll2_per = _gmm_single(X, 2, subkey, max_iter=50, tol=1e-4)
        ll2 = float(ll2_per * n)
        k2_params = 4 * d + 1
        bic2 = -2 * ll2 + k2_params * np.log(n)
        bic_improvement = float(bic1 - bic2)

        labels_np = np.array(labels_r)
        n_0 = int((labels_np == 0).sum())
        n_1 = n - n_0
        balance = min(n_0, n_1) / n  # in [0, 0.5]

        candidates.append((labels_r, bic_improvement, balance))
        log.debug("  Split %s restart %d: BIC improvement %.0f, balance %.2f",
                  node_path, r, bic_improvement, balance)

    if not candidates:
        return None, 0.0, 0.0

    # Find the best BIC improvement
    best_bic = max(c[1] for c in candidates)
    if best_bic <= 0:
        return None, 0.0, 0.0

    # Filter to candidates within tolerance of best BIC
    bic_floor = best_bic * (1.0 - bic_tolerance)
    eligible = [(i, c) for i, c in enumerate(candidates) if c[1] >= bic_floor]

    # Among eligible, pick the most balanced
    eligible.sort(key=lambda x: x[1][2], reverse=True)
    sel_idx, (sel_labels, sel_bic, sel_balance) = eligible[0]

    if n_restarts > 1:
        # Log the selection at DEBUG
        eligible_indices = [e[0] for e in eligible]
        log.debug("  Split %s: best BIC %.0f, within %d%% tolerance: %s. "
                  "Selected restart %d (BIC %.0f, balance %.2f).",
                  node_path, best_bic, int(bic_tolerance * 100),
                  eligible_indices, sel_idx, sel_bic, sel_balance)

    return sel_labels, sel_bic, sel_balance


def _run_k2_em_split(
    geno: np.ndarray,
    indices: np.ndarray,
    chrom_data: ChromData,
    n_iter: int = 3,
    gen_since_admix: float = 20.0,
    rng_seed: int = 42,
    seed_resp: jnp.ndarray | None = None,
) -> np.ndarray:
    """Run K=2 popout EM on a haplotype subset, return hard labels.

    Parameters
    ----------
    seed_resp : optional (H_subset, 2) initial responsibilities.
        If provided, skips spectral init and uses these directly.
    """
    from .em import init_model_soft
    from .hmm import forward_backward_em

    sub_geno = geno[indices]
    H_sub, T = sub_geno.shape

    geno_j = jnp.array(sub_geno)
    d_morgan = jnp.array(chrom_data.genetic_distances.astype(np.float64))

    if seed_resp is not None:
        resp = seed_resp
    else:
        from .spectral import seed_ancestry_soft
        _labels, resp, _n_anc, _proj = seed_ancestry_soft(
            sub_geno, n_ancestries=2, rng_seed=rng_seed,
        )

    model = init_model_soft(
        geno_j, resp, 2, gen_since_admix, window_refine=False,
    )

    from .em import update_allele_freq_from_stats, update_mu_from_stats
    from .datatypes import AncestryModel

    batch_size = min(H_sub, 10_000)

    for _it in range(n_iter):
        em_stats = forward_backward_em(geno_j, model, d_morgan, batch_size)
        new_freq = update_allele_freq_from_stats(em_stats)
        new_mu = update_mu_from_stats(em_stats)
        model = AncestryModel(
            n_ancestries=2,
            mu=new_mu,
            gen_since_admix=gen_since_admix,
            allele_freq=new_freq,
        )

    # Final decode to get per-haplotype labels
    from .hmm import forward_backward_decode
    decode = forward_backward_decode(geno_j, model, d_morgan, batch_size)
    hap_resp = decode.global_sums / T
    labels = np.array(jnp.argmax(jnp.array(hap_resp), axis=1), dtype=np.int32)

    return labels


# ---------------------------------------------------------------------------
# BIC delta computation
# ---------------------------------------------------------------------------

def _compute_bic_delta(X: jnp.ndarray, key: jax.Array) -> float:
    """Compute BIC(1-GMM) - BIC(2-GMM) for logging."""
    from .spectral import _gmm_single

    n, d = X.shape

    # 1-component BIC
    mean1 = X.mean(axis=0)
    var1 = jnp.maximum(jnp.var(X, axis=0), 1e-6)
    log_det1 = jnp.sum(jnp.log(var1))
    mahal1 = jnp.sum((X - mean1) ** 2 / var1, axis=1)
    ll1 = float(jnp.sum(-0.5 * (d * jnp.log(2 * jnp.pi) + log_det1 + mahal1)))
    k1 = 2 * d
    bic1 = -2 * ll1 + k1 * np.log(n)

    # 2-component BIC
    _labels, _resp, ll2_per = _gmm_single(X, 2, key, max_iter=50, tol=1e-4)
    ll2 = float(ll2_per * n)
    k2 = 4 * d + 1
    bic2 = -2 * ll2 + k2 * np.log(n)

    return float(bic1 - bic2)


# ---------------------------------------------------------------------------
# Tree visualisation logging
# ---------------------------------------------------------------------------

def _dump_pre_merge(
    geno: np.ndarray,
    leaf_labels: np.ndarray,
    leaf_info: list[LeafInfo],
    path_prefix: str,
    pseudocount: float = 0.5,
) -> None:
    """Write pre-merge leaf state to disk for post-mortem analysis."""
    H, T = geno.shape

    # 1. Per-haplotype leaf assignments
    with open(f"{path_prefix}.leaves.tsv", "w") as f:
        f.write("hap_idx\tleaf_label\tleaf_path\tleaf_depth\n")
        # Build lookup from label to info
        label_to_info = {li.label: li for li in leaf_info}
        for h in range(H):
            lbl = int(leaf_labels[h])
            li = label_to_info[lbl]
            f.write(f"{h}\t{lbl}\t{li.path}\t{li.depth}\n")

    # 2. Leaf metadata
    with open(f"{path_prefix}.leaf_meta.tsv", "w") as f:
        f.write("label\tpath\tdepth\tn_haps\tbic_score\n")
        for li in leaf_info:
            f.write(f"{li.label}\t{li.path}\t{li.depth}\t{li.n_haps}\t{li.bic_score:.1f}\n")

    # 3. Leaf allele frequencies (memory-safe: no float64 materialization)
    n_leaves = len(leaf_info)
    allele_freq = np.zeros((n_leaves, T), dtype=np.float32)
    labels_arr = np.array([li.label for li in leaf_info])
    paths_arr = np.array([li.path for li in leaf_info])
    for i, li in enumerate(leaf_info):
        mask = leaf_labels == li.label
        count = int(mask.sum())
        if count > 0:
            allele_freq[i] = (geno[mask].sum(axis=0, dtype=np.float64) + pseudocount) / (count + 2 * pseudocount)
    np.savez_compressed(
        f"{path_prefix}.leaf_freqs.npz",
        labels=labels_arr,
        paths=paths_arr,
        allele_freq=allele_freq,
    )


def _log_tree(split_log: list[dict], total_haps: int) -> None:
    """Log a text tree of the recursive splitting process."""
    if not split_log:
        return

    # Sort by path to get tree order
    entries = sorted(split_log, key=lambda e: e["path"])

    # Build set of paths that have a sibling with '1' at each depth
    all_paths = {e["path"] for e in entries}

    lines = []
    for entry in entries:
        path = entry["path"]
        depth = len(path) - 1  # 'L' is depth 0, 'L0' is depth 1, etc.
        indent = ""
        if depth > 0:
            parts = []
            for d in range(1, depth):
                # Check if the ancestor at this depth has an unfinished '1' sibling
                ancestor_path = path[:d + 1]
                # Does a sibling '1' exist at this depth?
                sibling = ancestor_path[:-1] + "1"
                if ancestor_path[-1] == "0" or (sibling in all_paths and ancestor_path[-1] != "1"):
                    parts.append("│   ")
                else:
                    parts.append("    ")
            last_bit = path[-1]
            if last_bit == "0":
                parts.append("├── ")
            else:
                parts.append("└── ")
            indent = "".join(parts)

        action = entry["action"]
        n = entry["n"]
        bic = entry["bic"]

        if "split" in action:
            line = f"{indent}{path} [{n} haps] split (BIC={bic:.0f})"
        else:
            reason = action.replace("LEAF ", "")
            line = f"{indent}{path} [{n} haps] LEAF {reason}"

        lines.append(line)

    log.info("Recursive splitting tree:")
    for line in lines:
        log.info("  %s", line)


def _log_leaf_summary(leaf_info: list[LeafInfo], total_haps: int) -> None:
    """Log a summary of discovered leaf populations."""
    log.info("Discovered %d leaf populations.", len(leaf_info))
    log.info("Leaf assignment summary:")
    for li in leaf_info:
        pct = 100 * li.n_haps / total_haps
        log.info("  Leaf %d (path=%s): %d haps (%.1f%%)",
                 li.label, li.path, li.n_haps, pct)
