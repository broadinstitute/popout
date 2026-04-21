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

import heapq
import itertools
import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .datatypes import ChromData
from .spectral import _bic_split_test, _genotypes_to_pca_projection

log = logging.getLogger(__name__)


def _available_device_memory() -> int:
    """Best-effort device memory estimate in bytes; 0 if unavailable."""
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("bytes_limit", 0)
    except (AttributeError, KeyError, RuntimeError):
        return 0


@dataclass
class LeafInfo:
    label: int
    n_haps: int
    depth: int
    path: str       # e.g. "L01" — sequence of 0/1 bits from root to leaf
    bic_score: float  # BIC delta of the split that produced this leaf.
                      # NaN for leaves created by a post-hoc merge.
                      # 0.0 for the root if no splits occurred.


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
# BIC helpers (diagonal-covariance Gaussian)
# ---------------------------------------------------------------------------

# NOTE: BIC parameter counts k1=2d, k2=4d+1 assume _gmm_single uses
# diagonal covariance (see spectral.py "Gaussian Mixture Model (diagonal
# covariance)" section). If _gmm_single is ever changed to full covariance,
# update these penalties to k1=d+d(d+1)/2, k2=2d+2·d(d+1)/2+1.


def _diag_gaussian_bic(X: jnp.ndarray) -> float:
    """BIC for a single diagonal-covariance Gaussian fit to X."""
    n, d = X.shape
    mu = X.mean(axis=0)
    var = jnp.maximum(jnp.var(X, axis=0), 1e-6)
    log_det = jnp.sum(jnp.log(var))
    mahal = jnp.sum((X - mu) ** 2 / var, axis=1)
    ll = float(jnp.sum(-0.5 * (d * jnp.log(2 * jnp.pi) + log_det + mahal)))
    k = 2 * d
    return -2 * ll + k * np.log(n)


def _diag_2gmm_bic_from_labels(X: jnp.ndarray, labels: jnp.ndarray) -> tuple[float, float]:
    """2-component diag-covariance BIC from hard labels, plus balance."""
    n, d = X.shape
    n_0 = int((labels == 0).sum())
    n_1 = n - n_0
    if n_0 == 0 or n_1 == 0:
        return float("inf"), 0.0
    X0, X1 = X[labels == 0], X[labels == 1]

    def _ll(Xc):
        mu = Xc.mean(axis=0)
        var = jnp.maximum(jnp.var(Xc, axis=0), 1e-6)
        log_det = jnp.sum(jnp.log(var))
        mahal = jnp.sum((Xc - mu) ** 2 / var, axis=1)
        return jnp.sum(-0.5 * (d * jnp.log(2 * jnp.pi) + log_det + mahal))

    w0, w1 = n_0 / n, n_1 / n
    ll = float(_ll(X0) + _ll(X1) + n_0 * np.log(w0) + n_1 * np.log(w1))
    k2 = 4 * d + 1
    bic2 = -2 * ll + k2 * np.log(n)
    return bic2, min(n_0, n_1) / n


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
    merge_max_tree_distance: int = 1,
    split_restarts: int = 5,
    balance_bic_tolerance: float = 0.10,
    max_haps_svd: int = 100_000,
    projection_batch: int = 50_000,
    dump_pre_merge_path: Optional[str] = None,
    device_memory_fraction: float = 0.5,
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
    merge_max_tree_distance : distance in the recursion tree at which two
        leaves may be considered for Hellinger merging. 1 = siblings only
        (conservative default). 2 = cousins.

    Returns
    -------
    leaf_labels : (H,) int32 — flat labels 0..(n_leaves - 1) per haplotype
    leaf_info : list of LeafInfo — metadata per leaf, in label order
    """
    H, T = geno.shape

    if H * T > 0:
        sample = geno[::max(1, H // 100), ::max(1, T // 100)]
        sample_max = int(sample.max())
        if sample_max > 1:
            raise ValueError(
                f"geno contains values > 1 (max seen in sample: {sample_max}). "
                "This module assumes a binary haplotype matrix. Filter missing "
                "genotypes or split diploid dosages upstream."
            )

    key = jax.random.PRNGKey(rng_seed)

    log.info("Recursive seed (%d haps, max_depth=%d, max_leaves=%d):",
             H, max_depth, max_leaves)

    # Build ChromData if not provided (for K=2 EM splits)
    if chrom_data is None:
        log.warning(
            "No chrom_data passed to recursive_split_seed; fabricating a uniform "
            "100 cM map over %d sites. HMM transition rates will not reflect true "
            "recombination and ancestry-block lengths will be biased. Pass the real "
            "ChromData when available.",
            T,
        )
        chrom_data = ChromData(
            geno=geno,
            pos_bp=np.arange(T, dtype=np.int64),
            pos_cm=np.linspace(0, 100.0, T),
            chrom="recursive",
        )

    # Device-resident geno: one-time host→device transfer if it fits.
    # At uint8, bytes == elements. Packed binary (jnp.packbits along axis=1)
    # would extend this by 8× — obvious next optimisation when GPU memory
    # stops growing as fast as biobanks.
    device_mem = _available_device_memory()
    geno_bytes = geno.size  # uint8, so bytes == elements
    if device_mem > 0 and geno_bytes < device_memory_fraction * device_mem:
        geno_j = jnp.asarray(geno)
        log.info("geno resident on device (%.1f GB / %.1f GB available)",
                 geno_bytes / 1e9, device_mem / 1e9)
    else:
        geno_j = None
        log.info("geno too large for device or device memory unavailable "
                 "(%.1f GB); using per-call transfer path",
                 geno_bytes / 1e9)

    # Priority queue: pop highest-BIC-delta nodes first so that, when
    # max_leaves binds, the budget goes to the most informative splits.
    counter = itertools.count()
    queue: list[tuple[float, int, int, _SplitNode]] = []

    def _push(nd: _SplitNode) -> None:
        heapq.heappush(
            queue,
            (-nd.bic_delta, -len(nd.indices), next(counter), nd),
        )

    _push(_SplitNode(
        indices=np.arange(H, dtype=np.int64),
        depth=0,
        path="L",
    ))

    leaves: list[_SplitNode] = []
    split_log: list[dict] = []  # for tree visualisation

    while queue:
        # Check if we've hit the leaf limit
        if len(leaves) + len(queue) >= max_leaves:
            # Remaining queue items become leaves (drain in priority order)
            while queue:
                _priority, _size_key, _tiebreak, node = heapq.heappop(queue)
                leaves.append(node)
                split_log.append({
                    "path": node.path, "n": len(node.indices),
                    "action": f"LEAF (max_leaves, priority={node.bic_delta:.0f})",
                    "bic": node.bic_delta,
                    "depth": node.depth,
                })
            break

        _priority, _size_key, _tiebreak, node = heapq.heappop(queue)
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
        if len(node.indices) == H:
            sub_geno = geno          # root node: avoid full-matrix host copy
        else:
            sub_geno = geno[node.indices]
        sub_proj = _geno_sub_pca(
            sub_geno, n_components=2, key=subkey,
            max_haps_svd=max_haps_svd, projection_batch=projection_batch,
        )

        # BIC split test (gates whether we attempt a split at all)
        key, subkey = jax.random.split(key)
        should_split, gate_labels = _bic_split_test(sub_proj, subkey, bic_per_sample)

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
        # The gate's labels count as the first restart.
        key, subkey = jax.random.split(key)
        selected_labels, bic_delta, sel_balance = _select_balanced_split(
            sub_proj, subkey, n_restarts=split_restarts,
            bic_tolerance=balance_bic_tolerance, node_path=node.path,
            prior_labels=gate_labels,
        )

        if selected_labels is None:
            leaves.append(node)
            split_log.append({
                "path": node.path, "n": n,
                "action": "LEAF (no valid split)", "bic": node.bic_delta,
                "depth": node.depth,
            })
            continue

        # Build seed responsibilities from GMM labels
        seed_resp_k2 = jax.nn.one_hot(selected_labels, 2, dtype=jnp.float32)

        child_labels = _run_k2_em_split(
            geno, node.indices, chrom_data,
            geno_j=geno_j,
            n_iter=em_iter_per_split,
            gen_since_admix=gen_since_admix,
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

        _push(_SplitNode(idx_0, node.depth + 1, node.path + "0", bic_delta))
        _push(_SplitNode(idx_1, node.depth + 1, node.path + "1", bic_delta))

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
        except OSError as e:
            log.warning("Failed to write pre-merge dump: %s", e)

    # Post-hoc merge of similar leaves
    if merge_hellinger_threshold > 0 and len(leaf_info) > 1:
        n_before = len(leaf_info)
        leaf_labels, leaf_info = _merge_close_leaves(
            geno, leaf_labels, leaf_info,
            hellinger_threshold=merge_hellinger_threshold,
            max_tree_distance=merge_max_tree_distance,
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

def _are_merge_candidates(
    path_a: str,
    path_b: str,
    max_tree_distance: int = 1,
) -> bool:
    """Check if two tree paths are candidates for Hellinger merging.

    Leaves can merge if their tree distance to LCA is <= max_tree_distance
    on each side. max_tree_distance=1 restricts to siblings (same parent);
    max_tree_distance=2 allows first-cousins.
    """
    if len(path_a) < 2 or len(path_b) < 2:
        return True  # root children are always candidates
    lca_len = 0
    for k in range(min(len(path_a), len(path_b))):
        if path_a[k] == path_b[k]:
            lca_len = k + 1
        else:
            break
    dist_a = len(path_a) - lca_len
    dist_b = len(path_b) - lca_len
    return dist_a <= max_tree_distance and dist_b <= max_tree_distance


def _merge_close_leaves(
    geno: np.ndarray,
    leaf_labels: np.ndarray,
    leaf_info: list[LeafInfo],
    hellinger_threshold: float = 0.08,
    pseudocount: float = 0.5,
    max_tree_distance: int = 1,
) -> tuple[np.ndarray, list[LeafInfo]]:
    """Merge leaf pairs whose allele-frequency Hellinger distance is below threshold.

    Frequencies are computed once and updated incrementally on each merge
    to avoid re-scanning the full genotype array.

    Default hellinger_threshold=0.08 matches the public recursive_split_seed API.
    Default max_tree_distance=1 restricts to siblings (same parent).
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

    # B2: Hoist sqrt computation outside the loop; refresh only merged rows.
    sqrt_p = np.sqrt(np.clip(freq, 0, 1))
    sqrt_1mp = np.sqrt(np.clip(1 - freq, 0, 1))
    n_active = len(info)

    while n_active > 1:
        # Find the closest merge-candidate pair
        best_dist = float('inf')
        best_i, best_j = -1, -1
        for i in range(n_active):
            for j in range(i + 1, n_active):
                if not _are_merge_candidates(info[i].path, info[j].path, max_tree_distance):
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

        # Refresh sqrt rows for the merged leaf
        sqrt_p[best_i] = np.sqrt(np.clip(freq[best_i], 0, 1))
        sqrt_1mp[best_i] = np.sqrt(np.clip(1 - freq[best_i], 0, 1))

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
            bic_score=float("nan"),
        )

        # B3: Swap-delete — move last active row into the vacated slot
        last = n_active - 1
        if best_j != last:
            freq[best_j] = freq[last]
            sqrt_p[best_j] = sqrt_p[last]
            sqrt_1mp[best_j] = sqrt_1mp[last]
            counts[best_j] = counts[last]
            info[best_j] = info[last]
        n_active -= 1
        info.pop()  # remove the (now-duplicated) last entry

    # Trim info to active entries
    info = info[:n_active]

    # Remap labels to contiguous 0..n_new-1
    old_labels = sorted({li.label for li in info})
    lut = np.empty(max(old_labels) + 1, dtype=np.int32)
    for new, old in enumerate(old_labels):
        lut[old] = new
    new_labels = lut[labels]
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
    prior_labels: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray | None, float, float]:
    """Run multiple K=2 GMM restarts and pick the most balanced among top-BIC.

    If prior_labels is provided (e.g. from the BIC gate), it counts as the
    first of n_restarts, preserving the user-facing restart budget.

    Returns (labels, bic_delta, balance_score) or (None, 0, 0) if no valid split.
    """
    from .spectral import _gmm_single

    n, d = X.shape
    bic1 = _diag_gaussian_bic(X)

    candidates = []  # (labels, bic_improvement, balance)

    if prior_labels is not None:
        bic2_prior, balance_prior = _diag_2gmm_bic_from_labels(X, prior_labels)
        candidates.append((prior_labels, float(bic1 - bic2_prior), balance_prior))
        remaining = max(n_restarts - 1, 0)
    else:
        remaining = n_restarts

    for r in range(remaining):
        key, subkey = jax.random.split(key)
        labels_r, _resp_r, ll2_per = _gmm_single(X, 2, subkey, max_iter=50, tol=1e-4)
        ll2 = float(ll2_per * n)
        k2_params = 4 * d + 1
        bic2 = -2 * ll2 + k2_params * np.log(n)
        bic_improvement = float(bic1 - bic2)

        labels_np = np.asarray(labels_r)
        n_0 = int((labels_np == 0).sum())
        balance = min(n_0, n - n_0) / n

        candidates.append((labels_r, bic_improvement, balance))
        log.debug("  Split %s restart %d: BIC improvement %.0f, balance %.2f",
                  node_path, r, bic_improvement, balance)

    if not candidates:
        return None, 0.0, 0.0

    # Fast path: single candidate
    if len(candidates) == 1:
        labels, bic_improvement, balance = candidates[0]
        if bic_improvement <= 0:
            return None, 0.0, 0.0
        return labels, bic_improvement, balance

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

    if len(candidates) > 1:
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
    *,
    geno_j: jnp.ndarray | None = None,
    n_iter: int = 3,
    gen_since_admix: float = 20.0,
    seed_resp: jnp.ndarray | None = None,
    rng_key: jax.Array | None = None,
) -> np.ndarray:
    """Run K=2 popout EM on a haplotype subset, return hard labels.

    Parameters
    ----------
    geno_j : optional device-resident full genotype array. When provided,
        the subset is sliced on-device instead of copying from host.
    seed_resp : optional (H_subset, 2) initial responsibilities.
        If provided, skips spectral init and uses these directly.
    rng_key : optional JAX PRNG key for fallback spectral init.
    """
    from .em import init_model_soft
    from .hmm import forward_backward_em

    if geno_j is not None:
        if len(indices) == geno_j.shape[0]:
            sub_geno_j = geno_j  # root node — no copy needed
        else:
            sub_geno_j = geno_j[jnp.asarray(indices)]
    else:
        sub_geno_j = jnp.asarray(geno[indices])
    H_sub, T = sub_geno_j.shape

    d_morgan = jnp.asarray(chrom_data.genetic_distances.astype(np.float64))

    if seed_resp is not None:
        resp = seed_resp
    else:
        from .spectral import seed_ancestry_soft
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        derived_seed = int(jax.random.bits(rng_key, dtype=jnp.uint32))
        sub_geno = geno[indices]  # host-side copy only for fallback init
        _labels, resp, _n_anc, _proj = seed_ancestry_soft(
            sub_geno, n_ancestries=2, rng_seed=derived_seed,
        )

    model = init_model_soft(
        sub_geno_j, resp, 2, gen_since_admix, window_refine=False,
    )

    from .em import update_allele_freq_from_stats, update_mu_from_stats
    from .datatypes import AncestryModel

    batch_size = min(H_sub, 10_000)

    for _it in range(n_iter):
        em_stats = forward_backward_em(sub_geno_j, model, d_morgan, batch_size)
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
    decode = forward_backward_decode(sub_geno_j, model, d_morgan, batch_size)
    if not bool(jnp.all(jnp.isfinite(decode.global_sums))):
        raise RuntimeError(
            "K=2 EM produced non-finite posteriors during final decode. "
            "This usually means a component collapsed (empty or near-empty "
            "cluster). Check gen_since_admix and seed responsibilities."
        )
    hap_resp = decode.global_sums / T
    labels = np.asarray(jnp.argmax(hap_resp, axis=1)).astype(np.int32)

    return labels


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

    lines = []
    for entry in entries:
        path = entry["path"]
        depth = len(path) - 1  # 'L' is depth 0, 'L0' is depth 1, etc.
        indent = ""
        if depth > 0:
            parts = []
            for d in range(1, depth):
                ancestor_step = path[d]  # the 0/1 bit chosen at depth d
                # If the ancestor branch was "0", a "1" sibling may appear later in sorted
                # order, so draw a vertical connector; otherwise leave blank.
                parts.append("│   " if ancestor_step == "0" else "    ")
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
