"""Label-permutation-invariant cluster-evaluation metrics.

Per ``LABEL_SPACE.md`` §8.1 — emitted next to every mapped confusion
view so analysts get an honest floor alongside the mapped percentages.

All three metrics are scikit-learn-equivalent but implemented in pure
numpy to avoid pulling sklearn into the labelspace package's runtime
dependency surface. Tests pin them to known reference values for a
handful of canonical cases.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ── Confusion matrix → metrics ──────────────────────────────────────────


def _confusion_matrix(labels_a: Sequence, labels_b: Sequence) -> np.ndarray:
    """Build a contingency table from two same-length label sequences."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"labels must be 1-D and same length; got {a.shape} vs {b.shape}")
    a_levels, a_codes = np.unique(a, return_inverse=True)
    b_levels, b_codes = np.unique(b, return_inverse=True)
    K_a, K_b = len(a_levels), len(b_levels)
    C = np.zeros((K_a, K_b), dtype=np.int64)
    for i, j in zip(a_codes, b_codes):
        C[i, j] += 1
    return C


def adjusted_rand_index(labels_a: Sequence, labels_b: Sequence) -> float:
    """Hubert & Arabie 1985. Range ~[-0.5, 1]; 0 = random; 1 = identical."""
    C = _confusion_matrix(labels_a, labels_b)
    n = int(C.sum())
    if n < 2:
        return float("nan")

    def comb2(x: np.ndarray) -> np.ndarray:
        return (x * (x - 1)) / 2

    sum_comb_c = comb2(C).sum()
    sum_comb_a = comb2(C.sum(axis=1)).sum()
    sum_comb_b = comb2(C.sum(axis=0)).sum()
    total_comb = n * (n - 1) / 2
    if total_comb == 0:
        return float("nan")
    expected = (sum_comb_a * sum_comb_b) / total_comb
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_idx - expected
    if denom == 0:
        return 1.0 if sum_comb_c == max_idx else 0.0
    return float((sum_comb_c - expected) / denom)


def mutual_information(labels_a: Sequence, labels_b: Sequence) -> float:
    """Raw (un-normalised) mutual information in nats."""
    C = _confusion_matrix(labels_a, labels_b).astype(np.float64)
    n = C.sum()
    if n == 0:
        return 0.0
    pi = C.sum(axis=1) / n
    pj = C.sum(axis=0) / n
    pij = C / n
    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if pij[i, j] > 0 and pi[i] > 0 and pj[j] > 0:
                mi += pij[i, j] * np.log(pij[i, j] / (pi[i] * pj[j]))
    return float(mi)


def _entropy(labels: Sequence) -> float:
    a = np.asarray(labels)
    _, counts = np.unique(a, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def normalized_mutual_info(labels_a: Sequence, labels_b: Sequence) -> float:
    """Arithmetic-mean normalised MI (the most common variant)."""
    mi = mutual_information(labels_a, labels_b)
    h_a = _entropy(labels_a)
    h_b = _entropy(labels_b)
    denom = 0.5 * (h_a + h_b)
    if denom == 0:
        return 0.0 if mi == 0 else float("nan")
    return float(mi / denom)


def v_measure(labels_a: Sequence, labels_b: Sequence, *, beta: float = 1.0) -> float:
    """Rosenberg & Hirschberg 2007. Harmonic mean of homogeneity + completeness."""
    mi = mutual_information(labels_a, labels_b)
    h_a = _entropy(labels_a)
    h_b = _entropy(labels_b)
    homogeneity = 0.0 if h_a == 0 else mi / h_a
    completeness = 0.0 if h_b == 0 else mi / h_b
    if homogeneity == 0 and completeness == 0:
        return 0.0
    return float(
        (1 + beta) * homogeneity * completeness
        / (beta * homogeneity + completeness)
    )


def cluster_eval(labels_a: Sequence, labels_b: Sequence) -> dict[str, float]:
    """Compute the canonical four metrics in one call.

    Use this from any harness that emits a mapped confusion view —
    surfaces the label-permutation-invariant floor next to the mapped
    counts so reviewers can spot deceptive accuracy from a bad match.
    """
    return {
        "ari": adjusted_rand_index(labels_a, labels_b),
        "nmi": normalized_mutual_info(labels_a, labels_b),
        "v_measure": v_measure(labels_a, labels_b),
        "n": int(len(labels_a)),
    }
