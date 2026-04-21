"""Benchmark metrics for comparing LAI tool outputs."""

from __future__ import annotations

import numpy as np

from popout.benchmark.common import MISSING_LABEL, TractSet


def per_ancestry_r2(a: TractSet, b: TractSet) -> dict[int, float]:
    """Per-ancestry r² between per-haplotype ancestry fractions.

    For each ancestry k, computes Pearson r² between per-haplotype
    fractions in a and b. This is the primary metric in 2025 LAI papers.
    """
    labels = sorted(k for k in a.label_map if k != MISSING_LABEL)
    result = {}
    for k in labels:
        frac_a = (a.calls == k).mean(axis=1)
        frac_b = (b.calls == k).mean(axis=1)
        if frac_a.std() == 0 or frac_b.std() == 0:
            result[k] = float("nan")
        else:
            r = np.corrcoef(frac_a, frac_b)[0, 1]
            result[k] = float(r ** 2)
    return result


def per_site_accuracy(a: TractSet, b: TractSet) -> float:
    """Fraction of (hap, site) pairs where labels agree."""
    # Exclude MISSING from comparison
    valid = (a.calls != MISSING_LABEL) & (b.calls != MISSING_LABEL)
    if valid.sum() == 0:
        return float("nan")
    return float((a.calls[valid] == b.calls[valid]).mean())


def per_haplotype_accuracy(a: TractSet, b: TractSet) -> np.ndarray:
    """Per-haplotype accuracy. Returns (H,) array."""
    valid = (a.calls != MISSING_LABEL) & (b.calls != MISSING_LABEL)
    accs = np.zeros(a.n_haps)
    for h in range(a.n_haps):
        h_valid = valid[h]
        if h_valid.sum() == 0:
            accs[h] = float("nan")
        else:
            accs[h] = (a.calls[h, h_valid] == b.calls[h, h_valid]).mean()
    return accs


def per_ancestry_precision_recall(
    a: TractSet, b: TractSet
) -> dict[int, dict[str, float]]:
    """Per-ancestry precision and recall.

    Precision: of sites a called k, what fraction did b also call k.
    Recall: of sites b called k, what fraction did a also call k.

    If b is ground truth, these have the usual meaning.
    """
    labels = sorted(k for k in a.label_map if k != MISSING_LABEL)
    result = {}
    for k in labels:
        a_is_k = a.calls == k
        b_is_k = b.calls == k
        agree = a_is_k & b_is_k
        precision = float(agree.sum() / a_is_k.sum()) if a_is_k.sum() > 0 else float("nan")
        recall = float(agree.sum() / b_is_k.sum()) if b_is_k.sum() > 0 else float("nan")
        result[k] = {"precision": precision, "recall": recall}
    return result


def global_fraction_error(a: TractSet, b: TractSet) -> np.ndarray:
    """Per-sample L1 distance between global ancestry fractions.

    Returns (n_samples,) array. Assumes haplotypes are paired as
    consecutive rows (0,1), (2,3), etc.
    """
    labels = sorted(k for k in a.label_map if k != MISSING_LABEL)
    K = len(labels)
    n_samples = a.n_haps // 2

    errors = np.zeros(n_samples)
    for s in range(n_samples):
        h0, h1 = 2 * s, 2 * s + 1
        for idx, k in enumerate(labels):
            frac_a = ((a.calls[h0] == k).mean() + (a.calls[h1] == k).mean()) / 2
            frac_b = ((b.calls[h0] == k).mean() + (b.calls[h1] == k).mean()) / 2
            errors[s] += abs(frac_a - frac_b)
    return errors


def tract_length_stats(tracts: TractSet) -> dict:
    """Summary statistics of tract lengths in bp and sites."""
    tract_list = tracts.to_tracts()
    if not tract_list:
        return {"count": 0}

    site_lengths = []
    bp_lengths = []
    for _, start_idx, end_idx, _ in tract_list:
        n_sites = end_idx - start_idx
        site_lengths.append(n_sites)
        bp_start = tracts.site_positions[start_idx]
        bp_end = tracts.site_positions[min(end_idx - 1, tracts.n_sites - 1)]
        bp_lengths.append(int(bp_end - bp_start))

    site_arr = np.array(site_lengths)
    bp_arr = np.array(bp_lengths)
    return {
        "count": len(site_lengths),
        "sites": {
            "min": int(site_arr.min()),
            "max": int(site_arr.max()),
            "mean": float(site_arr.mean()),
            "median": float(np.median(site_arr)),
            "q25": float(np.percentile(site_arr, 25)),
            "q75": float(np.percentile(site_arr, 75)),
        },
        "bp": {
            "min": int(bp_arr.min()),
            "max": int(bp_arr.max()),
            "mean": float(bp_arr.mean()),
            "median": float(np.median(bp_arr)),
            "q25": float(np.percentile(bp_arr, 25)),
            "q75": float(np.percentile(bp_arr, 75)),
        },
    }


def compute_all_metrics(
    a: TractSet,
    b: TractSet,
    b_is_truth: bool = False,
) -> dict:
    """Compute the full metric suite for an aligned pair of TractSets.

    If b_is_truth, precision/recall have their usual meaning and r²
    is the headline accuracy metric.
    """
    result = {
        "tool_a": a.tool_name,
        "tool_b": b.tool_name,
        "b_is_truth": b_is_truth,
        "n_haps": a.n_haps,
        "n_sites": a.n_sites,
        "per_site_accuracy": per_site_accuracy(a, b),
        "per_ancestry_r2": per_ancestry_r2(a, b),
        "per_ancestry_precision_recall": per_ancestry_precision_recall(a, b),
        "global_fraction_error_mean": float(global_fraction_error(a, b).mean()),
        "global_fraction_error_max": float(global_fraction_error(a, b).max()),
        "tract_stats_a": tract_length_stats(a),
        "tract_stats_b": tract_length_stats(b),
    }

    r2_vals = [v for v in result["per_ancestry_r2"].values() if not np.isnan(v)]
    result["mean_r2"] = float(np.mean(r2_vals)) if r2_vals else float("nan")

    return result
