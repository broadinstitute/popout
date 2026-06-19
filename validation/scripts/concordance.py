"""Shared per-ancestry concordance + hard-call confusion primitives.

Both ``compare_to_rye.py`` and ``compare_to_rf.py`` route through this
module so the per-(cluster, chrom, ancestry) metric row and the hard-
call confusion table have the same shape regardless of which other tool
FLARE is being compared to. Alignment is always by name from the
tools' headers; no sidecar label maps.

Per-ancestry metrics row schema (fed to ``write_concordance_metrics``):

    ancestry  cluster_mu  n_samples  pearson_r  ccc  cosine_mean
    mae_mean  mae_median  mae_p95
    jaccard_at_0.10  jaccard_at_0.25  jaccard_at_0.50  pass

Acceptance thresholds (R10 in PLAN2.md §2.2):

    Pearson r >= 0.95  AND  Lin's CCC >= 0.90  for each ancestry with
    cluster_mu >= MU_GATE; below MU_GATE the ancestry is degenerate
    (pass = None).

Hard-call confusion: ``a`` and ``b`` need not share a label set.
``write_hard_confusion`` writes a |a_labels| x |b_labels| table whose
[i, j] cell counts samples where a-argmax == a_labels[i] AND b-argmax
== b_labels[j]. The argmax for each tool is taken over that tool's own
column set (so RF can call MID even when the comparison tool has no
MID column).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ── constants ────────────────────────────────────────────────────────────

MU_GATE: float = 0.01
JACCARD_THRESHOLDS: tuple[float, ...] = (0.10, 0.25, 0.50)
PEARSON_THRESHOLD: float = 0.95
CCC_THRESHOLD: float = 0.90


# ── numeric helpers ──────────────────────────────────────────────────────


def lin_ccc(x, y) -> float:
    """Lin's concordance correlation coefficient (CCC). Pure numpy."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float((2 * cov) / denom)


def cosine_similarity_per_sample(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between matrices A and B
    (n_samples x n_dims)."""
    num = np.sum(A * B, axis=1)
    denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    denom = np.where(denom == 0, np.nan, denom)
    return num / denom


# ── alignment ────────────────────────────────────────────────────────────


def align_by_name(
    a_props: np.ndarray, a_names: list[str],
    b_props: np.ndarray, b_names: list[str],
    *, basis_order: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reorder both matrices' columns onto the shared name set and return
    them aligned, alongside the ordered shared label list.

    ``basis_order`` (optional) is the preferred column order for the
    shared labels. Any labels not in ``basis_order`` are appended in
    a_names order. When ``basis_order`` is None the order is a_names
    order on the intersection.
    """
    a_lookup = {n: i for i, n in enumerate(a_names)}
    b_lookup = {n: i for i, n in enumerate(b_names)}
    intersection = [n for n in a_names if n in b_lookup]
    if basis_order is not None:
        ordered = [n for n in basis_order if n in a_lookup and n in b_lookup]
        # tack on any intersection names that weren't in basis_order
        for n in intersection:
            if n not in ordered:
                ordered.append(n)
        shared = ordered
    else:
        shared = intersection
    if not shared:
        raise RuntimeError(
            f"No shared ancestry names. a={a_names!r} b={b_names!r}"
        )
    a_cols = [a_lookup[n] for n in shared]
    b_cols = [b_lookup[n] for n in shared]
    return a_props[:, a_cols], b_props[:, b_cols], shared


# ── per-ancestry metrics ─────────────────────────────────────────────────


def per_ancestry_metrics(
    a_props: np.ndarray, b_props: np.ndarray, ancestry_idx: int,
) -> dict[str, float | int | bool | None]:
    """Compute the canonical metric row for one ancestry index.

    Convention: ``a`` is FLARE; ``b`` is the other tool (Rye or RF).
    Both matrices are already aligned to the shared label set in the
    same column order. ``cluster_mu`` is mean(FLARE column) — a
    side-channel scalar used only for the mu-gate. r/CCC operate on
    the full per-sample vectors.
    """
    x = a_props[:, ancestry_idx]
    y = b_props[:, ancestry_idx]

    cluster_mu = float(x.mean())
    n = int(len(x))

    if x.std() == 0 or y.std() == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(x, y)[0, 1])

    ccc = lin_ccc(x, y)

    err = np.abs(x - y)
    mae_mean = float(err.mean())
    mae_median = float(np.median(err))
    mae_p95 = float(np.percentile(err, 95))

    jaccards: dict[float, float] = {}
    for tau in JACCARD_THRESHOLDS:
        a_mask = x >= tau
        b_mask = y >= tau
        inter = int(np.sum(a_mask & b_mask))
        union = int(np.sum(a_mask | b_mask))
        jaccards[tau] = float(inter / union) if union > 0 else float("nan")

    if cluster_mu < MU_GATE:
        passed: bool | None = None
    else:
        passed = bool(
            (not np.isnan(pearson_r) and pearson_r >= PEARSON_THRESHOLD)
            and (not np.isnan(ccc) and ccc >= CCC_THRESHOLD)
        )

    return {
        "cluster_mu": cluster_mu,
        "n_samples": n,
        "pearson_r": pearson_r,
        "ccc": ccc,
        # Per-ancestry single-column cosine is sign-only and not
        # meaningful. Global cosine over the K-dim row vectors lives in
        # concordance_summary.json (mean_cosine).
        "cosine_mean": None,
        "mae_mean": mae_mean,
        "mae_median": mae_median,
        "mae_p95": mae_p95,
        "jaccard_at_0.10": jaccards[0.10],
        "jaccard_at_0.25": jaccards[0.25],
        "jaccard_at_0.50": jaccards[0.50],
        "pass": passed,
    }


def build_concordance_rows(
    a_props: np.ndarray, b_props: np.ndarray, shared_labels: list[str],
) -> list[dict]:
    """One concordance row per shared label."""
    rows: list[dict] = []
    for j, label in enumerate(shared_labels):
        m = per_ancestry_metrics(a_props, b_props, j)
        m["ancestry"] = label
        rows.append(m)
    return rows


# ── writers ──────────────────────────────────────────────────────────────


_METRIC_COLS = [
    "ancestry", "cluster_mu", "n_samples", "pearson_r", "ccc",
    "cosine_mean", "mae_mean", "mae_median", "mae_p95",
    "jaccard_at_0.10", "jaccard_at_0.25", "jaccard_at_0.50", "pass",
]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if np.isnan(v):
            return "NA"
        return f"{v:.6f}"
    return str(v)


def write_concordance_metrics(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\t".join(_METRIC_COLS) + "\n")
        for r in rows:
            f.write("\t".join(_fmt(r[c]) for c in _METRIC_COLS) + "\n")


def write_concordance_summary(
    rows: list[dict], a_aligned: np.ndarray, b_aligned: np.ndarray,
    n_overlap: int, out_path: Path,
) -> None:
    """Cohort-level concordance summary. ``a_aligned``/``b_aligned`` are
    the per-sample matrices on the shared label set; the global CCC and
    mean cosine are computed across them."""
    import json

    eligible = [r for r in rows if r["cluster_mu"] >= MU_GATE]
    passing_r = [r["ancestry"] for r in eligible
                 if not np.isnan(r["pearson_r"]) and r["pearson_r"] >= PEARSON_THRESHOLD]
    passing_ccc = [r["ancestry"] for r in eligible
                   if not np.isnan(r["ccc"]) and r["ccc"] >= CCC_THRESHOLD]
    failing = [r["ancestry"] for r in eligible if r["pass"] is False]
    mean_pearson_r = (
        float(np.nanmean([r["pearson_r"] for r in eligible])) if eligible else float("nan")
    )
    global_ccc = lin_ccc(a_aligned.flatten(), b_aligned.flatten())
    cos_sim = cosine_similarity_per_sample(a_aligned, b_aligned)
    mean_cosine = float(np.nanmean(cos_sim)) if cos_sim.size else float("nan")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("json").dumps({
        "global_ccc": global_ccc,
        "mean_cosine": mean_cosine,
        "mean_pearson_r": mean_pearson_r,
        "n_samples_overlap": n_overlap,
        "labels_passing_r_ge_0.95": passing_r,
        "labels_passing_ccc_ge_0.90": passing_ccc,
        "labels_failing": failing,
    }, indent=2))


def write_merged_groups(rows: list[dict], out_path: Path) -> None:
    """Long-form one-row-per-ancestry summary (used by the report's
    merged-groups table)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("label\tn_samples\tcluster_mu\tpearson_r\tccc\n")
        for r in rows:
            pr = "NA" if np.isnan(r["pearson_r"]) else f"{r['pearson_r']:.6f}"
            cc = "NA" if np.isnan(r["ccc"]) else f"{r['ccc']:.6f}"
            f.write(f"{r['ancestry']}\t{r['n_samples']}\t"
                    f"{r['cluster_mu']:.6f}\t{pr}\t{cc}\n")


def write_full_matrix(
    a_aligned: np.ndarray, b_aligned: np.ndarray, shared_labels: list[str],
    out_path: Path, *, a_axis_name: str = "flare_ancestry",
) -> None:
    """Pearson r per a-column x b-column on the shared label set. The
    diagonal is the named-ancestry concordance; off-diagonals diagnose
    cross-label bleed (a sample called eur by tool A that tool B
    correlates with afr, etc)."""
    K = len(shared_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{a_axis_name}\t" + "\t".join(shared_labels) + "\n")
        for fi in range(K):
            x = a_aligned[:, fi]
            row = [shared_labels[fi]]
            for ri in range(K):
                y = b_aligned[:, ri]
                if x.std() == 0 or y.std() == 0:
                    row.append("NA")
                else:
                    row.append(f"{float(np.corrcoef(x, y)[0, 1]):.4f}")
            f.write("\t".join(row) + "\n")


def write_hard_confusion(
    a_props: np.ndarray, a_labels: list[str],
    b_props: np.ndarray, b_labels: list[str],
    out_path: Path,
    *, a_axis_name: str = "rf_label", b_axis_name: str = "flare_call",
) -> None:
    """Hard-call confusion. ``a`` and ``b`` may have different label
    sets; each tool's argmax is over its own columns. Cell [i, j]
    counts samples where a-argmax==a_labels[i] AND b-argmax==b_labels[j].

    Row totals + column totals + grand total are appended.
    """
    a_primary = np.argmax(a_props, axis=1)
    b_primary = np.argmax(b_props, axis=1)
    cm = np.zeros((len(a_labels), len(b_labels)), dtype=int)
    for i in range(len(a_primary)):
        cm[a_primary[i], b_primary[i]] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{a_axis_name}\t" + "\t".join(b_labels) + "\ttotal\n")
        for i, lab in enumerate(a_labels):
            row = [lab] + [str(cm[i, j]) for j in range(len(b_labels))] + [str(cm[i].sum())]
            f.write("\t".join(row) + "\n")
        col_totals = cm.sum(axis=0)
        f.write("total\t" + "\t".join(str(c) for c in col_totals)
                + f"\t{col_totals.sum()}\n")
