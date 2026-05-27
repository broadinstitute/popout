#!/usr/bin/env python3
"""§8.2 / R10: FLARE global proportions vs Rye (supervised ADMIXTURE) Q.

Rye is the team's supervised-ADMIXTURE variant; its Q file already carries
labeled ancestry columns (`eur, eas, amr, afr, sas`), so no column-bootstrap
is needed (contrast `compare_to_admixture.py` v1.0, which bootstrapped
unlabeled ADMIXTURE columns against the RF classifier). FLARE proportions
align to the same five-column basis after merging the FLARE component
ancestries via `labels.json["rf_to_popout_components"]`.

Per PLAN2.md §2.2, R10 acceptance is
    Pearson r ≥ 0.95 AND Lin's CCC ≥ 0.90
for every ancestry with cluster_mu ≥ 0.01 (μ-gating: degenerate ancestries
get a pass=null row, not a failure).

Source lift: `my_notes/lk_notebooks/lk-test-flare-afr.ipynb` cell 15
(`compare_rye_vs_flare()`, helpers `lin_ccc()`, `cosine_similarity_per_sample()`).
The cell-15 plotting helpers are dropped; per-ancestry scatter PNGs are
emitted directly to the schema layout. MAE quantiles, Jaccard@τ, and the
μ-gating logic are NEW (not in cell 15).

Usage:
    python compare_to_rye.py \\
        --global-tsv PATH/<prefix>.global.tsv \\
        --rye-q       PATH/aou_admixture_estimates_rye_pruned_v9.Q \\
        --labels-json PATH/labels.json \\
        --out-dir     PATH/diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "popout"))
from popout.viz._loaders import read_global_tsv


# Canonical Rye column order (PLAN2.md §2.1; MID dropped per Sharon's R4 rule).
RYE_LABELS: tuple[str, ...] = ("eur", "eas", "amr", "afr", "sas")
JACCARD_THRESHOLDS: tuple[float, ...] = (0.10, 0.25, 0.50)
MU_GATE = 0.01           # below this, an ancestry is a degenerate non-test
PEARSON_THRESHOLD = 0.95
CCC_THRESHOLD = 0.90


# ─── Lifted from lk-test-flare-afr.ipynb cell 15 ──────────────────────────


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
    """Row-wise cosine similarity between matrices A and B (n_samples x n_dims)."""
    num = np.sum(A * B, axis=1)
    denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    denom = np.where(denom == 0, np.nan, denom)
    return num / denom


# ─── Rye loader (file format from PLAN2.md §2.1) ──────────────────────────


def load_rye_q(path: Path) -> dict[str, np.ndarray]:
    """Read a Rye Q TSV. Column order is detected by name — `research_id`
    (or sample_id/SAMPLE/sample) can appear anywhere in the header, and
    the 5 ancestry columns are pulled by name from RYE_LABELS.

    Observed in production: `eur eas amr afr sas research_id` (id LAST).

    Returns dict mapping sample_id -> length-5 proportion array (in RYE_LABELS order).
    """
    ID_ALIASES = ("research_id", "sample_id", "sample")
    out: dict[str, np.ndarray] = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        lower = [h.lower() for h in header]
        id_col = next((i for i, h in enumerate(lower) if h in ID_ALIASES), None)
        if id_col is None:
            raise RuntimeError(
                f"{path}: no sample-id column found in header. Expected one of "
                f"{ID_ALIASES}; got {header}"
            )
        # Build a permutation from header order → RYE_LABELS order.
        col_idx: list[int] = []
        for label in RYE_LABELS:
            try:
                col_idx.append(lower.index(label))
            except ValueError:
                raise RuntimeError(
                    f"{path}: missing required Rye column {label!r}. Got: {header}"
                )
        max_col = max(id_col, max(col_idx))
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max_col:
                continue
            sid = parts[id_col].strip()
            if not sid:
                continue
            try:
                out[sid] = np.array([float(parts[i]) for i in col_idx], dtype=np.float64)
            except ValueError as e:
                raise RuntimeError(f"{path}: failed to parse {sid!r}: {e}")
    return out


# ─── FLARE → 5-column projection via labels.json ──────────────────────────


def project_flare_to_rye_basis(
    flare_proportions: np.ndarray,
    labels: dict,
) -> np.ndarray:
    """Sum FLARE component ancestries per RF label so the matrix lines up with Rye.

    `labels["rf_to_popout_components"]` maps each RF label name (afr, amr, eas, eur, mid, sas)
    to a list of FLARE column indices. We sum those columns per RYE_LABELS entry.

    Returns (n_samples, len(RYE_LABELS)) matrix in RYE_LABELS order. MID is dropped
    (PLAN2.md §2.1: Rye doesn't track MID).
    """
    n, k = flare_proportions.shape
    rf_to_pop = labels.get("rf_to_popout_components", {})
    if not rf_to_pop:
        raise RuntimeError(
            "labels.json missing 'rf_to_popout_components' — cannot project FLARE to Rye basis"
        )
    projected = np.zeros((n, len(RYE_LABELS)), dtype=np.float64)
    for j, label in enumerate(RYE_LABELS):
        component_idxs = rf_to_pop.get(label, [])
        if not component_idxs:
            continue  # this Rye label has no FLARE component; column stays zero
        projected[:, j] = flare_proportions[:, component_idxs].sum(axis=1)
    return projected


# ─── Metrics per ancestry (cell-15 logic + v1.1 additions) ────────────────


def per_ancestry_metrics(
    flare: np.ndarray, rye: np.ndarray, ancestry_idx: int,
) -> dict[str, float | None]:
    """Compute the full schema row for one ancestry index (0..len(RYE_LABELS)-1)."""
    x = flare[:, ancestry_idx]
    y = rye[:, ancestry_idx]

    cluster_mu = float(x.mean())
    n = int(len(x))

    # Pearson r — guard against zero-variance.
    if x.std() == 0 or y.std() == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(x, y)[0, 1])

    ccc = lin_ccc(x, y)

    # Cosine per-sample (degenerate row × row collapses; report mean over samples).
    # Restricted to this ancestry column (1-D), cosine is just sign(x)*sign(y)
    # per sample — not meaningful here. We instead report the column-wise
    # correlation as cosine_mean is misleading. Keep NA for single-column cosine
    # to avoid ambiguity; the global cosine is computed across all 5 dims below.
    cosine_mean = None  # see global cosine in `concordance_summary.json`

    # MAE quantiles (★ v1.1 — not in cell 15).
    err = np.abs(x - y)
    mae_mean = float(err.mean())
    mae_median = float(np.median(err))
    mae_p95 = float(np.percentile(err, 95))

    # Jaccard@τ (★ v1.1 — uncommented from cell 15's commented-out sweep).
    jaccards: dict[float, float] = {}
    for tau in JACCARD_THRESHOLDS:
        a_mask = x >= tau
        b_mask = y >= tau
        inter = int(np.sum(a_mask & b_mask))
        union = int(np.sum(a_mask | b_mask))
        jaccards[tau] = float(inter / union) if union > 0 else float("nan")

    # μ-gating (★ v1.1 — not in any notebook).
    if cluster_mu < MU_GATE:
        passed = None  # degenerate non-test
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
        "cosine_mean": cosine_mean,
        "mae_mean": mae_mean,
        "mae_median": mae_median,
        "mae_p95": mae_p95,
        "jaccard_at_0.10": jaccards[0.10],
        "jaccard_at_0.25": jaccards[0.25],
        "jaccard_at_0.50": jaccards[0.50],
        "pass": passed,
    }


# ─── Output writers ───────────────────────────────────────────────────────


def write_concordance_metrics(rows: list[dict], out_path: Path) -> None:
    cols = ["ancestry", "cluster_mu", "n_samples", "pearson_r", "ccc",
            "cosine_mean", "mae_mean", "mae_median", "mae_p95",
            "jaccard_at_0.10", "jaccard_at_0.25", "jaccard_at_0.50", "pass"]
    with open(out_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            def fmt(v):
                if v is None:
                    return ""
                if isinstance(v, bool):
                    return "true" if v else "false"
                if isinstance(v, float):
                    if np.isnan(v):
                        return "NA"
                    return f"{v:.6f}"
                return str(v)
            f.write("\t".join(fmt(r[c]) for c in cols) + "\n")


def write_concordance_summary(
    rows: list[dict], global_ccc: float, n_overlap: int, out_path: Path,
) -> None:
    eligible = [r for r in rows if r["cluster_mu"] >= MU_GATE]
    passing_r = [r["ancestry"] for r in eligible
                 if not np.isnan(r["pearson_r"]) and r["pearson_r"] >= PEARSON_THRESHOLD]
    passing_ccc = [r["ancestry"] for r in eligible
                   if not np.isnan(r["ccc"]) and r["ccc"] >= CCC_THRESHOLD]
    failing = [r["ancestry"] for r in eligible if r["pass"] is False]
    mean_pearson_r = (
        float(np.nanmean([r["pearson_r"] for r in eligible])) if eligible else float("nan")
    )
    out_path.write_text(json.dumps({
        "global_ccc": global_ccc,
        "mean_pearson_r": mean_pearson_r,
        "n_samples_overlap": n_overlap,
        "labels_passing_r_ge_0.95": passing_r,
        "labels_passing_ccc_ge_0.90": passing_ccc,
        "labels_failing": failing,
    }, indent=2))


def write_rye_full_matrix(
    flare: np.ndarray, rye: np.ndarray, out_path: Path,
) -> None:
    """Pearson r per FLARE ancestry × Rye ancestry. Both bases are RYE_LABELS-aligned;
    this is mostly diagonal-heavy but useful for diagnosing cross-label bleed."""
    K = len(RYE_LABELS)
    with open(out_path, "w") as f:
        f.write("flare_ancestry\t" + "\t".join(RYE_LABELS) + "\n")
        for fi in range(K):
            x = flare[:, fi]
            row = [RYE_LABELS[fi]]
            for ri in range(K):
                y = rye[:, ri]
                if x.std() == 0 or y.std() == 0:
                    row.append("NA")
                else:
                    row.append(f"{float(np.corrcoef(x, y)[0, 1]):.4f}")
            f.write("\t".join(row) + "\n")


def write_rye_merged_groups(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        f.write("rf_label\tn_samples\tcluster_mu\tpearson_r\tccc\n")
        for r in rows:
            pr = "NA" if np.isnan(r["pearson_r"]) else f"{r['pearson_r']:.6f}"
            cc = "NA" if np.isnan(r["ccc"]) else f"{r['ccc']:.6f}"
            f.write(f"{r['ancestry']}\t{r['n_samples']}\t"
                    f"{r['cluster_mu']:.6f}\t{pr}\t{cc}\n")


def write_rye_confusion(
    flare: np.ndarray, rye: np.ndarray, out_path: Path,
) -> None:
    """Hard primary-call confusion matrix: rows = FLARE primary, cols = Rye primary."""
    K = len(RYE_LABELS)
    flare_primary = np.argmax(flare, axis=1)
    rye_primary = np.argmax(rye, axis=1)
    cm = np.zeros((K, K), dtype=int)
    for i in range(len(flare_primary)):
        cm[flare_primary[i], rye_primary[i]] += 1
    with open(out_path, "w") as f:
        f.write("flare_primary\t" + "\t".join(RYE_LABELS) + "\ttotal\n")
        for i in range(K):
            row = [RYE_LABELS[i]] + [str(cm[i, j]) for j in range(K)] + [str(cm[i].sum())]
            f.write("\t".join(row) + "\n")
        col_totals = cm.sum(axis=0)
        f.write("total\t" + "\t".join(str(c) for c in col_totals) + f"\t{col_totals.sum()}\n")


def write_rye_scatter(
    flare: np.ndarray, rye: np.ndarray, rows: list[dict], out_dir: Path,
) -> None:
    """One scatter per ancestry with cluster_mu ≥ MU_GATE."""
    for j, label in enumerate(RYE_LABELS):
        row = rows[j]
        if row["cluster_mu"] < MU_GATE:
            continue
        x = rye[:, j]
        y = flare[:, j]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(x, y, s=6, alpha=0.4)
        lims = [0, max(1.0, float(max(x.max(), y.max())))]
        ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="y=x")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Rye {label}")
        ax.set_ylabel(f"FLARE {label}")
        title = f"{label}: FLARE vs Rye (r={row['pearson_r']:.3f}, CCC={row['ccc']:.3f}, n={row['n_samples']})"
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"rye_scatter_{label}.png", dpi=150)
        plt.close(fig)


def write_rye_admixture_comparison(
    flare: np.ndarray, rye: np.ndarray, out_dir: Path,
) -> None:
    """Mean-bar comparison across all ancestries."""
    flare_means = flare.mean(axis=0)
    rye_means = rye.mean(axis=0)
    x = np.arange(len(RYE_LABELS))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, rye_means, width, label="Rye", color="#4477AA")
    ax.bar(x + width / 2, flare_means, width, label="FLARE", color="#EE6677")
    ax.set_xticks(x)
    ax.set_xticklabels(RYE_LABELS)
    ax.set_ylabel("Mean proportion")
    ax.set_title("Cohort mean ancestry proportions: FLARE vs Rye")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rye_admixture_comparison.png", dpi=150)
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--global-tsv", type=Path, required=True,
                   help="FLARE/popout global.tsv (popout format)")
    p.add_argument("--rye-q", type=Path, required=True,
                   help="Rye Q TSV: header `research_id eur eas amr afr sas`")
    p.add_argument("--labels-json", type=Path, required=True,
                   help="labels.json from compare_to_rf.py (for rf_to_popout_components)")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.global_tsv, args.rye_q, args.labels_json):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Loading FLARE global.tsv from {args.global_tsv}")
    ga = read_global_tsv(args.global_tsv)
    flare_idx = {sid: i for i, sid in enumerate(ga.sample_names)}
    print(f"  {len(ga.sample_names):,} FLARE samples, {ga.n_ancestries} ancestries")

    print(f"Loading labels.json from {args.labels_json}")
    labels = json.loads(args.labels_json.read_text())

    print(f"Loading Rye Q from {args.rye_q}")
    rye_samples = load_rye_q(args.rye_q)
    print(f"  {len(rye_samples):,} Rye samples")

    # Intersect.
    common = sorted(set(flare_idx) & set(rye_samples))
    if not common:
        raise RuntimeError(
            "No overlapping samples between FLARE global.tsv and Rye Q. "
            f"Examples — FLARE: {list(flare_idx)[:3]}; Rye: {list(rye_samples)[:3]}"
        )
    n_overlap = len(common)
    print(f"  intersection: {n_overlap:,} samples")

    # Build aligned matrices in RYE_LABELS basis.
    flare_rows = np.array([flare_idx[sid] for sid in common])
    flare_full = ga.proportions[flare_rows]                            # (n, K_flare)
    flare = project_flare_to_rye_basis(flare_full, labels)             # (n, 5) on RYE_LABELS
    rye = np.array([rye_samples[sid] for sid in common])               # (n, 5) on RYE_LABELS

    # Per-ancestry metrics.
    rows = []
    for j, label in enumerate(RYE_LABELS):
        m = per_ancestry_metrics(flare, rye, j)
        m["ancestry"] = label
        rows.append(m)

    # Global CCC on the flattened (n_samples × K) matrices.
    global_ccc = lin_ccc(flare.flatten(), rye.flatten())

    # Cosine summary on the full 5-dim row vectors (this is well-defined,
    # unlike per-ancestry single-column cosine).
    cos_sim = cosine_similarity_per_sample(flare, rye)
    print(f"  global_ccc={global_ccc:.4f}  cosine mean={float(np.nanmean(cos_sim)):.4f}")
    for r in rows:
        gate = " (μ<0.01)" if r["cluster_mu"] < MU_GATE else ""
        ccc = "NA" if np.isnan(r["ccc"]) else f"{r['ccc']:.3f}"
        prr = "NA" if np.isnan(r["pearson_r"]) else f"{r['pearson_r']:.3f}"
        pass_txt = "—" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL")
        print(f"    {r['ancestry']:>3}{gate:<10}  mu={r['cluster_mu']:.4f}  "
              f"r={prr}  CCC={ccc}  {pass_txt}")

    # ── Write outputs ──
    write_concordance_metrics(rows, args.out_dir / "concordance_metrics.tsv")
    write_concordance_summary(rows, global_ccc, n_overlap,
                              args.out_dir / "concordance_summary.json")
    write_rye_full_matrix(flare, rye, args.out_dir / "rye_full_matrix.tsv")
    write_rye_merged_groups(rows, args.out_dir / "rye_merged_groups.tsv")
    write_rye_confusion(flare, rye, args.out_dir / "rye_confusion_matrix.tsv")
    write_rye_scatter(flare, rye, rows, args.out_dir)
    write_rye_admixture_comparison(flare, rye, args.out_dir)
    print(f"  wrote 7 output families to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
