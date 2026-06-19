#!/usr/bin/env python3
"""§8.2 / R10: FLARE global proportions vs Rye (supervised ADMIXTURE) Q.

Rye is the team's supervised-ADMIXTURE variant; its Q file carries named
ancestry columns from the set `{eur, eas, amr, afr, sas}`. FLARE's
global.tsv also carries named columns (the panel-population names from the
VCF ``##ANCESTRY=`` header, preserved verbatim by
``flare_to_popout_format.write_popout_global``). Alignment is by name on
the intersection of the two header sets. No sidecar label map is needed
and none is consulted.

Per PLAN2.md §2.2, R10 acceptance is
    Pearson r >= 0.95 AND Lin's CCC >= 0.90
for every ancestry with cluster_mu >= 0.01 (mu-gating: degenerate
ancestries get a pass=null row, not a failure).

Source lift: `my_notes/lk_notebooks/lk-test-flare-afr.ipynb` cell 15
(`compare_rye_vs_flare()`, helpers `lin_ccc()`, `cosine_similarity_per_sample()`).
The cell-15 plotting helpers are dropped; per-ancestry scatter PNGs are
emitted directly to the schema layout. MAE quantiles, Jaccard@tau, and the
mu-gating logic are NEW (not in cell 15).

Usage:
    python compare_to_rye.py \\
        --global-tsv PATH/<prefix>.global.tsv \\
        --rye-q       PATH/aou_admixture_estimates_rye_pruned_v9.Q \\
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

# Shared concordance primitives — same shape as compare_to_rf.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from concordance import (  # noqa: E402
    MU_GATE,
    align_by_name,
    build_concordance_rows,
    write_concordance_metrics,
    write_concordance_summary,
    write_full_matrix,
    write_hard_confusion,
    write_merged_groups,
)


# Rye's native column order (distinct from SP5 alphabetical). Rye does not
# carry a MID column.
RYE_LABELS: tuple[str, ...] = ("eur", "eas", "amr", "afr", "sas")


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


# ─── Output writers ───────────────────────────────────────────────────────


def write_rye_scatter(
    flare: np.ndarray, rye: np.ndarray, labels: list[str],
    rows: list[dict], out_dir: Path,
) -> None:
    """One scatter per ancestry on the shared label set with cluster_mu >= MU_GATE."""
    for j, label in enumerate(labels):
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
    flare: np.ndarray, rye: np.ndarray, labels: list[str], out_dir: Path,
) -> None:
    """Mean-bar comparison across the shared label set."""
    flare_means = flare.mean(axis=0)
    rye_means = rye.mean(axis=0)
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, rye_means, width, label="Rye", color="#4477AA")
    ax.bar(x + width / 2, flare_means, width, label="FLARE", color="#EE6677")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
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
                   help="FLARE global.tsv (popout-format, named columns)")
    p.add_argument("--rye-q", type=Path, required=True,
                   help="Rye Q TSV: header carries `research_id` plus named "
                        "ancestry columns from {eur, eas, amr, afr, sas}")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.global_tsv, args.rye_q):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Loading FLARE global.tsv from {args.global_tsv}")
    ga = read_global_tsv(args.global_tsv)
    flare_idx = {sid: i for i, sid in enumerate(ga.sample_names)}
    print(f"  {len(ga.sample_names):,} FLARE samples; ancestry columns "
          f"(verbatim from header): {ga.ancestry_names!r}")

    print(f"Loading Rye Q from {args.rye_q}")
    rye_samples = load_rye_q(args.rye_q)
    print(f"  {len(rye_samples):,} Rye samples")

    # Intersect samples.
    common = sorted(set(flare_idx) & set(rye_samples))
    if not common:
        raise RuntimeError(
            "No overlapping samples between FLARE global.tsv and Rye Q. "
            f"FLARE examples: {list(flare_idx)[:3]}; "
            f"Rye examples: {list(rye_samples)[:3]}"
        )
    n_overlap = len(common)
    print(f"  intersection: {n_overlap:,} samples")

    # Align FLARE and Rye columns by name. Rye's native order acts as
    # the preferred basis so cluster_mu and per-ancestry rows come out
    # in RYE_LABELS order on the intersection.
    flare_rows = np.array([flare_idx[sid] for sid in common])
    rye_full = np.array([rye_samples[sid] for sid in common])  # (n, 5) RYE_LABELS order
    flare, rye, shared_labels = align_by_name(
        ga.proportions[flare_rows], list(ga.ancestry_names),
        rye_full, list(RYE_LABELS),
        basis_order=list(RYE_LABELS),
    )

    rows = build_concordance_rows(flare, rye, shared_labels)

    for r in rows:
        gate = " (mu<0.01)" if r["cluster_mu"] < MU_GATE else ""
        ccc = "NA" if np.isnan(r["ccc"]) else f"{r['ccc']:.3f}"
        prr = "NA" if np.isnan(r["pearson_r"]) else f"{r['pearson_r']:.3f}"
        pass_txt = "-" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL")
        print(f"    {r['ancestry']:>3}{gate:<10}  mu={r['cluster_mu']:.4f}  "
              f"r={prr}  CCC={ccc}  {pass_txt}")

    # ── Write outputs (all rye-suffixed for symmetry with compare_to_rf) ──
    write_concordance_metrics(rows, args.out_dir / "concordance_metrics_rye.tsv")
    write_concordance_summary(rows, flare, rye, n_overlap,
                              args.out_dir / "concordance_summary_rye.json")
    write_full_matrix(flare, rye, shared_labels,
                      args.out_dir / "rye_full_matrix.tsv",
                      a_axis_name="flare_ancestry")
    write_merged_groups(rows, args.out_dir / "rye_merged_groups.tsv")
    write_hard_confusion(
        flare, list(shared_labels), rye, list(shared_labels),
        args.out_dir / "rye_confusion_matrix.tsv",
        a_axis_name="flare_primary",
    )
    write_rye_scatter(flare, rye, shared_labels, rows, args.out_dir)
    write_rye_admixture_comparison(flare, rye, shared_labels, args.out_dir)
    print(f"  wrote 7 output families to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
