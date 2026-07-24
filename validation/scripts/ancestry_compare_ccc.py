#!/usr/bin/env python3
"""Generic ancestry-table comparison: CCC + scatter on overlapping samples.

Both inputs are single TSVs with a sample-id column and N ancestry columns.
If label spaces match, compare 1:1.
If they differ, the larger label space is collapsed onto the smaller by
argmax-Pearson mapping (per-column) on the overlap, then summed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ID_CANDIDATES = ("sample_id", "sample", "research_id", "iid", "IID", "id", "ID")


def detect_id_col(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            sys.exit(f"id column {explicit!r} not in {list(df.columns)}")
        return explicit
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    sys.exit(f"could not detect id column among {list(df.columns)}; pass --id-a/--id-b")


def load_table(path: Path, id_col: str | None, label: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, sep=None, engine="python")
    ic = detect_id_col(df, id_col)
    anc_cols = [c for c in df.columns if c != ic]
    if not anc_cols:
        sys.exit(f"{path}: no ancestry columns after removing id {ic!r}")
    df = df[[ic, *anc_cols]].copy()
    df[ic] = df[ic].astype(str)
    df = df.rename(columns={ic: "sample_id"}).set_index("sample_id")
    print(f"{label}: {path.name}  id={ic}  n={len(df)}  labels={anc_cols}", file=sys.stderr)
    return df, anc_cols


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    vx = x.var(ddof=0)
    vy = y.var(ddof=0)
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float(2.0 * cov / denom)


def zscore(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (a - mu) / sd


def collapse_by_mapping(
    src: pd.DataFrame, src_cols: list[str], dst_cols: list[str],
    ref_src: pd.DataFrame, ref_dst: pd.DataFrame, src_label: str, dst_label: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assign each src column to its argmax-Pearson dst column on the overlap, then sum."""
    common = ref_src.index.intersection(ref_dst.index)
    if len(common) == 0:
        sys.exit(f"no overlap between {src_label} and {dst_label} for mapping")
    S = ref_src.loc[common, src_cols].to_numpy(dtype=float)
    D = ref_dst.loc[common, dst_cols].to_numpy(dtype=float)
    corr = (zscore(S).T @ zscore(D)) / len(common)

    mapping: dict[str, str] = {}
    print(f"\ncollapse mapping: {src_label}({len(src_cols)}) -> {dst_label}({len(dst_cols)})",
          file=sys.stderr)
    print(f"  n={len(common)} overlap samples used", file=sys.stderr)
    for i, s in enumerate(src_cols):
        j = int(np.argmax(corr[i]))
        best = dst_cols[j]
        mapping[s] = best
        row = "  " + s + "  " + "  ".join(
            f"{dst_cols[k]}={corr[i, k]:+.3f}" for k in range(len(dst_cols))
        ) + f"  ->  {best}"
        print(row, file=sys.stderr)

    unassigned = [d for d in dst_cols if d not in mapping.values()]
    if unassigned:
        print(f"WARNING: no {src_label} column mapped to: {unassigned}", file=sys.stderr)

    out = pd.DataFrame(index=src.index)
    for d in dst_cols:
        cols = [s for s, m in mapping.items() if m == d]
        if not cols:
            sys.exit(f"no {src_label} columns collapse to {d}")
        out[d] = src[cols].sum(axis=1)
    return out, mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path, help="table A")
    ap.add_argument("b", type=Path, help="table B")
    ap.add_argument("--id-a", type=str, default=None)
    ap.add_argument("--id-b", type=str, default=None)
    ap.add_argument("--name-a", type=str, default="A")
    ap.add_argument("--name-b", type=str, default="B")
    ap.add_argument("--out", type=Path, default=Path("ancestry_compare_ccc.png"))
    ap.add_argument("--n-plot", type=int, default=None,
                    help="max samples to plot (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a_df, a_cols = load_table(args.a, args.id_a, args.name_a)
    b_df, b_cols = load_table(args.b, args.id_b, args.name_b)

    same_space = set(a_cols) == set(b_cols)
    if same_space:
        labels = sorted(set(a_cols), key=a_cols.index)
        a_use = a_df[labels]
        b_use = b_df[labels]
        collapse_note = "label spaces identical (no collapse)"
        print(f"\n{collapse_note}", file=sys.stderr)
    else:
        if len(a_cols) == len(b_cols):
            sys.exit(f"label spaces differ but sizes equal ({len(a_cols)}): "
                     f"A={a_cols} B={b_cols}; refusing to guess a permutation")
        if len(a_cols) > len(b_cols):
            labels = list(b_cols)
            a_use, mapping = collapse_by_mapping(
                a_df, a_cols, labels, a_df, b_df, args.name_a, args.name_b,
            )
            b_use = b_df[labels]
            collapse_note = f"collapsed {args.name_a} ({len(a_cols)}) onto {args.name_b} labels ({len(labels)})"
        else:
            labels = list(a_cols)
            b_use, mapping = collapse_by_mapping(
                b_df, b_cols, labels, b_df, a_df, args.name_b, args.name_a,
            )
            a_use = a_df[labels]
            collapse_note = f"collapsed {args.name_b} ({len(b_cols)}) onto {args.name_a} labels ({len(labels)})"

    common = a_use.index.intersection(b_use.index)
    print(f"\noverlap: {len(common)} samples", file=sys.stderr)
    if len(common) == 0:
        sys.exit("no overlapping samples")

    A = a_use.loc[common, labels].to_numpy()
    B = b_use.loc[common, labels].to_numpy()

    per_ccc = {lab: lin_ccc(A[:, i], B[:, i]) for i, lab in enumerate(labels)}
    overall = lin_ccc(A.ravel(), B.ravel())

    print("\nCCC (Lin's concordance correlation coefficient):")
    for lab in labels:
        print(f"  {lab}: {per_ccc[lab]:.4f}")
    print(f"  overall (flattened): {overall:.4f}")

    if args.n_plot is None or args.n_plot >= len(common):
        n = len(common)
        Ap, Bp = A, B
    else:
        n = args.n_plot
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(common), size=n, replace=False)
        Ap, Bp = A[idx], B[idx]

    k = len(labels)
    ncols = 3
    nrows = int(np.ceil((k + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, lab in enumerate(labels):
        ax = axes[i]
        ax.scatter(Bp[:, i], Ap[:, i], s=3, alpha=0.35, edgecolor="none")
        ax.plot([0, 1], [0, 1], color="red", lw=1, ls="--")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"{args.name_b} {lab}")
        ax.set_ylabel(f"{args.name_a} {lab}")
        ax.set_title(f"{lab}: CCC={per_ccc[lab]:.4f}")
        ax.set_aspect("equal", adjustable="box")

    summary_ax = axes[k]
    summary_ax.axis("off")
    summary_ax.text(
        0.02, 0.95,
        f"N overlap: {len(common):,}\nN plotted: {n:,}\n\n"
        + "\n".join(f"CCC[{lab}] = {per_ccc[lab]:.4f}" for lab in labels)
        + f"\n\noverall CCC = {overall:.4f}",
        va="top", ha="left", family="monospace", fontsize=11,
        transform=summary_ax.transAxes,
    )

    for j in range(k + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{args.name_a} vs {args.name_b} global ancestry  |  {collapse_note}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
