#!/usr/bin/env python3
"""Layer 3.3: F_ST tree from converged model allele frequencies.

Computes pairwise Hudson F_ST between all ancestry pairs, builds a UPGMA
dendrogram, and plots alongside a heatmap. Expect: AFR deepest split,
EUR-AMR-EAS-SAS clade with sub-continental structure.

Usage:
    python validate_fst_tree.py --prefix data/recur_v2/aou_v9_hmm \
        --out-dir diagnostics/validation/recur_v2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "popout"))

from popout.viz._loaders import read_labels_json, read_model_npz
from popout.viz._style import ancestry_names


def hudson_fst(p_i: np.ndarray, p_j: np.ndarray) -> float:
    """Compute Hudson F_ST averaged across sites."""
    h_s = (p_i * (1 - p_i) + p_j * (1 - p_j)) / 2
    p_bar = (p_i + p_j) / 2
    h_t = p_bar * (1 - p_bar)
    # Avoid division by zero at monomorphic sites
    mask = h_t > 0
    if mask.sum() == 0:
        return 0.0
    return float(1 - h_s[mask].mean() / h_t[mask].mean())


def main():
    parser = argparse.ArgumentParser(description="Layer 3.3: F_ST tree")
    parser.add_argument("--prefix", type=Path, required=True,
                        help="Output prefix (e.g. data/recur_v2/aou_v9_hmm)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for plots")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated ancestry labels (optional)")
    parser.add_argument("--labels-json", type=Path, default=None,
                        help="Path to labels.json for ancestry names (optional)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = read_model_npz(args.prefix.with_name(args.prefix.name + ".model.npz"))
    freq = model["allele_freq"]  # (A, T)
    mu = model["mu"]
    n_anc = freq.shape[0]

    if args.labels_json and args.labels_json.exists():
        labels = read_labels_json(args.labels_json)
        names = ancestry_names(n_anc, labels)
    elif args.labels:
        names = args.labels.split(",")
    else:
        names = [f"{i}" for i in range(n_anc)]

    # Compute pairwise F_ST
    fst_matrix = np.zeros((n_anc, n_anc))
    for i in range(n_anc):
        for j in range(i + 1, n_anc):
            f = hudson_fst(freq[i], freq[j])
            fst_matrix[i, j] = f
            fst_matrix[j, i] = f

    # Print F_ST matrix
    print("Pairwise Hudson F_ST:")
    print(f"\n  {'':>5}", end="")
    for j in range(n_anc):
        print(f"  {names[j]:>6}", end="")
    print()
    for i in range(n_anc):
        print(f"  {names[i]:>5}", end="")
        for j in range(n_anc):
            if i == j:
                print(f"  {'---':>6}", end="")
            else:
                print(f"  {fst_matrix[i,j]:>6.4f}", end="")
        print(f"   (mu={mu[i]:.4f})")

    # Summary stats
    upper = fst_matrix[np.triu_indices(n_anc, k=1)]
    print(f"\n  F_ST range: [{upper.min():.4f}, {upper.max():.4f}]")
    print(f"  F_ST mean:  {upper.mean():.4f}")
    print(f"  F_ST median: {np.median(upper):.4f}")

    # UPGMA dendrogram
    condensed = squareform(fst_matrix)
    Z = linkage(condensed, method="average")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: dendrogram
    ax = axes[0]
    dn = dendrogram(Z, labels=names, orientation="left", ax=ax,
                    leaf_font_size=9, color_threshold=0)
    ax.set_xlabel("F_ST (UPGMA distance)")
    ax.set_title("UPGMA Tree from Pairwise F_ST")

    # Right: heatmap (reordered by dendrogram leaves)
    ax = axes[1]
    leaf_order = dn["leaves"]
    reordered = fst_matrix[np.ix_(leaf_order, leaf_order)]
    reordered_names = [names[i] for i in leaf_order]

    im = ax.imshow(reordered, cmap="YlOrRd", vmin=0, vmax=upper.max())
    ax.set_xticks(range(n_anc))
    ax.set_yticks(range(n_anc))
    ax.set_xticklabels(reordered_names, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(reordered_names, fontsize=8)
    ax.set_title("Pairwise F_ST (dendrogram order)")

    for i in range(n_anc):
        for j in range(n_anc):
            if i != j:
                val = reordered[i, j]
                color = "white" if val > upper.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="F_ST")
    fig.suptitle("Ancestry Population Structure from Converged Model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = args.out_dir / "fst_tree.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved {save_path}")

    # Save F_ST matrix as TSV
    tsv_path = args.out_dir / "fst_matrix.tsv"
    with open(tsv_path, "w") as f:
        f.write("ancestry\t" + "\t".join(names) + "\tmu\n")
        for i in range(n_anc):
            vals = "\t".join(f"{fst_matrix[i,j]:.5f}" for j in range(n_anc))
            f.write(f"{names[i]}\t{vals}\t{mu[i]:.5f}\n")
    print(f"  Saved {tsv_path}")


if __name__ == "__main__":
    main()
