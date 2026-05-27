#!/usr/bin/env python3
"""Concordance visualization: popout/FLARE vs the RF preliminary classifier.

Generates 9 figures comparing popout/FLARE per-sample ancestry assignments
against the RF classifier's predictions on the same cohort. See
diagnostics/GLOSSARY.md for the canonical vocabulary.

Usage:
    python plot_concordance.py \
        --popout-global data/<run>/<prefix>.global.tsv \
        --rf-ancestry PATH/foxtrot_v4.ancestry_preds.tsv \
        --labels-json diagnostics/<run>/labels.json \
        --out-dir diagnostics/<run>
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "popout"))

from popout.viz._loaders import read_labels_json
from popout.viz._style import ancestry_colors, ancestry_names, popout_style
from popout.viz.label_correlation import plot_label_correlation


# ── Argument parsing ─────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Concordance: popout/FLARE vs RF classifier")
parser.add_argument("--popout-global", type=Path, required=True,
                    help="Primary tool's per-sample global ancestry TSV (FLARE or popout)")
parser.add_argument("--rf-ancestry", type=Path, required=True,
                    help="RF ancestry predictions (foxtrot_v4.ancestry_preds.tsv)")
parser.add_argument("--labels-json", type=Path, required=True,
                    help="labels.json produced by compare_to_rf.py for the primary tool")
parser.add_argument("--secondary-global", type=Path, default=None,
                    help="OPTIONAL second tool's global.tsv (the OTHER of FLARE/popout). "
                         "When provided alongside --secondary-labels, figures gain a "
                         "third panel/strip for that tool (three-way comparison).")
parser.add_argument("--secondary-labels", type=Path, default=None,
                    help="OPTIONAL labels.json for the secondary tool (required when "
                         "--secondary-global is set; pairs them).")
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

if (args.secondary_global is None) != (args.secondary_labels is None):
    parser.error("--secondary-global and --secondary-labels must be supplied together.")


# ── Load data ────────────────────────────────────────────────────────────

print("Loading labels.json...")
labels = read_labels_json(args.labels_json)
rf_ref_labels = labels["rf_ref_labels"]  # e.g. ['afr','amr','eas','eur','mid','sas']
rf_to_popout = labels["rf_to_popout_components"]
n_ref = len(rf_ref_labels)
TOOL = labels.get("tool", "popout")  # "FLARE" or "popout"; user-facing label
print(f"  tool: {TOOL}")

# Real merging only happened when the tool's K is larger than the number of
# RF reference labels (popout K=16 → 6 collapses; FLARE K=5 → 5 does not).
# When K matches, "(merged)" is misleading; drop the qualifier.

print(f"Loading {TOOL} global TSV...")
popout_samples = {}
with open(args.popout_global) as f:
    header = f.readline().strip().split("\t")
    n_popout_anc = len(header) - 1
    for line in f:
        parts = line.strip().split("\t")
        popout_samples[parts[0]] = [float(x) for x in parts[1:]]
print(f"  {len(popout_samples):,} samples, {n_popout_anc} {TOOL} ancestries")

# "(merged)" qualifier — non-empty only when n_popout_anc > n_ref so merging
# actually happened. Used in axis labels and figure titles below.
MERGED = " (merged)" if n_popout_anc > n_ref else ""

print("Loading RF ancestry table...")
rf_data = {}
parse_errors = 0
with open(args.rf_ancestry) as f:
    rf_header = f.readline().strip().split("\t")
    rid_col = rf_header.index("research_id")
    pred_col = rf_header.index("ancestry_pred")
    prob_col = rf_header.index("probabilities")
    for line in f:
        parts = line.strip().split("\t")
        rid = parts[rid_col]
        try:
            rf_data[rid] = (parts[pred_col], ast.literal_eval(parts[prob_col]))
        except (ValueError, SyntaxError):
            parse_errors += 1
print(f"  {len(rf_data):,} samples ({parse_errors} parse errors)")

print("Joining...")
common_ids = sorted(set(popout_samples) & set(rf_data))
n = len(common_ids)
print(f"  {n:,} matched samples")

if n == 0:
    print("FATAL: No matching IDs")
    sys.exit(1)

# Build aligned matrices
popout_raw = np.zeros((n, n_popout_anc), dtype=np.float32)
rf_prob = np.zeros((n, n_ref), dtype=np.float32)
rf_hard_calls = []
for i, rid in enumerate(common_ids):
    popout_raw[i] = popout_samples[rid]
    label, probs = rf_data[rid]
    rf_prob[i] = probs
    rf_hard_calls.append(label)
rf_hard_calls = np.array(rf_hard_calls)


# ── Merge popout components → RF reference labels ──────────────────────

def merge_popout(raw, rf_to_popout, rf_ref_labels):
    """Sum popout columns per RF-label merge group → (N, n_ref) matrix."""
    merged = np.zeros((raw.shape[0], len(rf_ref_labels)), dtype=np.float32)
    for j, name in enumerate(rf_ref_labels):
        for idx in rf_to_popout.get(name, []):
            merged[:, j] += raw[:, idx]
    return merged

popout_merged = merge_popout(popout_raw, rf_to_popout, rf_ref_labels)
print(f"  Collapsed {TOOL} {n_popout_anc} → {n_ref} dimensions")

# Hard calls
popout_hard_idx = np.argmax(popout_merged, axis=1)
popout_hard = np.array([rf_ref_labels[i] for i in popout_hard_idx])
rf_max_prob = rf_prob.max(axis=1)


# ── Optional secondary tool (3-way comparison) ──────────────────────────
#
# When --secondary-global + --secondary-labels are supplied, load the other
# tool's per-sample proportions on the same samples and collapse to the
# 6 RF reference labels. Downstream figures pick up `SECONDARY_TOOL` and
# `secondary_merged` if `SECONDARY_TOOL` is non-None.

SECONDARY_TOOL: str | None = None
secondary_merged: np.ndarray | None = None
if args.secondary_global is not None:
    sec_labels = read_labels_json(args.secondary_labels)
    SECONDARY_TOOL = sec_labels.get("tool", "popout")
    if SECONDARY_TOOL == TOOL:
        raise RuntimeError(
            f"Secondary tool '{SECONDARY_TOOL}' is the same as primary '{TOOL}'. "
            "Provide the OTHER of FLARE/popout."
        )
    print(f"Loading secondary {SECONDARY_TOOL} global TSV...")
    sec_samples: dict[str, list[float]] = {}
    with open(args.secondary_global) as f:
        sec_header = f.readline().strip().split("\t")
        n_sec_anc = len(sec_header) - 1
        for line in f:
            parts = line.strip().split("\t")
            sec_samples[parts[0]] = [float(x) for x in parts[1:]]
    print(f"  {len(sec_samples):,} samples, {n_sec_anc} {SECONDARY_TOOL} ancestries")

    # Subset secondary to the primary-vs-RF common_ids. Fail loudly when
    # any sample is missing from the secondary — that's a real data gap
    # (e.g., the secondary run didn't cover this cohort).
    missing = [rid for rid in common_ids if rid not in sec_samples]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples in the {TOOL}/RF intersection are NOT in the "
            f"secondary {SECONDARY_TOOL} run. Example IDs: {missing[:5]}. "
            f"Three-way comparison requires the secondary run to cover the "
            f"same samples."
        )
    sec_raw = np.zeros((n, n_sec_anc), dtype=np.float32)
    for i, rid in enumerate(common_ids):
        sec_raw[i] = sec_samples[rid]
    sec_rf_to_popout = sec_labels["rf_to_popout_components"]
    secondary_merged = merge_popout(sec_raw, sec_rf_to_popout, rf_ref_labels)
    print(f"  Collapsed {SECONDARY_TOOL} {n_sec_anc} → {n_ref} dimensions")
    SECONDARY_MERGED_QUAL = " (merged)" if n_sec_anc > n_ref else ""
else:
    SECONDARY_MERGED_QUAL = ""

# Consistent colors for 6 continental labels
LABEL_COLORS = dict(zip(rf_ref_labels, ancestry_colors(n_ref)))
LABEL_COLORS["mixed"] = "#999999"


# ── F1: Correlation heatmap (20 x 6) ────────────────────────────────────

print("\nF1: Correlation heatmap...")
fig1 = plot_label_correlation(prefix=None, labels=labels)
fig1.savefig(args.out_dir / "correlation_heatmap.png", dpi=args.dpi)
plt.close(fig1)
print("  saved correlation_heatmap.png")


# ── F2: Merged confusion matrix heatmap (7 x 7) ────────────────────────

print("F2: Merged confusion matrix...")

# Build labels with "mixed" for low-confidence
rf_cm = np.where(rf_max_prob < 0.8, "mixed", rf_hard_calls)
popout_max_merged = popout_merged.max(axis=1)
popout_cm = np.where(popout_max_merged < 0.8, "mixed", popout_hard)

cm_labels = list(rf_ref_labels) + ["mixed"]
cm = np.zeros((len(cm_labels), len(cm_labels)), dtype=int)
for i in range(n):
    r = cm_labels.index(rf_cm[i])
    c = cm_labels.index(popout_cm[i])
    cm[r, c] += 1

# Row-normalize for recall
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

with popout_style():
    fig2, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cm_labels)))
    ax.set_yticks(range(len(cm_labels)))
    ax.set_xticklabels(cm_labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(cm_labels, fontsize=10)
    ax.set_xlabel(f"{TOOL}{MERGED}")
    ax.set_ylabel("RF")
    for i in range(len(cm_labels)):
        for j in range(len(cm_labels)):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if pct > 60 else "black"
            ax.text(j, i, f"{count:,}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=7, color=color)
    fig2.colorbar(im, ax=ax, shrink=0.8, label="Recall")
    ax.set_title(
        ("Merged " if MERGED else "") + "Confusion Matrix (row-normalized recall)",
        fontweight="bold",
    )
    fig2.tight_layout()

fig2.savefig(args.out_dir / "merged_confusion_matrix.png", dpi=args.dpi)
plt.close(fig2)
print("  saved merged_confusion_matrix.png")


# ── F3: Soft proportion hexbin grid (2 x 3) ─────────────────────────────

print("F3: Soft proportion hexbin grid...")

with popout_style():
    fig3, axes = plt.subplots(2, 3, figsize=(14, 9))
    for j, name in enumerate(rf_ref_labels):
        ax = axes[j // 3, j % 3]
        x = rf_prob[:, j]
        y = popout_merged[:, j]
        hb = ax.hexbin(x, y, gridsize=80, mincnt=1, norm=LogNorm(),
                        cmap="viridis", extent=[0, 1, 0, 1])
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, alpha=0.7)
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_xlabel(f"RF {name}")
        ax.set_ylabel(f"{TOOL} {name}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    fig3.suptitle("Soft Proportion Agreement (hexbin)", fontsize=13, fontweight="bold")
    fig3.tight_layout()

fig3.savefig(args.out_dir / "soft_proportion_hexbin.png", dpi=args.dpi)
plt.close(fig3)
print("  saved soft_proportion_hexbin.png")


# ── F4: Side-by-side admixture bars ──────────────────────────────────────

print("F4: Admixture comparison bars...")

# Sort by popout dominant ancestry, then by proportion
dominant = np.argmax(popout_merged, axis=1)
sort_idx = np.lexsort((popout_merged[np.arange(n), dominant], dominant))

# Subsample
max_display = 5000
if n > max_display:
    step = n / max_display
    sel = np.round(np.arange(max_display) * step).astype(int)
    sel = np.clip(sel, 0, n - 1)
else:
    sel = np.arange(n)
    max_display = n

popout_sorted = popout_merged[sort_idx][sel]
rf_sorted = rf_prob[sort_idx][sel]
n_display = len(sel)

rgba_colors = [to_rgba(LABEL_COLORS[name]) for name in rf_ref_labels]
n_vert = 200

def build_strip(props, n_anc, n_display, n_vert, rgba_colors):
    strip = np.zeros((n_vert, n_display, 4), dtype=np.float32)
    for si in range(n_display):
        cum = 0.0
        for a in range(n_anc):
            y_start = int(cum * n_vert)
            cum += props[si, a]
            y_end = int(cum * n_vert)
            for ch in range(4):
                strip[y_start:y_end, si, ch] = rgba_colors[a][ch]
    return strip

strip_popout = build_strip(popout_sorted, n_ref, n_display, n_vert, rgba_colors)
strip_rf = build_strip(rf_sorted, n_ref, n_display, n_vert, rgba_colors)

# Optional third strip when a secondary tool was supplied.
strip_secondary = None
if secondary_merged is not None:
    secondary_sorted = secondary_merged[sort_idx][sel]
    strip_secondary = build_strip(secondary_sorted, n_ref, n_display, n_vert, rgba_colors)

with popout_style():
    panels: list[tuple[np.ndarray, str]] = [(strip_popout, f"{TOOL}{MERGED}")]
    if strip_secondary is not None:
        panels.append((strip_secondary, f"{SECONDARY_TOOL}{SECONDARY_MERGED_QUAL}"))
    panels.append((strip_rf, "RF"))

    fig4, axes = plt.subplots(len(panels), 1,
                              figsize=(16, 2.4 * len(panels) + 0.2),
                              sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (strip, title) in zip(axes, panels):
        ax.imshow(strip, aspect="auto", origin="lower",
                  extent=[0, n_display, 0, 1], interpolation="nearest")
        ax.set_xlim(0, n_display)
        ax.set_ylim(0, 1)
        ax.set_ylabel(title, fontsize=11)
        ax.set_yticks([0, 0.5, 1])

    axes[-1].set_xlabel(f"Samples ({n_display:,} of {n:,}, sorted by {TOOL} dominant ancestry)")

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=LABEL_COLORS[name], label=name) for name in rf_ref_labels]
    fig4.legend(handles=handles, loc="upper center", ncol=n_ref, fontsize=9,
                frameon=False, bbox_to_anchor=(0.5, 1.0))
    title_parts = [TOOL]
    if SECONDARY_TOOL is not None:
        title_parts.append(SECONDARY_TOOL)
    title_parts.append("RF")
    fig4.suptitle("Admixture Comparison — " + " vs ".join(title_parts),
                  fontsize=13, fontweight="bold", y=1.04)
    fig4.tight_layout()

fig4.savefig(args.out_dir / "admixture_comparison.png", dpi=args.dpi, bbox_inches="tight")
plt.close(fig4)
print("  saved admixture_comparison.png")


# ── F5: Per-sample L1 distance violin ────────────────────────────────────

print("F5: L1 distance violin...")

l1 = np.abs(popout_merged - rf_prob).sum(axis=1)

# Stratify by RF hard label
groups = list(rf_ref_labels) + ["mixed"]
rf_cm_labels = np.where(rf_max_prob < 0.8, "mixed", rf_hard_calls)
group_data = []
group_labels = []
for g in groups:
    mask = rf_cm_labels == g
    if mask.sum() > 0:
        group_data.append(l1[mask])
        group_labels.append(f"{g}\n(n={mask.sum():,})")

with popout_style():
    fig5, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(group_data, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        color = LABEL_COLORS[groups[i]] if i < len(rf_ref_labels) else LABEL_COLORS["mixed"]
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel(f"L1 distance ({TOOL} vs RF)")
    ax.set_title("Per-Sample L1 Distance by RF Label", fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)
    # Annotate medians
    for i, gd in enumerate(group_data):
        med = np.median(gd)
        ax.text(i + 1, med + 0.02, f"{med:.3f}", ha="center", fontsize=8, color="black")
    fig5.tight_layout()

fig5.savefig(args.out_dir / "l1_distance_violin.png", dpi=args.dpi)
plt.close(fig5)
print("  saved l1_distance_violin.png")


# ── F6: Calibration curves (2 x 3) ──────────────────────────────────────

print("F6: Calibration curves...")

n_bins_cal = 20
bin_edges = np.linspace(0, 1, n_bins_cal + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

with popout_style():
    fig6, axes = plt.subplots(2, 3, figsize=(14, 9))
    for j, name in enumerate(rf_ref_labels):
        ax = axes[j // 3, j % 3]
        x = rf_prob[:, j]
        y = popout_merged[:, j]

        means = np.full(n_bins_cal, np.nan)
        stds = np.full(n_bins_cal, np.nan)
        for b in range(n_bins_cal):
            mask = (x >= bin_edges[b]) & (x < bin_edges[b + 1])
            if mask.sum() >= 10:
                means[b] = y[mask].mean()
                stds[b] = y[mask].std()

        valid = ~np.isnan(means)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.plot(bin_centers[valid], means[valid], "o-",
                color=LABEL_COLORS[name], linewidth=2, markersize=4)
        ax.fill_between(bin_centers[valid],
                        means[valid] - stds[valid],
                        means[valid] + stds[valid],
                        color=LABEL_COLORS[name], alpha=0.15)
        ax.set_xlabel(f"RF {name} probability")
        ax.set_ylabel(f"{TOOL} {name}{MERGED}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(name, fontweight="bold")

    fig6.suptitle(f"Calibration: RF Probability vs {TOOL}{MERGED} Proportion",
                  fontsize=13, fontweight="bold")
    fig6.tight_layout()

fig6.savefig(args.out_dir / "calibration_curves.png", dpi=args.dpi)
plt.close(fig6)
print("  saved calibration_curves.png")


# ── F6b: Per-sub-ancestry calibration (EUR breakdown) ─────────────────────

# For any RF label that maps to multiple popout ancestries, show each
# sub-ancestry's calibration individually to diagnose whether a U-shaped
# merged calibration curve is a real model defect or a merging artifact.

popout_to_rf_raw = {int(k): v for k, v in labels.get("popout_to_rf_label", {}).items()}
_lm_vals = list(popout_to_rf_raw.values())
_popout_names = [
    f"{popout_to_rf_raw[i]}.{i}" if _lm_vals.count(popout_to_rf_raw[i]) > 1 else popout_to_rf_raw[i]
    for i in range(n_popout_anc)
]

for j_ref, ref_label in enumerate(rf_ref_labels):
    sub_indices = rf_to_popout.get(ref_label, [])
    if len(sub_indices) <= 1:
        continue

    n_subs = len(sub_indices)
    ncols = min(n_subs, 3)
    nrows = (n_subs + ncols - 1) // ncols
    print(f"F6b: Per-sub-ancestry calibration for {ref_label} ({n_subs} sub-ancestries)...")

    x = rf_prob[:, j_ref]

    with popout_style():
        fig6b, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                   squeeze=False)
        for si, pa_idx in enumerate(sub_indices):
            ax = axes[si // ncols, si % ncols]
            y = popout_raw[:, pa_idx]
            pa_name = _popout_names[pa_idx]
            mu_frac = popout_raw[:, pa_idx].mean()

            means = np.full(n_bins_cal, np.nan)
            stds = np.full(n_bins_cal, np.nan)
            for b in range(n_bins_cal):
                bmask = (x >= bin_edges[b]) & (x < bin_edges[b + 1])
                if bmask.sum() >= 10:
                    means[b] = y[bmask].mean()
                    stds[b] = y[bmask].std()

            valid = ~np.isnan(means)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3)
            ax.plot(bin_centers[valid], means[valid], "o-",
                    color=LABEL_COLORS[ref_label], linewidth=2, markersize=4)
            ax.fill_between(bin_centers[valid],
                            means[valid] - stds[valid],
                            means[valid] + stds[valid],
                            color=LABEL_COLORS[ref_label], alpha=0.15)
            ax.set_xlabel(f"RF {ref_label} probability")
            ax.set_ylabel(f"{TOOL} {pa_name}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(0.5, y.max() * 1.1, means[valid].max() * 1.3) if valid.any() else 0.5)
            ax.set_title(f"{pa_name}  (mu={mu_frac:.3f})", fontweight="bold")

        # Hide unused axes
        for si in range(n_subs, nrows * ncols):
            axes[si // ncols, si % ncols].set_visible(False)

        fig6b.suptitle(
            f"Sub-ancestry Calibration: RF {ref_label} prob vs individual {TOOL} {ref_label} components",
            fontsize=13, fontweight="bold")
        fig6b.tight_layout()

    fname = f"calibration_{ref_label}_breakdown.png"
    fig6b.savefig(args.out_dir / fname, dpi=args.dpi)
    plt.close(fig6b)
    print(f"  saved {fname}")


# ── F6c: Calibration slope matrix (K × 6) ────────────────────────────────

print("F6c: Calibration slope matrix...")

MIN_BIN_N = 100
MIN_POPULATED_BINS = 5

slope_matrix = np.full((n_popout_anc, n_ref), np.nan)
max_cal_matrix = np.full((n_popout_anc, n_ref), np.nan)

for j_ref in range(n_ref):
    x = rf_prob[:, j_ref]
    bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, n_bins_cal - 1)
    for a in range(n_popout_anc):
        y = popout_raw[:, a]
        bx, by = [], []
        max_val = 0.0
        for b in range(n_bins_cal):
            mask = bin_idx == b
            if mask.sum() < MIN_BIN_N:
                continue
            m = y[mask].mean()
            bx.append(bin_centers[b])
            by.append(m)
            if m > max_val:
                max_val = m
        max_cal_matrix[a, j_ref] = max_val
        if len(bx) < MIN_POPULATED_BINS:
            continue
        slope, _ = np.polyfit(bx, by, 1)
        slope_matrix[a, j_ref] = slope

# Save TSV
with open(args.out_dir / "calibration_slope_matrix.tsv", "w") as f:
    f.write("ancestry\t" + "\t".join(f"{r}_slope" for r in rf_ref_labels)
            + "\t" + "\t".join(f"{r}_max" for r in rf_ref_labels) + "\n")
    for a in range(n_popout_anc):
        slopes = "\t".join(f"{slope_matrix[a, j]:.4f}" if not np.isnan(slope_matrix[a, j]) else "NA"
                           for j in range(n_ref))
        maxes = "\t".join(f"{max_cal_matrix[a, j]:.4f}" for j in range(n_ref))
        f.write(f"{_popout_names[a]}\t{slopes}\t{maxes}\n")
print("  saved calibration_slope_matrix.tsv")

# Heatmap
with popout_style():
    fig6c, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(14, max(6, n_popout_anc * 0.35 + 1)),
                                        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.4})

    # Slope heatmap
    slope_display = np.where(np.isnan(slope_matrix), 0, slope_matrix)
    vmax_s = max(0.5, np.nanmax(np.abs(slope_matrix)))
    im_s = ax_s.imshow(slope_display, cmap="RdBu_r", vmin=-vmax_s, vmax=vmax_s,
                       aspect="auto", interpolation="nearest")
    ax_s.set_xticks(range(n_ref))
    ax_s.set_xticklabels(rf_ref_labels, fontsize=9, rotation=45, ha="right")
    ax_s.set_yticks(range(n_popout_anc))
    ax_s.set_yticklabels(_popout_names, fontsize=8)
    for a in range(n_popout_anc):
        for j in range(n_ref):
            v = slope_matrix[a, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v) > vmax_s * 0.6 else "black"
            ax_s.text(j, a, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig6c.colorbar(im_s, ax=ax_s, shrink=0.7, label="Slope")
    ax_s.set_title("Calibration Slope", fontweight="bold")

    # Max calibrated value heatmap
    im_m = ax_m.imshow(max_cal_matrix, cmap="YlOrRd", vmin=0,
                       vmax=max(0.3, np.nanmax(max_cal_matrix)),
                       aspect="auto", interpolation="nearest")
    ax_m.set_xticks(range(n_ref))
    ax_m.set_xticklabels(rf_ref_labels, fontsize=9, rotation=45, ha="right")
    ax_m.set_yticks(range(n_popout_anc))
    ax_m.set_yticklabels(_popout_names, fontsize=8)
    for a in range(n_popout_anc):
        for j in range(n_ref):
            v = max_cal_matrix[a, j]
            if np.isnan(v):
                continue
            color = "white" if v > np.nanmax(max_cal_matrix) * 0.6 else "black"
            ax_m.text(j, a, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig6c.colorbar(im_m, ax=ax_m, shrink=0.7, label="Max value")
    ax_m.set_title("Max Calibrated Value", fontweight="bold")

    fig6c.suptitle("Calibration Slope & Max Value (per ancestry × RF label)",
                   fontsize=13, fontweight="bold")
    fig6c.tight_layout()

fig6c.savefig(args.out_dir / "calibration_slope_matrix.png", dpi=args.dpi)
plt.close(fig6c)
print("  saved calibration_slope_matrix.png")

# Console summary
print("\n  Calibration slope matrix (slope | max_cal):")
header = "              " + "  ".join(f"{r:>12}" for r in rf_ref_labels)
print(header)
for a in range(n_popout_anc):
    vals = "  ".join(
        f"{slope_matrix[a,j]:>+5.2f}|{max_cal_matrix[a,j]:>5.2f}"
        if not np.isnan(slope_matrix[a, j]) else "   NA|   NA"
        for j in range(n_ref)
    )
    print(f"  {_popout_names[a]:>12}  {vals}")

# Flag vestigial and straddling ancestries
vestigial = []
straddling = []
for a in range(n_popout_anc):
    max_slope = np.nanmax(slope_matrix[a, :])
    max_cal = np.nanmax(max_cal_matrix[a, :])
    n_above_03 = np.nansum(slope_matrix[a, :] > 0.3)
    if max_slope < 0.15 and max_cal < 0.15:
        vestigial.append(_popout_names[a])
    elif n_above_03 > 1:
        straddling.append(_popout_names[a])

if vestigial:
    print(f"\n  Vestigial (slope < 0.15, max_cal < 0.15): {', '.join(vestigial)}")
if straddling:
    print(f"  Straddling (slope > 0.3 against multiple labels): {', '.join(straddling)}")
if not vestigial and not straddling:
    print(f"\n  No vestigial or straddling ancestries detected.")

# Mass-weighted working-ancestry share: fraction of total mu in ancestries
# with at least one slope >= 0.5
mu_per_anc = popout_raw.mean(axis=0)  # genome-wide mean proportion per ancestry
working_mask = np.array([np.nanmax(slope_matrix[a, :]) >= 0.5
                         for a in range(n_popout_anc)])
working_mu = float(mu_per_anc[working_mask].sum())
working_names = [_popout_names[a] for a in range(n_popout_anc) if working_mask[a]]
print(f"\n  Mass-weighted working share: {working_mu:.3f} "
      f"({len(working_names)}/{n_popout_anc} ancestries: {', '.join(working_names)})")

# Continental coverage: summed mu and slope-weighted mu per RF label
merge_group_stats = labels.get("merge_group_stats", {})
print(f"\n  Continental coverage (summed mu per RF label vs cohort proportion):")
print(f"    {'RF label':>10}  {'Summed mu':>10}  {'Slope-wt mu':>12}")
print(f"    {'─'*10}  {'─'*10}  {'─'*12}")
for j, name in enumerate(rf_ref_labels):
    mg = merge_group_stats.get(name)
    if mg:
        indices = mg["indices"]
        s_mu = mg["summed_mu"]
    else:
        indices = rf_to_popout.get(name, [])
        s_mu = float(mu_per_anc[indices].sum()) if indices else 0.0
    # Slope-weighted mu: weight each component's mu by its slope against this label
    sw_mu = 0.0
    for idx in indices:
        s = slope_matrix[idx, j]
        if not np.isnan(s) and s > 0:
            sw_mu += mu_per_anc[idx] * s
    print(f"    {name:>10}  {s_mu:>10.4f}  {sw_mu:>12.4f}")


# ── F7: Residual violin (popout - RF per label) ────────────────────────

print("F7: Residual violin...")

residuals = popout_merged - rf_prob  # (N, 6)

with popout_style():
    fig7, ax = plt.subplots(figsize=(10, 6))
    residual_data = [residuals[:, j] for j in range(n_ref)]
    parts = ax.violinplot(residual_data, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(LABEL_COLORS[rf_ref_labels[i]])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(1, n_ref + 1))
    ax.set_xticklabels(rf_ref_labels, fontsize=11)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel(f"Residual ({TOOL}{MERGED} − RF)")
    ax.set_title("Residual Distribution per Continental Label", fontweight="bold")
    # Annotate mean and std
    for j in range(n_ref):
        mu = residuals[:, j].mean()
        std = residuals[:, j].std()
        ax.text(j + 1, ax.get_ylim()[1] * 0.9,
                f"μ={mu:+.4f}\nσ={std:.4f}",
                ha="center", fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    fig7.tight_layout()

fig7.savefig(args.out_dir / "residual_violin.png", dpi=args.dpi)
plt.close(fig7)
print("  saved residual_violin.png")


# ── F8: Concordance vs RF confidence ────────────────────────────────────

print("F8: Concordance vs confidence...")

n_bins_conf = 20
conf_edges = np.linspace(0, 1, n_bins_conf + 1)
conf_centers = (conf_edges[:-1] + conf_edges[1:]) / 2

concordance_rate = np.full(n_bins_conf, np.nan)
bin_counts = np.zeros(n_bins_conf, dtype=int)

match = (popout_hard == rf_hard_calls)

for b in range(n_bins_conf):
    mask = (rf_max_prob >= conf_edges[b]) & (rf_max_prob < conf_edges[b + 1])
    bin_counts[b] = mask.sum()
    if mask.sum() >= 10:
        concordance_rate[b] = match[mask].mean()

with popout_style():
    fig8, ax1 = plt.subplots(figsize=(10, 6))
    valid = ~np.isnan(concordance_rate)

    # Concordance line
    ax1.plot(conf_centers[valid], concordance_rate[valid] * 100, "o-",
             color="#4477AA", linewidth=2, markersize=5, label="Concordance rate")
    ax1.set_xlabel("RF max probability (confidence)")
    ax1.set_ylabel("Hard-label concordance (%)", color="#4477AA")
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, 1)
    ax1.tick_params(axis="y", labelcolor="#4477AA")

    # Sample count bars on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(conf_centers, bin_counts, width=1.0 / n_bins_conf * 0.8,
            alpha=0.2, color="#EE6677", label="Sample count")
    ax2.set_ylabel("Sample count per bin", color="#EE6677")
    ax2.tick_params(axis="y", labelcolor="#EE6677")

    # Overall concordance
    overall = match.mean() * 100
    ax1.axhline(overall, color="#4477AA", linewidth=1, linestyle="--", alpha=0.5)
    ax1.text(0.02, overall + 2, f"Overall: {overall:.1f}%",
             fontsize=9, color="#4477AA")

    ax1.set_title("Hard-Label Concordance vs RF Confidence", fontweight="bold")
    fig8.tight_layout()

fig8.savefig(args.out_dir / "concordance_vs_confidence.png", dpi=args.dpi)
plt.close(fig8)
print("  saved concordance_vs_confidence.png")


# ── F9: Entropy scatter (popout vs RF admixture level) ──────────────────

print("F9: Entropy scatter...")

def shannon_entropy(p):
    """Shannon entropy in bits, row-wise. 0*log(0) = 0."""
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=1)

h_popout = shannon_entropy(popout_merged)
h_rf = shannon_entropy(rf_prob)

with popout_style():
    fig9, ax = plt.subplots(figsize=(8, 8))
    max_h = np.log2(n_ref)  # max entropy for uniform 6-way
    hb = ax.hexbin(h_rf, h_popout, gridsize=80, mincnt=1, norm=LogNorm(),
                   cmap="viridis", extent=[0, max_h, 0, max_h])
    ax.plot([0, max_h], [0, max_h], "r--", linewidth=1, alpha=0.7)
    ax.set_xlabel("RF Shannon entropy (bits)")
    ax.set_ylabel(f"{TOOL} Shannon entropy (bits)")
    ax.set_xlim(0, max_h)
    ax.set_ylim(0, max_h)
    ax.set_aspect("equal")
    fig9.colorbar(hb, ax=ax, shrink=0.8, label="Count")
    ax.set_title(f"Admixture Level: {TOOL} vs RF (entropy)", fontweight="bold")

    # Annotate r
    r = np.corrcoef(h_rf, h_popout)[0, 1]
    ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=11,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig9.tight_layout()

fig9.savefig(args.out_dir / "entropy_scatter.png", dpi=args.dpi)
plt.close(fig9)
print("  saved entropy_scatter.png")


# ── Summary metrics ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CONCORDANCE SUMMARY")
print("=" * 60)

# ── Primary: Per-RF-label soft Pearson r (merged components) ──

merge_group_stats = labels.get("merge_group_stats", {})

print(f"\n── Primary: Soft Correlation (per RF label, merged components) ──")
print(f"\n  {'RF label':>10}  {'Merged r':>10}  {'Summed mu':>10}  {'Mean L1':>10}  {'Mean residual':>14}  Components")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*30}")

coverage_07 = 0
for j, name in enumerate(rf_ref_labels):
    mg = merge_group_stats.get(name)
    if mg:
        merged_r = mg["merged_r"]
        summed_mu = mg["summed_mu"]
        comp_str = ", ".join(mg["names"])
    else:
        merged_r = float(np.corrcoef(popout_merged[:, j], rf_prob[:, j])[0, 1])
        summed_mu = float(popout_merged[:, j].mean())
        comp_str = "(merged)"
    ml1 = np.abs(residuals[:, j]).mean()
    mr = residuals[:, j].mean()
    if merged_r > 0.7:
        coverage_07 += 1
    print(f"  {name:>10}  {merged_r:>+10.4f}  {summed_mu:>10.4f}  {ml1:>10.4f}  {mr:>+14.4f}  {comp_str}")

print(f"\n  Sub-continental coverage: {coverage_07}/{n_ref} RF labels "
      f"have merged r > 0.7")

# ── Supplementary: Merged concordance ──

print(f"\n── Supplementary: Merged Concordance ──")

print(f"\n  Overall hard-label concordance: {match.mean()*100:.1f}%")
print(f"  Mean per-sample L1 distance:   {l1.mean():.4f}")
print(f"  Median per-sample L1 distance: {np.median(l1):.4f}")

print(f"\n  {'RF label':>10}  {'n':>8}  {'Concordance':>12}")
print(f"  {'─'*10}  {'─'*8}  {'─'*12}")
for name in rf_ref_labels:
    mask = rf_hard_calls == name
    if mask.sum() > 0:
        conc = (popout_hard[mask] == rf_hard_calls[mask]).mean() * 100
        print(f"  {name:>10}  {mask.sum():>8,}  {conc:>11.1f}%")
mask = rf_max_prob < 0.8
if mask.sum() > 0:
    conc = (popout_hard[mask] == rf_hard_calls[mask]).mean() * 100
    print(f"  {'mixed':>10}  {mask.sum():>8,}  {conc:>11.1f}%")

print(f"\nAll outputs saved to {args.out_dir}/")
print("Done.")
