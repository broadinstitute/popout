#!/usr/bin/env python3
"""Compare popout/FLARE global ancestry output to the RF classifier's predictions.

The RF classifier is the preliminary random-forest ancestry tool the
AoU consortium runs as part of v9 (its output lives in
`foxtrot_v4.ancestry_preds.tsv`).
See `diagnostics/GLOSSARY.md` for the canonical vocabulary.

Usage:
    python compare_to_rf.py \\
        --popout-global PATH/aou_v9_hmm.global.tsv \\
        --rf-ancestry PATH/foxtrot_v4.ancestry_preds.tsv \\
        --popout-spectral PATH/aou_v9_hmm.spectral.npz \\
        --out-dir PATH/diagnostics
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Canonical RF reference labels ────────────────────────────────────────
#
# Phase 4 of the label-space retrofit: this is the SP6 superpop space
# (popout.labelspace.registry.SP6). The local name is preserved for the
# rest of this script; downstream code that imports it sees the same
# tuple as before.
from popout.labelspace.registry import SP6 as _SP6
RF_LABEL_ORDER: tuple[str, ...] = _SP6.members


def detect_tool_name(prefix: Path) -> str:
    """Return the canonical tool name for this run (`FLARE` or `popout`).

    Inspects ``{prefix}.summary.json`` for ``config.method``. Popout's
    native runs report ``method="hmm"``; FLARE-derived runs (via
    flare_to_popout_format.py) report ``method="flare"``.
    """
    summary_path = prefix.with_name(prefix.name + ".summary.json")
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        method = s.get("config", {}).get("method", "").lower()
        if method == "flare":
            return "FLARE"
        if method in ("hmm", "popout"):
            return "popout"
    return "popout"


# ── Argument parsing ──────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Compare popout/FLARE output to the RF classifier's per-sample ancestry predictions."
)
parser.add_argument("--popout-global", type=Path, required=True,
                    help="Path to popout/FLARE .global.tsv")
parser.add_argument("--rf-ancestry", type=Path, required=True,
                    help="Path to RF ancestry prediction table (foxtrot_v4.ancestry_preds.tsv)")
parser.add_argument("--popout-spectral", type=Path, default=None,
                    help="Path to popout's .spectral.npz (optional; falls back to RF pca_features)")
parser.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for results")
parser.add_argument(
    "--matching", choices=("postS", "by_name"), default="postS",
    help=(
        "Algorithm that maps tool components → SP6 RF labels. "
        "`postS` (default) is the legacy posterior-correlation + "
        "calibration-slope rule from popout.labelspace.matching; use it "
        "for popout DX where components are unnamed. `by_name` trusts "
        "the input ``global.tsv`` header verbatim and runs "
        "``popout.labelspace.matching.by_name`` against SP6 — use it "
        "for FLARE (v3+) bundles, where the panel-population names from "
        "the ``##ANCESTRY=`` VCF header are the source of truth."
    ),
)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

# Detect canonical tool name (FLARE / popout) from the run's summary.json.
TOOL = detect_tool_name(args.popout_global.with_name(args.popout_global.name.removesuffix(".global.tsv")))
print(f"Detected tool: {TOOL}")


# ── Load {TOOL} global TSV ────────────────────────────────────────────────

print(f"Loading {TOOL} global TSV...")
popout_samples = {}
with open(args.popout_global) as f:
    header = f.readline().strip().split("\t")
    popout_anc_cols = header[1:]  # e.g. ["ancestry_0", "ancestry_1", "ancestry_2"]
    n_popout_anc = len(popout_anc_cols)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        popout_samples[sid] = [float(x) for x in parts[1:]]

print(f"  {len(popout_samples):,} samples, {n_popout_anc} {TOOL} ancestries")


# ── Load RF ancestry table ───────────────────────────────────────────────

print("Loading RF ancestry table...")
rf_data: dict[str, tuple[str, list[float]]] = {}  # research_id -> (hard_label, prob_vector)
rf_raw_lines = 0
parse_errors = 0

with open(args.rf_ancestry) as f:
    rf_header = f.readline().strip().split("\t")
    print(f"  RF columns: {rf_header}")

    rid_col = rf_header.index("research_id")
    pred_col = rf_header.index("ancestry_pred")
    prob_col = rf_header.index("probabilities")

    for line in f:
        rf_raw_lines += 1
        parts = line.strip().split("\t")
        rid = parts[rid_col]
        label = parts[pred_col]
        try:
            probs = ast.literal_eval(parts[prob_col])
        except (ValueError, SyntaxError):
            parse_errors += 1
            continue
        rf_data[rid] = (label, probs)

print(f"  {len(rf_data):,} samples parsed ({parse_errors} parse errors)")
n_rf_labels = len(next(iter(rf_data.values()))[1])
print(f"  Probability vector length: {n_rf_labels}")

# Validate against the canonical RF reference order. We refuse to silently
# accept a different label count (CLAUDE.md: no fallbacks).
if n_rf_labels != len(RF_LABEL_ORDER):
    raise RuntimeError(
        f"RF probability vector has {n_rf_labels} entries but RF_LABEL_ORDER "
        f"declares {len(RF_LABEL_ORDER)} ({RF_LABEL_ORDER}). If the RF model "
        f"changed, update RF_LABEL_ORDER at the top of this file."
    )

rf_ref_labels: tuple[str, ...] = RF_LABEL_ORDER
print(f"  RF reference labels (canonical): {rf_ref_labels}")


# ── Join on sample_id / research_id ───────────────────────────────────────

print("\nJoining tables...")
common_ids = set(popout_samples.keys()) & set(rf_data.keys())
popout_only = set(popout_samples.keys()) - set(rf_data.keys())
rf_only_samples = set(rf_data.keys()) - set(popout_samples.keys())

print(f"  Common IDs: {len(common_ids):,}")
print(f"  {TOOL}-only: {len(popout_only):,}")
print(f"  RF-only: {len(rf_only_samples):,}")

if len(common_ids) == 0:
    print(f"\n*** FATAL: No matching IDs between {TOOL} and RF tables. ***")
    print(f"First 5 {TOOL} IDs:", list(popout_samples.keys())[:5])
    print("First 5 RF IDs:", list(rf_data.keys())[:5])
    sys.exit(1)

if len(common_ids) < len(popout_samples) * 0.5:
    print(f"\n*** WARNING: Only {len(common_ids)/len(popout_samples)*100:.1f}% of {TOOL} samples matched. ***")
    print(f"First 5 {TOOL}-only IDs:", list(popout_only)[:5])
    print("First 5 RF-only IDs:", list(rf_only_samples)[:5])


# ── Sanity-check the canonical RF label order against the data ────────────
#
# We hardcode RF_LABEL_ORDER but still validate against the data: for each
# hard label that *does* appear in the cohort, confirm that the canonical
# index for that label is in fact the argmax of its mean probability vector.
# If a future RF model rearranges columns, this check fires immediately.

print("\nSanity-checking RF_LABEL_ORDER against the data...")
label_to_prob_sums: dict[str, np.ndarray] = {}
label_counts: dict[str, int] = {}
for rid in common_ids:
    label, probs = rf_data[rid]
    label_to_prob_sums.setdefault(label, np.zeros(n_rf_labels))
    label_to_prob_sums[label] += np.array(probs)
    label_counts[label] = label_counts.get(label, 0) + 1

mismatches = []
for label, sums in label_to_prob_sums.items():
    if label not in rf_ref_labels:
        # Hard label outside the canonical set — e.g. the RF added a class.
        # Raise rather than silently skip.
        raise RuntimeError(
            f"RF hard label {label!r} is not in RF_LABEL_ORDER ({rf_ref_labels})."
        )
    expected_idx = rf_ref_labels.index(label)
    mean_prob = sums / label_counts[label]
    peak_idx = int(np.argmax(mean_prob))
    print(f"    {label:>6} (n={label_counts[label]:>7,}): peak at index {peak_idx} "
          f"(canonical={expected_idx}), mean = [{', '.join(f'{v:.3f}' for v in mean_prob)}]")
    if peak_idx != expected_idx:
        mismatches.append((label, expected_idx, peak_idx))

if mismatches:
    msg = "; ".join(f"{lbl}: canonical={c}, observed peak={p}" for lbl, c, p in mismatches)
    raise RuntimeError(
        f"RF probability index order disagrees with RF_LABEL_ORDER: {msg}. "
        f"Update RF_LABEL_ORDER at the top of compare_to_rf.py."
    )


# ── Build aligned arrays ──────────────────────────────────────────────────

print("\nBuilding aligned arrays...")
common_list = sorted(common_ids)
n = len(common_list)

popout_mat = np.zeros((n, n_popout_anc), dtype=np.float32)
rf_prob_matrix = np.zeros((n, n_rf_labels), dtype=np.float32)
rf_hard_calls_list: list[str] = []

for i, rid in enumerate(common_list):
    popout_mat[i] = popout_samples[rid]
    label, probs = rf_data[rid]
    rf_prob_matrix[i] = probs
    rf_hard_calls_list.append(label)

rf_hard_calls = np.array(rf_hard_calls_list)
print(f"  Aligned {n:,} samples")


# ── Pearson correlation between popout components and RF prob columns ────
#
# Always computed: feeds the soft_correlation.tsv diagnostic regardless
# of which matching algorithm is used to label components.

corr = np.zeros((n_popout_anc, n_rf_labels), dtype=np.float64)
for pa in range(n_popout_anc):
    for ri in range(n_rf_labels):
        a = popout_mat[:, pa]
        b = rf_prob_matrix[:, ri]
        if a.std() < 1e-12 or b.std() < 1e-12:
            corr[pa, ri] = 0.0
        else:
            corr[pa, ri] = float(np.corrcoef(a, b)[0, 1])


# ── Build the Assignment (component → SP6 label) ─────────────────────────
#
# Two methods, selectable via --matching:
#
#   - postS (default): popout.labelspace.matching.posterior_slope —
#     posterior-correlation + calibration-slope override. Use when the
#     tool's components are unnamed (popout DX). Adds correlations /
#     slope_matrix / max_cal_matrix / overrides to ``diagnostics``.
#
#   - by_name: popout.labelspace.matching.by_name — exact-name match
#     against SP6. Use for FLARE v3+ where the input ``global.tsv``
#     header already carries the FLARE panel-population names from the
#     ``##ANCESTRY=`` VCF header (see Phase 6 of the label-space
#     retrofit and validation/SCHEMA.md).
from popout.labelspace import get as _ls_get
from popout.labelspace.matching import (
    by_name as _by_name,
    posterior_slope as _posterior_slope,
)
from popout.labelspace.naming import ordered_subcomponent_names as _osc_names

if args.matching == "postS":
    _assignment = _posterior_slope(
        popout_mat, rf_prob_matrix, _ls_get("SP6"),
        source={"tool": TOOL},
    )
    slope_matrix = np.array(
        [[np.nan if v is None else v for v in row]
         for row in _assignment.diagnostics["slope_matrix"]]
    )
    max_cal_matrix = np.array(
        [[np.nan if v is None else v for v in row]
         for row in _assignment.diagnostics["max_cal_matrix"]]
    )
    overrides = [
        (o["component"], o["from_label"], o["to_label"],
         o["from_slope"], o["to_slope"])
        for o in _assignment.diagnostics["overrides"]
    ]
    popout_to_rf_label = dict(_assignment.component_to_label)
    popout_names = _osc_names(
        popout_to_rf_label,
        correlations=corr.tolist(),
        target_members=rf_ref_labels,
    )

    if overrides:
        print(f"\nSlope-based label overrides:")
        for pa, old, new, old_slope, new_slope in overrides:
            print(f"  ancestry {pa}: {old} (slope={old_slope:+.3f}) → "
                  f"{new} (slope={new_slope:+.3f})")
elif args.matching == "by_name":
    # Trust the input header verbatim. ``popout_anc_cols`` was read off
    # the popout/FLARE global.tsv header — for FLARE v3+ these are the
    # panel-population names (afr/amr/eas/eur/sas) declared in the VCF.
    _assignment = _by_name(
        popout_anc_cols, _ls_get("SP6"),
        source={"tool": TOOL},
    )
    _assignment.diagnostics["correlations"] = corr.tolist()
    _assignment.diagnostics["matching"] = "by_name"
    slope_matrix = None
    max_cal_matrix = None
    popout_to_rf_label = dict(_assignment.component_to_label)
    popout_names = list(popout_anc_cols)
else:
    raise RuntimeError(f"unknown --matching {args.matching!r}")
print(f"\n{TOOL} ancestry names: {popout_names}")

# RF label → list of popout ancestry indices (sorted by r against that label)
rf_to_popout_components: dict[str, list[int]] = {}
for pa, name in popout_to_rf_label.items():
    rf_to_popout_components.setdefault(name, []).append(pa)
for name, indices in rf_to_popout_components.items():
    ref_col = rf_ref_labels.index(name)
    indices.sort(key=lambda i: -corr[i, ref_col])

# Per-RF-label merged stats.
merged_stats: dict[str, dict] = {}
for ri, rf_name in enumerate(rf_ref_labels):
    indices = rf_to_popout_components.get(rf_name, [])
    summed = popout_mat[:, indices].sum(axis=1) if indices else np.zeros(n)
    merged_r = float(np.corrcoef(summed, rf_prob_matrix[:, ri])[0, 1]) if indices else 0.0
    summed_mu = float(summed.mean())
    merged_stats[rf_name] = {
        "indices": indices,
        "names": [popout_names[i] for i in indices],
        "merged_r": merged_r,
        "summed_mu": summed_mu,
    }


# ── 1. Confusion matrix ──────────────────────────────────────────────────

print("\n=== 1. Confusion Matrix ===")

# Tool hard call = argmax of posteriors.
popout_hard = np.argmax(popout_mat, axis=1)

# Phase 4 of the label-space retrofit: "mixed" is no longer a 7th category
# baked into the confusion matrix. Low-confidence samples (RF max prob < 0.8)
# are counted separately and reported as a footer, so the CM rows stay in
# the canonical SP6 space.
rf_max_prob = rf_prob_matrix.max(axis=1)
_LOW_CONF_THRESHOLD = 0.8
low_conf_mask = rf_max_prob < _LOW_CONF_THRESHOLD
n_low_confidence = int(low_conf_mask.sum())

rf_cm_labels = sorted(set(rf_hard_calls))
popout_cm_labels = list(range(n_popout_anc))

cm = np.zeros((len(rf_cm_labels), len(popout_cm_labels)), dtype=int)
for i in range(n):
    row = rf_cm_labels.index(rf_hard_calls[i])
    col = popout_hard[i]
    cm[row, col] += 1

col_header = "\t".join(popout_names)
print(f"{'rf_label':>12}\t{col_header}\ttotal")
cm_lines = [f"rf_label\t{col_header}\ttotal"]
for r, label in enumerate(rf_cm_labels):
    vals = "\t".join(str(cm[r, c]) for c in range(len(popout_cm_labels)))
    total = cm[r].sum()
    print(f"{label:>12}\t{vals}\t{total}")
    cm_lines.append(f"{label}\t{vals}\t{total}")

col_totals = cm.sum(axis=0)
vals = "\t".join(str(v) for v in col_totals)
print(f"{'total':>12}\t{vals}\t{col_totals.sum()}")
cm_lines.append(f"total\t{vals}\t{col_totals.sum()}")

print(f"\nLow-confidence samples (RF max prob < {_LOW_CONF_THRESHOLD}): "
      f"{n_low_confidence:,} ({100 * n_low_confidence / n:.2f}%)")
cm_lines.append(f"# n_low_confidence\t{n_low_confidence}\t{_LOW_CONF_THRESHOLD}")

with open(args.out_dir / "confusion_matrix.tsv", "w") as f:
    f.write("\n".join(cm_lines) + "\n")


# ── 2. Per-popout-ancestry composition ────────────────────────────────────

print(f"\n=== 2. Per-{TOOL}-Ancestry Composition ===")

comp_lines = ["popout_ancestry\tn_samples\t" + "\t".join(rf_ref_labels)]
for a in range(n_popout_anc):
    mask = popout_mat[:, a] > 0.8
    n_high = mask.sum()
    if n_high == 0:
        print(f"\n{popout_names[a]}: 0 samples with posterior > 0.8")
        comp_lines.append(f"{popout_names[a]}\t0\t" + "\t".join(["NA"] * n_rf_labels))
        continue

    mean_rf = rf_prob_matrix[mask].mean(axis=0)
    print(f"\n{popout_names[a]} (n={n_high:,}, mean RF probabilities):")
    vals_sorted = sorted(zip(rf_ref_labels, mean_rf), key=lambda x: -x[1])
    for lbl, val in vals_sorted:
        print(f"  {lbl:>6}: {val:.4f}")
    comp_lines.append(f"{popout_names[a]}\t{n_high}\t" + "\t".join(f"{mean_rf[i]:.4f}" for i in range(n_rf_labels)))

with open(args.out_dir / "popout_composition.tsv", "w") as f:
    f.write("\n".join(comp_lines) + "\n")


# ── 3. Soft-correlation table ─────────────────────────────────────────────

print("\n=== 3. Soft Correlation (Pearson r) ===")

header = "           " + "  ".join(f"{lbl:>6}" for lbl in rf_ref_labels)
print(header)
corr_lines = ["popout_ancestry\t" + "\t".join(rf_ref_labels)]
for pa in range(n_popout_anc):
    vals = "  ".join(f"{corr[pa, ri]:>+6.3f}" for ri in range(n_rf_labels))
    print(f"{popout_names[pa]:>11}  {vals}")
    corr_lines.append(f"{popout_names[pa]}\t" + "\t".join(f"{corr[pa, ri]:+.4f}" for ri in range(n_rf_labels)))

with open(args.out_dir / "soft_correlation.tsv", "w") as f:
    f.write("\n".join(corr_lines) + "\n")


# ── 4. PCA overlay ───────────────────────────────────────────────────────

print("\n=== 4. PCA Overlay ===")

sample_pca = None
pca_source = None

if args.popout_spectral is not None and args.popout_spectral.exists():
    spec = np.load(args.popout_spectral)
    pca_proj = spec["pca_proj"]  # (H, n_pc) — haplotypes
    sample_pca_full = (pca_proj[0::2] + pca_proj[1::2]) / 2  # avg both haps → (n_samples, n_pc)
    popout_id_order = list(popout_samples.keys())
    id_to_popout_idx = {sid: i for i, sid in enumerate(popout_id_order)}
    sample_pca = np.zeros((n, sample_pca_full.shape[1]))
    for i, rid in enumerate(common_list):
        pidx = id_to_popout_idx.get(rid)
        if pidx is not None:
            sample_pca[i] = sample_pca_full[pidx]
    pca_source = "popout spectral"
else:
    # Fall back to RF pca_features column.
    common_set = set(common_list)
    pca_by_rid = {}
    with open(args.rf_ancestry) as f:
        rf_header2 = f.readline().strip().split("\t")
        pca_col = rf_header2.index("pca_features") if "pca_features" in rf_header2 else -1
        if pca_col >= 0:
            for line in f:
                parts = line.strip().split("\t")
                rid = parts[rid_col]
                if rid in common_set:
                    try:
                        pca_by_rid[rid] = ast.literal_eval(parts[pca_col])
                    except (ValueError, SyntaxError):
                        pass

    if pca_by_rid:
        n_pc = len(next(iter(pca_by_rid.values())))
        sample_pca = np.zeros((n, n_pc))
        for i, rid in enumerate(common_list):
            if rid in pca_by_rid:
                sample_pca[i] = pca_by_rid[rid]
        pca_source = "RF pca_features"
    else:
        print("  SKIP: No PCA data available (no --popout-spectral and no pca_features in RF table)")

if sample_pca is not None:
    label_set = sorted(set(rf_hard_calls))
    cmap = plt.cm.get_cmap("tab10", max(10, len(label_set)))

    fig, ax = plt.subplots(figsize=(9, 8))
    for li, label in enumerate(label_set):
        indices = np.array([i for i in range(n) if rf_hard_calls[i] == label])
        if len(indices) == 0:
            continue
        ax.scatter(sample_pca[indices, 0], sample_pca[indices, 1],
                   s=0.5, alpha=0.3, c=[cmap(li)], label=f"{label} ({len(indices):,})",
                   rasterized=True)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(markerscale=10, frameon=False, fontsize=8, title="RF hard label")
    ax.set_title(f"PCA colored by RF hard label (source: {pca_source})")
    fig.tight_layout()
    fig.savefig(args.out_dir / "pca_by_rf_label.png", dpi=150)
    plt.close(fig)
    print(f"  saved pca_by_rf_label.png (PCA source: {pca_source})")


# ── 5. Verdict ────────────────────────────────────────────────────────────

print("\n=== 5. Computing Verdict ===")

best_matches = []
for pa in range(n_popout_anc):
    best_ri = int(np.argmax(corr[pa]))
    best_label = rf_ref_labels[best_ri]
    best_r = corr[pa, best_ri]
    absorbed = []
    for ri in range(n_rf_labels):
        if ri != best_ri and corr[pa, ri] > 0.3:
            absorbed.append((rf_ref_labels[ri], corr[pa, ri]))
    best_matches.append((pa, best_label, best_r, absorbed))

n_strong = sum(1 for _, _, r, _ in best_matches if r > 0.7)
n_weak = sum(1 for _, _, r, _ in best_matches if r < 0.3)
any_merging = any(len(absorbed) > 0 for _, _, _, absorbed in best_matches)

rf_coverage = set()
for pa, best_label, best_r, absorbed in best_matches:
    if best_r > 0.7:
        rf_coverage.add(best_label)

if n_strong == n_popout_anc and not any_merging:
    verdict = "clean_match"
    verdict_title = f"Clean K={n_popout_anc} match"
    verdict_text = (
        f"All {n_popout_anc} {TOOL} ancestries correspond cleanly to RF labels "
        f"(all r > 0.5, no merging). RF labels covered: {sorted(rf_coverage)}."
    )
elif n_weak > n_popout_anc * 0.3:
    verdict = "unclear"
    verdict_title = "Unclear match"
    verdict_text = (
        f"{n_weak}/{n_popout_anc} {TOOL} ancestries have best-match r < 0.3 to any "
        f"RF label. These may represent sub-continental structure the RF classifier "
        f"doesn't capture, or noise clusters."
    )
else:
    verdict = "sub_continental"
    verdict_title = f"Sub-continental structure (K={n_popout_anc})"
    merged_details = []
    for pa, best_label, best_r, absorbed in best_matches:
        if absorbed:
            abs_str = " + ".join(f"{lbl}(r={r:.2f})" for lbl, r in absorbed)
            merged_details.append(f"{popout_names[pa]} = {best_label}(r={best_r:.2f}) + {abs_str}")
    verdict_text = (
        f"{TOOL}'s K={n_popout_anc} ancestries provide sub-continental resolution. "
        f"{len(rf_coverage)}/{len(rf_ref_labels)} RF labels covered, "
        f"{n_strong}/{n_popout_anc} {TOOL} ancestries have r > 0.7 to an RF label."
    )
    if merged_details:
        verdict_text += "\n\nMerging detected:\n" + "\n".join(f"  - {d}" for d in merged_details)

print(f"\nRF label ← {TOOL} merge groups:")
for rf_name in rf_ref_labels:
    ms = merged_stats[rf_name]
    names_str = ", ".join(ms["names"])
    print(f"  {rf_name} ← [{names_str}]  "
          f"(merged r={ms['merged_r']:.3f}, summed mu={ms['summed_mu']:.3f})")

print(f"\nVerdict: {verdict_title}")
print(f"  {verdict_text}")


# ── Write SUMMARY.md ──────────────────────────────────────────────────────

cm_md = "| rf_label | " + " | ".join(popout_names) + " | total |\n"
cm_md += "|-----------|" + "|".join("--------:" for _ in popout_cm_labels) + "|------:|\n"
for r, label in enumerate(rf_cm_labels):
    vals = " | ".join(str(cm[r, c]) for c in range(len(popout_cm_labels)))
    cm_md += f"| {label} | {vals} | {cm[r].sum()} |\n"
vals = " | ".join(str(v) for v in col_totals)
cm_md += f"| **total** | {vals} | {col_totals.sum()} |\n"
cm_md += (f"\n*Low-confidence samples (RF max prob &lt; {_LOW_CONF_THRESHOLD}): "
          f"{n_low_confidence:,} ({100 * n_low_confidence / n:.2f}%). "
          f"These are counted in the matrix under their RF hard call; the "
          f"former 'mixed' pseudo-row has been retired.*\n")

comp_md = ""
for a in range(n_popout_anc):
    mask = popout_mat[:, a] > 0.8
    n_high = mask.sum()
    if n_high == 0:
        comp_md += f"\n**{popout_names[a]}:** 0 samples with posterior > 0.8\n"
        continue
    mean_rf = rf_prob_matrix[mask].mean(axis=0)
    comp_md += f"\n**{popout_names[a]}** (n={n_high:,}):\n```\n"
    for lbl, val in sorted(zip(rf_ref_labels, mean_rf), key=lambda x: -x[1]):
        comp_md += f"  {lbl:>6}: {val:.4f}\n"
    comp_md += "```\n"

corr_md = "| | " + " | ".join(rf_ref_labels) + " |\n"
corr_md += "|---|" + "|".join("------:" for _ in rf_ref_labels) + "|\n"
for pa in range(n_popout_anc):
    vals = " | ".join(f"{corr[pa, ri]:+.3f}" for ri in range(n_rf_labels))
    corr_md += f"| {popout_names[pa]} | {vals} |\n"

merge_group_md = f"| RF label | Merged r | Summed mu | {TOOL} components |\n"
merge_group_md += "|-----------|----------:|----------:|-------------------|\n"
coverage_07 = sum(1 for name in rf_ref_labels if merged_stats[name]["merged_r"] > 0.7)
for rf_name in rf_ref_labels:
    ms = merged_stats[rf_name]
    names_str = ", ".join(ms["names"])
    merge_group_md += f"| {rf_name} | {ms['merged_r']:+.4f} | {ms['summed_mu']:.4f} | {names_str} |\n"

summary = f"""# {TOOL} vs RF Classifier Concordance

Compares {TOOL} per-sample global ancestry to the RF preliminary
classifier's per-sample predictions on the same cohort. See
`diagnostics/GLOSSARY.md` for the canonical vocabulary (in particular: the
RF classifier is a random-forest tool applied to the AoU v9 cohort; it is
distinct from FLARE, popout, and ADMIXTURE).

## Data

- {TOOL} samples: {len(popout_samples):,}
- RF samples:     {len(rf_data):,}
- Matched:        {len(common_ids):,}
- {TOOL} ancestries: {n_popout_anc}
- RF reference labels: {list(rf_ref_labels)}

## 1. Soft Correlation (Pearson r) — Primary

### Per-RF-label merge groups

{merge_group_md}
**Sub-continental coverage: {coverage_07}/{n_rf_labels} RF labels have merged r > 0.7**

### Full correlation matrix

{corr_md}

## 2. Verdict: {verdict_title}

{verdict_text}

## 3. Per-{TOOL}-Ancestry Composition

Mean RF soft probability vector over samples with {TOOL} posterior > 0.8:
{comp_md}

## 4. Confusion Matrix — Supplementary

RF hard label (rows) vs {TOOL} hard call (columns). The legacy "mixed" pseudo-row has been retired (Phase 4 of the label-space retrofit); the low-confidence sample count is reported as a footnote instead.

{cm_md}

## 5. PCA Overlay

![PCA by RF label](pca_by_rf_label.png)
"""

(args.out_dir / "SUMMARY.md").write_text(summary)


# ── 6. Write labels.json ──────────────────────────────────────────────────

print("\n=== 6. Writing labels.json (v1+v2 dual-format) ===")

# Phase 5 of the label-space retrofit: writer emits both the legacy
# keys and the new v2 schema. Existing readers keep working; new
# readers consume the v2 block + provenance.tag (the figure shorthand).
from popout.labelspace.naming import name_components as _name_components
from popout.labelspace.shorthand import format as _format_tag

_subcomponent_names = _name_components(
    rf_to_popout_components,
    correlations=corr.tolist(),
    target_members=list(rf_ref_labels),
)
_assignment.subcomponent_names = _subcomponent_names
_assignment.diagnostics["merge_group_stats"] = {
    name: {
        "indices": ms["indices"],
        "names": ms["names"],
        "merged_r": ms["merged_r"],
        "summed_mu": ms["summed_mu"],
    }
    for name, ms in merged_stats.items()
}
_assignment.provenance["produced_by"] = "validation/scripts/compare_to_rf.py"
_assignment.provenance["tag"] = _format_tag(
    _assignment.target_space, [_assignment],
)
labels_path = args.out_dir / "labels.json"
_assignment.dump_v1_compatible(labels_path)
print(f"  saved {labels_path.name}  tag: {_assignment.provenance['tag']}")


print(f"\nAll outputs saved to {args.out_dir}/")
print("Done.")
