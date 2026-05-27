#!/usr/bin/env python3
"""Layer 1: Structural sanity checks on recursive-seeded LAI output.

Checks:
  1.1  Global proportions vs model mu (decode/model agreement)
  1.2  Tract length distribution per ancestry (exponential overlay)
  1.3  Switch rate per haplotype (bimodal expected)
  1.4  Posterior concentration (if available)

Usage:
    python validate_structural.py --prefix data/recur_v2/aou_v9_hmm \
        --out-dir diagnostics/validation/recur_v2
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "popout"))

from popout.viz._loaders import read_global_tsv, read_labels_json, read_model_text, read_summary, read_tracts
from popout.viz.tracts import plot_tract_lengths


def _ancestry_name(idx: int, labels: dict | None) -> str:
    """Resolve an ancestry index to a display name using labels.json."""
    if not labels:
        return f"ancestry_{idx}"
    raw = {int(k): v for k, v in labels.get("popout_to_rf_label", {}).items()}
    if not raw:
        return f"ancestry_{idx}"
    counts: dict[str, int] = {}
    for v in raw.values():
        counts[v] = counts.get(v, 0) + 1
    base = raw.get(idx, f"ancestry_{idx}")
    if base.startswith("ancestry_"):
        return base
    return f"{base}.{idx}" if counts.get(base, 0) > 1 else base


MU_DIFF_THRESHOLD = 0.01


def check_mu_agreement(prefix: Path, out_dir: Path, *, labels: dict | None = None) -> bool:
    """1.1: Compare genome-wide global proportions to model mu.

    Emits mu_vs_global_diff.json (schema §1.4) and returns overall pass.
    """
    print("=" * 60)
    print("Check 1.1: Global proportions vs model mu")
    print("=" * 60)

    ga = read_global_tsv(prefix.with_name(prefix.name + ".global.tsv"))
    global_mu = ga.proportions.mean(axis=0)

    # Use the .model text file (reflects post-consolidation K) rather than
    # summary.json's final_model.mu (which is pre-consolidation).
    model_path = prefix.with_name(prefix.name + ".model")
    if model_path.exists():
        model_data = read_model_text(model_path)
        model_mu = np.array(model_data["mu"], dtype=np.float32)
    else:
        summary = read_summary(prefix.with_name(prefix.name + ".summary.json"))
        model_mu = np.array(summary["final_model"]["mu"], dtype=np.float32)

    diff = np.abs(global_mu - model_mu)
    max_diff = float(diff.max())
    all_pass = max_diff < MU_DIFF_THRESHOLD

    print(f"\n  {'Ancestry':>10} {'Global':>10} {'Model mu':>10} {'Diff':>10}")
    print(f"  {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")
    per_ancestry = []
    for i in range(len(model_mu)):
        flag = " *" if diff[i] >= MU_DIFF_THRESHOLD else ""
        print(f"  {i:>10} {global_mu[i]:>10.5f} {model_mu[i]:>10.5f} {diff[i]:>10.5f}{flag}")
        per_ancestry.append({
            "ancestry": i,
            "name": _ancestry_name(i, labels),
            "global_mu": float(global_mu[i]),
            "model_mu": float(model_mu[i]),
            "abs_diff": float(diff[i]),
            "pass": bool(diff[i] < MU_DIFF_THRESHOLD),
        })

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  Max diff: {max_diff:.5f}  [{status}]")

    json_path = out_dir / "mu_vs_global_diff.json"
    json_path.write_text(json.dumps({
        "max_abs_diff": max_diff,
        "threshold": MU_DIFF_THRESHOLD,
        "overall_pass": bool(all_pass),
        "per_ancestry": per_ancestry,
    }, indent=2))
    print(f"  wrote {json_path}")
    return all_pass


def check_tract_lengths(prefix: Path, out_dir: Path, *, labels=None) -> None:
    """1.2: Tract length distribution with exponential overlay.

    Emits tract_length_summary.json (schema §1.9) and the PNG.
    """
    print("\n" + "=" * 60)
    print("Check 1.2: Tract length distribution per ancestry")
    print("=" * 60)

    fig = plot_tract_lengths(prefix, log_scale=True, show_theoretical=True, labels=labels)
    save_path = out_dir / "tract_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")

    # Per-ancestry length stats (streaming).
    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")
    lengths_by_anc: dict[int, list[int]] = defaultdict(list)
    total_tracts = 0
    for t in read_tracts(tracts_path):
        # FLARE tract is inclusive-inclusive on bp.
        lengths_by_anc[t.ancestry].append(t.end_bp - t.start_bp + 1)
        total_tracts += 1

    # Pull model T (gen_since_admix) for the implied_T comparison.
    model_path = prefix.with_name(prefix.name + ".model")
    model_T = None
    if model_path.exists():
        try:
            md = read_model_text(model_path)
            model_T = float(md.get("gen_since_admix") or md.get("T") or 0.0) or None
        except Exception:
            model_T = None
    if model_T is None:
        summary_path = prefix.with_name(prefix.name + ".summary.json")
        if summary_path.exists():
            try:
                s = read_summary(summary_path)
                model_T = float(s.get("final_model", {}).get("T") or 0.0) or None
            except Exception:
                model_T = None

    per_ancestry = []
    for anc in sorted(lengths_by_anc.keys()):
        lengths_bp = np.array(lengths_by_anc[anc], dtype=np.float64)
        lengths_Mb = lengths_bp / 1e6
        n = int(lengths_Mb.size)
        mean_Mb = float(lengths_Mb.mean()) if n else 0.0
        median_Mb = float(np.median(lengths_Mb)) if n else 0.0
        # Exponential MLE rate in 1/Mb. Implied T in generations assumes
        # 1 cM/Mb (so 1 Morgan = 100 Mb). Mean Morgan length = mean_Mb/100;
        # under a Wright-Fisher admixture model with K ancestries, mean
        # tract length in Morgans = 1 / (K * T) → T = 100 / (K * mean_Mb).
        if n >= 100 and mean_Mb > 0:
            exp_fit_rate = float(1.0 / mean_Mb)
            implied_T = 100.0 / (len(lengths_by_anc) * mean_Mb)
        else:
            exp_fit_rate = None
            implied_T = None
        per_ancestry.append({
            "ancestry": int(anc),
            "name": _ancestry_name(int(anc), labels),
            "n_tracts": n,
            "mean_Mb": mean_Mb,
            "median_Mb": median_Mb,
            "exp_fit_rate": exp_fit_rate,
            "implied_T_gen": implied_T,
            "model_T_gen": model_T,
        })
        print(f"  ancestry {anc} ({per_ancestry[-1]['name']}): "
              f"n_tracts={n:,} mean_Mb={mean_Mb:.3f} median_Mb={median_Mb:.3f} "
              f"rate(/Mb)={exp_fit_rate if exp_fit_rate is None else f'{exp_fit_rate:.4f}'}")

    note = "stats computed per-ancestry; exp_fit_rate is exponential MLE 1/mean_Mb; "
    note += "implied_T_gen assumes 1 cM/Mb. exp_fit_rate is null for ancestries with n_tracts < 100."
    json_path = out_dir / "tract_length_summary.json"
    json_path.write_text(json.dumps({
        "n_tracts_total": total_tracts,
        "per_ancestry": per_ancestry,
        "note": note,
    }, indent=2))
    print(f"  wrote {json_path}")


def check_switch_rate(prefix: Path, out_dir: Path):
    """1.3: Distribution of ancestry switches per haplotype."""
    print("\n" + "=" * 60)
    print("Check 1.3: Switch rate per haplotype")
    print("=" * 60)

    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")

    print("  Streaming tracts (this may take a minute)...")
    tracts_per_hap: Counter = Counter()
    for t in read_tracts(tracts_path):
        tracts_per_hap[(t.sample, t.haplotype)] += 1

    n_haps = len(tracts_per_hap)
    switches = np.array([count - 1 for count in tracts_per_hap.values()])

    print(f"  Total haplotypes: {n_haps:,}")
    print(f"  Mean switches/hap: {switches.mean():.1f}")
    print(f"  Median switches/hap: {np.median(switches):.0f}")
    print(f"  Min: {switches.min()}, Max: {switches.max()}")

    # Distribution summary
    bins_summary = [0, 3, 10, 20, 50, 100, switches.max() + 1]
    histogram = []
    print(f"\n  {'Range':>12} {'Count':>10} {'Percent':>8}")
    print(f"  {'-'*12:>12} {'-'*10:>10} {'-'*8:>8}")
    for lo, hi in zip(bins_summary[:-1], bins_summary[1:]):
        count = int(((switches >= lo) & (switches < hi)).sum())
        pct = 100.0 * count / n_haps
        label = f"{lo}-{hi-1}" if hi <= switches.max() else f"{lo}+"
        print(f"  {label:>12} {count:>10,} {pct:>7.1f}%")
        histogram.append({"bin_lo": int(lo), "bin_hi": int(hi), "count": count})

    json_path = out_dir / "switch_rate_summary.json"
    json_path.write_text(json.dumps({
        "n_haplotypes": int(n_haps),
        "mean": float(switches.mean()),
        "median": float(np.median(switches)),
        "p99": float(np.percentile(switches, 99)),
        "min": int(switches.min()),
        "max": int(switches.max()),
        "histogram": histogram,
    }, indent=2))
    print(f"  wrote {json_path}")

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(switches, bins=np.arange(0, min(switches.max() + 2, 201)),
            edgecolor="black", linewidth=0.3, color="#4477AA", alpha=0.8)
    ax.set_xlabel("Ancestry switches per haplotype")
    ax.set_ylabel("Count")
    ax.set_title(f"Switch Rate Distribution (n={n_haps:,} haplotypes)")
    ax.axvline(np.median(switches), color="red", linestyle="--", linewidth=1.5,
               label=f"median={np.median(switches):.0f}")
    ax.legend()
    fig.tight_layout()

    save_path = out_dir / "switch_rate_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved {save_path}")

    # Log-scale version for tail visibility
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(switches, bins=np.arange(0, min(switches.max() + 2, 201)),
            edgecolor="black", linewidth=0.3, color="#4477AA", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Ancestry switches per haplotype")
    ax.set_ylabel("Count (log)")
    ax.set_title(f"Switch Rate Distribution — log scale (n={n_haps:,} haplotypes)")
    ax.axvline(np.median(switches), color="red", linestyle="--", linewidth=1.5,
               label=f"median={np.median(switches):.0f}")
    ax.legend()
    fig.tight_layout()

    save_path = out_dir / "switch_rate_distribution_log.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def check_posterior(prefix: Path):
    """1.4: Posterior concentration (if available)."""
    print("\n" + "=" * 60)
    print("Check 1.4: Posterior concentration")
    print("=" * 60)

    tracts_path = prefix.with_name(prefix.name + ".tracts.tsv.gz")
    with open(tracts_path, "rb") as f:
        import gzip
        with gzip.open(f, "rt") as gz:
            header = gz.readline().strip()

    if "mean_posterior" in header:
        print("  Per-tract posteriors available — run popout.viz.posterior.plot_posterior_confidence()")
    else:
        print("  SKIP: Per-tract posteriors not available (no mean_posterior column).")
        print("  Re-run with --probs to enable this check.")

    summary_path = prefix.with_name(prefix.name + ".summary.json")
    summary = read_summary(summary_path)
    mean_conf = summary.get("output", {}).get("mean_posterior_confidence")
    if mean_conf is not None:
        print(f"  Summary-level mean posterior confidence: {mean_conf:.4f}")
    else:
        print("  No summary-level posterior confidence either.")


def main():
    parser = argparse.ArgumentParser(description="Layer 1: Structural sanity checks")
    parser.add_argument("--prefix", type=Path, required=True,
                        help="Output prefix (e.g. data/recur_v2/aou_v9_hmm)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for plots and results")
    parser.add_argument("--labels-json", type=Path, default=None,
                        help="Path to labels.json for ancestry names (optional)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = None
    labels_path = args.labels_json
    if labels_path is None:
        candidate = args.out_dir / "labels.json"
        if candidate.exists():
            labels_path = candidate
            print(f"Auto-discovered labels.json at {labels_path}")
    if labels_path and labels_path.exists():
        labels = read_labels_json(labels_path)

    results = {}

    results["mu_agreement"] = check_mu_agreement(args.prefix, args.out_dir, labels=labels)
    check_tract_lengths(args.prefix, args.out_dir, labels=labels)
    check_switch_rate(args.prefix, args.out_dir)
    check_posterior(args.prefix)

    print("\n" + "=" * 60)
    overall = "PASS" if results["mu_agreement"] else "FAIL"
    print(f"OVERALL: {overall}")
    print("=" * 60)


if __name__ == "__main__":
    main()
