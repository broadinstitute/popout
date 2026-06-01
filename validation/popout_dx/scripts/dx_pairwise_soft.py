#!/usr/bin/env python3
"""popout DX — soft-call pairwise concordance metrics.

For each requested comparison tool, project both popout and the other
tool to the canonical RF basis and compute per-RF-label metrics
(``pearson_r``, ``ccc``, ``mae_{mean,median,p95}``, ``jaccard@τ``,
``pass``). Pass is μ-gated on popout's mean (popout is the anchor).

Emits:
  ``popout_vs_<tool>.metrics.tsv``  — one row per RF label per tool pair
  ``per_sample_mae.tsv``            — one row per sample, columns mae_vs_<tool>
                                       (empty when a tool is absent — stable
                                        cohort-collation schema)
  ``summary.json``                  — pair counts (passing / failing / null)
                                       and per-tool mean ccc / mean pearson
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from validation.popout_dx.scripts.dx_loaders import (
    RF_LABELS_CANONICAL,
    LoaderError,
    fmt_tsv_value,
    load_flare_global,
    load_labels,
    load_popout_for_roster,
    load_rf_for_roster,
    load_rye_for_roster,
    per_label_metrics,
    project_to_rf_basis,
)


METRIC_COLUMNS: tuple[str, ...] = (
    "rf_label", "popout_mu", "n_samples_compared",
    "pearson_r", "ccc",
    "mae_mean", "mae_median", "mae_p95",
    "jaccard_0.10", "jaccard_0.25", "jaccard_0.50",
    "pass",
)


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"dx_pairwise_soft: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def compute_pair(popout_rf: np.ndarray, other_rf: np.ndarray) -> list[dict]:
    """Return one metric row per RF label (popout vs other in RF basis)."""
    rows: list[dict] = []
    for j, rf_label in enumerate(RF_LABELS_CANONICAL):
        row = per_label_metrics(popout_rf[:, j], other_rf[:, j])
        row["rf_label"] = rf_label
        rows.append(row)
    return rows


def write_metrics_tsv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\t".join(METRIC_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(fmt_tsv_value(r[c]) for c in METRIC_COLUMNS) + "\n")


def per_sample_mae(popout_rf: np.ndarray, other_rf: np.ndarray) -> np.ndarray:
    """Mean absolute error per sample over the RF-basis columns."""
    return np.abs(popout_rf - other_rf).mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--popout-global", required=True, type=Path)
    ap.add_argument("--popout-labels", required=True, type=Path)
    ap.add_argument("--flare-global", type=Path, default=None)
    ap.add_argument("--flare-labels", type=Path, default=None)
    ap.add_argument("--rye-q", type=Path, default=None)
    ap.add_argument("--rf", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    try:
        if args.flare_global is not None:
            roster, flare_q = load_flare_global(args.flare_global)
        else:
            from popout.viz._loaders import read_global_tsv
            anchor = read_global_tsv(args.popout_global)
            roster = list(anchor.sample_names)
            flare_q = None

        popout_labels = load_labels(args.popout_labels)
        popout_q = load_popout_for_roster(args.popout_global, roster)
        popout_rf = project_to_rf_basis(popout_q, "popout", popout_labels)

        per_sample = {sid: {"mae_vs_flare": "", "mae_vs_rye": "", "mae_vs_rf": ""}
                      for sid in roster}
        summary_pairs: list[dict] = []

        def add_pair(tool: str, other_rf: np.ndarray) -> None:
            rows = compute_pair(popout_rf, other_rf)
            write_metrics_tsv(rows, args.out_dir / f"popout_vs_{tool}.metrics.tsv")
            mae = per_sample_mae(popout_rf, other_rf)
            for sid, v in zip(roster, mae):
                per_sample[sid][f"mae_vs_{tool}"] = f"{v:.6f}"
            for r in rows:
                summary_pairs.append({
                    "tool": tool,
                    "rf_label": r["rf_label"],
                    "popout_mu": r["popout_mu"],
                    "pearson_r": r["pearson_r"],
                    "ccc": r["ccc"],
                    "mae_mean": r["mae_mean"],
                    "pass": r["pass"],
                })

        if args.flare_global is not None:
            if args.flare_labels is None:
                die("--flare-global supplied without --flare-labels")
            flare_labels = load_labels(args.flare_labels)
            flare_rf = project_to_rf_basis(flare_q, "flare", flare_labels)
            add_pair("flare", flare_rf)

        if args.rye_q is not None:
            rye_q = load_rye_for_roster(args.rye_q, roster)
            rye_rf = project_to_rf_basis(rye_q, "rye")
            add_pair("rye", rye_rf)

        if args.rf is not None:
            rf_q, _ = load_rf_for_roster(args.rf, roster)
            rf_rf = project_to_rf_basis(rf_q, "rf")
            add_pair("rf", rf_rf)

        # per_sample_mae.tsv — always emitted; absent-tool columns stay empty.
        per_sample_path = args.out_dir / "per_sample_mae.tsv"
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with open(per_sample_path, "w") as f:
            f.write("sample_id\tmae_vs_flare\tmae_vs_rye\tmae_vs_rf\n")
            for sid in roster:
                row = per_sample[sid]
                f.write(f"{sid}\t{row['mae_vs_flare']}\t{row['mae_vs_rye']}\t{row['mae_vs_rf']}\n")

        # summary.json — pass counts + per-tool means.
        def _safe_mean(xs: list[float]) -> float:
            arr = np.array([x for x in xs if x is not None and not np.isnan(x)], dtype=np.float64)
            return float(arr.mean()) if arr.size else float("nan")

        tools_present = sorted({p["tool"] for p in summary_pairs})
        per_tool_summary: dict[str, dict] = {}
        for tool in tools_present:
            pairs = [p for p in summary_pairs if p["tool"] == tool]
            per_tool_summary[tool] = {
                "n_labels": len(pairs),
                "n_passing": sum(1 for p in pairs if p["pass"] is True),
                "n_failing": sum(1 for p in pairs if p["pass"] is False),
                "n_null":    sum(1 for p in pairs if p["pass"] is None),
                "mean_pearson_r_eligible": _safe_mean(
                    [p["pearson_r"] for p in pairs if p["pass"] is not None]
                ),
                "mean_ccc_eligible": _safe_mean(
                    [p["ccc"] for p in pairs if p["pass"] is not None]
                ),
                "mean_mae_eligible": _safe_mean(
                    [p["mae_mean"] for p in pairs if p["pass"] is not None]
                ),
            }
        n_pass = sum(s["n_passing"] for s in per_tool_summary.values())
        n_fail = sum(s["n_failing"] for s in per_tool_summary.values())
        n_null = sum(s["n_null"] for s in per_tool_summary.values())
        summary = {
            "n_samples": len(roster),
            "pairs": summary_pairs,
            "n_pairs_passing": n_pass,
            "n_pairs_failing": n_fail,
            "n_pairs_null":    n_null,
            "per_tool": per_tool_summary,
        }
        # NaN → null for JSON-safety
        def _clean(o):
            if isinstance(o, float) and np.isnan(o):
                return None
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_clean(v) for v in o]
            return o
        (args.out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")

    except LoaderError as e:
        die(str(e))


if __name__ == "__main__":
    main()
