#!/usr/bin/env python3
"""popout DX — hard-call pairwise confusion matrices.

For each requested comparison tool, project both popout and the other
tool to the canonical RF basis, argmax each per sample, and write a
``popout_label`` × ``other_label`` confusion-matrix TSV with totals.

The argmax is taken on the RF-basis q (post-projection) for fairness:
popout's K_popout components get summed into RF buckets before argmax,
so a popout sample that's 0.6 in component_0 (afr) + 0.3 in component_3
(afr) is called "afr" (sum 0.9) rather than tied to its largest single
component.

Inputs are file paths plus an optional rye/rf and the labels.json for
popout/flare. The cluster's roster is the sample-id list of the FLARE
per-cluster ``global.tsv``; if FLARE is absent, the roster comes from
``--popout-global`` directly (and the orchestrator filters in-process).
"""

from __future__ import annotations

import argparse
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
    project_to_rf_basis,
)


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"dx_pairwise_hard: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def confusion_matrix(
    popout_call: np.ndarray, other_call: np.ndarray, label_names: tuple[str, ...],
) -> np.ndarray:
    """Return ``confusion[i, j] = count(popout==i, other==j)`` with row/col
    indices keyed to ``label_names``.
    """
    K = len(label_names)
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    cm = np.zeros((K, K), dtype=np.int64)
    pop_idx = np.array([name_to_idx[n] for n in popout_call], dtype=np.int64)
    oth_idx = np.array([name_to_idx[n] for n in other_call], dtype=np.int64)
    for p, o in zip(pop_idx, oth_idx):
        cm[p, o] += 1
    return cm


def write_confusion_tsv(cm: np.ndarray, label_names: tuple[str, ...], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    K = len(label_names)
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    grand = int(cm.sum())
    with open(out_path, "w") as f:
        f.write("popout_label\t" + "\t".join(label_names) + "\ttotal\n")
        for i, name in enumerate(label_names):
            row = [str(int(cm[i, j])) for j in range(K)]
            f.write(f"{name}\t" + "\t".join(row) + f"\t{int(row_totals[i])}\n")
        f.write("total\t" + "\t".join(str(int(col_totals[j])) for j in range(K)) + f"\t{grand}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--popout-global", required=True, type=Path)
    ap.add_argument("--popout-labels", required=True, type=Path,
                    help="popout-side labels.json (rf_to_popout_components)")
    ap.add_argument("--flare-global", type=Path, default=None,
                    help="popout-format FLARE global.tsv for this cluster (defines roster)")
    ap.add_argument("--flare-labels", type=Path, default=None,
                    help="FLARE-side labels.json (per-cluster, from cohort bundle)")
    ap.add_argument("--rye-q", type=Path, default=None)
    ap.add_argument("--rf", type=Path, default=None,
                    help="RF ancestry TSV (research_id, ancestry_pred, probabilities)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="emits popout_vs_<tool>.confusion.tsv files here")
    args = ap.parse_args()

    try:
        # Roster: FLARE's per-cluster global.tsv if available; else popout's.
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
        popout_argmax = popout_rf.argmax(axis=1)
        popout_call = np.array([RF_LABELS_CANONICAL[i] for i in popout_argmax], dtype=object)

        if args.flare_global is not None:
            if args.flare_labels is None:
                die("--flare-global supplied without --flare-labels; cannot project FLARE")
            flare_labels = load_labels(args.flare_labels)
            flare_rf = project_to_rf_basis(flare_q, "flare", flare_labels)
            flare_call = np.array(
                [RF_LABELS_CANONICAL[i] for i in flare_rf.argmax(axis=1)], dtype=object
            )
            cm = confusion_matrix(popout_call, flare_call, RF_LABELS_CANONICAL)
            write_confusion_tsv(cm, RF_LABELS_CANONICAL, args.out_dir / "popout_vs_flare.confusion.tsv")

        if args.rye_q is not None:
            rye_q = load_rye_for_roster(args.rye_q, roster)
            rye_rf = project_to_rf_basis(rye_q, "rye")
            rye_call = np.array(
                [RF_LABELS_CANONICAL[i] for i in rye_rf.argmax(axis=1)], dtype=object
            )
            cm = confusion_matrix(popout_call, rye_call, RF_LABELS_CANONICAL)
            write_confusion_tsv(cm, RF_LABELS_CANONICAL, args.out_dir / "popout_vs_rye.confusion.tsv")

        if args.rf is not None:
            _, rf_hard = load_rf_for_roster(args.rf, roster)
            # RF's hard call is the canonical label; use directly. Out-of-vocab
            # labels (e.g. samples called "mixed" by RF heuristics upstream)
            # would silently zero; assert they're in the canonical set.
            bad = sorted(set(rf_hard) - set(RF_LABELS_CANONICAL))
            if bad:
                die(f"RF hard calls contained out-of-vocab labels: {bad}")
            cm = confusion_matrix(popout_call, rf_hard, RF_LABELS_CANONICAL)
            write_confusion_tsv(cm, RF_LABELS_CANONICAL, args.out_dir / "popout_vs_rf.confusion.tsv")

    except LoaderError as e:
        die(str(e))


if __name__ == "__main__":
    main()
