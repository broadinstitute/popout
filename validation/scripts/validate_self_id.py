#!/usr/bin/env python3
"""§8.2 self-reported ancestry concordance.

Joins FLARE/popout per-sample global ancestry to a self-reported-ancestry
table and reports, per self-ID class, the mean μ per FLARE ancestry. A
clean ancestry call should show its self-ID-matched class dominating the
expected ancestry column.

Self-ID table format: TSV with a header row. First column is sample id
(any of: ``sample_id``, ``research_id``, ``person_id``, ``id``).
Self-ID column is one of: ``self_id``, ``self_reported_ancestry``,
``srace``, ``ancestry``. Other columns are ignored.

Usage:
    python validate_self_id.py \\
        --global-tsv PATH/<prefix>.global.tsv \\
        --self-id-tsv PATH/self_id.tsv \\
        --out-dir PATH/diagnostics \\
        [--labels-json PATH/labels.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "popout"))
from popout.viz._loaders import read_global_tsv


SAMPLE_ID_ALIASES = ("sample_id", "research_id", "person_id", "id")
SELF_ID_ALIASES = ("self_id", "self_reported_ancestry", "srace", "ancestry")


def load_self_id(path: Path) -> dict[str, str]:
    """Return sample_id -> self_id mapping."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        lower = [h.lower() for h in header]
        sid_col = next((i for i, h in enumerate(lower) if h in SAMPLE_ID_ALIASES), None)
        cls_col = next((i for i, h in enumerate(lower) if h in SELF_ID_ALIASES), None)
        if sid_col is None:
            raise RuntimeError(
                f"{path}: no sample-id column found (expected one of {SAMPLE_ID_ALIASES})"
            )
        if cls_col is None:
            raise RuntimeError(
                f"{path}: no self-id column found (expected one of {SELF_ID_ALIASES})"
            )
        out: dict[str, str] = {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(sid_col, cls_col):
                continue
            sid = parts[sid_col].strip()
            cls = parts[cls_col].strip()
            if not sid or not cls:
                continue
            out[sid] = cls
    return out


def _ancestry_name(idx: int, labels: dict | None) -> str:
    if not labels:
        return f"ancestry_{idx}"
    raw = {int(k): v for k, v in labels.get("popout_to_rf_label", {}).items()}
    if not raw:
        return f"ancestry_{idx}"
    counts: dict[str, int] = {}
    for v in raw.values():
        counts[v] = counts.get(v, 0) + 1
    base = raw.get(idx, f"ancestry_{idx}")
    return f"{base}.{idx}" if counts.get(base, 0) > 1 else base


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--global-tsv", type=Path, required=True)
    p.add_argument("--self-id-tsv", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.global_tsv, args.self_id_tsv):
        if not path.exists():
            raise FileNotFoundError(path)

    labels = None
    if args.labels_json is not None and args.labels_json.exists():
        labels = json.loads(args.labels_json.read_text())

    print(f"Loading global TSV from {args.global_tsv}")
    ga = read_global_tsv(args.global_tsv)
    K = ga.n_ancestries
    sample_idx = {sid: i for i, sid in enumerate(ga.sample_names)}
    print(f"  {len(ga.sample_names):,} samples, {K} ancestries")

    print(f"Loading self-ID table from {args.self_id_tsv}")
    self_id = load_self_id(args.self_id_tsv)
    print(f"  {len(self_id):,} self-ID rows")

    # Join.
    by_class: dict[str, list[int]] = defaultdict(list)
    n_joined = 0
    for sid, cls in self_id.items():
        idx = sample_idx.get(sid)
        if idx is None:
            continue
        by_class[cls].append(idx)
        n_joined += 1
    print(f"  joined: {n_joined:,} samples across {len(by_class)} self-ID classes")

    if n_joined == 0:
        raise RuntimeError(
            f"No overlap between global.tsv samples and self-ID table. "
            f"First 5 global IDs: {list(ga.sample_names)[:5]}; "
            f"first 5 self-ID IDs: {list(self_id.keys())[:5]}"
        )

    # ── check.tsv (long form) ──
    out_tsv = args.out_dir / "check.tsv"
    per_class_summary = []
    with open(out_tsv, "w") as f:
        f.write("self_id\tn\tancestry\tname\tmean_mu\n")
        for cls in sorted(by_class.keys()):
            idxs = by_class[cls]
            mean_mu = ga.proportions[idxs].mean(axis=0)
            for a in range(K):
                name = _ancestry_name(a, labels)
                f.write(f"{cls}\t{len(idxs)}\t{a}\t{name}\t{mean_mu[a]:.4f}\n")
            dom = int(np.argmax(mean_mu))
            per_class_summary.append({
                "self_id": cls,
                "n": int(len(idxs)),
                "dominant_ancestry_name": _ancestry_name(dom, labels),
                "dominant_mean_mu": float(mean_mu[dom]),
            })
    print(f"  wrote {out_tsv}")

    # ── summary.json ──
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps({
        "n_samples_joined": int(n_joined),
        "n_self_id_classes": int(len(by_class)),
        "per_class": per_class_summary,
    }, indent=2))
    print(f"  wrote {out_json}")
    for row in per_class_summary:
        print(f"    {row['self_id']:>10} (n={row['n']:>5}): "
              f"dominant {row['dominant_ancestry_name']} "
              f"({row['dominant_mean_mu']:.3f})")


if __name__ == "__main__":
    main()
