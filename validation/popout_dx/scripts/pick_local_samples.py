#!/usr/bin/env python3
"""Stratified-by-admixture sample picker for popout DX local mode.

Buckets the cluster's samples by their popout-derived dominant ancestry
(or "mixed" when no component reaches ``threshold``) and picks
``per_bucket_n`` samples per non-empty bucket, deterministically.

Inputs
------
``--popout-global``   popout whole-cohort ``.global.tsv``
``--labels``          ``labels.json`` (popout → RF label map)
``--cluster-roster``  one-sample-id-per-line file listing this cluster's samples
``--out``             output path for ``selected_samples.tsv``
``--cluster-id``      string, used to seed the RNG (so two clusters
                      sharing the same global RNG seed still pick
                      different subsets)
``--rng-seed``        int, base RNG seed (default 42)
``--per-bucket-n``    samples per non-empty bucket (default 25)
``--threshold``       max-proportion threshold; below this → "mixed"
                      (default 0.80)

Output
------
TSV with columns: ``sample_id``, ``bucket``, ``popout_dominant_anc``,
``popout_max_prop``. Sorted by ``bucket`` then ``sample_id`` for
diff-ability.

Determinism
-----------
Seed = ``sha256(f"{rng_seed}:{cluster_id}")[:8] → int``. The same
``(rng_seed, cluster_id)`` always produces the same selection
regardless of Python ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# popout is installed in the docker image (`pip install --no-deps`);
# the loaders are pure-numpy.
from popout.viz._loaders import read_global_tsv, read_labels_json


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"pick_local_samples: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def deterministic_seed(rng_seed: int, cluster_id: str) -> int:
    h = hashlib.sha256(f"{rng_seed}:{cluster_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--popout-global", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--cluster-roster", required=True, type=Path,
                    help="one sample_id per line")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--per-bucket-n", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    if not 0.0 < args.threshold <= 1.0:
        die(f"--threshold must be in (0, 1], got {args.threshold}")
    if args.per_bucket_n < 1:
        die(f"--per-bucket-n must be >= 1, got {args.per_bucket_n}")

    roster = [
        line.strip()
        for line in args.cluster_roster.read_text().splitlines()
        if line.strip()
    ]
    if not roster:
        die(f"--cluster-roster {args.cluster_roster} is empty")
    roster_set = set(roster)
    if len(roster_set) != len(roster):
        die(f"--cluster-roster {args.cluster_roster} contains duplicate sample ids")

    global_data = read_global_tsv(args.popout_global)
    sample_names = list(global_data.sample_names)
    proportions = np.asarray(global_data.proportions, dtype=np.float64)
    if proportions.ndim != 2 or proportions.shape[0] != len(sample_names):
        die(
            f"{args.popout_global}: malformed (proportions shape {proportions.shape} vs "
            f"{len(sample_names)} samples)"
        )

    labels = read_labels_json(args.labels)
    popout_to_rf: dict[int, str] = labels.get("popout_to_rf_label", {})
    if not isinstance(popout_to_rf, dict) or not popout_to_rf:
        die(f"{args.labels} missing or empty popout_to_rf_label")
    # Component index → RF label name. Components without a mapping fall
    # back to a synthetic name; that should not happen on a well-formed
    # labels.json.
    K = proportions.shape[1]
    component_name: list[str] = []
    for k in range(K):
        name = popout_to_rf.get(k)
        if name is None:
            die(f"labels.json popout_to_rf_label missing entry for component {k} (K={K})")
        component_name.append(str(name))

    # Subset to the cluster's roster, in roster order. Missing samples are
    # an error — silent drops would mask upstream sample-id drift.
    name_to_idx = {s: i for i, s in enumerate(sample_names)}
    missing = [s for s in roster if s not in name_to_idx]
    if missing:
        die(
            f"{len(missing)} cluster-roster sample(s) absent from popout global.tsv; "
            f"first: {missing[:5]}"
        )
    cluster_idx = np.array([name_to_idx[s] for s in roster], dtype=np.int64)
    cluster_props = proportions[cluster_idx]                  # (n_cluster, K)
    cluster_argmax = np.argmax(cluster_props, axis=1)         # (n_cluster,)
    cluster_max = cluster_props[np.arange(len(roster)), cluster_argmax]
    is_mixed = cluster_max < args.threshold

    buckets: dict[str, list[int]] = {}
    for local_i, sample in enumerate(roster):
        if is_mixed[local_i]:
            bucket = "mixed"
        else:
            bucket = f"high_{component_name[cluster_argmax[local_i]]}"
        buckets.setdefault(bucket, []).append(local_i)

    rng = np.random.default_rng(deterministic_seed(args.rng_seed, args.cluster_id))
    selected: list[tuple[str, str, str, float]] = []
    for bucket in sorted(buckets):
        members = buckets[bucket]
        take = min(args.per_bucket_n, len(members))
        chosen = rng.choice(np.array(members, dtype=np.int64), size=take, replace=False)
        chosen.sort()
        for local_i in chosen:
            local_i = int(local_i)
            selected.append((
                roster[local_i],
                bucket,
                component_name[cluster_argmax[local_i]],
                float(cluster_max[local_i]),
            ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("sample_id\tbucket\tpopout_dominant_anc\tpopout_max_prop\n")
        for row in selected:
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.6f}\n")

    print(
        f"pick_local_samples: wrote {len(selected)} samples from "
        f"{len(buckets)} non-empty bucket(s) to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
