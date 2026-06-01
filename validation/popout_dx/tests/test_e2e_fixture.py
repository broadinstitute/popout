#!/usr/bin/env python3
"""End-to-end popout DX smoke test on a synthetic fixture.

Builds an in-memory 2-cluster × chr1 fixture (60 samples, FLARE +
popout + Rye + RF), runs the full pipeline:

  discover_runs → run_dx_cluster (×2) → collate_dx

…and asserts the cohort bundle is schema-valid with the expected
files + headlines.

Run::

    PYTHONPATH=<gpulai>:<popout> \\
      python -m validation.popout_dx.tests.test_e2e_fixture [--mode global|global_local] [--keep WORKDIR]

By default the workdir is a ``tempfile.TemporaryDirectory`` that is
removed on exit; pass ``--keep`` with an explicit path to inspect the
artifacts after the run.

Local mode is a TODO: it requires a synthetic FLARE .anc.vcf.gz on
disk (the pipeline calls ``bcftools query`` which needs a real VCF
file). The fixture builder below is structured to accept that
addition later.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
GPULAI_ROOT = Path(__file__).resolve().parents[3]
POPOUT_ROOT = Path("/Users/ghall/code/work/broad/popout")
RF_LABELS = ["afr", "amr", "eas", "eur", "mid", "sas"]


def _env() -> dict:
    import os
    env = os.environ.copy()
    pp = [str(GPULAI_ROOT), str(POPOUT_ROOT)]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pp)
    return env


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, env=_env())


# ── Fixture builder ─────────────────────────────────────────────────────


def build_fixture(root: Path) -> dict:
    """Materialise the synthetic fixture under ``root``. Returns paths."""
    root.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    n = 60
    samples = [f"S{i:03d}" for i in range(n)]
    truth = np.random.dirichlet([1] * 6, size=n)

    # popout: K=4 components mapping (0,1)→afr; 2→amr; 3→eas. Cohort-wide.
    popout = np.zeros((n, 4))
    popout[:, 0] = truth[:, 0] * 0.5
    popout[:, 1] = truth[:, 0] * 0.5
    popout[:, 2] = truth[:, 1]
    popout[:, 3] = truth[:, 2]
    popout += np.random.normal(0, 0.02, popout.shape).clip(min=0)
    popout = popout / popout.sum(axis=1, keepdims=True)

    rye = np.stack([truth[:, 3], truth[:, 2], truth[:, 1], truth[:, 0], truth[:, 5]], axis=1)
    rf_hard = [RF_LABELS[i] for i in truth.argmax(axis=1)]

    # FLARE: K=5 mapped to {afr, amr, eas, eur, sas}. Per-cluster splits.
    clusters = {"cluster_000": samples[:30], "cluster_001": samples[30:]}
    flare = np.zeros((n, 5))
    flare[:, 0] = truth[:, 0]
    flare[:, 1] = truth[:, 1]
    flare[:, 2] = truth[:, 2]
    flare[:, 3] = truth[:, 3]
    flare[:, 4] = truth[:, 5]
    flare += np.random.normal(0, 0.02, flare.shape).clip(min=0)
    flare = flare / flare.sum(axis=1, keepdims=True)

    # Per-cluster bundle layout
    bsrc = root / "bundle_src"
    for cid, sids in clusters.items():
        cdir = bsrc / "per_cluster" / cid / "chr1"
        (cdir / "soft_correlation").mkdir(parents=True, exist_ok=True)
        sids_idx = [samples.index(s) for s in sids]
        with open(cdir / "global.tsv", "w") as f:
            f.write("sample_id\t" + "\t".join(f"ancestry_{i}" for i in range(5)) + "\n")
            for sid, i in zip(sids, sids_idx):
                f.write(sid + "\t" + "\t".join(f"{x:.6f}" for x in flare[i]) + "\n")
        labels = {
            "tool": "FLARE",
            "rf_ref_labels": RF_LABELS,
            "popout_to_rf_label": {"0": "afr", "1": "amr", "2": "eas", "3": "eur", "4": "sas"},
            "rf_to_popout_components": {
                "afr": [0], "amr": [1], "eas": [2], "eur": [3], "mid": [], "sas": [4],
            },
            "correlations": [], "n_overlapping_sites": 0,
        }
        (cdir / "soft_correlation" / "labels.json").write_text(json.dumps(labels))
    bundle_path = root / "cohort_bundle.v3.0.0.tar.gz"
    subprocess.run(
        ["tar", "czf", str(bundle_path), "-C", str(bsrc), "."],
        check=True,
    )

    # popout run_dir (whole-cohort)
    prun = root / "popout_run"
    prun.mkdir(parents=True, exist_ok=True)
    with open(prun / "aou.global.tsv", "w") as f:
        f.write("sample_id\t" + "\t".join(f"ancestry_{i}" for i in range(4)) + "\n")
        for s, p in zip(samples, popout):
            f.write(s + "\t" + "\t".join(f"{x:.6f}" for x in p) + "\n")
    (prun / "aou.model").write_text(
        "n_ancestries\t4\ngen_since_admix\t15.0\nmu\t0.25,0.25,0.25,0.25\n"
    )
    with gzip.open(prun / "aou.tracts.tsv.gz", "wt") as f:
        f.write("chrom\tstart_bp\tend_bp\tsample\thaplotype\tancestry\tn_sites\n")
        for s, p in zip(samples, popout):
            anc = int(np.argmax(p))
            for hap in (0, 1):
                f.write(f"chr1\t100000\t199000\t{s}\t{hap}\t{anc}\t100\n")

    # Rye + RF whole-cohort files
    rye_path = root / "rye.Q"
    with open(rye_path, "w") as f:
        f.write("eur\teas\tamr\tafr\tsas\tresearch_id\n")
        for s, r in zip(samples, rye):
            f.write("\t".join(f"{x:.6f}" for x in r) + f"\t{s}\n")
    rf_path = root / "rf.tsv"
    with open(rf_path, "w") as f:
        f.write("research_id\tancestry_pred\tprobabilities\n")
        for s, r, h in zip(samples, truth, rf_hard):
            f.write(f"{s}\t{h}\t{[float(x) for x in r]}\n")

    return {
        "root": root,
        "config": root / "dx_config.yaml",
        "bundle": bundle_path,
        "popout_run": prun,
        "rye": rye_path,
        "rf": rf_path,
        "clusters": list(clusters),
    }


def write_config(paths: dict, mode: str) -> None:
    extras = ""
    if mode == "global_local":
        extras = "\nlocal_sampling:\n  per_bucket_n: 5\n  threshold: 0.80\n  rng_seed: 42\n  chroms: [chr1]\n"
    paths["config"].write_text(f"""run_name: smoke_e2e_{mode}
schema_version: "1.0.0"
tools: [popout, flare, rye, rf]
flare:
  cohort_bundle: {paths["bundle"]}
rye:
  q_path: {paths["rye"]}
rf:
  ancestry_path: {paths["rf"]}
clusters: ['cluster_*']
chroms: ['chr*']{extras}
""")


# ── Assertions ──────────────────────────────────────────────────────────


def assert_cohort_bundle_valid(out_tar: Path, expected_clusters: list[str]) -> None:
    with tarfile.open(out_tar, "r:*") as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile())
    expected = {
        "cohort_dx/cohort_manifest.json",
        "cohort_dx/cohort_summary.json",
        "cohort_dx/cohort/manifest.tsv",
        "cohort_dx/cohort/tier1_metrics.tsv",
        "cohort_dx/cohort/per_sample_mae.tsv",
        "cohort_dx/cohort/pairwise_soft_summary.tsv",
        "cohort_dx/cohort/popout_vs_flare.confusion.tsv",
        "cohort_dx/cohort/popout_vs_flare.metrics.tsv",
        "cohort_dx/cohort/popout_vs_rye.confusion.tsv",
        "cohort_dx/cohort/popout_vs_rye.metrics.tsv",
        "cohort_dx/cohort/popout_vs_rf.confusion.tsv",
        "cohort_dx/cohort/popout_vs_rf.metrics.tsv",
    }
    missing = expected - set(names)
    if missing:
        raise AssertionError(f"cohort bundle missing files: {sorted(missing)}")

    # Spot-check cohort_manifest cluster list.
    with tarfile.open(out_tar, "r:*") as tar:
        m = json.loads(tar.extractfile("cohort_dx/cohort_manifest.json").read())
    if sorted(m["cluster_ids"]) != sorted(expected_clusters):
        raise AssertionError(
            f"cohort_manifest.cluster_ids = {m['cluster_ids']!r} != expected {expected_clusters!r}"
        )
    if m["schema_version"] != "1.0.0":
        raise AssertionError(f"schema_version = {m['schema_version']!r} != '1.0.0'")
    print(f"  ✓ cohort bundle has {len(names)} files, schema_version=1.0.0, "
          f"clusters={m['cluster_ids']}")


# ── Driver ──────────────────────────────────────────────────────────────


def run(work: Path, mode: str) -> None:
    paths = build_fixture(work / "fixture")
    write_config(paths, mode=mode)

    discover_out = work / "discover"
    _run([
        sys.executable, str(SCRIPTS / "discover_runs.py"),
        "--config", str(paths["config"]),
        "--popout-outputs", str(paths["popout_run"]),
        "--mode", mode,
        "--out-dir", str(discover_out),
    ])

    tarballs_dir = work / "tarballs"
    tarballs_dir.mkdir(parents=True, exist_ok=True)
    for cid in paths["clusters"]:
        out_tar = tarballs_dir / f"{cid}.chr1.popout_dx.v1.0.0.tar.gz"
        _run([
            sys.executable, str(SCRIPTS / "run_dx_cluster.py"),
            "--runs-manifest-tsv", str(discover_out / "runs_manifest.tsv"),
            "--cluster-id", cid,
            "--chrom", "chr1",
            "--mode", mode,
            "--run-name", f"smoke_e2e_{mode}",
            "--tools", "popout,flare,rye,rf",
            "--config-file", str(paths["config"]),
            "--work-dir", str(work / "per_cluster_work" / cid),
            "--max-workers", "4",
            "--emit-tarball", str(out_tar),
        ])

    cohort_tar = work / f"cohort_dx.smoke_e2e_{mode}.v1.0.0.tar.gz"
    _run([
        sys.executable, str(SCRIPTS / "collate_dx.py"),
        "--tarballs", *(str(p) for p in sorted(tarballs_dir.glob("*.tar.gz"))),
        "--run-name", f"smoke_e2e_{mode}",
        "--mode", mode,
        "--tools", "popout,flare,rye,rf",
        "--out-dir", str(work / "collate"),
        "--out-tarball", str(cohort_tar),
    ])

    assert_cohort_bundle_valid(cohort_tar, paths["clusters"])
    print(f"\n✓ popout DX e2e ({mode}) PASS — bundle: {cohort_tar}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="global", choices=("global",),
                    help="global_local is TODO (needs synthetic anc.vcf.gz)")
    ap.add_argument("--keep", type=Path, default=None,
                    help="if set, use this dir as the workspace and do not clean up")
    args = ap.parse_args()

    if args.keep is not None:
        args.keep.mkdir(parents=True, exist_ok=True)
        run(args.keep, mode=args.mode)
        print(f"  (kept workspace at {args.keep})")
    else:
        with tempfile.TemporaryDirectory(prefix="popout_dx_e2e_") as td:
            run(Path(td), mode=args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
