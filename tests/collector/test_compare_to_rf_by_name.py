"""Phase 6: verify ``compare_to_rf.py --matching by_name`` end-to-end.

Synthesises a tiny FLARE-shaped ``global.tsv`` whose columns are the
SP5 panel names + an ``rf_ancestry.tsv`` whose probabilities lock
each sample to one RF label, runs the script, then asserts that the
emitted ``labels.json`` is the deterministic 1-to-1 by_name mapping
(no postS-derived ``afr.1, afr.2`` subancestries).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "validation" / "scripts" / "compare_to_rf.py"


def _write_flare_global(path: Path, sample_ids: list[str], props: np.ndarray) -> None:
    """Emit a v3-shaped FLARE ``global.tsv`` with named panel columns."""
    cols = ("afr", "amr", "eas", "eur", "sas")
    assert props.shape == (len(sample_ids), len(cols))
    with open(path, "w") as f:
        f.write("sample_id\t" + "\t".join(cols) + "\n")
        for sid, row in zip(sample_ids, props):
            f.write(sid + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")


def _write_rf_table(path: Path, sample_ids: list[str], rf_probs: np.ndarray) -> None:
    """Emit a foxtrot_v4-shaped RF ancestry preds TSV."""
    sp6 = ("afr", "amr", "eas", "eur", "mid", "sas")
    assert rf_probs.shape == (len(sample_ids), len(sp6))
    with open(path, "w") as f:
        f.write("research_id\tancestry_pred\tprobabilities\n")
        for sid, probs in zip(sample_ids, rf_probs):
            pred = sp6[int(np.argmax(probs))]
            probs_str = "[" + ", ".join(f"{v:.6f}" for v in probs) + "]"
            f.write(f"{sid}\t{pred}\t{probs_str}\n")


def _write_summary(path: Path) -> None:
    """Minimal ``summary.json`` so ``compare_to_rf.detect_tool_name`` says FLARE."""
    path.write_text(json.dumps({"config": {"method": "flare"}}))


@pytest.fixture
def synthetic_flare_run(tmp_path: Path):
    """Build a tiny FLARE-shaped run + RF table with 30 lock-in samples per RF label."""
    rng = np.random.default_rng(7)
    sp5 = ("afr", "amr", "eas", "eur", "sas")
    sp6 = ("afr", "amr", "eas", "eur", "mid", "sas")
    sample_ids: list[str] = []
    flare_props: list[np.ndarray] = []
    rf_probs: list[np.ndarray] = []
    for li, lab in enumerate(sp5):
        for k in range(30):
            sid = f"{lab}_{k:03d}"
            sample_ids.append(sid)
            # FLARE column proportions: dominant in the lab's panel column.
            p = rng.dirichlet([0.5] * 5)
            p = 0.95 * np.eye(5)[li] + 0.05 * p
            p = p / p.sum()
            flare_props.append(p.astype(np.float32))
            # RF probabilities: dominant in the matching SP6 column.
            ri = sp6.index(lab)
            r = rng.dirichlet([0.5] * 6)
            r = 0.95 * np.eye(6)[ri] + 0.05 * r
            r = r / r.sum()
            rf_probs.append(r.astype(np.float32))

    flare_props_arr = np.vstack(flare_props)
    rf_probs_arr = np.vstack(rf_probs)

    global_tsv = tmp_path / "cluster_000.chr1.global.tsv"
    rf_tsv = tmp_path / "rf_ancestry.tsv"
    summary_json = tmp_path / "cluster_000.chr1.summary.json"
    _write_flare_global(global_tsv, sample_ids, flare_props_arr)
    _write_rf_table(rf_tsv, sample_ids, rf_probs_arr)
    _write_summary(summary_json)
    return {"tmp": tmp_path, "global_tsv": global_tsv,
            "rf_tsv": rf_tsv, "summary": summary_json}


def test_by_name_emits_one_to_one_labels(synthetic_flare_run, tmp_path: Path):
    """``--matching by_name`` should produce a 1-to-1 FLARE→SP6 mapping."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--popout-global", str(synthetic_flare_run["global_tsv"]),
         "--rf-ancestry", str(synthetic_flare_run["rf_tsv"]),
         "--matching", "by_name",
         "--out-dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert res.returncode == 0, (
        f"compare_to_rf.py exit {res.returncode}\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )

    labels = json.loads((out_dir / "labels.json").read_text())

    # The FLARE panel columns are afr/amr/eas/eur/sas in that order.
    expected = {"0": "afr", "1": "amr", "2": "eas", "3": "eur", "4": "sas"}
    actual = {str(k): v for k, v in labels["popout_to_rf_label"].items()}
    assert actual == expected, (
        f"by_name should map panel header verbatim. expected={expected!r} "
        f"actual={actual!r}"
    )

    # No fake subancestries (no `afr.1, afr.2` of postS).
    for rf, stats in labels.get("merge_group_stats", {}).items():
        names = stats.get("names", [])
        assert all("." not in n for n in names), (
            f"by_name should not invent subancestries; got names={names} for {rf}"
        )

    # Provenance: matching method is recorded.
    assert labels.get("provenance", {}).get("matching", labels.get("matching")) in (
        "by_name", "name",
    ) or labels.get("method") in ("by_name", "name"), (
        f"labels.json should record matching=by_name; got {labels!r}"
    )


def test_postS_path_still_works(synthetic_flare_run, tmp_path: Path):
    """Regression: ``--matching postS`` (the default) must still produce a labels.json."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--popout-global", str(synthetic_flare_run["global_tsv"]),
         "--rf-ancestry", str(synthetic_flare_run["rf_tsv"]),
         "--matching", "postS",
         "--out-dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert res.returncode == 0, (
        f"compare_to_rf.py postS exit {res.returncode}\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    labels = json.loads((out_dir / "labels.json").read_text())
    assert "popout_to_rf_label" in labels
