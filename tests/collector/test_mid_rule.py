"""Phase 6: verify the MID-rule fold in ``collate_runs.collate_confusion_rf``.

The collator applies one of three rules to the RF-side MID column when
unpivoting per-cluster ``rf_confusion_matrix.tsv`` into
``cohort/confusion_rf.tsv``:

  - ``none``         — pass-through (legacy v2 behavior)
  - ``drop``         — remove every MID row
  - ``fold_to_eur``  — sum MID counts into EUR per (cluster, flare_call)

The chosen rule is recorded in ``cohort_manifest.json.provenance.mid_rule``;
this test exercises just the unpivot helper, not the manifest writer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validation.scripts.collate_runs import (
    ClusterArtifact,
    SCHEMA_VERSION,
    collate_confusion_rf,
)


@pytest.fixture
def fake_artifact(tmp_path: Path) -> ClusterArtifact:
    """A bare-bones ClusterArtifact with one per-cluster confusion matrix."""
    art_dir = tmp_path / "art" / "cluster_007" / "chr1"
    (art_dir / "confusion").mkdir(parents=True)
    # rf_label rows × FLARE-call columns; MID has 100 counts to be folded.
    text = (
        "rf_label\tafr\tamr\teas\teur\tsas\ttotal\n"
        "afr\t500\t10\t0\t5\t0\t515\n"
        "amr\t10\t300\t0\t20\t0\t330\n"
        "eas\t0\t0\t250\t5\t0\t255\n"
        "eur\t5\t20\t0\t400\t10\t435\n"
        "mid\t0\t5\t0\t90\t5\t100\n"
        "sas\t0\t0\t0\t10\t200\t210\n"
        "total\t515\t335\t250\t530\t215\t1845\n"
    )
    (art_dir / "confusion" / "rf_confusion_matrix.tsv").write_text(text)
    return ClusterArtifact(
        cluster_id="cluster_007",
        chrom="chr1",
        artifact_dir=art_dir,
        manifest={"schema_version": SCHEMA_VERSION,
                  "cluster_id": "cluster_007", "chrom": "chr1"},
        sha256="00" * 32,
    )


def _read_long(path: Path) -> dict[tuple[str, str], int]:
    """Read the long-form output back into ``{(rf_label, flare_call): n}``."""
    out: dict[tuple[str, str], int] = {}
    with open(path) as f:
        next(f)
        for line in f:
            _cid, _chrom, rf, flare, n = line.rstrip().split("\t")
            out[(rf, flare)] = int(n)
    return out


def test_mid_rule_none_passes_through(fake_artifact, tmp_path: Path):
    out = tmp_path / "confusion_rf.tsv"
    collate_confusion_rf([fake_artifact], out, mid_rule="none")
    rows = _read_long(out)
    assert rows[("mid", "eur")] == 90, "mid row must survive when mid_rule=none"
    assert rows[("eur", "eur")] == 400, "eur row untouched"


def test_mid_rule_drop_removes_mid(fake_artifact, tmp_path: Path):
    out = tmp_path / "confusion_rf.tsv"
    collate_confusion_rf([fake_artifact], out, mid_rule="drop")
    rows = _read_long(out)
    mid_rows = [k for k in rows if k[0] == "mid"]
    assert not mid_rows, f"mid rows must be dropped, found {mid_rows}"
    assert rows[("eur", "eur")] == 400, "eur row untouched by drop"


def test_mid_rule_fold_to_eur_sums_into_eur(fake_artifact, tmp_path: Path):
    out = tmp_path / "confusion_rf.tsv"
    collate_confusion_rf([fake_artifact], out, mid_rule="fold_to_eur")
    rows = _read_long(out)
    mid_rows = [k for k in rows if k[0] == "mid"]
    assert not mid_rows, f"mid rows must be folded away, found {mid_rows}"
    # eur row now carries eur + mid per flare_call.
    assert rows[("eur", "afr")] == 5 + 0,    "eur·afr = eur(5)   + mid(0)"
    assert rows[("eur", "amr")] == 20 + 5,   "eur·amr = eur(20)  + mid(5)"
    assert rows[("eur", "eur")] == 400 + 90, "eur·eur = eur(400) + mid(90)"
    assert rows[("eur", "sas")] == 10 + 5,   "eur·sas = eur(10)  + mid(5)"


def test_mid_rule_unknown_raises(fake_artifact, tmp_path: Path):
    with pytest.raises(ValueError, match="unknown mid_rule"):
        collate_confusion_rf([fake_artifact], tmp_path / "x.tsv",
                             mid_rule="unrecognised")
