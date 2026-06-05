"""Phase 6 follow-up: ``cohort/cohort_global.tsv`` carries named columns.

The v2 collator emitted a single meta-header
``cluster_id<TAB>chrom<TAB>sample_id<TAB>ancestry_props_tab_separated``
regardless of how many ancestry columns each cluster's data carried.
v3 fixes the per-cluster panel naming, so the collator can now echo
the real panel names. All clusters in the cohort must share the same
panel (otherwise the wide table has no canonical column order); panel
mismatch is a hard error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validation.scripts.collate_runs import (
    ClusterArtifact,
    SCHEMA_VERSION,
    collate_cohort_global,
)


def _mk_artifact(
    tmp_path: Path, cluster_id: str, chrom: str, panel: list[str],
    sample_rows: list[tuple[str, ...]],
) -> ClusterArtifact:
    art_dir = tmp_path / "art" / cluster_id / chrom
    art_dir.mkdir(parents=True)
    with open(art_dir / "global.tsv", "w") as f:
        f.write("sample_id\t" + "\t".join(panel) + "\n")
        for row in sample_rows:
            f.write("\t".join(row) + "\n")
    return ClusterArtifact(
        cluster_id=cluster_id,
        chrom=chrom,
        artifact_dir=art_dir,
        manifest={"schema_version": SCHEMA_VERSION,
                  "cluster_id": cluster_id, "chrom": chrom},
        sha256="00" * 32,
    )


def test_cohort_global_header_carries_panel_names(tmp_path: Path):
    panel = ["eas", "amr", "eur", "afr", "sas"]
    a = _mk_artifact(tmp_path, "cluster_000", "chr1", panel,
                     [("S1", "0.10", "0.70", "0.10", "0.05", "0.05"),
                      ("S2", "0.05", "0.05", "0.80", "0.05", "0.05")])
    b = _mk_artifact(tmp_path, "cluster_001", "chr1", panel,
                     [("S3", "0.20", "0.20", "0.20", "0.30", "0.10")])

    out = tmp_path / "cohort_global.tsv"
    collate_cohort_global([a, b], out)

    lines = out.read_text().rstrip("\n").splitlines()
    assert lines[0] == "cluster_id\tchrom\tsample_id\teas\tamr\teur\tafr\tsas", (
        f"named-column header missing or wrong; got {lines[0]!r}"
    )
    # Each row carries the cluster + chrom prefix + sample + 5 props.
    assert lines[1] == "cluster_000\tchr1\tS1\t0.10\t0.70\t0.10\t0.05\t0.05"
    assert lines[2] == "cluster_000\tchr1\tS2\t0.05\t0.05\t0.80\t0.05\t0.05"
    assert lines[3] == "cluster_001\tchr1\tS3\t0.20\t0.20\t0.20\t0.30\t0.10"
    assert len(lines) == 4, "expected header + 3 data rows"


def test_cohort_global_rejects_mismatched_panel(tmp_path: Path):
    a = _mk_artifact(tmp_path, "cluster_000", "chr1",
                     ["eas", "amr", "eur", "afr", "sas"],
                     [("S1", "0.1", "0.7", "0.1", "0.05", "0.05")])
    b = _mk_artifact(tmp_path, "cluster_001", "chr1",
                     ["afr", "amr", "eas", "eur", "sas"],   # different order
                     [("S3", "0.2", "0.2", "0.2", "0.3", "0.1")])

    out = tmp_path / "cohort_global.tsv"
    with pytest.raises(RuntimeError, match="panel header.*disagrees with cohort panel"):
        collate_cohort_global([a, b], out)


def test_cohort_global_rejects_missing_sample_id_column(tmp_path: Path):
    art_dir = tmp_path / "art" / "cluster_000" / "chr1"
    art_dir.mkdir(parents=True)
    # Header doesn't lead with sample_id — that's a v2 cohort-shaped file
    # masquerading as per-cluster; refuse.
    (art_dir / "global.tsv").write_text(
        "research_id\tafr\tamr\teas\teur\tsas\n"
        "S1\t0.1\t0.1\t0.1\t0.6\t0.1\n"
    )
    art = ClusterArtifact(
        cluster_id="cluster_000", chrom="chr1",
        artifact_dir=art_dir,
        manifest={"schema_version": SCHEMA_VERSION,
                  "cluster_id": "cluster_000", "chrom": "chr1"},
        sha256="00" * 32,
    )
    with pytest.raises(RuntimeError, match="first column must be 'sample_id'"):
        collate_cohort_global([art], tmp_path / "cohort_global.tsv")


def test_cohort_global_empty_arts_is_noop(tmp_path: Path):
    out = tmp_path / "cohort_global.tsv"
    collate_cohort_global([], out)
    assert not out.exists(), "empty input should not create the file"
