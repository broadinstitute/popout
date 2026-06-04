"""Phase 2: skeleton report renders for both FLARE and popout DX.

These tests render the markdown only — pandoc invocation is gated
behind ``POPOUT_REPORT_TEST_PDF=1`` for environments that have
xelatex installed. The skeleton is what we get after Phase 2 (cover +
reading guide + label-space conventions + provenance); Phases 3 and 4
fill in the rest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from popout.reports import (
    ReportContext,
    load_report_config,
    render_report,
    run_pandoc,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"


def _fake_bundle(*, mode: str = "global") -> dict:
    return {
        "schema_version": "1.0.0",
        "run_name": "fixture_run",
        "mode": mode,
        "tools": ["popout", "flare", "rye", "rf"],
        "n_clusters": 2,
        "n_chroms": 1,
        "n_artifacts": 2,
        "generated_at": "2026-06-04T00:00:00Z",
        "cluster_ids": ["cluster_000", "cluster_001"],
        "chroms": ["chr1"],
        "sha256_per_artifact": {
            "cluster_000/chr1": "abcdef1234567890" * 4,
            "cluster_001/chr1": "fedcba0987654321" * 4,
        },
    }


@pytest.mark.parametrize("config_name,bundle_mode", [
    ("flare_validation_report.yaml", "global"),
    ("popout_dx_report.yaml", "global_local"),
])
def test_skeleton_md_renders(config_name, bundle_mode, tmp_path: Path):
    cfg = load_report_config(CONFIGS / config_name)
    ctx = ReportContext(
        bundle=_fake_bundle(mode=bundle_mode),
        bundle_dir=tmp_path / "fake_bundle",
        config=cfg,
        estimates={},
        assets_dir=tmp_path / "assets",
    )
    ctx.assets_dir.mkdir(parents=True, exist_ok=True)
    md = render_report(ctx)

    # All four spine sections render.
    assert "# popout DX report" in md or "# FLARE validation report" in md
    assert "# How to read the metrics" in md
    assert "# Label-space conventions" in md
    assert "# Provenance" in md

    # No raw Jinja tokens left.
    assert "{{" not in md
    assert "{%" not in md

    # No popout references in the FLARE skeleton.
    if config_name == "flare_validation_report.yaml":
        # The conventions page mentions popout only in the corrH example
        # (flare_3) — confirm no marketing of popout in the body.
        assert "popout DX report" not in md
        # And the cover doesn't mention popout.
        cover_block = md.split("# Label-space conventions")[0]
        assert "popout" not in cover_block.lower()


@pytest.mark.skipif(
    not (shutil.which("pandoc") and shutil.which("xelatex")
         and os.environ.get("POPOUT_REPORT_TEST_PDF")),
    reason="POPOUT_REPORT_TEST_PDF=1 with pandoc + xelatex required",
)
@pytest.mark.parametrize("config_name", [
    "flare_validation_report.yaml",
    "popout_dx_report.yaml",
])
def test_skeleton_pdf_renders(config_name, tmp_path: Path):
    cfg = load_report_config(CONFIGS / config_name)
    ctx = ReportContext(
        bundle=_fake_bundle(mode="global"),
        bundle_dir=tmp_path / "fake_bundle",
        config=cfg,
        estimates={},
        assets_dir=tmp_path / "assets",
    )
    ctx.assets_dir.mkdir(parents=True, exist_ok=True)
    md = render_report(ctx)
    md_path = tmp_path / "report.md"
    md_path.write_text(md)
    pdf_path = tmp_path / "report.pdf"
    run_pandoc(md_path, pdf_path, style=cfg.style)
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 5_000
