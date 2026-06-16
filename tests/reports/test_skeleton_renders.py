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


def test_tag_policy_minimal_suppresses_default_sections(tmp_path: Path):
    """FLARE config: SP5 + by-name sections emit no tag; SP6 overrides do."""
    cfg = load_report_config(CONFIGS / "flare_validation_report.yaml")
    assert cfg.tag_policy == "minimal"
    ctx = ReportContext(
        bundle=_fake_bundle(),
        bundle_dir=tmp_path / "fake_bundle",
        config=cfg,
        estimates={},
        assets_dir=tmp_path / "assets",
    )
    ctx.assets_dir.mkdir(parents=True, exist_ok=True)

    # FLARE-only SP5 sections suppress the tag entirely.
    assert ctx.tag("cohort_composition") == ""
    assert ctx.tag("ccc_chrom_drift") == ""
    assert ctx.tag("flare_vs_rye_concordance") == ""

    # SP6 cross-tool sections still emit a slimmed tag. The FLARE
    # config's defaults.mid_rule="drop" is inherited by all sections;
    # on SP6 (which has MID) that drop is informative and survives.
    rf_tag = ctx.tag("flare_vs_rf_calibration")
    assert rf_tag.startswith("L=SP6/MID- | v=") and "=>" not in rf_tag, rf_tag

    # Verbose tag still available via the provenance helper.
    full = ctx.verbose_tag("cohort_composition")
    assert full.startswith("L=SP5/MID- | flare=>name | v=")


def test_tag_policy_verbose_preserves_full_tag(tmp_path: Path):
    """popout_dx config: every section still emits the full verbose tag."""
    cfg = load_report_config(CONFIGS / "popout_dx_report.yaml")
    assert cfg.tag_policy == "verbose"
    ctx = ReportContext(
        bundle=_fake_bundle(mode="global_local"),
        bundle_dir=tmp_path / "fake_bundle",
        config=cfg,
        estimates={},
        assets_dir=tmp_path / "assets",
    )
    ctx.assets_dir.mkdir(parents=True, exist_ok=True)

    tag = ctx.tag("traffic_light_grid")
    assert tag.startswith("L=SP6/MID+ | ")
    assert "popout=>name" in tag                # default fallback when no estimate


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
