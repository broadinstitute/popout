"""CCC suite: compute + render smoke tests against a synthetic bundle.

The eight new charts (three ``cohort_structure_*`` and five ``ccc_*``)
all read ``cohort/cohort_global.tsv`` + ``cohort/merged_groups_rf.tsv``
through ``popout.reports._helpers.load_cohort_cube``. The fixture
fabricates a 4-cluster × 22-chrom bundle with samples drawn from each
of the SP5 ancestries plus a handful of admixed samples, enough to
populate every chart without exercising the production scale.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from popout.reports.charts import (
    ccc_chrom_drift,
    ccc_chrom_stdev,
    ccc_dose_response,
    ccc_pair_mad,
    cohort_structure_3way_face,
    cohort_structure_pair_census,
    cohort_structure_top1_top2,
)
from popout.reports.config import ReportConfig, ReportStyle
from popout.reports.context import ReportContext

ANC = ["afr", "amr", "eas", "eur", "sas"]
N_PER_DOMINANT = 60       # 60 X-dominant samples per ancestry → 300 total
CHROMS = [f"chr{i}" for i in range(1, 23)]
CLUSTER_ID = "cluster_000"


def _synthetic_bundle(tmp_path: Path, seed: int = 7) -> Path:
    """Write cohort_global.tsv + merged_groups_rf.tsv into tmp_path."""
    rng = random.Random(seed)
    bundle_dir = tmp_path / "bundle"
    cohort = bundle_dir / "cohort"
    cohort.mkdir(parents=True, exist_ok=True)

    # merged_groups_rf.tsv — one mapping per (cluster, chrom).
    mg_rows = ["cluster_id\tchrom\trf_label\tcomponent_indices"]
    for chrom in CHROMS:
        for ai, lab in enumerate(ANC):
            mg_rows.append(f"{CLUSTER_ID}\t{chrom}\t{lab}\t{ai}")
    (cohort / "merged_groups_rf.tsv").write_text("\n".join(mg_rows) + "\n")

    # cohort_global.tsv — production column order: cluster_id, chrom,
    # sample_id, then the proportion vector.
    rows = ["cluster_id\tchrom\tsample_id\t0\t1\t2\t3\t4"]
    sid = 0
    for ai, lab in enumerate(ANC):
        for _ in range(N_PER_DOMINANT):
            sid += 1
            sample_id = f"s{sid:05d}_{lab}"
            for chrom in CHROMS:
                vec = [rng.uniform(0.0, 0.02) for _ in ANC]
                if chrom == "chr1":
                    vec[ai] = rng.uniform(0.95, 0.99)
                else:
                    noise = 0.10 if lab in ("afr", "amr") else 0.02
                    vec[ai] = max(0.50, rng.gauss(0.95 - noise, noise / 2))
                total = sum(vec)
                vec = [v / total for v in vec]
                cols = "\t".join(f"{v:.6f}" for v in vec)
                rows.append(f"{CLUSTER_ID}\t{chrom}\t{sample_id}\t{cols}")
    for _ in range(40):
        sid += 1
        sample_id = f"s{sid:05d}_admix"
        for chrom in CHROMS:
            vec = [rng.uniform(0.0, 0.02) for _ in ANC]
            vec[ANC.index("afr")] = rng.uniform(0.40, 0.55)
            vec[ANC.index("eur")] = rng.uniform(0.40, 0.55)
            total = sum(vec)
            vec = [v / total for v in vec]
            cols = "\t".join(f"{v:.6f}" for v in vec)
            rows.append(f"{CLUSTER_ID}\t{chrom}\t{sample_id}\t{cols}")
    (cohort / "cohort_global.tsv").write_text("\n".join(rows) + "\n")
    return bundle_dir


@pytest.fixture
def ctx(tmp_path: Path) -> ReportContext:
    bundle_dir = _synthetic_bundle(tmp_path)
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return ReportContext(
        bundle={},
        bundle_dir=bundle_dir,
        config=ReportConfig(title="t", style=ReportStyle(),
                            sections=(), raw={}),
        estimates={},
        assets_dir=assets_dir,
    )


ALL_CHARTS = [
    cohort_structure_pair_census,
    cohort_structure_top1_top2,
    cohort_structure_3way_face,
    ccc_chrom_drift,
    ccc_chrom_stdev,
    ccc_pair_mad,
    ccc_dose_response,
]


@pytest.mark.parametrize("mod", ALL_CHARTS,
                         ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_compute_and_render(mod, ctx):
    data = mod.compute(ctx, section=None)
    assert isinstance(data, dict)
    assert data.get("present") is True, (
        f"{mod.__name__} returned present=False on the synthetic bundle")
    fig = mod.render(data, palette=ctx.palette)
    assert fig is not None
    assert hasattr(fig, "savefig")
    w, h = fig.get_size_inches()
    assert 0 < w < 30 and 0 < h < 30, f"{mod.__name__} figure size out of bounds"
    matplotlib.pyplot.close(fig)


def test_chrom_drift_finds_asymmetry(ctx):
    """afr-dominant stratum should show larger drift than eur-dominant.

    The drift metric was folded into ccc_chrom_drift: each stratum now
    carries ``chr1_median``, ``others_mean``, and ``drift = chr1 − others``.
    """
    data = ccc_chrom_drift.compute(ctx, section=None)
    by_label = {s["label"]: s for s in data["strata"]
                if s["drift"] is not None}
    assert "afr" in by_label and "eur" in by_label
    assert by_label["afr"]["drift"] > by_label["eur"]["drift"], (
        f"expected afr drift > eur drift, got "
        f"afr={by_label['afr']['drift']:.3f}, "
        f"eur={by_label['eur']['drift']:.3f}")


def test_missing_bundle_returns_not_present(tmp_path: Path):
    """If cohort_global.tsv is absent, every chart returns present=False."""
    empty_dir = tmp_path / "empty"
    (empty_dir / "cohort").mkdir(parents=True)
    ctx_empty = ReportContext(
        bundle={},
        bundle_dir=empty_dir,
        config=ReportConfig(title="t", style=ReportStyle(),
                            sections=(), raw={}),
        estimates={},
        assets_dir=tmp_path / "assets",
    )
    ctx_empty.assets_dir.mkdir(parents=True, exist_ok=True)
    for mod in ALL_CHARTS:
        data = mod.compute(ctx_empty, section=None)
        assert data.get("present") is False, (
            f"{mod.__name__} should return present=False without bundle")
        # render() must still produce a figure without exception
        fig = mod.render(data, palette=ctx_empty.palette)
        assert fig is not None
        matplotlib.pyplot.close(fig)
