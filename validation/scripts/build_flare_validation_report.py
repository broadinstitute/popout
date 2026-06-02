#!/usr/bin/env python3
"""FLARE validation Stage 3 — build a single PDF report from a cohort bundle.

Consumes the output of `collate_runs.py` (a `cohort_bundle/` directory or
companion per-cluster tarballs) and emits one PDF rolling up cohort-level
charts.

Architecture mirrors `validation/scripts/build_report_pdf.py`:

  1. Walk the cohort bundle and build a single markdown document.
  2. Generate per-section charts via matplotlib ``savefig`` (one chart per
     page) into the report's side-car asset directory; embed with markdown
     ``![](...)``.
  3. Shell out to ``pandoc --pdf-engine=xelatex`` to produce the PDF.

Each section follows the same template: heading, "What this measures"
paragraph, "How to read this chart" paragraph, the chart, an auto-derived
"In this run" callout naming specific outliers from the actual data, and
(optionally) a supporting table.

Usage:
    python build_flare_validation_report.py \\
        --cohort-bundle <dir>                   # contains cohort/, cohort_*.json, per_cluster/ (optional)
        --tarball-dir   <dir>                   # alternative source for per_cluster figures
        --out           <report.pdf>            \\
        [--clusters cluster_001,cluster_007]    # subset
        [--max-clusters 10]                     # cap (clusters sorted by id)
        [--per-cluster]                         # append per-cluster long-form sections
        [--keep-md]                             # leave the intermediate .md next to the PDF

Contract:
  * Required cohort artifacts (per ``validation.schema.REQUIRED_COHORT_FILES``)
    are checked at load time; a missing one is a hard error with the exact
    path.
  * Required per-cluster files render a ``MISSING: <relative path>`` line in
    place of the figure/table but the report still renders.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

VALIDATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATION_DIR.parent))
from validation.schema import (  # noqa: E402
    OPTIONAL_COHORT_FILES,
    REQUIRED_COHORT_FILES,
    SCHEMA_VERSION,
)

POPOUT_REPO = Path("/Users/ghall/code/work/broad/popout")
if (POPOUT_REPO / "popout").is_dir():
    sys.path.insert(0, str(POPOUT_REPO))
from popout.viz._style import (  # noqa: E402
    ANCESTRY_PALETTE,
    chrom_length,
    chrom_sort_key,
    popout_style,
)


# ── Canonical color mappings ──────────────────────────────────────────────
#
# The same physical color always means the same ancestry across every chart
# in the report. RF labels appear first; numbered sub-ancestries (afr.1,
# eur.3, …) inherit their parent label's color.

# Phase 4 of the label-space retrofit: SP6 from popout.labelspace.registry.
from popout.labelspace.registry import SP6 as _SP6
ANCESTRY_ORDER: tuple[str, ...] = _SP6.members
ANCESTRY_COLOR: dict[str, str] = {
    "afr": ANCESTRY_PALETTE[0],
    "amr": ANCESTRY_PALETTE[1],
    "eas": ANCESTRY_PALETTE[2],
    "eur": ANCESTRY_PALETTE[3],
    "mid": ANCESTRY_PALETTE[4],
    "sas": ANCESTRY_PALETTE[5],
}
# A neutral palette for per-cluster coloring on charts where the ancestry
# palette is already in use (tract length, switch rate).
CLUSTER_PALETTE: tuple[str, ...] = ANCESTRY_PALETTE[6:] + ANCESTRY_PALETTE[:6]


def ancestry_color(name: str) -> str:
    """Map an ancestry name (eur, afr, afr.1, etc.) to its canonical color."""
    base = name.split(".")[0].lower()
    return ANCESTRY_COLOR.get(base, "#888888")


def cluster_color(idx: int) -> str:
    return CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]


# Render quality: every saved chart uses this DPI.
_DPI = 240


# ── Label-space tag (Phase 5 of the label-space retrofit) ────────────────
#
# The tag is computed once per report run from the cohort bundle's
# provenance (when present) or synthesized from the FLARE-validate
# defaults. Every chart gets stamped with the tag in the bottom-strip
# footer via _save_fig_with_tag() so a reader who sees `afr.1` on a
# scatterplot can trace it back to the exact mapped label space.

_LABEL_SPACE_TAG: str = ""


def set_label_space_tag(tag: str) -> None:
    """Set the module-level figure tag (called once from build_markdown)."""
    global _LABEL_SPACE_TAG
    _LABEL_SPACE_TAG = tag or ""


def get_label_space_tag() -> str:
    return _LABEL_SPACE_TAG


def resolve_label_space_tag(bundle: "CohortBundle") -> str:
    """Pick the canonical figure tag for a FLARE-validate cohort.

    Reads ``provenance.tag`` from ``cohort_manifest.json`` when the
    bundle was built post-Phase-5; otherwise synthesises the SP6 tag
    that matches FLARE-validate's default pipeline (corrH against the
    1KG reference; name matches for the RF and Rye comparators).
    """
    for src in (bundle.manifest, bundle.summary):
        prov = (src or {}).get("provenance") or {}
        tag = prov.get("tag")
        if isinstance(tag, str) and tag:
            return tag
    return "L=SP6/MID+ | flare=>corrH | rf=>name | rye=>name"


def _stamp_tag(fig) -> None:
    """Inject the figure-tag shorthand as a bottom-strip footer."""
    tag = get_label_space_tag()
    if not tag:
        return
    fig.text(
        0.5, 0.005, tag,
        ha="center", va="bottom",
        fontsize=6.5, color="#666",
        family="monospace", alpha=0.85,
    )


def _save_fig_with_tag(fig, out_png: Path, **kwargs) -> None:
    """Wrap fig.savefig: stamp the label-space footer first."""
    _stamp_tag(fig)
    fig.savefig(out_png, **kwargs)


def cluster_codes(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Return a (cc → "c1"/"c2"/…) map and a sorted (code, cc) legend list.

    "cc" is the full ``f"{cluster_id}·{chrom}"`` string. Short codes are
    used as inline labels on dense plots so long names don't overlap.
    """
    cc_keys = sorted({f"{cid}·{chrom}" for cid, chrom in pairs})
    mapping = {cc: f"c{i + 1}" for i, cc in enumerate(cc_keys)}
    legend = [(mapping[cc], cc) for cc in cc_keys]
    return mapping, legend


def _read_merged_groups_rf(bundle: CohortBundle) -> dict[tuple[str, str], dict[int, str]]:
    """(cluster_id, chrom) → {FLARE component index → RF label}.

    Source: ``cohort/merged_groups_rf.tsv``. The ``component_indices`` column
    is a comma-separated list of integers; ``rf_label`` is the merged RF
    label they collapse to.
    """
    path = bundle.bundle_dir / "cohort" / "merged_groups_rf.tsv"
    out: dict[tuple[str, str], dict[int, str]] = {}
    if not path.exists():
        return out
    header, rows = _read_tsv(path)
    if not rows:
        return out
    col = {h: i for i, h in enumerate(header)}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            rf = r[col["rf_label"]]
            idxs = r[col["component_indices"]]
        except (IndexError, KeyError):
            continue
        d = out.setdefault((cid, chrom), {})
        for token in idxs.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                d[int(token)] = rf
            except ValueError:
                continue
    return out


# ── Logging ───────────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


# ── Cohort bundle in-memory view ──────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class CohortBundle:
    bundle_dir: Path
    manifest: dict[str, Any]               # cohort_manifest.json
    summary: dict[str, Any]                # cohort_summary.json

    @property
    def run_name(self) -> str:
        return self.manifest["run_name"]

    @property
    def cluster_ids(self) -> list[str]:
        return list(self.manifest["cluster_ids"])

    @property
    def chroms(self) -> list[str]:
        return list(self.manifest["chroms"])

    @property
    def cluster_chrom_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for key in self.manifest.get("sha256_per_artifact", {}):
            cid, _, chrom = key.rpartition(".")
            if cid and chrom:
                pairs.append((cid, chrom))
        return sorted(pairs)


def load_cohort_bundle(bundle_dir: Path) -> CohortBundle:
    """Read required cohort JSONs; raise on any missing required file."""
    if not bundle_dir.is_dir():
        raise NotADirectoryError(bundle_dir)

    missing = [rel for rel in REQUIRED_COHORT_FILES if not (bundle_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(
            "cohort_bundle is missing required artifacts:\n  "
            + "\n  ".join(str(bundle_dir / rel) for rel in missing)
        )

    manifest = json.loads((bundle_dir / "cohort_manifest.json").read_text())
    summary = json.loads((bundle_dir / "cohort_summary.json").read_text())

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"cohort_manifest.json schema_version {manifest.get('schema_version')!r} "
            f"!= expected {SCHEMA_VERSION!r}"
        )

    return CohortBundle(bundle_dir=bundle_dir, manifest=manifest, summary=summary)


# ── Per-cluster directory resolution ──────────────────────────────────────


def resolve_cluster_dirs(
    bundle_dir: Path,
    tarball_dir: Path | None,
    tmpdir: Path,
    cluster_chrom_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    per_cluster_root = bundle_dir / "per_cluster"

    for cid, chrom in cluster_chrom_pairs:
        unpacked = per_cluster_root / cid / chrom
        if unpacked.is_dir():
            out[(cid, chrom)] = unpacked
            continue
        if tarball_dir is None:
            continue
        candidates = sorted(tarball_dir.glob(f"{cid}.{chrom}.validation.*.tar.gz"))
        if not candidates:
            continue
        tarball = candidates[-1]
        dest = tmpdir / cid / chrom
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:*") as tar:
            tar.extractall(dest, filter="data")
        drilled = dest / cid / chrom
        out[(cid, chrom)] = drilled if drilled.is_dir() else dest
    return out


# ── CLI filters ───────────────────────────────────────────────────────────


def filter_and_cap(
    pairs: list[tuple[str, str]],
    cluster_filter: list[str] | None,
    max_clusters: int | None,
) -> list[tuple[str, str]]:
    if cluster_filter:
        wanted = set(cluster_filter)
        unknown = wanted - {cid for cid, _ in pairs}
        if unknown:
            raise ValueError(
                f"--clusters references unknown cluster_id(s): {sorted(unknown)}"
            )
        pairs = [(cid, chrom) for cid, chrom in pairs if cid in wanted]
    if max_clusters is not None and max_clusters > 0:
        kept_cids: list[str] = []
        out: list[tuple[str, str]] = []
        for cid, chrom in pairs:
            if cid not in kept_cids:
                if len(kept_cids) >= max_clusters:
                    continue
                kept_cids.append(cid)
            out.append((cid, chrom))
        pairs = out
    return pairs


# ── Small markdown / TSV helpers ──────────────────────────────────────────


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("_", "\\_")


def _fmt_num(v: Any, places: int = 3) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return _md_escape(str(v))
    if fv != fv:
        return "—"
    return f"{fv:.{places}f}"


def _fmt_int(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{int(float(v)):,}"
    except (TypeError, ValueError):
        return _md_escape(str(v))


def _fmt_pct(v: Any, places: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{places}f}%"
    except (TypeError, ValueError):
        return _md_escape(str(v))


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    text = path.read_text().rstrip("\n")
    if not text:
        return [], []
    lines = text.split("\n")
    header = lines[0].split("\t")
    body = [line.split("\t") for line in lines[1:]]
    return header, body


def _md_table(header: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    h = [str(c) for c in header]
    out = ["| " + " | ".join(h) + " |",
           "|" + "|".join(["---"] * len(h)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_md_escape(str(c)) for c in row) + " |")
    return "\n".join(out) + "\n"


def _embed_image(png: Path, width: str = "95%") -> str:
    return f"![]({png}){{ width={width} }}\n"


def _missing(rel_path: str) -> str:
    return f"> **MISSING:** `{rel_path}`\n"


# ── Auto-callout helpers ──────────────────────────────────────────────────


def topn(rows: list[tuple[str, float]], n: int = 3, reverse: bool = True) -> list[tuple[str, float]]:
    """Return the n highest (reverse=True) or lowest (reverse=False) (label, value) pairs."""
    clean = [(lab, v) for lab, v in rows if v is not None and v == v]
    return sorted(clean, key=lambda r: r[1], reverse=reverse)[:n]


def fmt_topn(rows: list[tuple[str, float]], n: int = 3, reverse: bool = True,
             val_fmt: str = "{:.3f}") -> str:
    items = topn(rows, n=n, reverse=reverse)
    if not items:
        return "—"
    return ", ".join(f"{lab} ({val_fmt.format(v)})" for lab, v in items)


def n_weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    """[(weight, value), ...] → n-weighted mean, ignoring NaN values / zero weights."""
    num = 0.0
    den = 0.0
    for w, v in pairs:
        if v != v or w != w or w <= 0:
            continue
        num += w * v
        den += w
    return num / den if den > 0 else None


def overlay_ticks(ax, y_center: float, values: list[float], *,
                  color: str = "#222", tick_height: float = 0.32,
                  alpha: float = 0.95, lw: float = 1.5) -> None:
    """Draw short vertical tick marks at each value on a horizontal-bar chart.

    Used to overlay per-cluster values on top of a cohort-pooled bar. Ticks
    are drawn at exact values; coincident values produce overlapping ticks
    on purpose (truthful representation of per-cluster agreement).
    """
    if not values:
        return
    import numpy as np
    arr = np.array(values, dtype=float)
    arr = arr[arr == arr]  # drop NaN
    if arr.size == 0:
        return
    y0 = y_center - tick_height / 2
    y1 = y_center + tick_height / 2
    ax.vlines(arr, y0, y1, colors=color, alpha=alpha, linewidth=lw, zorder=5)


# ── Markdown frontmatter ──────────────────────────────────────────────────


def md_frontmatter(bundle: CohortBundle) -> str:
    return (
        "---\n"
        f"title: \"FLARE validation report: {bundle.run_name}\"\n"
        f"date: \"{bundle.manifest.get('generated_at', '')}\"\n"
        "geometry: margin=0.75in\n"
        "colorlinks: true\n"
        "header-includes:\n"
        "  - \\usepackage{adjustbox}\n"
        "  - \\usepackage{longtable}\n"
        "  - \\let\\oldtable\\table\n"
        "  - \\let\\endoldtable\\endtable\n"
        "  - \\renewenvironment{table}[1][]{\\oldtable[#1]\\adjustbox{max width=\\textwidth}\\bgroup}{\\egroup\\endoldtable}\n"
        "---\n"
    )


# ── §1. Cover & identity ──────────────────────────────────────────────────


def _sum_samples_from_cohort_global(bundle: CohortBundle) -> int:
    """Total sample-rows across every (cluster, chrom) artifact."""
    path = bundle.bundle_dir / "cohort" / "cohort_global.tsv"
    if not path.exists():
        return 0
    header, rows = _read_tsv(path)
    if not rows or "sample_id" not in header:
        return 0
    return len(rows)


def _sum_markers_from_manifest(bundle: CohortBundle) -> int:
    """Sum n_markers across all per-cluster manifests visible in cohort/manifest.tsv."""
    path = bundle.bundle_dir / "cohort" / "manifest.tsv"
    if not path.exists():
        return 0
    header, rows = _read_tsv(path)
    if not rows or "n_markers" not in header:
        return 0
    idx = header.index("n_markers")
    total = 0
    for r in rows:
        try:
            total += int(r[idx])
        except (IndexError, ValueError):
            continue
    return total


def section_cover(bundle: CohortBundle) -> str:
    summary = bundle.summary
    manifest = bundle.manifest
    n_samples_total = _sum_samples_from_cohort_global(bundle)
    n_markers_total = _sum_markers_from_manifest(bundle)
    chroms = ", ".join(bundle.chroms)

    lede = (
        f"This report summarises a FLARE local-ancestry run that spans "
        f"**{summary.get('n_clusters', '?')} clusters** on chromosome(s) "
        f"**{chroms or '?'}**, covering **{n_samples_total:,} samples** and "
        f"**{n_markers_total:,} marker · cluster pairs**. FLARE's global "
        "ancestry estimates are compared primarily against the Rye "
        "supervised-ADMIXTURE Q matrix (soft-vs-soft); the categorical "
        "superpop label assigned by the upstream RF (random-forest) "
        "classifier is used in the confusion, calibration, and "
        "hap-disagreement views. Subsequent pages walk through tract "
        "structure and a cross-cluster regional scan."
    )

    tag = get_label_space_tag() or resolve_label_space_tag(bundle)
    identity_rows = [
        ("run name",        manifest.get("run_name", "?")),
        ("schema version",  manifest.get("schema_version", "?")),
        ("generated",       manifest.get("generated_at", "?")),
        ("collation mode",  manifest.get("collation_mode", "?")),
        ("label space",     f"`{tag}`"),
    ]
    scale_rows = [
        ("clusters",          summary.get("n_clusters", "?")),
        ("chromosomes",       summary.get("n_chroms", "?")),
        ("artifacts",         summary.get("n_artifacts", "?")),
        ("samples (sum)",     f"{n_samples_total:,}"),
        ("markers (sum)",     f"{n_markers_total:,}"),
        ("wallclock hours",   _fmt_num(summary.get("total_wallclock_hours"), 2)),
        ("peak RSS GB (max)", _fmt_num(summary.get("total_peak_rss_gb_max"), 2)),
    ]

    # Render identity + scale as two side-by-side markdown tables in a row by
    # using a 2-column markdown table; pandoc handles this fine without extra
    # LaTeX scaffolding.
    pairs = max(len(identity_rows), len(scale_rows))
    side_rows = []
    for i in range(pairs):
        l = identity_rows[i] if i < len(identity_rows) else ("", "")
        r = scale_rows[i] if i < len(scale_rows) else ("", "")
        # _md_table escapes its own cells; passing pre-escaped values doubles up.
        side_rows.append((l[0], str(l[1]), r[0], str(r[1])))

    out = [
        "# FLARE validation report\n",
        f"### `{manifest.get('run_name', '')}`\n",
        lede + "\n",
        "## Run identity & scale\n",
        _md_table(
            ["", "identity", "", "scale"],
            side_rows,
        ),
        f"\n_Source: `{bundle.bundle_dir}`_\n",
    ]
    return "\n".join(out) + "\n"


# ── §2. How to read this report (static prose) ────────────────────────────


READING_GUIDE_MD = """# How to read this report

**What FLARE does.** FLARE is a local-ancestry inference (LAI) tool. Given a
phased VCF and a reference panel of known-ancestry haplotypes, it labels
every base pair of every sample's haplotype with a probability distribution
over the K reference ancestries. "Global" ancestry is the sample's mean
over the genome of those local calls; "local" is the per-base label.

**Three tools, three different outputs.** This report involves three
tools that each assign ancestry information to a sample.

- **FLARE** — local-ancestry inference; soft ancestry proportions per
  base, summarised to a per-sample global proportion vector.
- **RF classifier** — an upstream random-forest model on PCA loadings.
  It emits a probability vector over six superpopulations; the argmax of
  that vector is the sample's categorical **superpop label**
  (`afr`, `amr`, `eas`, `eur`, `mid`, `sas`).
- **Rye** — a supervised ADMIXTURE variant; emits a continuous per-sample
  Q matrix over five ancestries.

**Comparator vs stratifier.** When the report compares two tools' calls,
the page names the tools: the concordance page is **FLARE vs Rye**
(soft-vs-soft), the confusion and calibration pages are **FLARE vs RF**
(both produce categorical superpop labels and the matrix compares the
two tools' calls). When the report just groups samples by their
categorical superpop label (the hap-disagreement page), the page talks
about "the superpop label" — that's a grouping operation, not a tool
comparison. Rye is preferred over the RF probability vector for soft
concordance because RF's loss function pushes most of its probability
mass onto one ancestry, mechanically penalising FLARE on truly admixed
samples.

**What "cluster" means.** The AoU cohort was partitioned upstream into K
clusters (each cluster is a sample list). One FLARE run validates one
cluster on one chromosome; the report rolls up every `(cluster, chrom)`
artifact in the bundle.

**Global vs local ancestry, and `cluster_mu`.** All the concordance
metrics in this report compare per-sample _global_ proportions (FLARE's
genome-wide mean) against the comparator's per-sample probability vector.
`cluster_mu` is the mean proportion of an ancestry across the cluster's
samples; a low `cluster_mu` (< 0.01) means there is essentially no signal
for that ancestry in that cluster, so any metric computed there is
degenerate. The report μ-gates everywhere this matters: low-μ rows are
either dropped or annotated.

**Two stats you will see a lot — Pearson r and Lin's CCC.** Pearson r
measures how well two vectors _co-vary_ (rank correlation): if FLARE goes
up when the comparator goes up, r ≈ 1, regardless of whether the absolute
values agree. Lin's Concordance Correlation Coefficient (CCC) is stricter
— it asks whether the two vectors lie on the y=x line, so a systematic
bias (FLARE consistently 10 percentage points higher than the comparator)
tanks CCC even when Pearson r is near 1. The largest r-vs-CCC gap in a
cluster is the single best summary of calibration drift.
"""


def section_reading_guide() -> str:
    return READING_GUIDE_MD


# ── §2.5 Label-space conventions ─────────────────────────────────────────


LABEL_SPACE_CONVENTIONS_MD = """# Label-space conventions

Every figure in this report carries a one-line *figure tag* in the
bottom-strip footer, e.g.

```
L=SP6/MID+ | flare=>corrH | rf=>name | rye=>name
```

The tag is the **complete answer** to the question "what label space is
this graph in, and how did each tool get there?" Two charts with
identical tags are directly comparable; two charts with different tags
are *not* — even if they share the same axis labels.

## The five label spaces in play

| Space | Members (ordered) | Where it lives natively |
|---|---|---|
| `SP6` | afr, amr, eas, eur, mid, sas | the RF classifier's six classes; the canonical superpop target most figures live in |
| `SP5` | afr, amr, eas, eur, sas      | SP6 with MID removed; Rye's native target |
| `SP6.sub` | afr.1, afr.2, eur.1, …      | a subcontinental refinement of SP6, used when one tool's components fold into one SP6 label |
| `TRUTH` | anc_0, anc_1, … (per-run)  | simulator classes; not in production reports |
| `<tool>.native` | tool-defined              | FLARE panel codes, popout indices, Rye column names |

The shared space is named `SP` (for *superpopulation*) — deliberately
*not* after any single tool. The RF classifier is just one tool that
maps into `SP6` like any other; the old `RF_LABEL_ORDER` convention
wrongly implied ownership.

## How each tool gets into the target space

Each clause `<tool>=><method>` in the tag names the **matching method**
that produced that tool's mapping:

| Code | Algorithm | Operates on | Cardinality |
|---|---|---|---|
| `corrH` | Pearson correlation + Hungarian assignment | inferred allele frequencies vs reference frequencies | bijective; merges when K_inf > K_ref |
| `postS` | correlation argmax + calibration-slope override | per-sample posterior proportions vs RF probabilities | many-to-one |
| `confH` | Hungarian on a hard-call confusion matrix | per-site hard calls | bijective |
| `name`  | exact-name match (case-insensitive) | declared names | bijective |
| `manual` | analyst-supplied CSV | — | arbitrary |

In this report FLARE's components reach SP6 via **corrH** against the
1KG superpop reference. RF and Rye reach SP6 via **name** because both
already carry the canonical labels in their column / class headers.

## How label spaces are surfaced in the report

- **Cover.** The "Run identity" sub-table carries a `label_space` row
  with the full tag, the target space, and the MID flag.
- **Figure footers.** Every chart (cohort composition, concordance
  strip, calibration heatmap, confusion matrix, tract length, switch
  rate, hap-disagreement, regional Manhattan, all per-cluster figures)
  prints the tag at the bottom in a small monospace strip.
- **Subcomponent suffixes.** When a tool's K is larger than the target
  space's |L|, multiple components fold into one continental label and
  pick up dense **rank** suffixes: `afr.1` is the component most
  strongly correlated with AFR, `afr.2` the next, regardless of the raw
  EM index or the seed. (Pre-Phase-3 reports used the raw global
  index, so `afr.0 / afr.5` instead of `afr.1 / afr.2`. If you are
  cross-referencing an older PDF, the rank-based names are stable
  across reseeds; the index-based ones are not.)
- **MID handling.** Comparisons against Rye live in `SP5` and the tag
  reads `MID-`. Going from `SP6` to `SP5` is an explicit collapse with
  a declared rule (currently `drop`), never an implicit zero-fill.

The grammar, registry, naming rule, and version hash are defined in
`popout.labelspace` (`my_notes/labels/LABEL_SPACE.md`). A figure that
you cannot trace back to a specific tag is a figure you cannot defend.
"""


def section_label_space_conventions() -> str:
    return LABEL_SPACE_CONVENTIONS_MD


# ── §3. Cohort composition ────────────────────────────────────────────────


def section_cohort_composition(bundle: CohortBundle, assets_dir: Path) -> str:
    """Cohort-first composition: one large cohort bar + slim per-cluster bars."""
    path = bundle.bundle_dir / "cohort" / "cohort_global.tsv"
    if not path.exists():
        return _missing("cohort/cohort_global.tsv")
    header, rows = _read_tsv(path)
    if not rows:
        return "# Cohort composition\n\n_No per-sample data._\n"
    col = {h: i for i, h in enumerate(header)}

    # cohort_global.tsv: packed ancestry_props column (1 header col but K
    # numeric fields per body row after sample_id).
    n_meta = col["sample_id"] + 1
    flare_to_rf = _read_merged_groups_rf(bundle)

    counts: dict[tuple[str, str], dict[str, int]] = {}
    totals: dict[tuple[str, str], int] = {}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
        except (IndexError, KeyError):
            continue
        if len(r) <= n_meta:
            continue
        try:
            vals = [float(x) for x in r[n_meta:]]
        except ValueError:
            continue
        if not vals:
            continue
        primary_idx = max(range(len(vals)), key=lambda i: vals[i])
        rf = flare_to_rf.get((cid, chrom), {}).get(primary_idx)
        if rf is None:
            continue
        d = counts.setdefault((cid, chrom), {})
        d[rf] = d.get(rf, 0) + 1
        totals[(cid, chrom)] = totals.get((cid, chrom), 0) + 1

    if not counts:
        return "# Cohort composition\n\n_No mappable FLARE-component data._\n"

    # Cohort-pooled counts per RF label.
    cohort_counts: dict[str, int] = {}
    for cc, d in counts.items():
        for rf, n in d.items():
            cohort_counts[rf] = cohort_counts.get(rf, 0) + n
    cohort_total = sum(cohort_counts.values()) or 1

    cc_keys = sorted(counts.keys(), key=lambda k: -totals.get(k, 0))
    rf_set = [a for a in ANCESTRY_ORDER if a in cohort_counts] + sorted(
        {rf for rf in cohort_counts if rf not in ANCESTRY_ORDER}
    )
    cc_labels = [f"{cid}·{chrom}" for cid, chrom in cc_keys]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_clusters = len(cc_keys)
    # Two-row layout via gridspec: top = single cohort bar (tall), bottom =
    # slim per-cluster bars. Bottom-edge legend strip prevents collisions.
    fig = plt.figure(figsize=(9.0, max(4.0, 1.4 + 0.45 * n_clusters + 1.6)))
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[1.2, max(1.0, 0.4 * n_clusters), 0.55],
        hspace=0.45,
    )
    ax_cohort = fig.add_subplot(gs[0, 0])
    ax_clusters = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.axis("off")

    # ── Cohort bar ────────────────────────────────────────────────────────
    left = 0.0
    legend_handles = []
    for rf in rf_set:
        v = cohort_counts.get(rf, 0)
        if v <= 0:
            continue
        color = ancestry_color(rf)
        bar = ax_cohort.barh(0, v, left=left, color=color, edgecolor="white",
                             linewidth=0.6, height=0.7)
        legend_handles.append((rf, bar[0]))
        pct = 100.0 * v / cohort_total
        # In-segment label only if the segment is wide enough; otherwise skip
        # (the legend + supporting table carry the small ones).
        if pct >= 4.0:
            ax_cohort.text(left + v / 2, 0,
                           f"{rf}\n{int(v):,} ({pct:.1f}%)",
                           ha="center", va="center", fontsize=9,
                           color="white", fontweight="bold")
        left += v
    ax_cohort.set_xlim(0, cohort_total * 1.005)
    ax_cohort.set_ylim(-0.6, 0.6)
    ax_cohort.set_yticks([0])
    ax_cohort.set_yticklabels([f"cohort\n(n={cohort_total:,})"], fontsize=10,
                              fontweight="bold")
    ax_cohort.set_xticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax_cohort.spines[spine].set_visible(False)
    ax_cohort.set_title("Cohort sample count per FLARE primary ancestry",
                        fontsize=12, loc="left")

    # ── Per-cluster bars ──────────────────────────────────────────────────
    y = np.arange(n_clusters)
    left = np.zeros(n_clusters)
    for rf in rf_set:
        vals = np.array([counts[cc].get(rf, 0) for cc in cc_keys])
        if not vals.any():
            continue
        color = ancestry_color(rf)
        ax_clusters.barh(y, vals, left=left, color=color, edgecolor="white",
                         linewidth=0.5, height=0.72)
        left += vals
    ax_clusters.set_yticks(y)
    ax_clusters.set_yticklabels(cc_labels, fontsize=9)
    ax_clusters.invert_yaxis()
    total_max = max(totals.values()) if totals else 1
    for i, cc in enumerate(cc_keys):
        ax_clusters.text(left[i] + total_max * 0.01, i,
                         f"  n={totals.get(cc, 0):,}",
                         ha="left", va="center", fontsize=8, color="#222")
    ax_clusters.set_xlim(0, total_max * 1.12)
    ax_clusters.set_xlabel("number of samples", fontsize=10)
    ax_clusters.set_title("Decomposition by cluster · chrom (supporting view)",
                          fontsize=10, loc="left", color="#444")
    for spine in ("top", "right"):
        ax_clusters.spines[spine].set_visible(False)

    # ── Legend strip ──────────────────────────────────────────────────────
    ax_legend.legend(
        [h for _, h in legend_handles],
        [rf for rf, _ in legend_handles],
        title="ancestry",
        loc="center", ncol=min(len(legend_handles), 6),
        fontsize=9, title_fontsize=9, frameon=False,
    )

    out_png = assets_dir / "cohort_composition.png"
    _save_fig_with_tag(fig, out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    # In-this-run callout: dominant cohort ancestry, second, and any cluster
    # whose composition diverges from the cohort by >2× on some ancestry.
    cohort_fracs = {rf: cohort_counts[rf] / cohort_total for rf in cohort_counts}
    sorted_rf = sorted(cohort_fracs.items(), key=lambda kv: -kv[1])
    dom = sorted_rf[0] if sorted_rf else None
    snd = sorted_rf[1] if len(sorted_rf) > 1 else None
    divergent: list[str] = []
    for cc in cc_keys:
        cc_total = totals[cc]
        if cc_total < 50:
            continue
        for rf, cf in cohort_fracs.items():
            if cf < 0.02:
                continue
            cluster_f = counts[cc].get(rf, 0) / cc_total
            if cluster_f >= 2.0 * cf or (cf > 0 and cluster_f <= 0.5 * cf):
                ratio = cluster_f / cf if cf > 0 else float("inf")
                divergent.append(
                    f"`{cc[0]}·{cc[1]}` `{rf}` {cluster_f*100:.1f}% "
                    f"vs cohort {cf*100:.1f}% ({ratio:.1f}×)"
                )
    in_this_run: list[str] = []
    if dom:
        in_this_run.append(
            f"Cohort is dominated by `{dom[0]}` "
            f"({cohort_counts[dom[0]]:,} samples, {dom[1]*100:.1f}%)"
            + (f"; second is `{snd[0]}` ({cohort_counts[snd[0]]:,}, {snd[1]*100:.1f}%)."
               if snd else ".")
        )
    if divergent:
        in_this_run.append(
            "Clusters whose composition diverges >2× from the cohort on some "
            "ancestry: " + "; ".join(divergent[:6]) + "."
        )

    out = [
        "# Cohort composition\n",
        "**What this is.** The top bar treats all clusters as a single cohort: "
        "every sample's FLARE primary ancestry (argmax of its FLARE "
        "proportion vector) is named via `cohort/merged_groups_rf.tsv`, "
        "and the segments show the cohort-wide count per ancestry. The "
        "bottom strip is the same decomposition broken down by cluster · "
        "chrom, sorted by sample count, for readers who want to see how "
        "each cluster contributes.\n",
        "**How to read this.** Read the top bar first — that is the cohort's "
        "ancestral makeup as FLARE sees it. The cluster bars below show how "
        "evenly each ancestry is distributed across the clusters that built "
        "that cohort: roughly proportional widths means the cohort number is "
        "stable; lopsided cluster widths (one cluster holding most of an "
        "ancestry) means the cohort number is driven by that one cluster and "
        "downstream per-ancestry stats for it should be read with that in "
        "mind.\n",
        _embed_image(out_png),
        "**In this run.** " + " ".join(in_this_run) + "\n",
    ]
    return "\n".join(out) + "\n"


# ── §4. Concordance (Pearson r + Lin's CCC) ───────────────────────────────


def section_concordance(bundle: CohortBundle, assets_dir: Path) -> str:
    """Cohort-pooled bars per ancestry (r + CCC), per-cluster tick overlay."""
    path = bundle.bundle_dir / "cohort" / "concordance_metrics.tsv"
    if not path.exists():
        return _missing("cohort/concordance_metrics.tsv")
    header, rows = _read_tsv(path)
    if not rows:
        return ""
    col = {h: i for i, h in enumerate(header)}

    # anc → list of (cc, n, mu, r, ccc) for μ-gated rows.
    by_anc: dict[str, list[tuple[str, int, float, float, float]]] = {}
    for r in rows:
        try:
            cc = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
            anc = r[col["ancestry"]]
            mu = _safe_float(r[col["cluster_mu"]])
            n = int(float(r[col["n_samples"]]))
            pr = _safe_float(r[col["pearson_r"]])
            cccv = _safe_float(r[col["ccc"]])
        except (IndexError, KeyError, ValueError):
            continue
        if mu is None or mu < 0.01:
            continue
        if pr is None and cccv is None:
            continue
        by_anc.setdefault(anc, []).append((
            cc, n, mu,
            pr if pr is not None else float("nan"),
            cccv if cccv is not None else float("nan"),
        ))
    if not by_anc:
        return ("# FLARE vs Rye concordance\n\n"
                "_No (cluster · ancestry) pair has `cluster_mu ≥ 0.01`; "
                "nothing to plot._\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(
        by_anc.keys(),
        key=lambda a: ANCESTRY_ORDER.index(a) if a in ANCESTRY_ORDER else 99,
    )

    # Cohort-pool per ancestry: n_samples-weighted mean of r and CCC.
    pooled_r: dict[str, float | None] = {}
    pooled_ccc: dict[str, float | None] = {}
    for anc in labels:
        items = by_anc[anc]
        pooled_r[anc] = n_weighted_mean([(n, pr) for _, n, _, pr, _ in items])
        pooled_ccc[anc] = n_weighted_mean([(n, cc) for _, n, _, _, cc in items])

    # Per-cluster scatter values for tick overlay.
    cluster_r: dict[str, list[float]] = {
        anc: [pr for _, _, _, pr, _ in by_anc[anc] if pr == pr] for anc in labels
    }
    cluster_ccc: dict[str, list[float]] = {
        anc: [cc for _, _, _, _, cc in by_anc[anc] if cc == cc] for anc in labels
    }

    # Auto-zoom x range to data and reference lines.
    all_r = [v for vals in cluster_r.values() for v in vals] + \
            [v for v in pooled_r.values() if v is not None]
    all_c = [v for vals in cluster_ccc.values() for v in vals] + \
            [v for v in pooled_ccc.values() if v is not None]
    x_lo = max(0.0, min([*all_r, *all_c, 0.90]) - 0.04)
    x_hi = 1.01

    # Gridspec: two chart panels + a dedicated legend strip.
    n_anc = len(labels)
    chart_h = max(2.0, 0.55 * n_anc + 0.6)
    fig = plt.figure(figsize=(8.5, 2 * chart_h + 1.4))
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[chart_h, chart_h, 0.5],
        hspace=0.35,
    )
    ax_r = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_legend = fig.add_subplot(gs[2, 0]); ax_legend.axis("off")

    def _draw(ax, pooled: dict[str, float | None], per_cluster: dict[str, list[float]],
              ref: float, title: str, xlabel: str) -> None:
        y = list(range(n_anc))
        for i, anc in enumerate(labels):
            color = ancestry_color(anc)
            v = pooled.get(anc)
            if v is not None:
                ax.barh(i, v, color=color, edgecolor="white", linewidth=0.6,
                        height=0.62, zorder=2)
                ax.text(v + (x_hi - x_lo) * 0.005, i,
                        f"{v:.3f}", ha="left", va="center",
                        fontsize=9, color="#222")
            overlay_ticks(ax, i, per_cluster.get(anc, []),
                          color="#111", tick_height=0.46, lw=1.6, alpha=0.9)
        ax.axvline(ref, color="#666", linestyle="--", linewidth=0.9, zorder=1)
        ax.text(ref, n_anc - 0.4, f" PLAN2.md §2.2: {ref:.2f}",
                fontsize=8, color="#555", ha="left", va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11, loc="left")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    _draw(ax_r, pooled_r, cluster_r, 0.95,
          "Pearson r — rank linearity (cohort-pooled, n-weighted)",
          "Pearson r")
    _draw(ax_c, pooled_ccc, cluster_ccc, 0.90,
          "Lin's CCC — linearity + calibration (cohort-pooled, n-weighted)",
          "Lin's CCC")

    # Legend strip: explains the two visual elements.
    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    tick_proxy = plt.Line2D([0], [0], color="#111", linewidth=1.6)
    ax_legend.legend(
        [bar_proxy, tick_proxy],
        ["bar length = cohort-pooled value (n-weighted across clusters)",
         "vertical tick = one cluster's value"],
        loc="center", ncol=2, fontsize=9, frameon=False,
    )

    out_png = assets_dir / "concordance_strip.png"
    _save_fig_with_tag(fig, out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    # In-this-run callout (uses cohort-pooled values).
    pool_r_pairs = [(a, v) for a, v in pooled_r.items() if v is not None]
    pool_c_pairs = [(a, v) for a, v in pooled_ccc.items() if v is not None]
    # Calibration drift candidates: per (cluster, ancestry) r − CCC.
    gaps: list[tuple[str, float]] = []
    for anc, items in by_anc.items():
        for cc, _n, _mu, pr, cccv in items:
            if pr == pr and cccv == cccv:
                gaps.append((f"{cc}·{anc}", pr - cccv))
    top_gap = topn(gaps, n=3)
    in_this_run = [
        f"Cohort-pooled r per ancestry: {fmt_topn(pool_r_pairs, n=len(labels))}.",
        f"Cohort-pooled CCC per ancestry: {fmt_topn(pool_c_pairs, n=len(labels))}.",
    ]
    if top_gap:
        in_this_run.append(
            "Largest per-cluster r−CCC gap (calibration drift candidates): "
            + ", ".join(f"`{lab}` ({v:+.3f})" for lab, v in top_gap)
            + " — positive gap = FLARE tracks Rye in rank but is offset in absolute value."
        )

    # Supporting table: per (cluster · chrom · ancestry).
    mu_rows: list[tuple[str, str, str, str, str, str]] = []
    for anc in labels:
        for cc, n, mu, pr, cccv in sorted(by_anc[anc], key=lambda t: t[0]):
            mu_rows.append((
                anc, cc, f"{mu:.3f}", f"{n:,}",
                "—" if pr != pr else f"{pr:.3f}",
                "—" if cccv != cccv else f"{cccv:.3f}",
            ))
    mu_table = _md_table(
        ["ancestry", "cluster · chrom", "cluster_mu", "n_samples",
         "Pearson r", "Lin's CCC"],
        mu_rows,
    )

    out = [
        "# FLARE vs Rye concordance\n",
        "**What this is.** For each ancestry, two stats describing how "
        "FLARE's per-sample global proportion lines up against Rye's "
        "per-sample Q value for that ancestry. **Pearson r** measures rank "
        "co-variation (do they move together?). **Lin's CCC** additionally "
        "measures absolute agreement (do they lie on y=x?). A systematic "
        "offset — say FLARE consistently calling 10 percentage points "
        "higher than Rye — leaves r near 1 but tanks CCC. Rye is used "
        "instead of the superpop label here because both Rye and FLARE "
        "produce soft proportions, so the comparison is apples-to-apples; "
        "the categorical superpop label would penalise FLARE on truly "
        "admixed samples mechanically.\n",
        "**How to read this chart.** Each row is one ancestry, present in "
        "the cohort. The horizontal bar is the cohort-pooled value, computed "
        "as the n_samples-weighted mean across clusters (rows with "
        "`cluster_mu < 0.01` are dropped before pooling — they are "
        "degenerate). The vertical black ticks on each bar mark each "
        "cluster's individual value — they show how tightly the clusters "
        "agree on that pooled number. The dashed vertical line is a "
        "reference value from PLAN2.md §2.2 (0.95 for r, 0.90 for CCC) — a "
        "guideline, not a hard cutoff. The supporting table below carries "
        "per-cluster `cluster_mu` and `n_samples` so a small or low-μ row "
        "is easy to spot.\n",
        _embed_image(out_png),
        "**In this run.** " + " ".join(in_this_run) + "\n",
        "",
        mu_table,
    ]
    return "\n".join(out) + "\n"



# ── §6. Calibration matrices ──────────────────────────────────────────────


def _cluster_mu_from_concordance(bundle: CohortBundle) -> dict[tuple[str, str, str], float]:
    """(cluster_id, chrom, ancestry) → cluster_mu, from concordance_metrics.tsv."""
    path = bundle.bundle_dir / "cohort" / "concordance_metrics.tsv"
    out: dict[tuple[str, str, str], float] = {}
    if not path.exists():
        return out
    header, rows = _read_tsv(path)
    if not rows:
        return out
    col = {h: i for i, h in enumerate(header)}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            anc = r[col["ancestry"]]
            mu = _safe_float(r[col["cluster_mu"]])
        except (IndexError, KeyError):
            continue
        if mu is not None:
            out[(cid, chrom, anc)] = mu
    return out


def section_calibration(bundle: CohortBundle, assets_dir: Path) -> str:
    """One cohort-pooled (FLARE component × RF-assigned superpop label) calibration matrix."""
    path = bundle.bundle_dir / "cohort" / "calibration_slope.tsv"
    if not path.exists():
        return _missing("cohort/calibration_slope.tsv")
    header, rows = _read_tsv(path)
    if not rows:
        return ""
    col = {h: i for i, h in enumerate(header)}

    # (anc, rf) -> list of (cluster_mu, slope, max_cal) per cluster·chrom.
    cells: dict[tuple[str, str], list[tuple[float, float | None, float | None]]] = {}
    anc_set: list[str] = []
    rf_order: list[str] = []
    mu_map = _cluster_mu_from_concordance(bundle)
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            anc = r[col["ancestry_name"]]
            rf = r[col["rf_label"]]
            slope = _safe_float(r[col["slope"]])
            max_cal = _safe_float(r[col["max_cal"]])
        except (IndexError, KeyError):
            continue
        # Weight: cluster_mu for that FLARE component in that cluster (≈
        # representation of the ancestry in the cluster). Default 1.0 when
        # μ not available.
        mu = mu_map.get((cid, chrom, anc.split(".")[0]), 1.0)
        if mu is None:
            mu = 1.0
        cells.setdefault((anc, rf), []).append((mu, slope, max_cal))
        if anc not in anc_set:
            anc_set.append(anc)
        if rf not in rf_order:
            rf_order.append(rf)
    if not cells:
        return ""

    # Order FLARE components: known ancestries first (in canonical order),
    # then numbered sub-components alphabetically.
    def _anc_key(a: str) -> tuple[int, str]:
        base = a.split(".")[0]
        primary = ANCESTRY_ORDER.index(base) if base in ANCESTRY_ORDER else 99
        return (primary, a)
    anc_rows = sorted(anc_set, key=_anc_key)
    rf_order = [r for r in ANCESTRY_ORDER if r in rf_order] + \
               sorted(r for r in rf_order if r not in ANCESTRY_ORDER)

    # Cohort pool per (anc, rf): n-weighted (μ-weighted) mean max_cal and
    # median slope across clusters that defined it.
    pool_max: dict[tuple[str, str], float | None] = {}
    pool_slope: dict[tuple[str, str], float | None] = {}
    pool_n: dict[tuple[str, str], int] = {}
    for key, recs in cells.items():
        pool_max[key] = n_weighted_mean([(w, m) for w, _, m in recs if m is not None])
        slope_vals = sorted(s for _, s, _ in recs if s is not None)
        pool_slope[key] = (slope_vals[len(slope_vals) // 2]
                           if slope_vals else None)
        pool_n[key] = sum(1 for w, _, m in recs if m is not None)

    # μ per FLARE component, cohort-averaged across clusters that defined it.
    mu_by_anc: dict[str, float | None] = {}
    for anc in anc_rows:
        base = anc.split(".")[0]
        per_cluster_mu = [mu_map[k] for k in mu_map
                          if k[2] == base and mu_map[k] is not None]
        mu_by_anc[anc] = (sum(per_cluster_mu) / len(per_cluster_mu)
                          if per_cluster_mu else None)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    n_anc = len(anc_rows)
    n_rf = len(rf_order)
    M = np.full((n_anc, n_rf), np.nan)
    over1_cells: list[tuple[int, int]] = []
    for i, anc in enumerate(anc_rows):
        for j, rf in enumerate(rf_order):
            v = pool_max.get((anc, rf))
            if v is None:
                continue
            if v > 1.0:
                over1_cells.append((i, j))
                M[i, j] = 1.0
            else:
                M[i, j] = v

    # Layout: μ strip on the left, then heatmap, then a thin colorbar.
    fig = plt.figure(figsize=(2.8 + 0.85 * n_rf + 1.0,
                              max(3.5, 0.55 * n_anc + 1.5)))
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.2, max(n_rf * 0.85, 3.0), 0.5],
        wspace=0.05,
    )
    ax_mu = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    # ── μ strip ───────────────────────────────────────────────────────────
    bar_max_w = 1.0
    for i, anc in enumerate(anc_rows):
        mu = mu_by_anc.get(anc)
        if mu is None:
            ax_mu.text(0.5, i, "μ=—", ha="center", va="center",
                       fontsize=8, color="#666")
            continue
        w = max(0.02, bar_max_w * min(mu, 1.0))
        ax_mu.add_patch(mpatches.Rectangle(
            (0.0, i - 0.32), w, 0.64,
            facecolor=ancestry_color(anc), edgecolor="#444444", linewidth=0.4,
        ))
        ax_mu.text(bar_max_w + 0.06, i, f"μ={mu:.2f}",
                   ha="left", va="center", fontsize=8, color="#222")
    ax_mu.set_xlim(0, bar_max_w + 0.7)
    ax_mu.set_ylim(n_anc - 0.5, -0.5)
    ax_mu.set_yticks(range(n_anc))
    ax_mu.set_yticklabels(anc_rows, fontsize=9)
    ax_mu.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax_mu.spines[spine].set_visible(False)
    ax_mu.set_title("cohort μ", fontsize=9, loc="left", color="#444")

    # ── Heatmap ───────────────────────────────────────────────────────────
    im = ax_heat.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax_heat.set_xticks(range(n_rf))
    ax_heat.set_xticklabels(rf_order, fontsize=10)
    ax_heat.set_xlabel("RF-assigned superpop label", fontsize=10)
    ax_heat.set_yticks(range(n_anc))
    ax_heat.set_yticklabels([])
    ax_heat.set_title("Cohort-pooled calibration  ·  color = μ-weighted mean max_cal  ·  text = max_cal (m = median slope)",
                      fontsize=11, loc="left")
    for i in range(n_anc):
        for j in range(n_rf):
            v = pool_max.get((anc_rows[i], rf_order[j]))
            if v is None:
                continue
            s = pool_slope.get((anc_rows[i], rf_order[j]))
            txt_color = "white" if min(v, 1.0) < 0.55 else "black"
            if s is not None:
                txt = f"{v:.2f}\nm={s:.2f}"
            else:
                txt = f"{v:.2f}"
            ax_heat.text(j, i, txt, ha="center", va="center",
                         fontsize=8, color=txt_color, linespacing=0.95)
    for (i, j) in over1_cells:
        ax_heat.add_patch(mpatches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1, fill=False,
            hatch="///", edgecolor="#222", linewidth=0.6,
        ))

    fig.colorbar(im, cax=ax_cbar, label="max_cal (0–1)")

    out_png = assets_dir / "calibration.png"
    _save_fig_with_tag(fig, out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    # In-this-run: largest off-diagonal cohort-pooled cells.
    off_diag: list[tuple[str, float]] = []
    for (anc, rf), v in pool_max.items():
        if v is None:
            continue
        base = anc.split(".")[0]
        if base == rf:
            continue
        off_diag.append((f"FLARE={anc} → RF={rf}", v))
    top_off = topn(off_diag, n=3)
    n_over1 = len(over1_cells)
    in_this_run: list[str] = []
    if top_off:
        in_this_run.append(
            "Largest cohort-pooled off-diagonal cells (ancestry leakage "
            "candidates): " + ", ".join(f"{lab} = {v:.2f}" for lab, v in top_off)
            + "."
        )
    if n_over1:
        in_this_run.append(
            f"{n_over1} cell(s) have pooled `max_cal > 1` (hatched) — "
            "usually means the reference panel under-represents that ancestry "
            "and the calibration response is amplified beyond the input "
            "proportion."
        )

    out = [
        "# Calibration\n",
        "**What this is.** For every (FLARE component, RF-assigned superpop "
        "label) pair, two numbers describe how that FLARE component "
        "reacts to the RF probability for that label: **`max_cal`** is "
        "the largest mean FLARE proportion across binned RF probability "
        "(the strongest response of the component to the RF tool's "
        "signal for that label) and **`slope` (m)** is the linear-"
        "regression slope of FLARE proportion on RF probability across "
        "samples (only defined where both axes have enough variance). A "
        "well-behaved FLARE component answers exactly one RF label on the "
        "diagonal with `m ≈ 1`.\n",
        "**How to read this chart.** One row per FLARE component (cohort μ "
        "shown on the left as a small bar); one column per RF-assigned "
        "superpop label. Cells are the cohort-pooled `max_cal` "
        "(μ-weighted mean across clusters that defined that pair); the "
        "subscript `m=` is the cohort-median slope across the same "
        "clusters. A bright diagonal cell with `m ≈ 1` = the FLARE "
        "component is calibrated against the RF tool's call for that "
        "label. A bright off-diagonal = the FLARE component is being "
        "driven by a different RF label (leakage). Hatched cells = "
        "`max_cal > 1`, which usually means the reference panel "
        "under-represents that ancestry. Components with a tiny μ on the "
        "left are mostly noise regardless of cell colour.\n",
        _embed_image(out_png),
        "**In this run.** " + " ".join(in_this_run) + "\n",
    ]
    return "\n".join(out) + "\n"


# ── §7. Cohort-summed FLARE vs RF confusion ───────────────────────────────


def section_confusion(bundle: CohortBundle, assets_dir: Path) -> str:
    path = bundle.bundle_dir / "cohort" / "confusion_rf.tsv"
    if not path.exists():
        return _missing("cohort/confusion_rf.tsv")
    header, rows = _read_tsv(path)
    if not rows:
        return ""
    col = {h: i for i, h in enumerate(header)}

    cell: dict[tuple[str, str], int] = {}
    rf_labels: list[str] = []
    flare_calls: list[str] = []
    for r in rows:
        try:
            rf = r[col["rf_label"]]
            fc = r[col["flare_call"]]
            n = int(r[col["n"]])
        except (IndexError, KeyError, ValueError):
            continue
        cell[(rf, fc)] = cell.get((rf, fc), 0) + n
        if rf not in rf_labels:
            rf_labels.append(rf)
        if fc not in flare_calls:
            flare_calls.append(fc)
    rf_labels.sort(key=lambda x: ANCESTRY_ORDER.index(x) if x in ANCESTRY_ORDER else 99)
    flare_calls.sort(key=lambda x: ANCESTRY_ORDER.index(x) if x in ANCESTRY_ORDER else 99)

    import numpy as np
    M = np.zeros((len(rf_labels), len(flare_calls)), dtype=float)
    for i, rf in enumerate(rf_labels):
        for j, fc in enumerate(flare_calls):
            M[i, j] = cell.get((rf, fc), 0)
    row_sums = M.sum(axis=1, keepdims=True)
    col_sums = M.sum(axis=0, keepdims=True)
    recall = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
    precision = np.divide(M, col_sums, out=np.zeros_like(M), where=col_sums > 0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Layout: a gridspec with a dedicated precision row beneath the heatmap
    # and a dedicated recall column to its right, so margin labels never
    # collide with tick labels.
    fig = plt.figure(figsize=(max(7.5, 0.85 * len(flare_calls) + 4.5),
                              max(5.0, 0.75 * len(rf_labels) + 3.0)),
                     constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.18, 0.05],
                          height_ratios=[1.0, 0.10],
                          wspace=0.08, hspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_recall = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_prec = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_corner = fig.add_subplot(gs[1, 1])

    im = ax.imshow(recall, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(flare_calls)))
    ax.set_xticklabels(flare_calls, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rf_labels)))
    ax.set_yticklabels(rf_labels, fontsize=10)
    ax.set_ylabel("RF call (reference)", fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            n = int(M[i, j])
            txt = f"{recall[i, j]:.2f}\n({n:,})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if recall[i, j] > 0.55 else "black")
            base_rf = rf_labels[i].split(".")[0]
            base_fc = flare_calls[j].split(".")[0]
            if base_rf != base_fc and recall[i, j] > 0.05:
                rect = mpatches.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                          edgecolor="#c62828", linewidth=1.6)
                ax.add_patch(rect)
    # Right-hand recall column (its own axis).
    ax_recall.set_xlim(0, 1)
    ax_recall.set_xticks([])
    ax_recall.tick_params(left=False, labelleft=False)
    ax_recall.set_title("recall", fontsize=9, pad=4)
    for spine in ("top", "right", "bottom", "left"):
        ax_recall.spines[spine].set_visible(False)
    for i in range(M.shape[0]):
        if i < recall.shape[1]:
            v = recall[i, i]
            ax_recall.text(0.5, i, f"{v:.2f}" if v == v else "—",
                           ha="center", va="center", fontsize=10, color="#222")
        else:
            ax_recall.text(0.5, i, "—", ha="center", va="center", fontsize=10, color="#222")
    # Bottom precision row (its own axis).
    ax_prec.set_ylim(0, 1)
    ax_prec.set_yticks([])
    ax_prec.tick_params(bottom=False, labelbottom=False)
    for spine in ("top", "right", "bottom", "left"):
        ax_prec.spines[spine].set_visible(False)
    ax_prec.set_ylabel("precision", fontsize=9, rotation=0, ha="right", va="center", labelpad=12)
    for j in range(M.shape[1]):
        v = precision[j, j] if j < precision.shape[0] else float("nan")
        ax_prec.text(j, 0.5, f"{v:.2f}" if v == v else "—",
                     ha="center", va="center", fontsize=10, color="#222")
    ax_prec.set_xlim(ax.get_xlim())
    # Corner spacer (axes off).
    ax_corner.axis("off")
    fig.colorbar(im, cax=ax_cbar, label="row-normalized recall")
    fig.suptitle("Cohort-summed FLARE vs RF confusion matrix\n"
                 "(rows = RF argmax superpop label; cols = FLARE argmax  ·  "
                 "diagonals = correct  ·  red-bordered cells = systematic confusion > 5%)",
                 fontsize=11)
    ax.set_xlabel("FLARE hard call", fontsize=10)
    out_png = assets_dir / "confusion.png"
    _save_fig_with_tag(fig, out_png, dpi=_DPI)
    plt.close(fig)

    diag_recall = {rf_labels[i]: recall[i, i] for i in range(min(M.shape))}
    worst_recall = topn(list(diag_recall.items()), n=3, reverse=False)
    # Top systematic confusion = max off-diagonal recall.
    off_pairs: list[tuple[str, float]] = []
    for i, rf in enumerate(rf_labels):
        for j, fc in enumerate(flare_calls):
            if rf.split(".")[0] == fc.split(".")[0]:
                continue
            if recall[i, j] > 0.02:
                off_pairs.append((f"RF={rf} → FLARE={fc} (n={int(M[i, j]):,})", recall[i, j]))
    top_conf = topn(off_pairs, n=3)
    in_this_run = [
        ("Lowest-recall RF calls: " + fmt_topn(worst_recall, n=3, reverse=False, val_fmt="{:.2f}") + ".")
        if worst_recall else "",
        ("Top systematic off-diagonal: "
         + ", ".join(f"{lab} ({v:.2f})" for lab, v in top_conf) + ".")
        if top_conf else "No off-diagonal cell with > 2% recall.",
    ]

    out = [
        "# FLARE vs RF — cohort confusion matrix\n",
        "**What this is.** This page compares two tools that both assign a "
        "categorical superpop label per sample: the **RF classifier** (its "
        "argmax over its superpop probability vector) and **FLARE** (its "
        "argmax over its global ancestry proportion vector). The cohort "
        "table sums the per-cluster (RF call, FLARE call) counts; each "
        "row is the RF reference call, each column the FLARE call. The "
        "cell carries the cohort-wide count and the row-normalised recall "
        "fraction; diagonals are agreement.\n",
        "**How to read this chart.** Bright diagonal + high row-recall = "
        "FLARE's call agrees with the RF call cohort-wide for that "
        "superpop label. A bright off-diagonal cell, **outlined in red**, "
        "is a systematic disagreement: samples the RF classifier put in "
        "one label are being placed by FLARE in a different label. "
        "Compare row-recall (right margin: diag / row total) with "
        "column-precision (bottom margin: diag / column total). A label "
        "with high recall but low precision means FLARE over-calls it "
        "relative to RF.\n",
        _embed_image(out_png),
        "**In this run.** " + " ".join(s for s in in_this_run if s) + "\n",
    ]
    return "\n".join(out) + "\n"


# ── §8. Tract length ──────────────────────────────────────────────────────


def section_tract_length(bundle: CohortBundle, assets_dir: Path) -> str:
    """Cohort-pooled mean tract length per ancestry as horizontal bar + ticks."""
    path = bundle.bundle_dir / "cohort" / "tract_length_stats.tsv"
    if not path.exists():
        return _missing("cohort/tract_length_stats.tsv")
    header, rows = _read_tsv(path)
    if not rows:
        return ""
    col = {h: i for i, h in enumerate(header)}

    # anc → list of (cc, n_tracts, mean_Mb, implied_T, model_T)
    by_anc: dict[str, list[tuple[str, int, float, float | None, float | None]]] = {}
    for r in rows:
        try:
            cc = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
            name = r[col["ancestry_name"]]
            n_tracts = int(float(r[col["n_tracts"]]))
            mean_mb = _safe_float(r[col["mean_Mb"]])
            implied_t = _safe_float(r[col["implied_T_gen"]])
            model_t = _safe_float(r[col["model_T_gen"]])
        except (IndexError, KeyError, ValueError):
            continue
        if mean_mb is None or mean_mb <= 0:
            continue
        by_anc.setdefault(name, []).append((cc, n_tracts, mean_mb, implied_t, model_t))
    if not by_anc:
        return ""

    K_ANC = 5  # assumed reference panel ancestries
    labels = sorted(
        by_anc.keys(),
        key=lambda a: ANCESTRY_ORDER.index(a.split(".")[0])
        if a.split(".")[0] in ANCESTRY_ORDER else 99,
    )

    # Cohort pool per ancestry: n_tracts-weighted mean of mean_Mb.
    pooled_mb: dict[str, float | None] = {}
    pooled_n: dict[str, int] = {}
    model_ref_mb: dict[str, float] = {}
    for anc in labels:
        items = by_anc[anc]
        pooled_mb[anc] = n_weighted_mean([(n, mb) for _, n, mb, _, _ in items])
        pooled_n[anc] = sum(n for _, n, _, _, _ in items)
        # Reference: median cluster's model_T_gen → expected mean tract Mb
        # under 100/(T*K) (cM → Mb at chr1 ≈ 1:1 coarse approximation).
        models = [t for _, _, _, _, t in items if t is not None and t > 0]
        if models:
            med_t = sorted(models)[len(models) // 2]
            model_ref_mb[anc] = 100.0 / (med_t * K_ANC)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_anc = len(labels)
    chart_h = max(2.6, 0.7 * n_anc + 0.6)
    fig = plt.figure(figsize=(9.0, chart_h + 0.8))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[chart_h, 0.5], hspace=0.3,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0]); ax_legend.axis("off")

    # X scale: log because tract length spans orders of magnitude per ancestry.
    ax.set_xscale("log")
    # Compute x extent.
    all_x = [v for _, _, mb, _, _ in (it for items in by_anc.values() for it in items)
             for v in [mb]]
    all_x += [v for v in pooled_mb.values() if v is not None]
    all_x += list(model_ref_mb.values())
    x_lo = max(0.1, min(all_x) * 0.7) if all_x else 0.1
    x_hi = (max(all_x) * 1.5) if all_x else 100.0

    for i, anc in enumerate(labels):
        color = ancestry_color(anc)
        v = pooled_mb.get(anc)
        if v is not None:
            ax.barh(i, v - x_lo, left=x_lo, color=color, edgecolor="white",
                    linewidth=0.6, height=0.62, zorder=2)
            ax.text(v * 1.04, i, f"{v:.2f} Mb",
                    ha="left", va="center", fontsize=9, color="#222")
        # Per-cluster ticks at exact empirical mean.
        per_cluster = [mb for _, _, mb, _, _ in by_anc[anc]]
        overlay_ticks(ax, i, per_cluster, color="#111",
                      tick_height=0.48, lw=1.5, alpha=0.95)
        # Model expectation as a thin vertical line on the same row.
        if anc in model_ref_mb:
            yref = model_ref_mb[anc]
            ax.vlines(yref, i - 0.34, i + 0.34, color="#666", linewidth=1.4,
                      linestyle="--", zorder=4)
    ax.set_yticks(range(n_anc))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("mean tract length (Mb, log scale)", fontsize=10)
    ax.set_title("Tract length per ancestry  ·  bar = cohort-pooled (n_tracts-weighted)  ·  "
                 "ticks = per-cluster empirical means  ·  dashed = model expectation",
                 fontsize=11, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    tick_proxy = plt.Line2D([0], [0], color="#111", linewidth=1.5)
    model_proxy = plt.Line2D([0], [0], color="#666", linewidth=1.4, linestyle="--")
    ax_legend.legend(
        [bar_proxy, tick_proxy, model_proxy],
        ["cohort-pooled empirical mean (n_tracts-weighted)",
         "per-cluster empirical mean",
         "model expectation = 100 / (median model_T_gen × K=5)"],
        loc="center", ncol=3, fontsize=9, frameon=False,
    )

    out_png = assets_dir / "tract_length.png"
    _save_fig_with_tag(fig, out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    # Supporting table.
    table_rows: list[tuple[str, str, str, str]] = []
    for anc in labels:
        v = pooled_mb.get(anc)
        ref = model_ref_mb.get(anc)
        table_rows.append((
            anc,
            f"{pooled_n[anc]:,}",
            f"{v:.2f}" if v is not None else "—",
            f"{ref:.2f}" if ref is not None else "—",
        ))
    pool_table = _md_table(
        ["ancestry", "n_tracts (cohort)", "cohort mean_Mb", "model expectation (Mb)"],
        table_rows,
    )

    # In-this-run: biggest deviation of cohort-pooled mean from model.
    devs: list[tuple[str, float]] = []
    for anc, v in pooled_mb.items():
        ref = model_ref_mb.get(anc)
        if v is None or ref is None:
            continue
        devs.append((f"`{anc}` (empirical {v:.2f} Mb vs model {ref:.2f} Mb)",
                     abs(v - ref)))
    top_dev = topn(devs, n=3)
    in_this_run = [
        ("Largest cohort-pooled deviation from the model expectation: "
         + ", ".join(f"{lab} (Δ={v:.2f} Mb)" for lab, v in top_dev) + ".")
        if top_dev else "Cohort-pooled means sit close to the model expectation for every ancestry.",
        "A cohort bar much **shorter** than the dashed model line = over-segmentation "
        "(FLARE switching more than the model predicts); much **longer** = "
        "under-segmentation (FLARE missing switches the model expects).",
    ]

    out = [
        "# Tract length per ancestry\n",
        "**What this is.** Per-ancestry mean tract length describes FLARE's "
        "segmentation behaviour. Under a simple exponential admixture model, "
        "mean tract length in cM is roughly 100 / (T_gen × K_ancestries); "
        "the cluster's own `model_T_gen` (seeded into FLARE) therefore "
        "implies an expected mean. Empirical means much shorter than that = "
        "FLARE is over-segmenting (fast switches that look like LAI noise). "
        "Much longer = under-segmenting (FLARE missing real switches).\n",
        "**How to read this chart.** Each row is one FLARE ancestry "
        "component. The horizontal bar is the cohort-pooled mean tract "
        "length (weighted by `n_tracts` across clusters); the small vertical "
        "ticks on the bar mark each cluster's empirical mean — when they "
        "cluster tightly the cohort number is stable. The short dashed "
        "vertical mark is the model expectation derived from the median "
        "cluster's `model_T_gen` for that ancestry. Read the bar versus the "
        "dashed mark first; the per-cluster ticks tell you how united the "
        "clusters are behind that picture.\n",
        _embed_image(out_png),
        "**In this run.** " + " ".join(in_this_run) + "\n",
        "",
        pool_table,
    ]
    return "\n".join(out) + "\n"


# ── §9. Switch rate (forest layout) ───────────────────────────────────────


def section_switch_rate(bundle: CohortBundle, assets_dir: Path) -> str:
    path = bundle.bundle_dir / "cohort" / "switch_rate_stats.tsv"
    if not path.exists():
        return _missing("cohort/switch_rate_stats.tsv")
    header, rows = _read_tsv(path)
    if not rows:
        return ""
    col = {h: i for i, h in enumerate(header)}

    items: list[tuple[str, int, float, float, float, float, float]] = []
    # (label, n_haplotypes, min, median, mean, p99, max)
    for r in rows:
        try:
            lab = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
            n_hap = int(float(r[col["n_haplotypes"]]))
            mn = float(r[col["min"]])
            med = float(r[col["median"]])
            mean = float(r[col["mean"]])
            p99 = float(r[col["p99"]])
            mx = float(r[col["max"]])
        except (IndexError, KeyError, ValueError):
            continue
        items.append((lab, n_hap, mn, med, mean, p99, mx))
    if not items:
        return ""
    items.sort(key=lambda t: -t[4])  # sort by mean desc

    # Cohort pool (n_haplotypes-weighted mean / median for the central stats;
    # extremes are min / max across the whole cohort).
    cohort_mean = n_weighted_mean([(it[1], it[4]) for it in items]) or 0.0
    cohort_med = n_weighted_mean([(it[1], it[3]) for it in items]) or 0.0
    cohort_min = min(it[2] for it in items)
    cohort_p99 = max(it[5] for it in items)  # cohort-wide p99 ≈ max of cluster p99
    cohort_max = max(it[6] for it in items)
    cohort_n_hap = sum(it[1] for it in items)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(items)
    # Rows: cohort row at the top (bold) + n cluster rows below.
    rows_total = 1 + n
    chart_h = max(2.6, 0.4 * rows_total + 1.0)
    fig = plt.figure(figsize=(9.5, chart_h + 0.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[chart_h, 0.45], hspace=0.3,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0]); ax_legend.axis("off")

    # X scale.
    x_max = max(cohort_max, max(it[6] for it in items)) * 1.05

    def _draw_row(y: int, mn: float, med: float, mean: float, p99: float,
                  mx: float, *, bold: bool) -> None:
        line_color = "#222" if bold else "#bbbbbb"
        lw = 2.4 if bold else 1.2
        ax.hlines(y, mn, p99, color=line_color, linewidth=lw, zorder=2 if bold else 1)
        # Mean: vertical tick (no circle).
        ax.vlines(mean, y - 0.30, y + 0.30,
                  color="#3366A8" if bold else "#5a8acb",
                  linewidth=2.2 if bold else 1.4, zorder=5)
        # Median: small hollow square.
        s = 90 if bold else 50
        ax.scatter(med, y, marker="s", facecolor="white",
                   edgecolor=line_color, linewidth=1.4 if bold else 0.9,
                   s=s, zorder=4)
        # Max: ✕ in red.
        ax.scatter(mx, y, marker="x",
                   color="#c62828" if bold else "#d97a7a",
                   s=110 if bold else 60, linewidth=2.4 if bold else 1.4,
                   zorder=4)

    # Row 0 = cohort (always at the top).
    _draw_row(0, cohort_min, cohort_med, cohort_mean, cohort_p99, cohort_max,
              bold=True)
    # Per-cluster rows.
    for i, (_lab, _n, mn, med, mean, p99, mx) in enumerate(items, start=1):
        _draw_row(i, mn, med, mean, p99, mx, bold=False)

    yticks = [0] + list(range(1, n + 1))
    yticklabels = [f"cohort\n(n_hap={cohort_n_hap:,})"] + [it[0] for it in items]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    # Bold cohort label.
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.set_xlabel("ancestry switches per haplotype", fontsize=10)
    ax.set_title("Switch rate  ·  bold top row = cohort  ·  dim rows = per cluster · chrom",
                 fontsize=11, loc="left")
    # Annotate cohort stats inline at the right edge.
    ax.text(x_max * 0.998, 0,
            f"  mean {cohort_mean:.1f}  ·  median {cohort_med:.1f}  ·  "
            f"p99 {cohort_p99:.0f}  ·  max {cohort_max:.0f}",
            ha="right", va="center", fontsize=8.5, color="#222",
            fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend strip.
    line_proxy = plt.Line2D([0], [0], color="#222", linewidth=2.4)
    mean_proxy = plt.Line2D([0], [0], color="#3366A8", linewidth=2.2)
    med_proxy = plt.scatter([], [], marker="s", facecolor="white",
                            edgecolor="#222", s=80)
    max_proxy = plt.scatter([], [], marker="x", color="#c62828",
                            s=100, linewidth=2.4)
    ax_legend.legend(
        [line_proxy, mean_proxy, med_proxy, max_proxy],
        ["min → p99", "mean (vertical tick)", "median (hollow square)", "max (✕)"],
        loc="center", ncol=4, fontsize=9, frameon=False,
    )

    out_png = assets_dir / "switch_rate.png"
    _save_fig_with_tag(fig, out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    # Supporting table.
    table_rows = [(
        "**cohort**", f"{cohort_n_hap:,}", f"{cohort_mean:.2f}",
        f"{cohort_med:.2f}", f"{cohort_p99:.0f}", f"{cohort_max:.0f}",
    )]
    for it in items:
        table_rows.append((
            it[0], f"{it[1]:,}", f"{it[4]:.2f}",
            f"{it[3]:.2f}", f"{it[5]:.0f}", f"{it[6]:.0f}",
        ))
    table = _md_table(
        ["row", "n_haplotypes", "mean", "median", "p99", "max"],
        table_rows,
    )

    gap = [(it[0], it[6] - it[5]) for it in items]
    top_gap = topn(gap, n=1)
    in_this_run = [
        f"Cohort mean switch rate: **{cohort_mean:.2f}** per haplotype "
        f"(median {cohort_med:.2f}, p99 {cohort_p99:.0f}, max {cohort_max:.0f}).",
        (f"Largest max-vs-p99 gap (single-haplotype outlier candidate): "
         f"`{top_gap[0][0]}` (max − p99 = {top_gap[0][1]:.1f})." if top_gap else ""),
    ]

    out = [
        "# Switch rate\n",
        "**What this is.** Number of ancestry switches per haplotype on this "
        "chromosome. Pure-ancestry haplotypes sit near zero; recently "
        "admixed haplotypes spread to dozens. Outliers in the 100+ tail are "
        "typically driven by phasing errors or LAI noise rather than real "
        "biology.\n",
        "**How to read this chart.** The bold top row is the cohort — the "
        "horizontal line spans cohort `min` to cohort `p99`, the vertical "
        "blue tick is the n_haplotypes-weighted mean, the hollow square is "
        "the n-weighted median, and the red ✕ is the cohort-wide max. The "
        "dim rows below show the same forest for each cluster · chrom. Read "
        "the cohort row first; if the per-cluster rows are tightly bunched "
        "around it, the cohort number is stable. A cluster whose ✕ jumps "
        "far past its own p99 has at least one haplotype with pathological "
        "switching worth pulling out.\n",
        _embed_image(out_png),
        "**In this run.** " + " ".join(s for s in in_this_run if s) + "\n",
        "",
        table,
    ]
    return "\n".join(out) + "\n"


# ── §10. Hap disagreement ─────────────────────────────────────────────────


def section_hap_disagreement(bundle: CohortBundle, assets_dir: Path) -> str:
    path = bundle.bundle_dir / "cohort" / "hap_disagreement.tsv"
    if not path.exists():
        return _missing("cohort/hap_disagreement.tsv")
    header, rows = _read_tsv(path)
    if not rows:
        return ""
    col = {h: i for i, h in enumerate(header)}

    by_rf: dict[str, list[tuple[str, int, float]]] = {}  # rf → (cc, n, mean)
    for r in rows:
        try:
            cc = f"{r[col['cluster_id']]}·{r[col['chrom']]}"
            rf = r[col["rf_label"]]
            n = int(r[col["n"]])
            mean = float(r[col["mean"]])
        except (IndexError, KeyError, ValueError):
            continue
        by_rf.setdefault(rf, []).append((cc, n, mean))
    if not by_rf:
        return ""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(by_rf.keys(),
                    key=lambda a: ANCESTRY_ORDER.index(a) if a in ANCESTRY_ORDER else 99)
    # Cohort pool per superpop label: n-weighted mean, drop low-n rows (< 5).
    pooled: dict[str, float | None] = {}
    pooled_n: dict[str, int] = {}
    for rf in labels:
        items = [(n, m) for _, n, m in by_rf[rf] if n >= 5]
        pooled[rf] = n_weighted_mean(items)
        pooled_n[rf] = sum(n for n, _ in items)

    n_lab = len(labels)
    chart_h = max(2.6, 0.7 * n_lab + 0.6)
    fig = plt.figure(figsize=(9.0, chart_h + 0.8))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[chart_h, 0.5], hspace=0.3,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0]); ax_legend.axis("off")

    # X limits.
    all_x = [m for items in by_rf.values() for _, _, m in items]
    all_x += [v for v in pooled.values() if v is not None]
    x_hi = max(0.45, (max(all_x) if all_x else 0.4) * 1.15)

    # Baseline shaded band, drawn behind everything.
    ax.axvspan(0.10, 0.30, color="#cccccc", alpha=0.30, zorder=0)

    for i, rf in enumerate(labels):
        color = ancestry_color(rf)
        v = pooled.get(rf)
        if v is not None:
            ax.barh(i, v, color=color, edgecolor="white", linewidth=0.6,
                    height=0.62, zorder=2)
            ax.text(v + x_hi * 0.005, i, f"{v:.3f}",
                    ha="left", va="center", fontsize=9, color="#222")
        per_cluster = [m for _, _, m in by_rf[rf]]
        overlay_ticks(ax, i, per_cluster, color="#111",
                      tick_height=0.46, lw=1.6, alpha=0.95)
    ax.set_yticks(range(n_lab))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, x_hi)
    ax.set_xlabel("mean hap-disagreement fraction (bp-weighted)", fontsize=10)
    ax.set_title("Hap disagreement per superpop label  ·  bar = cohort-pooled (n-weighted)  ·  "
                 "ticks = per cluster · chrom",
                 fontsize=11, loc="left")
    # Inside-axes annotation for the baseline band.
    ax.text(0.20, n_lab - 0.4, "expected baseline for admixed labels (0.10–0.30)",
            ha="center", va="bottom", fontsize=8, color="#666")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    bar_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#888")
    tick_proxy = plt.Line2D([0], [0], color="#111", linewidth=1.6)
    band_proxy = plt.Rectangle((0, 0), 1, 0.6, color="#cccccc", alpha=0.5)
    ax_legend.legend(
        [bar_proxy, tick_proxy, band_proxy],
        ["cohort-pooled mean (n-weighted)",
         "per-cluster mean",
         "expected baseline for admixed labels (0.10–0.30)"],
        loc="center", ncol=3, fontsize=9, frameon=False,
    )

    out_png = assets_dir / "hap_disagreement.png"
    _save_fig_with_tag(fig, out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    # In-this-run: highest disagreement on pure-ancestry labels (proxy for phasing noise).
    pure_labels = set(_SP6.members)
    pure_hits: list[tuple[str, float]] = []
    for rf in pure_labels:
        if rf not in by_rf:
            continue
        for cc, _n, mean in by_rf[rf]:
            pure_hits.append((f"{cc} · RF={rf}", mean))
    top_pure = topn(pure_hits, n=3)
    spreads: list[tuple[str, float]] = []
    for rf, items in by_rf.items():
        means = [m for _, _, m in items]
        if len(means) > 1:
            spreads.append((rf, max(means) - min(means)))
    top_spread = topn(spreads, n=2)
    in_this_run = [
        ("Highest disagreement on pure-ancestry labels (suggestive of phasing "
         "noise): " + ", ".join(f"{lab} ({v:.2f})" for lab, v in top_pure) + ".")
        if top_pure else "",
        ("Largest cluster-to-cluster spread per superpop label: "
         + ", ".join(f"`{lab}` (Δ={v:.2f})" for lab, v in top_spread) + ".")
        if top_spread else "",
    ]

    # Supporting n table: one row per (cluster · chrom · superpop label) with
    # the mean and the sample count.
    n_rows: list[tuple[str, str, str, str]] = []
    for rf in labels:
        for cc, n, mean in sorted(by_rf[rf], key=lambda t: t[0]):
            n_rows.append((rf, cc, f"{n:,}", f"{mean:.3f}"))
    n_table = _md_table(
        ["superpop label", "cluster · chrom", "n samples", "mean disagreement"],
        n_rows,
    )

    out = [
        "# Hap disagreement\n",
        "**What this is.** For each sample, what fraction of the genome "
        "(bp-weighted) gets a _different_ ancestry call on hap1 vs hap2? "
        "Two non-exclusive interpretations: real biological asymmetry "
        "(admixed individuals can have one parent contribute a different "
        "ancestry from the other), or a phasing-error proxy (switch errors "
        "in upstream phasing artificially split a single-ancestry "
        "haplotype).\n",
        "**How to read this chart.** Each row is one superpop label "
        "(assigned by the upstream RF classifier). The horizontal bar is "
        "the cohort-pooled mean disagreement, n_samples-weighted across "
        "clusters (rows with fewer than 5 samples are dropped before "
        "pooling). The vertical black ticks on each bar mark each "
        "cluster's individual mean. The shaded grey band (0.10–0.30) is "
        "the expected baseline for genuinely admixed labels; pure-ancestry "
        "labels (e.g. `eas` in a non-admixed cohort) should sit near zero. "
        "A bar that pushes well past the band on a pure-ancestry label "
        "points more toward phasing noise than biology; tightly clustered "
        "ticks behind that bar say the whole cohort sees it.\n",
        _embed_image(out_png),
        "**In this run.** " + " ".join(s for s in in_this_run if s) + "\n",
        "",
        n_table,
    ]
    return "\n".join(out) + "\n"


# ── §11. Regional meta-analysis Manhattan + top-20 ────────────────────────


def section_regional(bundle: CohortBundle, assets_dir: Path) -> str:
    path = bundle.bundle_dir / "cohort" / "regional_meta.tsv"
    if not path.exists():
        return _missing("cohort/regional_meta.tsv")
    header, rows = _read_tsv(path)
    if not rows:
        return ""
    col = {h: i for i, h in enumerate(header)}

    points: list[tuple[str, float, float, int, str, str, float, float, str, str]] = []
    # (anc_base, mid_mb, neglogq, n_flagged, chrom, anc_name, start_mb, end_mb, mask_region, raw_q)
    for r in rows:
        try:
            chrom = r[col["chrom"]]
            start = float(r[col["start"]])
            end = float(r[col["end"]])
            anc_name = r[col["ancestry_name"]]
            q = float(r[col["stouffer_q"]])
            n_flagged = int(r[col["n_clusters_flagged"]])
            mask = r[col["mask_region"]] or ""
        except (IndexError, KeyError, ValueError):
            continue
        if q <= 0:
            continue
        anc_base = anc_name.split(".")[0]
        mid_mb = (start + end) / 2 / 1e6
        neglogq = -math.log10(q)
        points.append((anc_base, mid_mb, neglogq, n_flagged, chrom, anc_name,
                       start / 1e6, end / 1e6, mask, str(q)))
    if not points:
        return ""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chroms_in_use = sorted({p[4] for p in points}, key=chrom_sort_key)
    # Build a position offset map for multi-chrom layouts.
    offsets: dict[str, float] = {}
    cur = 0.0
    for chrom in chroms_in_use:
        offsets[chrom] = cur
        cur += chrom_length(chrom) / 1e6 or 250.0
    total_extent = cur

    fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)

    # Mask shading: only on windows that crossed the FDR threshold, so the
    # plot doesn't drown in segdup bands for non-significant windows.
    sig_threshold = -math.log10(0.05)
    drawn_masks: set[tuple[str, float, float]] = set()
    for _anc_base, _mid, neglogq, _nf, chrom, _anc_name, start_mb, end_mb, mask, _q in points:
        if neglogq < sig_threshold or not mask or chrom not in offsets:
            continue
        key = (chrom, start_mb, end_mb)
        if key in drawn_masks:
            continue
        drawn_masks.add(key)
        x0 = offsets[chrom] + start_mb
        x1 = offsets[chrom] + end_mb
        ax.axvspan(x0, x1, color="#e9e9e9", alpha=0.55, linewidth=0, zorder=0)

    # Lollipops: thin vertical line from 0 → -log10(q) with a short
    # horizontal tick cap. Line width scales with n_clusters_flagged so
    # heavily-flagged windows visually pop. No circular markers.
    seen_legend: set[str] = set()
    legend_handles: list = []
    max_flag = max((p[3] for p in points), default=1) or 1
    for anc_base, mid_mb, neglogq, n_flagged, chrom, _, _, _, _, _ in points:
        if chrom not in offsets:
            continue
        x = offsets[chrom] + mid_mb
        color = ancestry_color(anc_base)
        # Map n_flagged to line width: 0.6 (lone) → 2.6 (cohort-wide).
        lw = 0.6 + 2.0 * (n_flagged / max_flag)
        ax.vlines(x, 0, neglogq, color=color, linewidth=lw, alpha=0.85,
                  zorder=2)
        # Cap tick: small horizontal dash at the top of the lollipop.
        cap_half = max(0.6, total_extent * 0.0035)
        ax.hlines(neglogq, x - cap_half, x + cap_half, color=color,
                  linewidth=lw, alpha=0.95, zorder=3)
        if anc_base not in seen_legend:
            seen_legend.add(anc_base)
            legend_handles.append(
                plt.Line2D([0], [0], color=color, linewidth=2.4, label=anc_base)
            )

    # Annotate top peaks. Cluster nearby (by genomic mid) peaks into one
    # combined annotation so we don't stack three labels on the same column.
    top_peaks = sorted(points, key=lambda p: -p[2])[:5]
    # Group peaks whose midpoints are within 5 Mb of each other.
    clusters: list[list[tuple]] = []
    for p in top_peaks:
        placed = False
        for c in clusters:
            if abs(c[0][1] - p[1]) < 5.0 and c[0][4] == p[4]:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    # Y-axis headroom for stacked labels — annotation bboxes have ~1 line of
    # padding above their y coord, so y_top must sit well above the highest
    # annotation row or the box clips the upper spine.
    y_max = max(p[2] for p in points) if points else 1.0
    y_top = y_max * 1.7
    ax.set_ylim(0, y_top)
    for cluster_idx, group in enumerate(clusters[:3]):
        anc_base, mid_mb, neglogq, n_flagged, chrom, anc_name, _, _, mask, q = group[0]
        x = offsets[chrom] + mid_mb
        # Stagger labels vertically and horizontally so they don't collide.
        x_jitter = (cluster_idx - 1) * total_extent * 0.16
        y_offset = y_top * (0.80 - cluster_idx * 0.11)
        mask_str = mask if mask else "no mask"
        more_peaks = "" if len(group) == 1 else f" (+{len(group) - 1} nearby)"
        text = (f"#{cluster_idx + 1}: {chrom}:{mid_mb:.1f} Mb\n"
                f"{anc_name} · q={float(q):.1e}\n"
                f"{n_flagged} cluster(s) · {mask_str}{more_peaks}")
        ax.annotate(text,
                    xy=(x, neglogq), xytext=(x + x_jitter, y_offset),
                    fontsize=8, color="#222", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#888", linewidth=0.6),
                    arrowprops=dict(arrowstyle="->", color="#666", linewidth=0.7))
    ax.axhline(sig_threshold, color="#888", linestyle="--", linewidth=0.9, zorder=1)
    ax.annotate("FDR 5%", xy=(total_extent, sig_threshold),
                xytext=(4, 0), textcoords="offset points",
                fontsize=8, color="#555", ha="left", va="center")
    if len(chroms_in_use) > 1:
        for chrom in chroms_in_use[1:]:
            ax.axvline(offsets[chrom], color="#cccccc", linewidth=0.6, zorder=0)
    # Chrom labels: only when > 1 chrom; single-chrom layouts just use the x-axis.
    if len(chroms_in_use) > 1:
        for chrom in chroms_in_use:
            ax.text(offsets[chrom] + (chrom_length(chrom) / 1e6 or 250.0) / 2,
                    1.02, chrom, fontsize=9, color="#444", ha="center", va="bottom",
                    transform=ax.get_xaxis_transform())
    # Constrain xlim to actual data, with a touch of padding.
    if len(chroms_in_use) == 1:
        only = chroms_in_use[0]
        ax.set_xlim(0, chrom_length(only) / 1e6 or total_extent)
        ax.set_xlabel(f"genomic position on {only} (Mb)", fontsize=10)
    else:
        ax.set_xlim(0, total_extent)
        ax.set_xlabel("genomic position (Mb, chromosomes concatenated)", fontsize=10)
    ax.set_ylabel("−log10(stouffer q)", fontsize=10)
    ax.set_title("Cross-cluster regional meta-analysis  ·  lollipop height = significance  ·  "
                 "line width = clusters flagging the window",
                 fontsize=11, loc="left")
    ax.legend(handles=legend_handles, title="ancestry", fontsize=8,
              title_fontsize=9, loc="upper right", frameon=False)
    out_png = assets_dir / "regional_manhattan.png"
    _save_fig_with_tag(fig, out_png, dpi=_DPI)
    plt.close(fig)

    # Top-20 table for the supporting detail.
    def _mb(s: str) -> str:
        try:
            return f"{int(s)/1e6:.2f}"
        except (TypeError, ValueError):
            return s
    enriched: list[tuple[float, list[str]]] = []
    for r in rows:
        try:
            q = float(r[col["stouffer_q"]])
        except (IndexError, KeyError, ValueError):
            continue
        enriched.append((q, r))
    enriched.sort(key=lambda t: t[0])
    table_rows: list[tuple[str, ...]] = []
    for q, r in enriched[:20]:
        table_rows.append((
            r[col["chrom"]],
            _mb(r[col["start"]]),
            _mb(r[col["end"]]),
            r[col["ancestry_name"]],
            r[col["n_clusters_flagged"]],
            r[col["n_clusters_total"]],
            _fmt_num(r[col["stouffer_z"]], 2),
            _fmt_num(r[col["stouffer_p"]], 4),
            _fmt_num(r[col["stouffer_q"]], 4),
            r[col["mask_region"]] or "—",
        ))

    # In-this-run callout from the top peak + outside-mask count.
    n_outside_mask = sum(1 for p in points if not p[8])
    top = top_peaks[0] if top_peaks else None
    in_this_run = []
    if top:
        in_this_run.append(
            f"Top peak: `{top[4]}:{top[1]:.1f} Mb` for `{top[5]}` "
            f"(q = {float(top[9]):.1e}, flagged by {top[3]} cluster(s), "
            f"mask = {top[8] or 'none'})."
        )
    in_this_run.append(f"Windows outside any pre-registered mask: **{n_outside_mask:,}**.")

    out = [
        "# Regional meta-analysis\n",
        "**What this measures.** For every 1 Mb sliding window on each "
        "chromosome (250 kb step), the bp-weighted mean ancestry proportion "
        "across all haplotypes is computed per ancestry. Each (window, "
        "ancestry) is z-scored against the chromosome-wide mean for that "
        "ancestry, BH-corrected for FDR, then combined across clusters via "
        "Stouffer's method. A peak here is a window where multiple clusters "
        "independently see a deviation in the same direction.\n",
        "**How to read this chart.** Each point is one window × ancestry, "
        "plotted at its genomic midpoint and meta-analysis q-value (−log10, "
        "so up = more significant). Color = ancestry. Marker size scales "
        "with the number of clusters that independently flagged that window. "
        "Light-gray bands behind the points are pre-registered masks (HLA, "
        "centromere flank, segdup, high-LD). A tall peak inside a mask is "
        "expected; a tall peak outside any mask is real signal worth "
        "investigating. The dashed line is FDR 5%; the top three peaks are "
        "annotated by position and ancestry.\n",
        _embed_image(out_png),
        "**In this run.** " + " ".join(in_this_run) + "\n",
        "## Top 20 windows by FDR-corrected q\n",
        _md_table(
            ["chrom", "start (Mb)", "end (Mb)", "ancestry",
             "n flagged", "n total", "z", "p", "q", "mask"],
            table_rows,
        ),
    ]
    return "\n".join(out) + "\n"


# ── §12. Provenance appendix ──────────────────────────────────────────────


def section_provenance(bundle: CohortBundle) -> str:
    manifest = bundle.manifest
    sha = manifest.get("sha256_per_artifact", {})
    rows = [(k, v[:12]) for k, v in sha.items()]
    out = [
        "# Provenance\n",
        "| Field | Value |",
        "|---|---|",
        f"| schema_version | {manifest.get('schema_version', '?')} |",
        f"| run_name       | {_md_escape(manifest.get('run_name', '?'))} |",
        f"| collation_mode | {manifest.get('collation_mode', '?')} |",
        f"| n_artifacts    | {manifest.get('n_artifacts', '?')} |",
        f"| diff_against   | {manifest.get('diff_against') or '—'} |",
        "",
        "## sha256 per artifact (first 12 chars)\n",
        _md_table(["cluster.chrom", "sha256[:12]"], rows),
    ]
    return "\n".join(out) + "\n"


# ── Per-cluster opt-in sections (unchanged from v1) ───────────────────────


def section_cluster_header(cluster_id: str, chrom: str, cluster_dir: Path,
                           bundle: CohortBundle) -> str:
    manifest_path = cluster_dir / "manifest.json"
    if not manifest_path.exists():
        return f"# Cluster `{cluster_id}` / `{chrom}`\n\n" + _missing("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    opt = manifest.get("optional_inputs", {})
    optional_present = [k for k, v in opt.items() if v]
    optional_absent = [k for k, v in opt.items() if not v]

    rows = [
        ("n_samples",         manifest.get("n_samples", "?")),
        ("n_markers",         manifest.get("n_markers", "?")),
        ("n_ancestries",      manifest.get("n_ancestries", "?")),
        ("wallclock seconds", _fmt_num(manifest.get("total_wallclock_seconds"), 1)),
        ("peak RSS GB",       _fmt_num(manifest.get("peak_rss_gb"), 2)),
        ("cpu/wall",          _fmt_num(manifest.get("cpu_wall_ratio"), 2)),
        ("flare version",     manifest.get("flare_version", "?")),
        ("panel id",          manifest.get("panel_id") or "—"),
        ("input_vcf_sha",     (manifest.get("input_vcf_sha") or "")[:12]),
        ("generated_at",      manifest.get("generated_at", "?")),
    ]

    out = [
        f"# Cluster `{cluster_id}` · `{chrom}`\n",
        _md_table(["field", "value"], rows),
        "\n",
        f"**Optional inputs present:** {', '.join(optional_present) or 'none'}",
        f"  ·  **absent:** {', '.join(optional_absent) or 'none'}\n",
    ]

    return "\n".join(out) + "\n"


def section_cluster_concordance(cluster_dir: Path, manifest: dict[str, Any]) -> str:
    if not manifest.get("optional_inputs", {}).get("rye_q"):
        return ""

    out: list[str] = ["## Rye-vs-FLARE concordance\n"]

    summary_path = cluster_dir / "concordance" / "concordance_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        rows: list[tuple[str, str]] = []
        for k, v in s.items():
            if isinstance(v, list):
                rows.append((k, ", ".join(v) if v else "—"))
            else:
                rows.append((k, _md_escape(str(v))))
        out.append("### Summary\n")
        out.append(_md_table(["field", "value"], rows))

    metrics_path = cluster_dir / "concordance" / "concordance_metrics.tsv"
    if metrics_path.exists():
        header, rows = _read_tsv(metrics_path)
        out.append("\n### Per-ancestry metrics\n")
        out.append(_md_table(header, rows))

    comparison_png = cluster_dir / "concordance" / "rye_admixture_comparison.png"
    if comparison_png.exists():
        out.append("\n### Mean comparison\n")
        out.append(_embed_image(comparison_png))

    scatters = sorted((cluster_dir / "concordance").glob("rye_scatter_*.png"))
    if scatters:
        out.append("\n### Per-ancestry scatter\n")
        for img in scatters:
            label = img.stem.replace("rye_scatter_", "").upper()
            out.append(f"#### {label}\n")
            out.append(_embed_image(img))

    return "\n".join(out) + "\n"


def section_cluster_calibration(cluster_dir: Path) -> str:
    cal = cluster_dir / "calibration"
    if not cal.is_dir():
        return ""

    figures = [
        ("calibration_curves.png",        "Calibration curves"),
        ("calibration_slope_matrix.png",  "Calibration slope matrix"),
        ("soft_proportion_hexbin.png",    "Soft proportion hexbin"),
        ("merged_confusion_matrix.png",   "Merged confusion matrix"),
        ("residual_violin.png",           "Residual violin"),
        ("entropy_scatter.png",           "Entropy scatter"),
    ]
    out: list[str] = ["## Calibration\n"]

    slope_tsv = cal / "slope_matrix.tsv"
    if slope_tsv.exists():
        header, rows = _read_tsv(slope_tsv)
        out.append("### Slope matrix\n")
        out.append(_md_table(header, rows))

    for fname, title in figures:
        img = cal / fname
        if img.exists():
            out.append(f"\n### {title}\n")
            out.append(_embed_image(img))

    for img in sorted(cal.glob("calibration_*_breakdown.png")):
        label = img.stem.replace("calibration_", "").replace("_breakdown", "").upper()
        out.append(f"\n### Sub-ancestry calibration: {label}\n")
        out.append(_embed_image(img))

    return "\n".join(out) + "\n"


def section_cluster_structural(cluster_dir: Path) -> str:
    sub = cluster_dir / "structural"
    if not sub.is_dir():
        return ""
    out: list[str] = ["## Structural\n"]

    summary = sub / "tract_length_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        flat: list[tuple[str, str]] = []
        for k, v in data.items():
            if isinstance(v, list):
                flat.append((k, json.dumps(v)[:200]))
            else:
                flat.append((k, _md_escape(str(v))))
        out.append("### Tract length summary\n")
        out.append(_md_table(["field", "value"], flat))

    tract_png = sub / "tract_length_distribution.png"
    if tract_png.exists():
        out.append("\n### Tract length distribution\n")
        out.append(_embed_image(tract_png))

    sw1 = sub / "switch_rate_distribution.png"
    sw2 = sub / "switch_rate_distribution_log.png"
    if sw1.exists():
        out.append("\n### Switch rate distribution\n")
        out.append(_embed_image(sw1))
    if sw2.exists():
        out.append("\n### Switch rate distribution (log)\n")
        out.append(_embed_image(sw2))

    return "\n".join(out) + "\n"


def section_cluster_hap_and_regional(cluster_dir: Path) -> str:
    out: list[str] = ["## Hap disagreement + regional QC\n"]

    hap_summary = cluster_dir / "hap_disagreement" / "summary.json"
    if hap_summary.exists():
        data = json.loads(hap_summary.read_text())
        rows = [(k, _md_escape(str(v))) for k, v in data.items()
                if not isinstance(v, (list, dict))]
        out.append("### Hap disagreement summary\n")
        out.append(_md_table(["field", "value"], rows))

    hap_png = cluster_dir / "hap_disagreement" / "by_rf_label.png"
    if hap_png.exists():
        out.append("\n### Hap disagreement by superpop label\n")
        out.append(_embed_image(hap_png))

    region_png = next(iter((cluster_dir / "regional").glob("regional_qc_*.png")), None)
    if region_png is not None:
        out.append("\n### Regional QC\n")
        out.append(_embed_image(region_png))

    region_summary = cluster_dir / "regional" / "summary.json"
    if region_summary.exists():
        data = json.loads(region_summary.read_text())
        rows = [(k, _md_escape(str(v))) for k, v in data.items()
                if not isinstance(v, (list, dict))]
        out.append("\n### Regional summary\n")
        out.append(_md_table(["field", "value"], rows))

    return "\n".join(out) + "\n"


def per_cluster_section(cluster_id: str, chrom: str, cluster_dir: Path | None,
                        bundle: CohortBundle) -> str:
    if cluster_dir is None:
        return (
            f"# Cluster `{cluster_id}` · `{chrom}`\n\n"
            f"> **MISSING:** no `per_cluster/{cluster_id}/{chrom}/` directory or "
            f"tarball available; section omitted.\n"
        )
    manifest_path = cluster_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists() else {}
    )
    parts = [
        section_cluster_header(cluster_id, chrom, cluster_dir, bundle),
        section_cluster_concordance(cluster_dir, manifest),
        section_cluster_calibration(cluster_dir),
        section_cluster_structural(cluster_dir),
        section_cluster_hap_and_regional(cluster_dir),
    ]
    return "\n\\newpage\n".join(p for p in parts if p)


# ── Master assembler ──────────────────────────────────────────────────────


def build_markdown(
    bundle: CohortBundle,
    pairs: list[tuple[str, str]],
    cluster_dirs: dict[tuple[str, str], Path],
    assets_dir: Path,
    include_per_cluster: bool,
) -> str:
    # Resolve and stash the label-space tag once per run; every chart
    # picks it up via _save_fig_with_tag.
    set_label_space_tag(resolve_label_space_tag(bundle))

    page_break = "\n\\newpage\n"
    sections: list[str] = [
        md_frontmatter(bundle),
        section_cover(bundle),
        page_break, section_reading_guide(),
        page_break, section_label_space_conventions(),
        page_break, section_cohort_composition(bundle, assets_dir),
        page_break, section_concordance(bundle, assets_dir),
        page_break, section_calibration(bundle, assets_dir),
        page_break, section_confusion(bundle, assets_dir),
        page_break, section_tract_length(bundle, assets_dir),
        page_break, section_switch_rate(bundle, assets_dir),
        page_break, section_hap_disagreement(bundle, assets_dir),
        page_break, section_regional(bundle, assets_dir),
    ]
    if include_per_cluster:
        for cid, chrom in pairs:
            _log(f"rendering per-cluster section for {cid}/{chrom}")
            sections.append(page_break)
            sections.append(
                per_cluster_section(cid, chrom, cluster_dirs.get((cid, chrom)), bundle)
            )
    sections.append(page_break)
    sections.append(section_provenance(bundle))
    return "\n".join(s for s in sections if s)


# ── pandoc invocation ─────────────────────────────────────────────────────


def run_pandoc(md_path: Path, out_pdf: Path) -> None:
    cmd = [
        "pandoc", str(md_path),
        "-o", str(out_pdf),
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=0.75in",
        "-V", "fontsize=10pt",
        "-V", "mainfont=Helvetica",
        "-V", "monofont=Menlo",
        "--highlight-style=tango",
    ]
    _log(f"pandoc → {out_pdf}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"pandoc exit {result.returncode}")


# ── Argparse + main ───────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a PDF validation report from a flare_validate cohort bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cohort-bundle", type=Path, required=True,
                   help="path to an unpacked cohort_bundle/ directory")
    p.add_argument("--tarball-dir", type=Path, default=None,
                   help="optional directory of per-cluster *.validation.*.tar.gz tarballs "
                        "(used when cohort_bundle/per_cluster/ is absent)")
    p.add_argument("--out", type=Path, required=True,
                   help="destination PDF path")
    p.add_argument("--clusters", default=None,
                   help="comma-separated list of cluster_ids to include")
    p.add_argument("--max-clusters", type=int, default=None,
                   help="cap on number of distinct cluster_ids included")
    p.add_argument("--per-cluster", action="store_true",
                   help="append per-cluster long-form sections after the cohort pages")
    p.add_argument("--keep-md", action="store_true",
                   help="keep the intermediate .md document next to the output PDF")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    bundle = load_cohort_bundle(args.cohort_bundle)
    _log(f"loaded cohort_bundle: {bundle.run_name} "
         f"({bundle.summary.get('n_clusters', '?')} clusters, "
         f"{bundle.summary.get('n_chroms', '?')} chroms)")

    pairs = bundle.cluster_chrom_pairs
    cluster_filter = (
        [c.strip() for c in args.clusters.split(",") if c.strip()]
        if args.clusters else None
    )
    pairs = filter_and_cap(pairs, cluster_filter, args.max_clusters)
    _log(f"rendering {len(pairs)} cluster/chrom pair(s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_dir = args.out.parent
    assets_dir = out_dir / f"{args.out.stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.out.stem}.md"

    with tempfile.TemporaryDirectory(prefix="flare_validate_report_") as tmpdir:
        tmp = Path(tmpdir)
        cluster_dirs = resolve_cluster_dirs(
            args.cohort_bundle, args.tarball_dir, tmp, pairs,
        )
        missing = [p for p in pairs if p not in cluster_dirs]
        if missing:
            _log(f"WARNING: no per-cluster directory or tarball found for "
                 f"{len(missing)} pair(s); per-cluster opt-in will skip them")

        with popout_style():
            md = build_markdown(
                bundle, pairs, cluster_dirs, assets_dir,
                include_per_cluster=args.per_cluster,
            )
        md_path.write_text(md)
        _log(f"wrote {md_path}")

        run_pandoc(md_path, args.out)

    if not args.keep_md:
        md_path.unlink(missing_ok=True)

    _log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
