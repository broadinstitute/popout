#!/usr/bin/env python3
"""Build a PDF report from a popout DX cohort bundle.

Sibling of ``validation/scripts/build_flare_validation_report.py`` —
same lift-and-adapt template, popout-DX-specific sections.

Usage::

    python build_popout_dx_report.py \\
        --cohort-bundle cohort_dx.<run_name>.v1.0.0.tar.gz \\
        --out report.pdf \\
        [--clusters cluster_000,cluster_007] \\
        [--max-clusters 10] \\
        [--per-cluster]   \\
        [--keep-md]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

# popout.viz._style is optional; if importable we use its palette.
try:
    from popout.viz._style import ANCESTRY_PALETTE  # type: ignore
except ImportError:
    # Paul Tol qualitative palette (copy of popout.viz._style.ANCESTRY_PALETTE).
    ANCESTRY_PALETTE = [
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB", "#EE8866",
        "#44BB99", "#DDCC77", "#882255", "#332288",
    ]


SCHEMA_VERSION = "1.0.0"
ANCHOR_TOOL = "popout"
# Phase 4 of the label-space retrofit: SP6 is the canonical superpop
# space from popout.labelspace.registry.
from popout.labelspace.registry import SP6 as _SP6
RF_LABELS: tuple[str, ...] = _SP6.members
RF_LABEL_COLOR: dict[str, str] = {
    "afr": ANCESTRY_PALETTE[0],
    "amr": ANCESTRY_PALETTE[1],
    "eas": ANCESTRY_PALETTE[2],
    "eur": ANCESTRY_PALETTE[3],
    "mid": ANCESTRY_PALETTE[4],
    "sas": ANCESTRY_PALETTE[5],
}

# Traffic-light cell colors (Tol-safe).
TL_GREEN = "#117733"
TL_YELLOW = "#DDCC77"
TL_RED = "#CC3311"
TL_GREY = "#888888"

_DPI = 220


def _log(msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] build_popout_dx_report: {msg}", file=sys.stderr, flush=True)


# ── Cohort bundle in-memory view ─────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class CohortBundle:
    bundle_dir: Path                 # the directory containing cohort_manifest.json
    manifest: dict[str, Any]
    summary: dict[str, Any]

    @property
    def run_name(self) -> str:
        return self.manifest.get("run_name", "<unknown>")

    @property
    def mode(self) -> str:
        return self.manifest.get("mode", "global")

    @property
    def tools(self) -> list[str]:
        return list(self.manifest.get("tools", []))

    @property
    def comparison_tools(self) -> list[str]:
        return [t for t in self.tools if t != ANCHOR_TOOL]

    @property
    def cluster_ids(self) -> list[str]:
        return list(self.manifest.get("cluster_ids", []))

    @property
    def chroms(self) -> list[str]:
        return list(self.manifest.get("chroms", []))


def _untar_to(src: Path, dest: Path) -> Path:
    """Extract a popout DX cohort tarball; return the inner ``cohort_dx`` dir."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:*") as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError(f"{src} is empty")
        top = members[0].name.split("/", 1)[0]
        tar.extractall(dest, filter="data")
    inner = dest / top
    if not (inner / "cohort_manifest.json").is_file():
        raise RuntimeError(
            f"{src}: extracted {inner} does not contain cohort_manifest.json"
        )
    return inner


def resolve_bundle_dir(arg: Path, tmpdir: Path) -> Path:
    """Accept either an unpacked cohort_dx/ dir or a *.tar.gz tarball."""
    if arg.is_dir():
        if (arg / "cohort_manifest.json").is_file():
            return arg
        inner = arg / "cohort_dx"
        if (inner / "cohort_manifest.json").is_file():
            return inner
        raise FileNotFoundError(
            f"{arg}: no cohort_manifest.json at top level or inside cohort_dx/"
        )
    if arg.is_file() and (arg.name.endswith(".tar.gz") or arg.name.endswith(".tgz")):
        return _untar_to(arg, tmpdir)
    raise FileNotFoundError(f"{arg}: not a directory or tarball")


def load_cohort_bundle(bundle_dir: Path) -> CohortBundle:
    manifest = json.loads((bundle_dir / "cohort_manifest.json").read_text())
    summary = json.loads((bundle_dir / "cohort_summary.json").read_text())
    sv = manifest.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"cohort_manifest.json schema_version {sv!r} != expected {SCHEMA_VERSION!r}"
        )
    return CohortBundle(bundle_dir=bundle_dir, manifest=manifest, summary=summary)


# ── TSV / markdown helpers ───────────────────────────────────────────────


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header, rows). Empty file → ([], [])."""
    if not path.exists():
        return [], []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        text = f.read().rstrip("\n")
    if not text:
        return [], []
    lines = text.split("\n")
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


def _to_float(s: Any) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        f = float(s)
        return None if f != f else f
    s = str(s).strip()
    if s == "" or s == "NA" or s.lower() == "nan":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if f != f else f


def _md_escape(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("_", "\\_")


def _fmt_num(v: Any, places: int = 3) -> str:
    f = _to_float(v)
    if f is None:
        return "—"
    return f"{f:.{places}f}"


def _fmt_int(v: Any) -> str:
    f = _to_float(v)
    if f is None:
        return "—"
    return f"{int(f):,}"


def _fmt_pct(v: Any, places: int = 1) -> str:
    f = _to_float(v)
    if f is None:
        return "—"
    return f"{100 * f:.{places}f}%"


def _md_table(header: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    h = [str(c) for c in header]
    out = ["| " + " | ".join(h) + " |",
           "|" + "|".join(["---"] * len(h)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_md_escape(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _embed_image(png: Path, width: str = "95%") -> str:
    return f"\n![]({png}){{ width={width} }}\n\n"


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _page_break() -> str:
    return "\n\\newpage\n\n"


# ── Plot helpers ─────────────────────────────────────────────────────────


def _tl_color(pair: dict[str, Any]) -> str:
    """Traffic-light color for a `cohort_summary.json:pairs[]` entry."""
    n_pass = int(pair.get("n_clusters_passing", 0) or 0)
    n_fail = int(pair.get("n_clusters_failing", 0) or 0)
    n_null = int(pair.get("n_clusters_null", 0) or 0)
    n_eval = n_pass + n_fail
    if n_eval == 0:
        return TL_GREY  # μ-gated everywhere
    frac = n_pass / n_eval
    if frac >= 0.9:
        return TL_GREEN
    if frac >= 0.5:
        return TL_YELLOW
    return TL_RED


def plot_traffic_light_grid(
    pairs: list[dict[str, Any]],
    tools: list[str],
    rf_labels: tuple[str, ...],
) -> plt.Figure:
    by_key: dict[tuple[str, str], dict[str, Any]] = {
        (p["tool"], p["rf_label"]): p for p in pairs
    }
    n_rows, n_cols = len(tools), len(rf_labels)
    fig, ax = plt.subplots(figsize=(0.95 * n_cols + 1.2, 0.7 * n_rows + 1.0))

    for i, tool in enumerate(tools):
        for j, lab in enumerate(rf_labels):
            p = by_key.get((tool, lab))
            color = _tl_color(p) if p else TL_GREY
            ax.add_patch(plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                       facecolor=color, edgecolor="white", lw=1.0))
            text = "—"
            if p is not None:
                n_pass = int(p.get("n_clusters_passing", 0) or 0)
                n_fail = int(p.get("n_clusters_failing", 0) or 0)
                n_null = int(p.get("n_clusters_null", 0) or 0)
                n_eval = n_pass + n_fail
                if n_eval > 0:
                    text = f"{n_pass}/{n_eval}"
                elif n_null > 0:
                    text = "μ·∅"
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, text,
                    ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(rf_labels, fontsize=10)
    ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(tools, fontsize=10)
    ax.set_aspect("equal")
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Pass rate per (tool, RF label)\ncell = #passing / #μ-evaluable clusters", fontsize=10)
    fig.tight_layout()
    return fig


def plot_per_label_ccc(
    metrics_rows: list[dict[str, Any]],
    tool: str,
    rf_labels: tuple[str, ...],
) -> plt.Figure:
    """Per-RF-label CCC distribution across clusters (boxplot when n>5)."""
    by_label: dict[str, list[float]] = {lab: [] for lab in rf_labels}
    for r in metrics_rows:
        lab = r.get("rf_label", "")
        v = _to_float(r.get("ccc"))
        if v is not None and lab in by_label:
            by_label[lab].append(v)

    fig, ax = plt.subplots(figsize=(1.1 * len(rf_labels) + 1.5, 3.4))
    n_clusters_max = max((len(v) for v in by_label.values()), default=0)
    if n_clusters_max > 5:
        data = [by_label[lab] for lab in rf_labels]
        bp = ax.boxplot(data, positions=range(len(rf_labels)),
                        patch_artist=True, widths=0.55,
                        medianprops={"color": "#222"}, showfliers=True)
        for patch, lab in zip(bp["boxes"], rf_labels):
            patch.set_facecolor(RF_LABEL_COLOR.get(lab, TL_GREY))
            patch.set_alpha(0.6)
    else:
        for i, lab in enumerate(rf_labels):
            vals = by_label[lab]
            if vals:
                ax.scatter([i] * len(vals), vals, s=36,
                           color=RF_LABEL_COLOR.get(lab, TL_GREY),
                           edgecolors="#222", linewidths=0.6, alpha=0.85)
    ax.set_xticks(range(len(rf_labels)))
    ax.set_xticklabels(rf_labels)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.9, color="#117733", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0.5, color="#CC3311", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Lin's CCC")
    ax.set_title(f"popout vs {tool} — per-RF-label CCC across clusters")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_per_sample_mae_violin(
    per_sample_rows: list[dict[str, Any]],
    tools: list[str],
) -> plt.Figure:
    cols = [f"mae_vs_{t}" for t in tools]
    data: dict[str, np.ndarray] = {}
    for tool, col in zip(tools, cols):
        vals = [_to_float(r.get(col)) for r in per_sample_rows]
        arr = np.array([v for v in vals if v is not None and v > 0], dtype=float)
        if arr.size:
            data[tool] = arr

    if not data:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.text(0.5, 0.5, "no per-sample MAE data", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(2.5 * len(data) + 1.5, 3.2))
    positions = list(range(len(data)))
    vp = ax.violinplot([np.log10(data[t]) for t in data],
                       positions=positions, widths=0.85, showmedians=True)
    for body, tool in zip(vp["bodies"], data):
        body.set_facecolor(TL_GREY)
        body.set_alpha(0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"popout vs {t}\nn={len(data[t])}" for t in data])
    ax.set_ylabel("log10 MAE per sample")
    ax.set_title("Per-sample MAE distribution (log10)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_confusion_heatmap(
    cm: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    tool: str,
) -> plt.Figure:
    """Row-normalised heatmap of a hard-call confusion matrix."""
    fig, ax = plt.subplots(figsize=(0.7 * len(col_labels) + 2.0,
                                    0.7 * len(row_labels) + 1.5))
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if row_sums[i, 0] <= 0:
                ax.text(j, i, "—", ha="center", va="center",
                        color="#888", fontsize=8)
            else:
                v = norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.55 else "#222", fontsize=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(f"{tool} label →")
    ax.set_ylabel("popout label ↓")
    ax.set_title(f"Hard-call confusion (popout vs {tool}, row-normalised)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="row fraction")
    fig.tight_layout()
    return fig


def plot_boundary_distance_hist(rows: list[dict[str, Any]]) -> plt.Figure:
    distances = [_to_float(r.get("distance_bp")) for r in rows]
    distances = [abs(d) for d in distances if d is not None]
    matched = [bool(r.get("flanking_label_match") in ("true", "True", True))
               for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    if distances:
        bins = np.logspace(2, 8, 30)
        axes[0].hist(distances, bins=bins, color=TL_GREEN, alpha=0.85)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("|distance_bp| to nearest popout switch")
        axes[0].set_ylabel("# FLARE switches")
        axes[0].set_title("Boundary localization (log-scale)")
    else:
        axes[0].text(0.5, 0.5, "no boundary data", ha="center", va="center")
        axes[0].axis("off")
    if matched:
        n_match = sum(matched)
        n_total = len(matched)
        axes[1].bar(["matched", "unmatched"], [n_match, n_total - n_match],
                    color=[TL_GREEN, TL_RED])
        axes[1].set_ylabel("# FLARE switches")
        axes[1].set_title(f"Flanking-label match ({_fmt_pct(n_match / n_total)})")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    return fig


def plot_coarse_grid_curves(rows: list[dict[str, Any]]) -> plt.Figure:
    series: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for r in rows:
        try:
            res = _to_float(r.get("resolution_mb"))
            diag = _to_float(r.get("diagonal_fraction"))
            sample = str(r.get("sample"))
            hap = str(r.get("hap"))
            chrom = str(r.get("chrom"))
        except KeyError:
            continue
        if res is None or diag is None:
            continue
        series.setdefault((chrom, sample, hap), []).append((res, diag))

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    all_xs: set[float] = set()
    per_x: dict[float, list[float]] = {}
    for pts in series.values():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=TL_GREY, alpha=0.15, lw=0.8)
        for x, y in pts:
            all_xs.add(x)
            per_x.setdefault(x, []).append(y)
    if per_x:
        med_xs = sorted(per_x)
        med_ys = [float(np.median(per_x[x])) for x in med_xs]
        ax.plot(med_xs, med_ys, color=TL_GREEN, lw=2.2, label="median")
        ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("resolution (Mb)")
    ax.set_ylabel("diagonal fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coarse-grid sweep: agreement vs resolution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ── Section builders ────────────────────────────────────────────────────


def md_frontmatter(bundle: CohortBundle) -> str:
    return (
        "---\n"
        f"title: \"popout DX report: {_md_escape(bundle.run_name)}\"\n"
        f"date: \"{bundle.manifest.get('generated_at', '')}\"\n"
        "geometry: margin=0.75in\n"
        "colorlinks: true\n"
        "header-includes:\n"
        "  - \\usepackage{adjustbox}\n"
        "  - \\usepackage{longtable}\n"
        "---\n\n"
    )


def _resolve_figure_tag(bundle: CohortBundle) -> str | None:
    """Return the canonical figure-shorthand tag for this cohort, or None.

    Phase 5 of the label-space retrofit. Looks for ``provenance.tag`` in
    ``cohort_manifest.json``, then in ``cohort_summary.json``. Falls back
    to a synthesised SP6 tag if neither carries one (which keeps reports
    on pre-Phase-5 bundles readable).
    """
    for src in (bundle.manifest, bundle.summary):
        prov = (src or {}).get("provenance") or {}
        tag = prov.get("tag")
        if isinstance(tag, str) and tag:
            return tag
    tools = [t for t in bundle.tools if t != ANCHOR_TOOL]
    if not tools:
        return None
    return ("L=SP6/MID+ | " + " | ".join(
        ["popout=>postS"] + [f"{t}=>name" for t in tools]
    ))


def section_cover(bundle: CohortBundle) -> str:
    m = bundle.manifest
    s = bundle.summary
    lines = [
        f"# popout DX cohort report",
        "",
        f"**Run:** `{bundle.run_name}`  ",
        f"**Generated:** {m.get('generated_at', '?')}  ",
        f"**Schema:** v{m.get('schema_version', '?')}  ",
        f"**Mode:** `{bundle.mode}`  ",
        f"**Tools:** {', '.join(bundle.tools)}  ",
        f"**Scale:** {m.get('n_clusters', '?')} clusters × {m.get('n_chroms', '?')} chroms "
        f"({m.get('n_artifacts', '?')} artifacts)  ",
    ]
    tag = _resolve_figure_tag(bundle)
    if tag:
        lines.append(f"**Label space:** `{tag}`  ")
    lines.extend([
        "",
        "Cluster ids: `" + ", ".join(bundle.cluster_ids) + "`  ",
        "Chroms: `" + ", ".join(bundle.chroms) + "`  ",
        "",
    ])
    n_pairs = len(s.get("pairs", []))
    if n_pairs:
        lines.append(f"`cohort_summary.json` carries **{n_pairs}** (tool × RF-label) pair "
                     f"aggregates summarising every per-cluster comparison.")
    return "\n".join(lines) + "\n"


def section_reading_guide() -> str:
    return (
        "# Reading guide\n\n"
        "**Concordance metrics.**\n\n"
        "- *Pearson r* — linear correlation of paired ancestry proportions; "
        "insensitive to systematic bias.\n"
        "- *CCC* (Lin's concordance correlation coefficient) — agreement "
        "around the y=x line; penalises bias and miscalibration. A value of 1.0 "
        "is perfect agreement; ≥ 0.9 is excellent.\n"
        "- *MAE* — mean absolute error per sample, in proportion units (0–1).\n"
        "- *Jaccard@τ* — set agreement when each side's proportion is "
        "thresholded at τ. Robust to small floating-point differences.\n\n"
        "**μ-gating.** Per (cluster, RF-label) pairs with cluster_μ < 0.01 are "
        "reported with `pass = null` — there isn't enough of that ancestry in the "
        "cluster to make CCC meaningful. They contribute to `n_clusters_null`, "
        "not to passing/failing counts. In the traffic-light grid they appear as "
        "the grey `μ·∅` marker when *every* cluster was μ-gated.\n\n"
        "**RF caveat.** Random Forest is fundamentally a hard classifier; its "
        "per-class probabilities are not calibrated as proportions of admixture. "
        "Low CCC against RF is expected and not a defect of popout. The hard-call "
        "confusion matrix is the more meaningful comparison against RF.\n"
    )


def section_traffic_light(bundle: CohortBundle, assets: Path) -> str:
    pairs = bundle.summary.get("pairs", [])
    tools = [t for t in bundle.tools if t != ANCHOR_TOOL]
    if not pairs or not tools:
        return ""
    fig = plot_traffic_light_grid(pairs, tools, RF_LABELS)
    png = _save_fig(fig, assets / "traffic_light.png")
    return (
        "# Headline pass-rate grid\n\n"
        "Cells are coloured by the fraction of μ-evaluable clusters that "
        "passed the per-(cluster, RF-label) μ-gated threshold for each tool. "
        "Cell label is `#passing/#μ-evaluable`; **grey** cells (`μ·∅`) had "
        "every cluster μ-gated and could not be evaluated. **Green** ≥ 90% "
        "pass; **yellow** 50–90%; **red** < 50%.\n"
        + _embed_image(png, width="80%")
    )


def section_per_tool_concordance(
    bundle: CohortBundle, assets: Path,
) -> str:
    out: list[str] = ["# Per-tool global concordance\n"]
    summary_by_pair = {
        (p["tool"], p["rf_label"]): p for p in bundle.summary.get("pairs", [])
    }
    for tool in bundle.comparison_tools:
        metrics_path = bundle.bundle_dir / "cohort" / f"popout_vs_{tool}.metrics.tsv"
        header, rows_raw = _read_tsv(metrics_path)
        if not rows_raw:
            out.append(f"## popout vs {tool}\n\n> _no `{metrics_path.name}` found_\n")
            continue
        rows = [dict(zip(header, r)) for r in rows_raw]

        # Per-RF-label cohort summary table.
        tbl_rows: list[list[str]] = []
        for lab in RF_LABELS:
            p = summary_by_pair.get((tool, lab))
            if p is None:
                tbl_rows.append([lab, "—", "—", "—", "—", "—", "—"])
                continue
            n_pass = int(p.get("n_clusters_passing", 0) or 0)
            n_fail = int(p.get("n_clusters_failing", 0) or 0)
            n_null = int(p.get("n_clusters_null", 0) or 0)
            n_eval = n_pass + n_fail
            verdict = "—"
            if n_eval == 0:
                verdict = f"all {n_null} μ-gated"
            else:
                verdict = f"{n_pass}/{n_eval} pass"
                if n_null:
                    verdict += f" (+{n_null} μ·∅)"
            tbl_rows.append([
                lab,
                _fmt_num(p.get("mean_ccc_across_clusters")),
                _fmt_num(p.get("mean_pearson_r_across_clusters")),
                _fmt_int(n_pass),
                _fmt_int(n_fail),
                _fmt_int(n_null),
                verdict,
            ])

        out.append(f"## popout vs {tool}\n")
        out.append(_md_table(
            ["RF label", "mean CCC", "mean Pearson r", "n pass", "n fail", "n μ·∅", "verdict"],
            tbl_rows,
        ))

        # Per-cluster CCC plot.
        fig = plot_per_label_ccc(rows, tool, RF_LABELS)
        png = _save_fig(fig, assets / f"per_label_ccc_{tool}.png")
        out.append(_embed_image(png, width="90%"))
    return "\n".join(out)


def section_per_sample_mae(bundle: CohortBundle, assets: Path) -> str:
    path = bundle.bundle_dir / "cohort" / "per_sample_mae.tsv"
    header, rows_raw = _read_tsv(path)
    if not rows_raw:
        return ""
    rows = [dict(zip(header, r)) for r in rows_raw]
    out: list[str] = ["# Per-sample MAE distribution\n"]

    tbl: list[list[str]] = []
    for tool in bundle.comparison_tools:
        col = f"mae_vs_{tool}"
        vals = [_to_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            tbl.append([f"popout vs {tool}", "—", "—", "—", "—"])
            continue
        a = np.array(vals)
        tbl.append([
            f"popout vs {tool}",
            f"{a.size:,}",
            _fmt_num(np.median(a), 4),
            _fmt_num(np.percentile(a, 95), 4),
            _fmt_num(a.max(), 4),
        ])
    out.append(_md_table(["pair", "n samples", "median MAE",
                          "p95 MAE", "max MAE"], tbl))

    fig = plot_per_sample_mae_violin(rows, bundle.comparison_tools)
    png = _save_fig(fig, assets / "per_sample_mae.png")
    out.append(_embed_image(png, width="85%"))
    return "\n".join(out)


def section_confusion(bundle: CohortBundle, assets: Path) -> str:
    out: list[str] = ["# Hard-call confusion\n",
                      "Pooled across every (cluster, chrom) in the cohort. Heatmaps "
                      "are row-normalised; diagonal cells show popout↔tool agreement.\n"]
    any_rendered = False
    for tool in bundle.comparison_tools:
        path = bundle.bundle_dir / "cohort" / f"popout_vs_{tool}.confusion.tsv"
        header, rows_raw = _read_tsv(path)
        if not rows_raw:
            continue
        col_idx = {h: i for i, h in enumerate(header)}
        pop_col = "popout_label"
        if pop_col not in col_idx:
            continue
        col_labels = [c for c in header
                      if c not in ("cluster_id", "chrom", pop_col, "total")]
        row_label_set: list[str] = []
        pooled: dict[str, np.ndarray] = {}
        for r in rows_raw:
            lab = r[col_idx[pop_col]]
            if lab == "total":
                continue
            vec = np.zeros(len(col_labels), dtype=float)
            for j, c in enumerate(col_labels):
                vec[j] = _to_float(r[col_idx[c]]) or 0.0
            if lab not in pooled:
                pooled[lab] = vec.copy()
                row_label_set.append(lab)
            else:
                pooled[lab] += vec

        row_labels = [lab for lab in RF_LABELS if lab in pooled] + \
                     [lab for lab in row_label_set if lab not in RF_LABELS]
        cm = np.vstack([pooled[lab] for lab in row_labels])
        diag = 0.0
        total = cm.sum()
        for i, lab in enumerate(row_labels):
            if lab in col_labels:
                diag += cm[i, col_labels.index(lab)]
        agree = diag / total if total > 0 else 0.0

        fig = plot_confusion_heatmap(cm, row_labels, col_labels, tool)
        png = _save_fig(fig, assets / f"confusion_{tool}.png")
        out.append(f"## popout vs {tool} — diagonal agreement {_fmt_pct(agree)}\n")
        out.append(_embed_image(png, width="75%"))
        any_rendered = True
    if not any_rendered:
        out.append("> _no confusion matrices found_\n")
    return "\n".join(out)


def section_per_cluster_grid(bundle: CohortBundle) -> str:
    """Per (cluster_id, chrom) one-row summary."""
    path = bundle.bundle_dir / "cohort" / "manifest.tsv"
    header, rows_raw = _read_tsv(path)
    if not rows_raw:
        return ""
    col = {h: i for i, h in enumerate(header)}

    tier_path = bundle.bundle_dir / "cohort" / "tier1_metrics.tsv"
    th, tr = _read_tsv(tier_path)
    tier_by_key: dict[tuple[str, str], dict[str, str]] = {}
    if tr:
        tcol = {h: i for i, h in enumerate(th)}
        for row in tr:
            k = (row[tcol["cluster_id"]], row[tcol["chrom"]])
            tier_by_key.setdefault(k, {})[row[tcol["key"]]] = row[tcol["value"]]

    tools_no_anchor = bundle.comparison_tools
    out_header = ["cluster", "chrom", "n samples", "wallclock (s)", "peak RSS (GB)"]
    for t in tools_no_anchor:
        out_header.append(f"mean CCC vs {t}")
        out_header.append(f"pass vs {t}")

    rows: list[list[str]] = []
    for r in rows_raw:
        cid = r[col["cluster_id"]]
        chrom = r[col["chrom"]]
        row_out = [
            cid, chrom,
            _fmt_int(r[col.get("n_samples", -1)] if "n_samples" in col else None),
            _fmt_num(r[col["total_wallclock_seconds"]] if "total_wallclock_seconds" in col else None, 1),
            _fmt_num(r[col["peak_rss_gb"]] if "peak_rss_gb" in col else None, 2),
        ]
        t_metrics = tier_by_key.get((cid, chrom), {})
        for t in tools_no_anchor:
            ccc_key = f"popout_dx.mean_ccc_vs_{t}"
            pass_key = f"popout_dx.n_pairs_passing_vs_{t}"
            fail_key = f"popout_dx.n_pairs_failing_vs_{t}"
            ccc_v = t_metrics.get(ccc_key)
            n_pass = _to_float(t_metrics.get(pass_key))
            n_fail = _to_float(t_metrics.get(fail_key))
            if n_pass is None and n_fail is None:
                pass_str = "—"
            else:
                p = int(n_pass or 0)
                f = int(n_fail or 0)
                pass_str = f"{p}/{p + f}" if (p + f) > 0 else "all μ·∅"
            row_out.append(_fmt_num(ccc_v))
            row_out.append(pass_str)
        rows.append(row_out)

    return ("# Per-(cluster, chrom) performance grid\n"
            + _md_table(out_header, rows))


def section_provenance(bundle: CohortBundle) -> str:
    m = bundle.manifest
    s256 = m.get("sha256_per_artifact", {}) or {}
    out: list[str] = [
        "# Provenance\n",
        f"- schema_version: `{m.get('schema_version', '?')}`",
        f"- run_name: `{bundle.run_name}`",
        f"- mode: `{bundle.mode}`",
        f"- generated_at: `{m.get('generated_at', '?')}`",
        f"- n_artifacts: {m.get('n_artifacts', '?')}",
        "",
        "## Per-artifact SHA-256",
        "",
    ]
    if not s256:
        out.append("> _none recorded_")
    else:
        out.append(_md_table(
            ["artifact", "sha256"],
            [[k, v] for k, v in sorted(s256.items())],
        ))
    return "\n".join(out)


# ── Local-mode sections ──────────────────────────────────────────────────


def section_local_summary(bundle: CohortBundle) -> str:
    """Per-cluster local-mode metrics, sourced from cohort/local_per_sample.tsv."""
    path = bundle.bundle_dir / "cohort" / "local_per_sample.tsv"
    header, rows_raw = _read_tsv(path)
    if not rows_raw:
        return ""
    col = {h: i for i, h in enumerate(header)}
    by_cluster: dict[tuple[str, str], list[list[float | None]]] = {}
    for r in rows_raw:
        cid = r[col["cluster_id"]]
        chrom = r[col["chrom"]]
        vals = [
            _to_float(r[col["agree_pct"]]) if "agree_pct" in col else None,
            _to_float(r[col["jaccard_tracts"]]) if "jaccard_tracts" in col else None,
        ]
        by_cluster.setdefault((cid, chrom), []).append(vals)

    tbl: list[list[str]] = []
    for (cid, chrom), grp in sorted(by_cluster.items()):
        agree = [v[0] for v in grp if v[0] is not None]
        jaccard = [v[1] for v in grp if v[1] is not None]
        tbl.append([
            cid, chrom, _fmt_int(len(grp)),
            _fmt_pct(float(np.mean(agree))) if agree else "—",
            _fmt_num(float(np.mean(jaccard))) if jaccard else "—",
        ])
    return ("# Local-mode per-cluster summary\n"
            + "agree % = per-site agreement after `align_sites`; jaccard = tract-level\n\n"
            + _md_table(["cluster", "chrom", "n samples",
                         "mean per-site agree", "mean jaccard"], tbl))


def section_local_bp_confusion(bundle: CohortBundle, assets: Path) -> str:
    path = bundle.bundle_dir / "cohort" / "bp_confusion_segments.tsv.gz"
    header, rows_raw = _read_tsv(path)
    if not rows_raw:
        return ""
    col = {h: i for i, h in enumerate(header)}
    # Resolve column names (schema drift: file uses *_rf_label / seg_*_bp).
    flare_col = "flare_rf_label" if "flare_rf_label" in col else "flare_anc"
    popout_col = "popout_rf_label" if "popout_rf_label" in col else "popout_anc"
    start_col = "seg_start_bp" if "seg_start_bp" in col else "start_bp"
    end_col = "seg_end_bp" if "seg_end_bp" in col else "end_bp"
    if any(c not in col for c in (flare_col, popout_col, start_col, end_col)):
        return ""

    flare_labels: list[str] = []
    popout_labels: list[str] = []
    bp_grid: dict[tuple[str, str], int] = {}
    for r in rows_raw:
        fa = r[col[flare_col]]
        pa = r[col[popout_col]]
        try:
            length = int(r[col[end_col]]) - int(r[col[start_col]])
        except ValueError:
            continue
        if length <= 0:
            continue
        bp_grid[(fa, pa)] = bp_grid.get((fa, pa), 0) + length
        if fa not in flare_labels:
            flare_labels.append(fa)
        if pa not in popout_labels:
            popout_labels.append(pa)

    flare_labels.sort()
    popout_labels.sort()
    cm = np.array([
        [bp_grid.get((fa, pa), 0) for pa in popout_labels]
        for fa in flare_labels
    ], dtype=float)
    fig = plot_confusion_heatmap(cm, flare_labels, popout_labels, "popout (bp-weighted)")
    png = _save_fig(fig, assets / "bp_confusion.png")
    return (
        "# bp-confusion heatmap (local-mode View A)\n"
        "Pooled across every selected sample × haplotype; cells are bp-weighted "
        "row-fractions (rows = FLARE label, columns = popout label).\n"
        + _embed_image(png, width="75%")
    )


def section_boundary_localization(bundle: CohortBundle, assets: Path) -> str:
    path = bundle.bundle_dir / "cohort" / "boundary_localization.tsv"
    header, rows_raw = _read_tsv(path)
    if not rows_raw:
        return ""
    rows = [dict(zip(header, r)) for r in rows_raw]
    fig = plot_boundary_distance_hist(rows)
    png = _save_fig(fig, assets / "boundary_localization.png")
    return (
        "# Boundary localization (local-mode View B)\n"
        "Each row = one FLARE switch; we look for the nearest popout switch "
        "and record (a) bp distance and (b) whether the flanking RF labels "
        "agree on both sides.\n"
        + _embed_image(png, width="95%")
    )


def section_coarse_grid(bundle: CohortBundle, assets: Path) -> str:
    path = bundle.bundle_dir / "cohort" / "coarse_grid_summary.tsv"
    header, rows_raw = _read_tsv(path)
    if not rows_raw:
        return ""
    rows = [dict(zip(header, r)) for r in rows_raw]
    fig = plot_coarse_grid_curves(rows)
    png = _save_fig(fig, assets / "coarse_grid.png")
    return (
        "# Coarse-grid sweep (local-mode View C)\n"
        "Each faint line = one (cluster, sample, haplotype); the bold line is "
        "the per-resolution median. A high diagonal fraction at small "
        "resolutions means popout and FLARE agree at high spatial resolution; "
        "a flat-and-low curve means disagreement persists at every scale.\n"
        + _embed_image(png, width="90%")
    )


# ── Per-cluster long-form ────────────────────────────────────────────────


def section_per_cluster_pages(bundle: CohortBundle, assets: Path) -> str:
    """One sub-page per (cluster_id, chrom). Filters cohort TSVs by cluster_id."""
    manifest_path = bundle.bundle_dir / "cohort" / "manifest.tsv"
    header, rows_raw = _read_tsv(manifest_path)
    if not rows_raw:
        return ""
    col = {h: i for i, h in enumerate(header)}
    out: list[str] = []
    for r in rows_raw:
        cid = r[col["cluster_id"]]
        chrom = r[col["chrom"]]
        out.append(_page_break())
        out.append(f"# Per-cluster page: `{cid}` / `{chrom}`\n")
        for tool in bundle.comparison_tools:
            mp = bundle.bundle_dir / "cohort" / f"popout_vs_{tool}.metrics.tsv"
            mh, mr = _read_tsv(mp)
            if not mr:
                continue
            mcol = {h: i for i, h in enumerate(mh)}
            subset = [row for row in mr
                      if row[mcol["cluster_id"]] == cid
                      and row[mcol["chrom"]] == chrom]
            if not subset:
                continue
            tbl: list[list[str]] = []
            for row in subset:
                tbl.append([
                    row[mcol["rf_label"]],
                    _fmt_num(row[mcol["popout_mu"]]),
                    _fmt_num(row[mcol["pearson_r"]]),
                    _fmt_num(row[mcol["ccc"]]),
                    _fmt_num(row[mcol["mae_mean"]], 4),
                    _fmt_num(row[mcol["jaccard_0.25"]]),
                    row[mcol["pass"]] or "μ·∅",
                ])
            out.append(f"## popout vs {tool}\n")
            out.append(_md_table(
                ["RF label", "popout μ", "Pearson r", "CCC", "MAE mean",
                 "Jaccard@0.25", "pass"],
                tbl,
            ))
    return "\n".join(out)


# ── Document assembly ────────────────────────────────────────────────────


def build_markdown(
    bundle: CohortBundle,
    assets: Path,
    include_per_cluster: bool,
) -> str:
    parts: list[str] = [md_frontmatter(bundle)]
    parts.append(section_cover(bundle))
    parts.append(_page_break())
    parts.append(section_reading_guide())
    parts.append(_page_break())
    parts.append(section_traffic_light(bundle, assets))
    parts.append(_page_break())
    parts.append(section_per_tool_concordance(bundle, assets))
    parts.append(_page_break())
    parts.append(section_per_sample_mae(bundle, assets))
    parts.append(_page_break())
    parts.append(section_confusion(bundle, assets))
    parts.append(_page_break())
    parts.append(section_per_cluster_grid(bundle))

    if bundle.mode == "global_local":
        parts.append(_page_break())
        parts.append(section_local_summary(bundle))
        parts.append(_page_break())
        parts.append(section_local_bp_confusion(bundle, assets))
        parts.append(_page_break())
        parts.append(section_boundary_localization(bundle, assets))
        parts.append(_page_break())
        parts.append(section_coarse_grid(bundle, assets))

    parts.append(_page_break())
    parts.append(section_provenance(bundle))

    if include_per_cluster:
        parts.append(section_per_cluster_pages(bundle, assets))

    return "\n".join(p for p in parts if p)


# ── pandoc invocation ────────────────────────────────────────────────────


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


# ── CLI ─────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a PDF report from a popout DX cohort bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cohort-bundle", type=Path, required=True,
                   help="path to cohort_dx.<run>.tar.gz or an unpacked cohort_dx/ dir")
    p.add_argument("--out", type=Path, required=True, help="destination PDF (or .md) path")
    p.add_argument("--clusters", default=None,
                   help="comma-separated cluster_ids to subset (default: all)")
    p.add_argument("--max-clusters", type=int, default=None,
                   help="cap on number of distinct cluster_ids included")
    p.add_argument("--per-cluster", action="store_true",
                   help="append per-(cluster, chrom) long-form pages")
    p.add_argument("--keep-md", action="store_true",
                   help="keep the intermediate .md document next to the output PDF")
    return p.parse_args(argv)


def _apply_filters(
    bundle: CohortBundle,
    cluster_filter: list[str] | None,
    max_clusters: int | None,
) -> CohortBundle:
    cids = bundle.cluster_ids
    if cluster_filter:
        wanted = [c for c in cluster_filter if c in cids]
        unknown = [c for c in cluster_filter if c not in cids]
        if unknown:
            raise ValueError(
                f"--clusters references unknown cluster_id(s): {unknown}; "
                f"available: {cids}"
            )
        cids = wanted
    if max_clusters is not None and max_clusters > 0:
        cids = cids[:max_clusters]
    if cids == bundle.cluster_ids:
        return bundle
    new_manifest = dict(bundle.manifest)
    new_manifest["cluster_ids"] = cids
    return CohortBundle(
        bundle_dir=bundle.bundle_dir,
        manifest=new_manifest,
        summary=bundle.summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    md_only = args.out.suffix.lower() == ".md"
    assets_dir = args.out.parent / f"{args.out.stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out if md_only else args.out.parent / f"{args.out.stem}.md"

    with tempfile.TemporaryDirectory(prefix="popout_dx_report_") as tmpdir:
        bundle_dir = resolve_bundle_dir(args.cohort_bundle, Path(tmpdir))
        bundle = load_cohort_bundle(bundle_dir)
        _log(f"loaded bundle: {bundle.run_name} "
             f"(mode={bundle.mode}, {len(bundle.cluster_ids)} clusters, "
             f"tools={bundle.tools})")

        cluster_filter = (
            [c.strip() for c in args.clusters.split(",") if c.strip()]
            if args.clusters else None
        )
        bundle = _apply_filters(bundle, cluster_filter, args.max_clusters)
        _log(f"rendering {len(bundle.cluster_ids)} cluster(s)")

        md = build_markdown(bundle, assets_dir,
                            include_per_cluster=args.per_cluster)
        md_path.write_text(md)
        _log(f"wrote {md_path} ({md.count(chr(10))} lines)")

        if md_only:
            _log("--out is .md; skipping pandoc")
        else:
            run_pandoc(md_path, args.out)
            if not args.keep_md:
                md_path.unlink(missing_ok=True)
            _log(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
