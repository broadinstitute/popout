#!/usr/bin/env python3
"""Build a self-contained PDF analysis report for a popout run.

Combines run metadata, SUMMARY.md, and all diagnostic images into a single
markdown document, then converts to PDF via pandoc.

Usage:
    python build_report_pdf.py --data-dir /path/to/main_v4 --run-name main_v4
"""

import argparse
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path


def read_if_exists(path: Path) -> str | None:
    if path.exists():
        return path.read_text()
    return None


def parse_command_line(stdout_text: str) -> str | None:
    for line in stdout_text.splitlines():
        if line.startswith("=== Running:"):
            return line.removeprefix("=== Running: ").removesuffix(" ===").strip()
    return None


def detect_run_prefix(data_dir: Path) -> str:
    """Return the per-run output prefix used by popout in this dir.

    Stem-globs for ``*.summary.json`` and strips the suffix. Falls back to
    the legacy ``aou_v9_hmm`` prefix when no summary is present so existing
    reports keep building.
    """
    summaries = sorted(data_dir.glob("*.summary.json"))
    if summaries:
        name = summaries[0].name
        return name.removesuffix(".summary.json")
    return "aou_v9_hmm"


def render_priors_assignments(path: Path) -> str | None:
    """Render the ``--priors-dump-assignments`` audit TSV as a markdown table.

    The first line is a comment with the per-component nearest-1KG
    annotations (``# nearest_1KG\\tEUR(r=0.99)\\t...``). The second is the
    column header (``prior\\tcomp_0\\tcomp_1\\t...``). Body rows are one per
    prior with K weight columns.

    The table is **transposed** (rows = components, cols = priors) so
    the K=18-component case still fits page width.  Each prior's
    argmax cell is bolded; diffuse priors show no bolded row.
    """
    if not path.exists():
        return None
    text = path.read_text().rstrip("\n")
    if not text:
        return None
    lines = text.split("\n")
    if len(lines) < 2:
        return None
    annot_line = lines[0]
    header_line = lines[1]
    body = lines[2:]

    annots: list[str] = []
    if annot_line.startswith("#"):
        annots = annot_line.lstrip("#").strip().split("\t")[1:]

    comp_labels = header_line.split("\t")[1:]

    # Parse the (priors, components) weight matrix from the TSV body
    # then transpose to (components, priors) for rendering.
    prior_names: list[str] = []
    weights_pc: list[list[float]] = []
    for line in body:
        cells = line.split("\t")
        if not cells:
            continue
        prior_names.append(cells[0])
        try:
            row_vals = [float(v) for v in cells[1:]]
        except ValueError:
            row_vals = [float("nan")] * len(comp_labels)
        # Pad / truncate to K so the transpose below is well-shaped.
        if len(row_vals) < len(comp_labels):
            row_vals = row_vals + [float("nan")] * (len(comp_labels) - len(row_vals))
        weights_pc.append(row_vals[: len(comp_labels)])

    if not weights_pc or not comp_labels:
        return None

    # Per-prior argmax across components — used to bold the right cell.
    priors_argmax: list[int] = []
    for p_idx in range(len(prior_names)):
        col_vals = [weights_pc[p_idx][c_idx] for c_idx in range(len(comp_labels))]
        priors_argmax.append(int(max(range(len(col_vals)), key=lambda i: col_vals[i])))

    out = ["## Priors → Component Assignment\n"]
    out.append(
        "Soft assignment of each prior to discovered components from the "
        "final EM iteration.  Rows are components (with the 1KG/superpop "
        "nearest-neighbor annotation); columns are priors.  Bolded cell = "
        "argmax for that prior — i.e. where its mass landed.  Multiple "
        "priors bolding the same row means they bind to the same "
        "component (e.g. priors_phase_2a: MID and AFR both bind to "
        "comp_2 — the MID-on-AFR anomaly).\n"
    )

    # Header
    out.append("| Component | nearest_1KG | " + " | ".join(prior_names) + " |")
    out.append("|---|---|" + "|".join(["---:" for _ in prior_names]) + "|")

    # Body — one row per component
    for c_idx, comp in enumerate(comp_labels):
        annot = annots[c_idx] if c_idx < len(annots) else "?"
        cells = [comp, annot]
        for p_idx in range(len(prior_names)):
            v = weights_pc[p_idx][c_idx]
            fmt = f"{v:.3f}"
            if priors_argmax[p_idx] == c_idx:
                fmt = f"**{fmt}**"
            cells.append(fmt)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def render_pre_merge_leaves(meta_path: Path) -> str | None:
    """Render the ``recursive_pre_merge.leaf_meta.tsv`` table.

    Columns: label, path, depth, n_haps, bic_score. Sorted by hap count
    (descending) so the dominant clusters surface at the top.
    """
    if not meta_path.exists():
        return None
    text = meta_path.read_text().rstrip("\n")
    if not text:
        return None
    lines = text.split("\n")
    if len(lines) < 2:
        return None
    header = lines[0].split("\t")
    body = [l.split("\t") for l in lines[1:] if l]

    col = {h: i for i, h in enumerate(header)}
    if not all(k in col for k in ("label", "path", "depth", "n_haps")):
        return None

    def n_haps(row: list[str]) -> int:
        try:
            return int(row[col["n_haps"]])
        except (ValueError, IndexError):
            return 0

    body.sort(key=n_haps, reverse=True)

    out = ["## Pre-Merge Recursive Leaves\n"]
    out.append(
        "Components produced by the recursive K=2 EM splitter before "
        "post-EM consolidation. Hap counts add up to the seeding-eligible "
        "haplotype total. Tree paths encode the binary partition history "
        "(``L00`` = left-of-left at depth 2, etc.).\n"
    )
    out.append("| Label | Path | Depth | n_haps | BIC |")
    out.append("|------:|------|------:|-------:|----:|")
    for row in body:
        try:
            bic = f"{float(row[col['bic_score']]):,.0f}" if "bic_score" in col else "?"
        except (ValueError, IndexError):
            bic = "?"
        try:
            n = f"{int(row[col['n_haps']]):,}"
        except (ValueError, IndexError):
            n = row[col["n_haps"]] if "n_haps" in col else "?"
        out.append(
            f"| {row[col['label']]} | "
            f"{row[col['path']]} | "
            f"{row[col['depth']]} | "
            f"{n} | "
            f"{bic} |"
        )
    return "\n".join(out) + "\n"


def render_coverage_checks(check_path: Path, per_chrom_path: Path) -> str | None:
    """Render `coverage_check.tsv` + `per_chrom_consistency.tsv` as a section."""
    if not check_path.exists():
        return None
    lines = check_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    out = [
        "# FLARE QC\n",
        "**What this section shows.** Four structural sanity checks (§8.1 of "
        "the validation plan) on this FLARE run. All four must PASS for a run "
        "to be considered valid:\n\n"
        "- *input_set_equals_output_set* — every sample in the FLARE input VCF "
        "  also appears in the global ancestry output; no silent sample drops.\n"
        "- *qc_sample_count_consistent* — FLARE's own QC TSV agrees with the "
        "  output sample count.\n"
        "- *output_site_count_matches_log* — every panel x target site "
        "  intersection FLARE reported in its log was actually written.\n"
        "- *site_coverage_ge_95pct_of_intersection* — site retention &gt;= 95% of "
        "  the intersection. A near-zero ratio means FLARE silently dropped "
        "  most variants and the run should be re-investigated.\n\n"
        "The per-chromosome table below cross-checks the input vs output "
        "record counts from FLARE's qc.tsv — useful for spotting a chromosome "
        "where Stage A delivered a malformed input.\n",
        "## Output coverage\n",
        "| Check | Status | Detail |",
        "|---|:---:|---|",
    ]
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        check, status, detail = parts[0], parts[1], parts[2]
        out.append(f"| {check} | **{status}** | `{detail}` |")
    out.append("")
    if per_chrom_path.exists():
        pc_lines = per_chrom_path.read_text().strip().splitlines()
        if len(pc_lines) >= 2:
            out.append("## Per-chromosome record counts\n")
            out.append("| Chrom | gt_records | out_records |")
            out.append("|---|---:|---:|")
            for line in pc_lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 3:
                    out.append(f"| {parts[0]} | {int(parts[1]):,} | {int(parts[2]):,} |")
            out.append("")
    return "\n".join(out) + "\n"


def render_admixture_section(corr_path: Path, scatter_img: Path) -> str | None:
    if not corr_path.exists() and not scatter_img.exists():
        return None
    out = [
        "# Cross-tool consistency: ADMIXTURE\n",
        "**What this section shows.** Per-sample ancestry proportions from "
        "this run vs the team's ADMIXTURE Q values (the software, distinct "
        "from FLARE/popout/RF — used here as an independent cross-check; "
        "ADMIXTURE is also what AoU uses for cohort selection). For each RF "
        "label we report a Pearson r and a regression slope between this "
        "tool's proportion and the ADMIXTURE column that bootstraps to the "
        "same label. **What 'good' looks like:** r near 1, slope near 1, "
        "intercept near 0 — the two methods are agreeing. r near 0 means "
        "they disagree about that ancestry's signal; slope far from 1 means "
        "one method consistently over- or under-calls it.\n",
    ]
    if corr_path.exists():
        lines = corr_path.read_text().strip().splitlines()
        if len(lines) >= 2:
            out.append("| FLARE idx | FLARE label | Admixture col | Pearson r | Slope | Intercept | n |")
            out.append("|---:|---:|---:|---:|---:|---:|---:|")
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 7:
                    out.append("| " + " | ".join(parts) + " |")
            out.append("")
    if scatter_img.exists():
        out.append(f"![ADMIXTURE vs FLARE]({scatter_img})\n")
    return "\n".join(out) + "\n"


def render_regional_section(diag: Path) -> str | None:
    """Top-10 most significant windows + per-chrom manhattan plots."""
    win_path = diag / "regional_windows.tsv.gz"
    if not win_path.exists():
        return None
    import gzip as _gz
    rows = []
    with _gz.open(win_path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            try:
                q = float(parts[col["q"]])
            except (ValueError, KeyError):
                continue
            rows.append((q, parts))
    rows.sort(key=lambda r: r[0])

    # Format positions in Mb for readability (Start/End in raw bp can be 9 digits).
    def _mb(s: str) -> str:
        try:
            return f"{int(s)/1e6:.2f}"
        except (ValueError, TypeError):
            return s

    out = [
        "# Regional QC (§8.3)\n",
        "**What this section shows.** For every 1 Mb sliding window on each "
        "chromosome (250 kb step), the bp-weighted mean ancestry proportion "
        "across all haplotypes is computed per ancestry. Each (window, ancestry) "
        "pair is then z-scored against the chromosome-wide mean for that "
        "ancestry, and a Benjamini-Hochberg FDR correction is applied across "
        "all tests. A significant window means **the inferred ancestry "
        "proportion in that segment deviates from what would be expected given "
        "the rest of the chromosome** — a sign that the LAI call may be "
        "unreliable there (segdup, high-LD, centromere flank, panel-coverage "
        "hole, etc.). Each panel of the manhattan plot below is one ancestry; "
        "red dots are FDR-significant.\n",
        "Top-10 windows by FDR-corrected q-value (lower is more deviant):\n",
        "| Chrom | Start (Mb) | End (Mb) | Ancestry | mean_anc | z | p | q | mask |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for q, parts in rows[:10]:
        anc_col = "ancestry_name" if "ancestry_name" in col else "ancestry"
        cells = [
            parts[col["chrom"]],
            _mb(parts[col["start"]]),
            _mb(parts[col["end"]]),
            parts[col[anc_col]],
            parts[col["mean_anc"]],
            parts[col["z"]],
            parts[col["p"]],
            parts[col["q"]],
            parts[col["mask_region"]] or "—",
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    for img in sorted(diag.glob("regional_qc_*.png")):
        out.append(f"![{img.stem}]({img})\n")
    return "\n".join(out) + "\n"


def render_hap_disagreement_section(diag: Path) -> str | None:
    by_label = diag / "hap_disagreement_by_rf_label.png"
    by_region = diag / "hap_disagreement_by_region.png"
    if not by_label.exists() and not by_region.exists():
        return None
    out = [
        "# Hap-disagreement rate (§8.1 / §5.3)\n",
        "**What this section shows.** Per-sample, what fraction of the "
        "genome (bp-weighted) gets a *different* ancestry call on hap1 vs "
        "hap2? Two non-exclusive interpretations:\n\n"
        "- **Real ancestry asymmetry**: in an admixed diploid one parent's "
        "  contribution can be a different ancestry from the other's. Higher "
        "  rates in known-admixed labels (afr, amr, mixed) are expected.\n"
        "- **Phasing-error proxy**: switch errors in the upstream phasing "
        "  artificially split a single-ancestry haplotype across hap1 and "
        "  hap2, inflating disagreement. Single-ancestry buckets (e.g. "
        "  eas-only in cluster_007) should sit near 0%; large rates there "
        "  point at phasing or LAI noise, not real biology.\n\n"
        "By-label panel: violins of per-sample disagreement, grouped by the "
        "sample's RF hard call. By-region panel (when a region BED is "
        "supplied): bp-weighted disagreement inside each named region "
        "(centromere flank, segdup, high-LD, …) for region-specific QC.\n",
    ]
    if by_label.exists():
        out.append(f"## By RF hard label\n\n![hap disagreement by RF label]({by_label})\n")
    if by_region.exists():
        out.append(f"## By named genomic region\n\n![hap disagreement by region]({by_region})\n")
    return "\n".join(out) + "\n"


def render_compare_tracts_section(diag: Path) -> str | None:
    """Cross-tool tract concordance section (FLARE vs popout).

    Looks for the compare_tracts/ subdir produced by compare_tracts.py. Each
    figure is preceded by the same interpretation paragraph the script writes
    into SUMMARY.md, so the PDF stands alone.
    """
    sub = diag / "compare_tracts"
    summary = sub / "SUMMARY.md"
    if not summary.exists():
        return None

    out: list[str] = [
        "# Cross-tool tract concordance (FLARE vs popout)\n",
        "**What this section shows.** Per-bp local-ancestry comparison "
        "between two LAI tools (FLARE and popout) run on the same phased "
        "VCF. Three complementary views decompose raw disagreement into "
        "**calibration drift** (tools dispute identity at the same bp) vs "
        "**boundary-localization error** (tools agree on flanking ancestries "
        "but disagree on where the switch lies). See "
        "diagnostics/GLOSSARY.md for the canonical terminology.\n",
    ]

    audit = sub / "hap_pairing_audit.tsv"
    if audit.exists():
        with open(audit) as f:
            f.readline()
            parts = f.readline().rstrip("\n").split("\t")
        if len(parts) >= 4:
            out.append(
                f"**Hap-pairing audit:** primary={parts[0]}, "
                f"alternate={parts[1]}, verdict={parts[3]}. "
                "(If alternate > primary + 0.05 the script raises; "
                "this verdict is post-hoc evidence the convention is intact.)\n"
            )

    a_png = sub / "view_a_bp_confusion.png"
    a_violin = sub / "view_a_agreement_violin.png"
    if a_png.exists():
        out.append(
            "## View A — raw bp-agreement (headline)\n\n"
            "_Each cell is the total bp across the cohort where FLARE called "
            "the row's RF label and popout called the column's. Diagonal = "
            "bp where they agree on identity; off-diagonals = bp where they "
            "disagree. The diagonal fraction is the literal cross-tool "
            "bp-agreement rate. This mixes calibration drift and "
            "boundary-localization — Views B and C decompose them._\n\n"
            f"![View A bp confusion]({a_png})\n"
        )
    if a_violin.exists():
        out.append(
            "_Per-sample bp-agreement fraction, stratified by RF hard "
            "label. A wide low-mean violin in a label = many samples in "
            "that bucket disagree across tools; narrow high violin = the "
            "tools converge for that label._\n\n"
            f"![View A agreement violin]({a_violin})\n"
        )

    b_png = sub / "view_b_boundary_offset_histogram.png"
    if b_png.exists():
        out.append(
            "## View B — boundary-localization audit\n\n"
            "_For each FLARE ancestry switch, find the nearest popout "
            "switch on the same haplotype with the same flanking RF-label "
            "pair (within 5 Mb). Signed offset = popout_bp - flare_bp. "
            "Histogram tight around zero = tools agree on boundaries; broad "
            "with heavy tails = boundary-localization drives View A's "
            "disagreement. Low overall match rate = popout doesn't resolve "
            "the transition (often a site-density issue)._\n\n"
            f"![View B offsets]({b_png})\n"
        )

    c_sweep = sub / "view_c_grid_sweep.png"
    c_coarse_pngs = sorted(sub.glob("view_c_coarse_confusion_*.png"))
    if c_sweep.exists():
        out.append(
            "## View C — resolution-controlled coarse-grid sweep\n\n"
            "_At each grid size, take the bp-weighted dominant RF label "
            "per (sample, hap, window) for each tool and compare. As grid "
            "size grows, fine boundary disagreements get absorbed into the "
            "dominant call; the diagonal fraction rises if disagreement is "
            "boundary-driven. A flat sweep curve means disagreement is "
            "calibration drift — the tools dispute identity, not boundary "
            "placement. The gap between View A and the 5 Mb diagonal is "
            "the boundary-localization contribution; the gap from the 5 Mb "
            "diagonal to 1.0 is calibration drift._\n\n"
            f"![View C sweep]({c_sweep})\n"
        )
    for img in c_coarse_pngs:
        out.append(f"![View C coarse confusion]({img})\n")

    hell_violin = sub / "hellinger_by_rf_label.png"
    hell_cdf = sub / "hellinger_cdf.png"
    if hell_violin.exists() or hell_cdf.exists():
        out.append(
            "## Per-sample Hellinger\n\n"
            "_Each sample contributes a 6-vector of bp-fractions per RF "
            "label per tool (normalized per tool's covered bp). Hellinger "
            "in [0, 1] measures the distance between these two "
            "distributions. 0 = identical genome composition; 1 = disjoint "
            "support. Tight low-distance violins for labels where FLARE "
            "has full support are expected; `mixed` is widest._\n"
        )
        if hell_violin.exists():
            out.append(f"![Hellinger by RF label]({hell_violin})\n")
        if hell_cdf.exists():
            out.append(f"![Hellinger CDF]({hell_cdf})\n")

    tl = sub / "tract_length_distribution.png"
    if tl.exists():
        out.append(
            "## Tract-length distribution\n\n"
            "_Site-density disparity in physical terms: FLARE typically "
            "uses many more sites per chromosome but tracts are only "
            "modestly shorter because most extra sites land inside same-"
            "ancestry tracts. Wide divergence at the short end = FLARE "
            "is resolving short switches popout can't see._\n\n"
            f"![Tract length distribution]({tl})\n"
        )

    manhattan_main = next(iter(sub.glob("manhattan_*.png")), None)
    manhattan_facet = next(iter(sub.glob("manhattan_*_by_rf_label.png")), None)
    if manhattan_main:
        out.append(
            f"## Manhattan: per-window agreement on chromosome\n\n"
            "_Cohort average of bp-agreement per sliding window. Dips "
            "indicate regions of systematic disagreement (centromere, "
            "low-complexity, panel gaps). Faceted view breaks it out by "
            "RF hard label so you can see which ancestry's samples are "
            "driving any localized drop._\n\n"
            f"![Manhattan]({manhattan_main})\n"
        )
    if manhattan_facet:
        out.append(f"![Manhattan by RF label]({manhattan_facet})\n")

    return "\n".join(out) + "\n"


def render_run_identity(data_dir: Path, prefix: str, run_name: str) -> str | None:
    """Render the "Run Identity" sub-table.

    The canonical answer to "which tool? which version? which cohort?".
    Pulls from <prefix>.summary.json (+ <prefix>.flare_manifest.json for
    FLARE-derived runs). See diagnostics/GLOSSARY.md.
    """
    summary_path = data_dir / f"{prefix}.summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    config = summary.get("config", {})

    method = config.get("method", "?")
    if method == "flare":
        tool = "FLARE"
        version_raw = summary.get("popout_version", "")
        # popout_version looks like "FLARE  flare version 0.6.0 [..]"
        m = re.search(r"flare version (\S+)", version_raw)
        version = m.group(1) if m else "?"
    else:
        tool = "popout"
        version = summary.get("popout_version", "?")

    flare_log = summary.get("flare_log", {})
    raw_params = flare_log.get("raw_params", {})
    n_samples = flare_log.get("target_samples") or summary.get("n_samples", "?")

    # Chroms: derive from the site funnel if popout, otherwise from prefix.
    funnel = summary.get("site_filter_funnel", {})
    if funnel:
        chroms = ",".join(sorted(funnel.keys()))
    else:
        # Best-effort: pull "chrN" from the prefix or the gt path.
        m = re.search(r"\b(chr[0-9XYM]+)\b", prefix) or re.search(r"\b(chr[0-9XYM]+)\b", raw_params.get("gt", ""))
        chroms = m.group(1) if m else "?"

    # Cohort name: derive from the gt path's filename when present.
    gt_path = raw_params.get("gt", "")
    cohort = "?"
    if gt_path:
        gt_stem = Path(gt_path).name
        # Drop ".vcf.gz" / ".aou.v9.phased.vcf.gz" / etc.
        for suffix in (".aou.v9.phased.vcf.gz", ".vcf.gz", ".pgen", ".bgz"):
            if gt_stem.endswith(suffix):
                gt_stem = gt_stem[: -len(suffix)]
                break
        cohort = gt_stem
    elif "cohort" in summary:
        cohort = summary["cohort"]

    # Reference panel name: take basename of `ref` argument.
    ref = raw_params.get("ref", "")
    panel = "—"
    if ref:
        ref_name = Path(ref).name
        # gnomad_lai_90 from `chr1.gnomad_lai_90.vcf.bgz`
        m = re.search(r"\.(gnomad_[a-z0-9_]+)\.", ref_name)
        if m:
            panel = m.group(1)
        else:
            panel = ref_name

    wall_str = flare_log.get("wallclock_str") or summary.get("wallclock_str")
    if not wall_str:
        wall_s = summary.get("total_wall_clock_sec", 0)
        if wall_s:
            h, rem = divmod(int(wall_s), 3600)
            m, s = divmod(rem, 60)
            wall_str = f"{h}h {m}m {s}s"
        else:
            wall_str = "?"

    rows = [
        f"| Tool        | {tool} {version} |",
        f"| Run name    | {run_name} |",
        f"| Cohort      | {cohort} |",
        f"| n_samples   | {n_samples} |",
        f"| Chrom(s)    | {chroms} |",
        f"| Ref panel   | {panel} |",
        f"| Wallclock   | {wall_str} |",
    ]
    out = ["## Run Identity\n",
           "Canonical run identity. See `diagnostics/GLOSSARY.md` for the "
           "vocabulary used in this report.\n",
           "| Field | Value |\n|---|---|\n" + "\n".join(rows) + "\n"]
    return "\n".join(out) + "\n"


def detect_tool(data_dir: Path, prefix: str) -> str:
    """Return canonical tool name ('FLARE' or 'popout') from summary.json.

    Popout's native runs report config.method="hmm"; FLARE-derived runs
    (via flare_to_popout_format.py) report method="flare".
    """
    summary_path = data_dir / f"{prefix}.summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        method = s.get("config", {}).get("method", "").lower()
        if method == "flare":
            return "FLARE"
        if method in ("hmm", "popout"):
            return "popout"
    return "popout"


def build_markdown(data_dir: Path, run_name: str) -> str:
    diag = data_dir / "diagnostics"
    sections = []
    prefix = detect_run_prefix(data_dir)
    tool = detect_tool(data_dir, prefix)

    # ── Title ──
    sections.append(
        f"---\n"
        f"title: \"{tool} Analysis Report: {run_name}\"\n"
        f"geometry: margin=0.75in\n"
        f"colorlinks: true\n"
        f"header-includes:\n"
        f"  - \\usepackage{{adjustbox}}\n"
        f"  - \\usepackage{{longtable}}\n"
        f"  - \\let\\oldtable\\table\n"
        f"  - \\let\\endoldtable\\endtable\n"
        f"  - \\renewenvironment{{table}}[1][]{{\\oldtable[#1]\\adjustbox{{max width=\\textwidth}}\\bgroup}}{{\\egroup\\endoldtable}}\n"
        f"---\n"
    )

    # ── 1. Run Configuration ──
    sections.append("# Run Configuration\n")

    run_id_md = render_run_identity(data_dir, prefix, run_name)
    if run_id_md is not None:
        sections.append(run_id_md)

    stdout = read_if_exists(data_dir / "stdout")
    if stdout:
        cmd = parse_command_line(stdout)
        if cmd:
            # Break into flag per line for readability
            parts = cmd.split(" --")
            cmd_fmt = parts[0] + " \\\n    --" + " \\\n    --".join(parts[1:])
            sections.append(f"```\n{cmd_fmt}\n```\n")

    model_text = read_if_exists(data_dir / f"{prefix}.model")
    if model_text:
        sections.append(f"**Model:**\n\n```\n{model_text.strip()}\n```\n")

    summary_path = data_dir / f"{prefix}.summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        config = summary.get("config", {})
        runtime = summary.get("runtime", {})
        site_funnel = summary.get("site_filter_funnel", {})

        rows = []
        rows.append(f"| popout version | `{summary.get('popout_version', '?')}` |")
        rows.append(f"| method | {config.get('method', '?')} |")
        rows.append(f"| n_em_iter | {config.get('n_em_iter', '?')} |")
        rows.append(f"| gen_since_admix (initial) | {config.get('gen_since_admix', '?')} |")
        rows.append(f"| thin_cm | {config.get('thin_cm', '?')} |")
        rows.append(f"| seed | {config.get('seed', '?')} |")
        rows.append(f"| wall clock | {summary.get('total_wall_clock_sec', 0):.0f}s |")

        devices = runtime.get("device_info", {}).get("devices", [])
        if devices:
            rows.append(f"| GPU | {devices[0].get('kind', '?')} |")

        for chrom, funnel in site_funnel.items():
            rows.append(f"| chr{chrom} sites (biallelic/thinned/final) | "
                        f"{funnel.get('sites_biallelic', '?')} / "
                        f"{funnel.get('sites_after_thinning', '?')} / "
                        f"{funnel.get('sites_final', '?')} |")
            rows.append(f"| chr{chrom} genetic length | {funnel.get('genetic_length_cm', '?')} cM |")

        sections.append("| Parameter | Value |\n|---|---|\n" + "\n".join(rows) + "\n")

    # Post-EM consolidation — simplified table
    consol = read_if_exists(data_dir / f"{prefix}.post_em_consolidation.tsv")
    if consol:
        lines = consol.strip().split("\n")
        header = lines[0].split("\t")
        col = {h: i for i, h in enumerate(header)}
        table_lines = [
            "## Post-EM Consolidation\n",
            "| Source | Target | Reason | mu (source) | FST to target |",
            "|------:|------:|--------|----------:|--------------:|",
        ]
        for line in lines[1:]:
            c = line.split("\t")
            src = c[col.get("source_idx", 0)]
            tgt = c[col.get("target_idx", 1)]
            reason = c[col.get("reason", 2)]
            mu_s = c[col.get("mu_source", 3)]
            fst = c[col.get("fst_to_target", 5)]
            table_lines.append(f"| {src} | {tgt} | {reason} | {mu_s} | {fst} |")
        sections.append("\n".join(table_lines) + "\n")

    # Priors → component assignment audit (priors-only runs).
    pa_md = render_priors_assignments(
        data_dir / f"{prefix}.priors_assignments.tsv"
    )
    if pa_md is not None:
        sections.append(pa_md)

    # Pre-merge recursive seeding leaves.
    leaf_md = render_pre_merge_leaves(
        data_dir / f"{prefix}.recursive_pre_merge.leaf_meta.tsv"
    )
    if leaf_md is not None:
        sections.append(leaf_md)

    # ── FLARE QC (§8.1: coverage, per-chrom consistency) ──
    qc_md = render_coverage_checks(
        diag / "coverage_check.tsv",
        diag / "per_chrom_consistency.tsv",
    )
    if qc_md is not None:
        sections.append(qc_md)

    # ── 2. FST Tree ──
    fst_img = diag / "fst_tree.png"
    if fst_img.exists():
        sections.append("# FST Tree\n")
        sections.append(f"![FST Tree]({fst_img})\n")

    # ── 3. SUMMARY.md (soft correlation, verdict, confusion matrix) ──
    summary_md = read_if_exists(diag / "SUMMARY.md")
    if summary_md:
        # Strip the first H1 line (we'll re-title it)
        lines = summary_md.split("\n")
        body = "\n".join(lines[1:] if lines[0].startswith("# ") else lines)
        sections.append(f"# {tool} vs RF Classifier Concordance\n" + body + "\n")

    # ── 4. Concordance Figures ──
    figure_order = [
        ("correlation_heatmap.png", "Correlation Heatmap (K x 6)",
         "Pearson r between each of this tool's K raw ancestry columns and "
         "each of the 6 RF reference labels. Look for one strong positive "
         "match per row; large negative off-diagonals suggest the slope-"
         "override may have relabeled a component."),
        ("merged_confusion_matrix.png", "Confusion Matrix",
         "Row = RF hard label (the 'true' label per the RF classifier). "
         "Column = this tool's hard call after collapsing to 6 RF labels. "
         "Each cell shows count and row-normalized recall. Diagonals near "
         "100% = clean per-label agreement; 'mixed' row covers samples "
         "with RF max prob < 0.8."),
        ("soft_proportion_hexbin.png", "Soft Proportion Agreement",
         "One hexbin per RF reference label. X = RF probability for that "
         "label; Y = this tool's proportion. Density colored on a log "
         "scale. Tight on the y=x diagonal = soft proportions match across "
         "the cohort. Bowing above/below the diagonal = systematic over/under-"
         "call by this tool for that label."),
        ("admixture_comparison.png", "Admixture Comparison Bars",
         "Stacked-color strip plot. Each vertical line is one sample; colors "
         "are ancestry proportions stacked to 1.0. **When a secondary tool is "
         "supplied this figure has three rows** (this tool, secondary tool, "
         "RF) so you can visually compare all three estimators on the same "
         "samples in the same order. Two-row mode is two-way."),
        ("l1_distance_violin.png", "Per-Sample L1 Distance",
         "For each sample, L1 distance between this tool's soft proportions "
         "and RF's probability vector. Lower = better agreement. Grouped by "
         "RF hard label so you can see whether disagreement concentrates in "
         "particular ancestries (typically the under-represented ones)."),
        ("calibration_curves.png", "Calibration Curves (per RF label)",
         "For each RF label, bin samples by their RF probability for that "
         "label (X axis), and plot this tool's mean proportion (Y axis) per "
         "bin. The y=x line is perfect calibration. Below the line = this "
         "tool under-calls samples that RF assigns high probability for that "
         "label. Above the line = over-calls."),
        ("calibration_slope_matrix.png", "Calibration Slope Matrix",
         "Per-(this-tool-ancestry x RF-label) calibration slope and max value. "
         "Slope > 0.5 = component meaningfully responds to that RF label. "
         "Slope ~ 0 with low max = vestigial component. Slope > 0.3 against "
         "multiple labels = straddling component (mass split across more than "
         "one RF identity)."),
        ("residual_violin.png", "Residual Distribution",
         "Per-RF-label violin of (this tool's proportion - RF probability). "
         "Centered at 0 = unbiased. Centered above 0 = this tool over-calls; "
         "below = under-calls. Wide violins = high per-sample variance."),
        ("concordance_vs_confidence.png", "Concordance vs RF Confidence",
         "Hard-label concordance (this tool's hard call == RF's hard call) "
         "vs RF max probability per sample, binned. Concordance typically "
         "rises with RF confidence — when RF is uncertain about a sample, "
         "the tools disagree more. Sample-count bars on the right axis show "
         "how the cohort distributes across confidence."),
        ("entropy_scatter.png", "Entropy Scatter",
         "Per-sample Shannon entropy of this tool's soft vector vs RF's. "
         "Lower entropy = more confidently single-ancestry. Tight on y=x = "
         "the two tools agree about how admixed each sample is, even if "
         "they sometimes disagree about which ancestries are mixed."),
    ]

    sections.append("# Concordance Figures\n")
    sections.append(
        "**How to read this section.** Each figure compares this run's "
        "per-sample ancestry calls to the RF classifier's predictions on "
        "the same samples. Section §8.2 of the validation plan calls this "
        "the primary cross-tool concordance check. Each subsection below "
        "has a one-sentence caption explaining what the figure shows and "
        "how to read it.\n"
    )
    for entry in figure_order:
        fname, title = entry[0], entry[1]
        caption = entry[2] if len(entry) >= 3 else None
        img = diag / fname
        if img.exists():
            block = f"## {title}\n\n"
            if caption:
                block += f"_{caption}_\n\n"
            block += f"![{title}]({img})\n"
            sections.append(block)

    # ── Cross-tool: ADMIXTURE (§8.2) ──
    adm_md = render_admixture_section(
        diag / "admixture_correlations.tsv",
        diag / "admixture_scatter.png",
    )
    if adm_md is not None:
        sections.append(adm_md)

    # ── 5. Calibration Breakdowns ──
    breakdown_files = sorted(diag.glob("calibration_*_breakdown.png"))
    if breakdown_files:
        sections.append("# Calibration Breakdowns (per sub-ancestry)\n")
        for bf in breakdown_files:
            label = bf.stem.replace("calibration_", "").replace("_breakdown", "").upper()
            sections.append(f"## {label} Sub-ancestry Calibration\n\n![{label} breakdown]({bf})\n")

    # ── 6. Structural Checks ──
    struct_figs = [
        ("tract_length_distribution.png", "Tract Length Distribution",
         "Per-ancestry empirical tract-length distribution (log-y). The "
         "overlaid theoretical exponential reflects what's expected at the "
         "model's gen_since_admix. Tracts much longer than the theoretical "
         "tail suggest under-segmentation; much shorter suggests "
         "over-segmentation (LAI noise)."),
        ("switch_rate_distribution.png", "Switch Rate Distribution",
         "Per-haplotype count of ancestry switches (= number of tracts - 1). "
         "Pure-ancestry haps cluster near 0–2; admixed haps spread to dozens. "
         "Outliers in the 100+ tail are typically samples with heavy LAI "
         "uncertainty or phasing artifacts and warrant inspection."),
        ("switch_rate_distribution_log.png", "Switch Rate Distribution (log)",
         "Same distribution, log-y. Use this when most haps have few "
         "switches and the long tail would otherwise be invisible."),
    ]
    has_struct = any((diag / f[0]).exists() for f in struct_figs)
    if has_struct:
        sections.append("# Structural Checks (§8.1)\n")
        sections.append(
            "**What this section shows.** Tract-level structural sanity "
            "checks on the per-haplotype ancestry calls. These are "
            "tool-agnostic on the structure (any LAI tool emits tracts) and "
            "catch broad anomalies in how the model is segmenting the genome.\n"
        )
        for entry in struct_figs:
            fname, title = entry[0], entry[1]
            caption = entry[2] if len(entry) >= 3 else None
            img = diag / fname
            if img.exists():
                block = f"## {title}\n\n"
                if caption:
                    block += f"_{caption}_\n\n"
                block += f"![{title}]({img})\n"
                sections.append(block)

    # ── Hap-disagreement (§8.1 / §5.3) ──
    hap_md = render_hap_disagreement_section(diag)
    if hap_md is not None:
        sections.append(hap_md)

    # ── Cross-tool tract concordance (FLARE vs popout) ──
    compare_tracts_md = render_compare_tracts_section(diag)
    if compare_tracts_md is not None:
        sections.append(compare_tracts_md)

    # ── Regional QC (§8.3) ──
    regional_md = render_regional_section(diag)
    if regional_md is not None:
        sections.append(regional_md)

    # ── 7. PCA ──
    pca_img = diag / "pca_by_rf_label.png"
    if pca_img.exists():
        sections.append("# PCA Overlay\n")
        sections.append(
            "**What this section shows.** Sample positions in PC1xPC2 (RF's "
            "or popout's spectral space, whichever was supplied), colored by "
            "the RF hard label. Use this to verify that the RF labels carve "
            "the PCA space into coherent clusters in this cohort. If RF "
            "labels are scrambled in PC space the upstream classifier may "
            "be miscalibrated for the cohort's structure.\n"
        )
        sections.append(f"![PCA by RF label]({pca_img})\n")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Build PDF analysis report")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PDF path (default: <data-dir>/diagnostics/<run-name>_report.pdf)")
    args = parser.parse_args()

    diag = args.data_dir / "diagnostics"
    if not diag.exists():
        print(f"FATAL: {diag} does not exist", file=sys.stderr)
        sys.exit(1)

    out_pdf = args.output or (diag / f"{args.run_name}_report.pdf")
    md_path = diag / f"{args.run_name}_report.md"

    print(f"Building report for {args.run_name}...")
    md_content = build_markdown(args.data_dir, args.run_name)
    md_path.write_text(md_content)
    print(f"  wrote {md_path}")

    # Convert to PDF via pandoc
    cmd = [
        "pandoc", str(md_path),
        "-o", str(out_pdf),
        "--pdf-engine=pdflatex",
        "-V", "geometry:margin=0.75in",
        "-V", "fontsize=10pt",
        "--highlight-style=tango",
    ]
    print(f"  running pandoc...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  pandoc FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"  saved {out_pdf}")


if __name__ == "__main__":
    main()
