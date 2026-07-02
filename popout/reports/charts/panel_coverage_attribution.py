"""Per-chromosome FLARE-minus-Rye residual, stratified by Rye-dominant ancestry.

One panel per SP5 ancestry. Each panel shows 22 box plots, one per
autosome, of the per-sample residual

    residual(i, c, X) = FLARE chr_c X-prop(i) - Rye X-prop(i)

restricted to samples where Rye calls them dominantly X (Rye X >= 0.85).
chr1 is highlighted on each panel because FLARE trains the HMM there
and applies it on chr2..chr22.

The chart corroborates the panel-composition experiment's finding from
an independent estimator on the unmodified production cohort: the afr
facet has a wider residual spread than any other on every chromosome
and a negative median throughout; the eur facet has a positive median
throughout with substantially tighter spread.

Data:
  - ``cohort/cohort_global.tsv`` — per-(cluster, chrom, sample) FLARE
    proportions, named columns.
  - external Rye Q TSV — per-sample SP5 proportions, passed via
    ``ReportContext.rye_q`` (set from the report builder's ``--rye-q``).

The chart degrades gracefully when ``rye_q`` is not provided: it
returns ``present=False`` and the template skips the figure.

Adapted from
``cluster-composition-experiment-2026-06-15/build_smoking_gun_chart.py``
chart 1, recast onto v6's named-column bundle.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP5 as _SP5

from .._helpers import read_tsv


SP5_ORDER: tuple[str, ...] = tuple(_SP5.members)
RYE_DOM_THRESHOLD: float = 0.85
N_CHROMS: int = 22


# ── helpers ──────────────────────────────────────────────────────────────


def _load_per_sample_xprop(bundle_dir: Path) -> dict[str, np.ndarray]:
    """Return ``{sample_id: (N_CHROMS, len(SP5_ORDER)) matrix}``.

    Reads ``cohort/cohort_global.tsv`` directly. The TSV header carries
    the FLARE panel-population names verbatim (v6 schema); only columns
    whose name is in ``SP5_ORDER`` are kept, in SP5_ORDER. Samples
    missing any of the 22 autosomes are dropped at the end.
    """
    path = bundle_dir / "cohort" / "cohort_global.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {}
    col = {h: i for i, h in enumerate(header)}
    # SP5 columns that exist in this bundle, mapped to header positions.
    sp_cols: dict[str, int] = {}
    for lab in SP5_ORDER:
        if lab in col:
            sp_cols[lab] = col[lab]
    if not sp_cols:
        return {}
    chrom_idx = {f"chr{i}": i - 1 for i in range(1, N_CHROMS + 1)}
    per_sample: dict[str, np.ndarray] = {}
    for r in rows:
        try:
            sid = r[col["sample_id"]]
            chrom = r[col["chrom"]]
        except (IndexError, KeyError):
            continue
        ci = chrom_idx.get(chrom)
        if ci is None:
            continue
        row = np.zeros(len(SP5_ORDER), dtype=float)
        for j, lab in enumerate(SP5_ORDER):
            ci_col = sp_cols.get(lab)
            if ci_col is None:
                continue
            try:
                row[j] = float(r[ci_col])
            except (IndexError, ValueError):
                row[j] = np.nan
        if sid not in per_sample:
            per_sample[sid] = np.full(
                (N_CHROMS, len(SP5_ORDER)), np.nan, dtype=float,
            )
        per_sample[sid][ci, :] = row
    complete = {sid: arr for sid, arr in per_sample.items()
                if not np.isnan(arr).any()}
    return complete


def _load_rye_q(path: Path) -> dict[str, dict[str, float]]:
    """Return ``{research_id: {sp5_label: proportion}}`` from a Rye Q TSV.

    Header must carry ``research_id`` plus columns named for SP5 members.
    """
    out: dict[str, dict[str, float]] = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        lower = [h.lower() for h in header]
        id_aliases = ("research_id", "sample_id", "sample")
        id_col = next((i for i, h in enumerate(lower) if h in id_aliases), None)
        if id_col is None:
            raise RuntimeError(
                f"{path}: no sample-id column in Rye Q header {header!r}"
            )
        anc_cols = [(i, header[i]) for i in range(len(header))
                    if header[i] in SP5_ORDER]
        if not anc_cols:
            raise RuntimeError(
                f"{path}: no SP5 ancestry columns in Rye Q header {header!r}"
            )
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= id_col:
                continue
            sid = parts[id_col].strip()
            if not sid:
                continue
            try:
                out[sid] = {lab: float(parts[i]) for i, lab in anc_cols}
            except (IndexError, ValueError):
                continue
    return out


def _rye_dominant_subsets(
    per_sample: dict[str, np.ndarray],
    rye: dict[str, dict[str, float]],
) -> dict[str, list[str]]:
    """Return ``{ancestry: [sample_ids with Rye_X >= RYE_DOM_THRESHOLD]}``."""
    out: dict[str, list[str]] = {a: [] for a in SP5_ORDER}
    for sid in per_sample:
        r = rye.get(sid)
        if r is None:
            continue
        for a in SP5_ORDER:
            if r.get(a, 0.0) >= RYE_DOM_THRESHOLD:
                out[a].append(sid)
    return out


# ── compute ──────────────────────────────────────────────────────────────


def compute(ctx, section=None) -> dict:
    rye_q = getattr(ctx, "rye_q", None)
    if not rye_q or not Path(rye_q).exists():
        return {
            "present": False,
            "reason": "rye_q not provided to report builder",
        }
    per_sample = _load_per_sample_xprop(ctx.bundle_dir)
    if not per_sample:
        return {"present": False, "reason": "no complete-chr1..22 samples in bundle"}
    rye = _load_rye_q(Path(rye_q))
    if not rye:
        return {"present": False, "reason": "Rye Q file empty or unreadable"}
    rye_dom = _rye_dominant_subsets(per_sample, rye)

    per_panel: dict[str, dict] = {}
    for a in SP5_ORDER:
        sids = rye_dom[a]
        if len(sids) < 50:
            per_panel[a] = {"n": len(sids), "insufficient": True}
            continue
        ai = SP5_ORDER.index(a)
        flare_chrom = np.array([per_sample[sid][:, ai] for sid in sids])
        rye_vals = np.array([rye[sid][a] for sid in sids])
        residuals = flare_chrom - rye_vals[:, None]
        by_chrom = [residuals[:, c] for c in range(N_CHROMS)]
        chr1 = residuals[:, 0]
        chr2_22 = residuals[:, 1:]
        per_panel[a] = {
            "n": len(sids),
            "insufficient": False,
            "by_chrom": by_chrom,
            "chr1_median": float(np.median(chr1)),
            "chr1_iqr": float(
                np.subtract(*np.percentile(chr1, [75, 25]))
            ),
            "chr2_22_median": float(np.median(chr2_22)),
            "chr2_22_iqr": float(
                np.subtract(*np.percentile(chr2_22, [75, 25]))
            ),
        }

    return {
        "present": True,
        "per_panel": per_panel,
        "labels": list(SP5_ORDER),
        "rye_dom_threshold": RYE_DOM_THRESHOLD,
    }


# ── render ───────────────────────────────────────────────────────────────


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(8, 1.4))
        msg = data.get("reason", "no data")
        ax.text(
            0.5, 0.5,
            f"panel-coverage attribution figure unavailable ({msg})",
            ha="center", va="center", fontsize=10,
        )
        ax.axis("off")
        return fig

    labels = data["labels"]
    per_panel = data["per_panel"]

    fig, axes = plt.subplots(1, len(labels), figsize=(22, 6), sharey=True)
    for ax, a in zip(axes, labels):
        panel = per_panel[a]
        color = palette.get(a, "#888")
        if panel.get("insufficient"):
            ax.set_title(
                f"X = {a}\n(n={panel.get('n', 0):,}, insufficient)",
                color=color, fontsize=12, fontweight="bold",
            )
            ax.set_xlabel("chromosome")
            continue
        by_chrom = panel["by_chrom"]
        bp = ax.boxplot(
            by_chrom, positions=range(1, N_CHROMS + 1),
            widths=0.6, showfliers=False, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.0),
        )
        for c, box in enumerate(bp["boxes"]):
            box.set_facecolor("#d94a4a" if c == 0 else color)
            box.set_alpha(0.85)
            box.set_edgecolor("white")
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(
            0.02, 0.02,
            (
                f"n = {panel['n']:,}\n"
                f"chr1: med={panel['chr1_median']:+.3f}, "
                f"IQR={panel['chr1_iqr']:.3f}\n"
                f"chr2..22: med={panel['chr2_22_median']:+.3f}, "
                f"IQR={panel['chr2_22_iqr']:.3f}"
            ),
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
            color="#222",
            bbox=dict(
                facecolor="white", edgecolor="#bbb",
                boxstyle="round,pad=0.3",
            ),
        )
        ax.set_title(f"X = {a}", color=color, fontsize=14, fontweight="bold")
        ax.set_xlabel("chromosome")
        ax.set_xticks(range(1, N_CHROMS + 1, 2))
        ax.set_xticklabels([str(c) for c in range(1, N_CHROMS + 1, 2)])
        ax.set_ylim(-0.35, 0.35)
        ax.grid(True, axis="y", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(
        f"FLARE chr_c X-prop - Rye X-prop\n"
        f"(samples with Rye X >= {data['rye_dom_threshold']})"
    )
    fig.suptitle(
        "Per-chromosome FLARE-minus-Rye residual, stratified by Rye-dominant ancestry",
        y=1.00, fontsize=14,
    )
    fig.tight_layout()
    return fig
