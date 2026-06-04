"""Cross-cluster regional meta-analysis — Manhattan + top peaks.

Data: ``cohort/regional_meta.tsv`` (per (window, ancestry) Stouffer
meta-analysis across clusters). Lollipops at genomic midpoint;
height = −log10(q); width scales with `n_clusters_flagged`. Mask
shading on significant windows that fall inside pre-registered masks
(HLA, centromere flank, segdup, high-LD).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from popout.viz._style import chrom_length, chrom_sort_key

from .._helpers import read_tsv, to_float


FDR_THRESHOLD = 0.05


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "regional_meta.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}

    points: list[dict] = []
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
        points.append({
            "anc_base": anc_base, "anc_name": anc_name,
            "chrom": chrom, "start_mb": start / 1e6, "end_mb": end / 1e6,
            "mid_mb": mid_mb, "neglogq": neglogq, "q": q,
            "n_flagged": n_flagged, "mask": mask,
        })
    if not points:
        return {"present": False}

    chroms_in_use = sorted({p["chrom"] for p in points}, key=chrom_sort_key)
    offsets: dict[str, float] = {}
    cur = 0.0
    for chrom in chroms_in_use:
        offsets[chrom] = cur
        cur += chrom_length(chrom) / 1e6 or 250.0
    total_extent = cur

    top_peaks = sorted(points, key=lambda p: -p["neglogq"])[:20]
    n_outside_mask = sum(1 for p in points if not p["mask"])

    # Top-20 supporting table (sorted by q ascending).
    enriched: list[tuple[float, dict]] = []
    for p in points:
        enriched.append((p["q"], p))
    enriched.sort(key=lambda t: t[0])
    table_rows = []
    for q, p in enriched[:20]:
        # Look up the raw row's stouffer_z and stouffer_p too, from the
        # original TSV. We didn't preserve them per-point; re-read.
        pass

    # Re-scan rows for the full per-window detail (z, p, n_total).
    table_rows = []
    enriched_full: list[tuple[float, dict]] = []
    for r in rows:
        try:
            q = float(r[col["stouffer_q"]])
        except (IndexError, KeyError, ValueError):
            continue
        enriched_full.append((q, {
            "chrom": r[col["chrom"]],
            "start_mb": float(r[col["start"]]) / 1e6,
            "end_mb": float(r[col["end"]]) / 1e6,
            "ancestry_name": r[col["ancestry_name"]],
            "n_flagged": int(r[col["n_clusters_flagged"]]),
            "n_total": int(r[col["n_clusters_total"]]),
            "z": to_float(r[col["stouffer_z"]]),
            "p": to_float(r[col["stouffer_p"]]),
            "q": q,
            "mask": r[col["mask_region"]] or "",
        }))
    enriched_full.sort(key=lambda t: t[0])
    table_rows = [row for _, row in enriched_full[:20]]

    return {
        "present": True,
        "points": points,
        "chroms_in_use": chroms_in_use,
        "offsets": offsets,
        "total_extent": total_extent,
        "top_peaks": top_peaks,
        "n_outside_mask": n_outside_mask,
        "table_rows": table_rows,
        "fdr_threshold": FDR_THRESHOLD,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no regional-meta data", ha="center", va="center")
        ax.axis("off")
        return fig

    points = data["points"]
    chroms_in_use = data["chroms_in_use"]
    offsets = data["offsets"]
    total_extent = data["total_extent"]
    fdr_threshold = data["fdr_threshold"]

    fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    sig_threshold = -math.log10(fdr_threshold)

    def _color(anc: str) -> str:
        return palette.get(anc.split(".")[0], "#888888")

    # Mask shading on significant windows.
    drawn_masks: set[tuple[str, float, float]] = set()
    for p in points:
        if p["neglogq"] < sig_threshold or not p["mask"]:
            continue
        if p["chrom"] not in offsets:
            continue
        key = (p["chrom"], p["start_mb"], p["end_mb"])
        if key in drawn_masks:
            continue
        drawn_masks.add(key)
        x0 = offsets[p["chrom"]] + p["start_mb"]
        x1 = offsets[p["chrom"]] + p["end_mb"]
        ax.axvspan(x0, x1, color="#e9e9e9", alpha=0.55, linewidth=0, zorder=0)

    # Lollipops.
    seen_legend: set[str] = set()
    legend_handles: list = []
    max_flag = max((p["n_flagged"] for p in points), default=1) or 1
    for p in points:
        if p["chrom"] not in offsets:
            continue
        x = offsets[p["chrom"]] + p["mid_mb"]
        color = _color(p["anc_base"])
        lw = 0.6 + 2.0 * (p["n_flagged"] / max_flag)
        ax.vlines(x, 0, p["neglogq"], color=color, linewidth=lw,
                  alpha=0.85, zorder=2)
        cap_half = max(0.6, total_extent * 0.0035)
        ax.hlines(p["neglogq"], x - cap_half, x + cap_half,
                  color=color, linewidth=lw, alpha=0.95, zorder=3)
        if p["anc_base"] not in seen_legend:
            seen_legend.add(p["anc_base"])
            legend_handles.append(
                plt.Line2D([0], [0], color=color, linewidth=2.4,
                           label=p["anc_base"])
            )

    # Annotate top 3 peaks.
    top_peaks = data["top_peaks"][:3]
    y_max = max(p["neglogq"] for p in points) if points else 1.0
    y_top = y_max * 1.7
    ax.set_ylim(0, y_top)
    for i, p in enumerate(top_peaks):
        if p["chrom"] not in offsets:
            continue
        x = offsets[p["chrom"]] + p["mid_mb"]
        x_jitter = (i - 1) * total_extent * 0.16
        y_offset = y_top * (0.80 - i * 0.11)
        mask_str = p["mask"] if p["mask"] else "no mask"
        text = (
            f"#{i + 1}: {p['chrom']}:{p['mid_mb']:.1f} Mb\n"
            f"{p['anc_name']} · q={p['q']:.1e}\n"
            f"{p['n_flagged']} cluster(s) · {mask_str}"
        )
        ax.annotate(text, xy=(x, p["neglogq"]),
                    xytext=(x + x_jitter, y_offset),
                    fontsize=8, color="#222", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#888", linewidth=0.6),
                    arrowprops=dict(arrowstyle="->", color="#666",
                                    linewidth=0.7))

    ax.axhline(sig_threshold, color="#888", linestyle="--",
               linewidth=0.9, zorder=1)
    ax.annotate(f"FDR {int(fdr_threshold * 100)}%",
                xy=(total_extent, sig_threshold), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color="#555",
                ha="left", va="center")

    if len(chroms_in_use) > 1:
        for chrom in chroms_in_use[1:]:
            ax.axvline(offsets[chrom], color="#cccccc", linewidth=0.6, zorder=0)
        for chrom in chroms_in_use:
            ax.text(offsets[chrom] + (chrom_length(chrom) / 1e6 or 250.0) / 2,
                    1.02, chrom, fontsize=9, color="#444",
                    ha="center", va="bottom",
                    transform=ax.get_xaxis_transform())

    if len(chroms_in_use) == 1:
        only = chroms_in_use[0]
        ax.set_xlim(0, chrom_length(only) / 1e6 or total_extent)
        ax.set_xlabel(f"genomic position on {only} (Mb)", fontsize=10)
    else:
        ax.set_xlim(0, total_extent)
        ax.set_xlabel("genomic position (Mb, chromosomes concatenated)",
                      fontsize=10)
    ax.set_ylabel("−log10(stouffer q)", fontsize=10)
    ax.set_title(
        "Cross-cluster regional meta-analysis  ·  lollipop height = "
        "significance  ·  line width = clusters flagging the window",
        fontsize=11, loc="left",
    )
    ax.legend(handles=legend_handles, title="ancestry", fontsize=8,
              title_fontsize=9, loc="upper right", frameon=False)
    return fig
