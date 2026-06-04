"""Per-sample MAE distribution — popout vs each comparison tool.

Reads ``cohort/per_sample_mae.tsv`` (one row per sample, one
``mae_vs_<tool>`` column per comparison tool) and renders a violin
plot of ``log10(MAE)`` for each available pair. The data dict also
carries summary statistics for the template's table.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .._helpers import fmt_int, fmt_num, read_tsv, to_float


ANCHOR_TOOL = "popout"
TL_GREY = "#888888"


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "per_sample_mae.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    tools = [t for t in (ctx.bundle.get("tools") or []) if t != ANCHOR_TOOL]
    col = {h: i for i, h in enumerate(header)}

    by_tool: dict[str, list[float]] = {}
    for tool in tools:
        key = f"mae_vs_{tool}"
        if key not in col:
            continue
        vals: list[float] = []
        for r in rows:
            v = to_float(r[col[key]] if col[key] < len(r) else None)
            if v is not None and v > 0:
                vals.append(v)
        if vals:
            by_tool[tool] = vals

    if not by_tool:
        return {"present": False}

    table_rows: list[dict] = []
    for tool in tools:
        vals = by_tool.get(tool)
        if not vals:
            table_rows.append({
                "tool": tool, "n": "—",
                "median": "—", "p95": "—", "max": "—",
            })
            continue
        a = np.array(vals)
        table_rows.append({
            "tool": tool,
            "n": fmt_int(a.size),
            "median": fmt_num(float(np.median(a)), 4),
            "p95": fmt_num(float(np.percentile(a, 95)), 4),
            "max": fmt_num(float(a.max()), 4),
        })

    return {
        "present": True,
        "by_tool": by_tool,
        "tools_in_order": [t for t in tools if t in by_tool],
        "table_rows": table_rows,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.4))
        ax.text(0.5, 0.5, "no per-sample MAE data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    tools = data["tools_in_order"]
    by_tool = data["by_tool"]

    fig, ax = plt.subplots(figsize=(2.4 * len(tools) + 1.6, 3.4))
    positions = list(range(len(tools)))
    vp = ax.violinplot(
        [np.log10(np.array(by_tool[t])) for t in tools],
        positions=positions, widths=0.82, showmedians=True,
    )
    for body in vp["bodies"]:
        body.set_facecolor(TL_GREY)
        body.set_alpha(0.55)
        body.set_edgecolor("#444")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"popout vs {t}\nn={len(by_tool[t]):,}" for t in tools],
        fontsize=9,
    )
    ax.set_ylabel("log10 MAE per sample")
    ax.set_title("Per-sample MAE distribution (log10)", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
