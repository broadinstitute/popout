"""Per-RF-label CCC distribution across clusters, one panel per tool.

For each comparison tool (flare/rye/rf), read
``cohort/popout_vs_<tool>.metrics.tsv`` and render the per-RF-label
CCC distribution — boxplots when there are >5 clusters per label,
stripplots otherwise. The data dict also carries a per-tool cohort
summary table sourced from ``cohort_summary.json:pairs``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from popout.labelspace.registry import SP6

from .._helpers import fmt_int, fmt_num, read_tsv, to_float


ANCHOR_TOOL = "popout"
TL_GREY = "#888888"


def _palette(p, lab):
    return p.get(lab, TL_GREY)


def compute(ctx, section=None) -> dict:
    tools = [t for t in (ctx.bundle.get("tools") or []) if t != ANCHOR_TOOL]
    if not tools:
        return {"present": False}
    rf_labels = list(SP6.members)
    pairs = ctx.bundle.get("pairs") or []
    summary_by_pair = {(p["tool"], p["rf_label"]): p for p in pairs}

    tool_panels: list[dict] = []
    for tool in tools:
        path = ctx.bundle_dir / "cohort" / f"popout_vs_{tool}.metrics.tsv"
        header, rows = read_tsv(path)
        if not rows:
            tool_panels.append({
                "tool": tool, "present": False,
                "summary_rows": [], "by_label": {},
            })
            continue
        col = {h: i for i, h in enumerate(header)}
        by_label: dict[str, list[float]] = {lab: [] for lab in rf_labels}
        for r in rows:
            try:
                lab = r[col["rf_label"]]
                v = to_float(r[col["ccc"]])
            except (KeyError, IndexError):
                continue
            if v is not None and lab in by_label:
                by_label[lab].append(v)

        summary_rows: list[dict] = []
        for lab in rf_labels:
            p = summary_by_pair.get((tool, lab))
            if p is None:
                summary_rows.append({
                    "rf_label": lab,
                    "mean_ccc": "—", "mean_r": "—",
                    "n_pass": "—", "n_fail": "—", "n_null": "—",
                    "verdict": "—",
                })
                continue
            n_pass = int(p.get("n_clusters_passing", 0) or 0)
            n_fail = int(p.get("n_clusters_failing", 0) or 0)
            n_null = int(p.get("n_clusters_null", 0) or 0)
            n_eval = n_pass + n_fail
            if n_eval == 0:
                verdict = f"all {n_null} μ-gated"
            else:
                verdict = f"{n_pass}/{n_eval} pass"
                if n_null:
                    verdict += f" (+{n_null} μ·∅)"
            summary_rows.append({
                "rf_label": lab,
                "mean_ccc": fmt_num(p.get("mean_ccc_across_clusters")),
                "mean_r": fmt_num(p.get("mean_pearson_r_across_clusters")),
                "n_pass": fmt_int(n_pass),
                "n_fail": fmt_int(n_fail),
                "n_null": fmt_int(n_null),
                "verdict": verdict,
            })

        tool_panels.append({
            "tool": tool, "present": True,
            "summary_rows": summary_rows, "by_label": by_label,
        })

    if not any(p["present"] for p in tool_panels):
        return {"present": False}

    return {
        "present": True,
        "rf_labels": rf_labels,
        "tool_panels": tool_panels,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.4))
        ax.text(0.5, 0.5, "no per-tool metrics data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    rf_labels = data["rf_labels"]
    panels = [p for p in data["tool_panels"] if p["present"]]
    n_panels = len(panels)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(0.9 * len(rf_labels) + 2.0, 2.6 * n_panels + 0.6),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        by_label = panel["by_label"]
        tool = panel["tool"]
        n_max = max((len(v) for v in by_label.values()), default=0)
        if n_max > 5:
            bp = ax.boxplot(
                [by_label[lab] for lab in rf_labels],
                positions=range(len(rf_labels)),
                patch_artist=True, widths=0.55,
                medianprops={"color": "#222"}, showfliers=True,
            )
            for patch, lab in zip(bp["boxes"], rf_labels):
                patch.set_facecolor(_palette(palette, lab))
                patch.set_alpha(0.6)
        else:
            for i, lab in enumerate(rf_labels):
                vals = by_label[lab]
                if vals:
                    ax.scatter(
                        [i] * len(vals), vals, s=36,
                        color=_palette(palette, lab),
                        edgecolors="#222", linewidths=0.6, alpha=0.85,
                    )
        ax.set_xticks(range(len(rf_labels)))
        ax.set_xticklabels(rf_labels)
        ax.axhline(0.9, color="#117733", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(0.5, color="#CC3311", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"popout vs {tool}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes:
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Lin's CCC")
    fig.suptitle("Per-RF-label CCC across clusters", fontsize=11, y=1.0)
    fig.tight_layout()
    return fig
