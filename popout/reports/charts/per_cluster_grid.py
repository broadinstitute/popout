"""Per-(cluster, chrom) performance grid — table-only.

Joins ``cohort/manifest.tsv`` (samples / wallclock / RSS) with
``cohort/tier1_metrics.tsv`` (per-tool mean CCC + pass/fail counts).
No chart; the section template renders the rows as a pipe table.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .._helpers import fmt_int, fmt_num, read_tsv, to_float


ANCHOR_TOOL = "popout"


def compute(ctx, section=None) -> dict:
    manifest_path = ctx.bundle_dir / "cohort" / "manifest.tsv"
    header, rows = read_tsv(manifest_path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}

    tier_path = ctx.bundle_dir / "cohort" / "tier1_metrics.tsv"
    th, tr = read_tsv(tier_path)
    tier_by_key: dict[tuple[str, str], dict[str, str]] = {}
    if tr:
        tcol = {h: i for i, h in enumerate(th)}
        for r in tr:
            k = (r[tcol["cluster_id"]], r[tcol["chrom"]])
            tier_by_key.setdefault(k, {})[r[tcol["key"]]] = r[tcol["value"]]

    tools = [t for t in (ctx.bundle.get("tools") or []) if t != ANCHOR_TOOL]

    # Pre-flatten the header + rows into plain string lists so the
    # template can render a simple pipe table without nested for loops
    # (nested loops + trim_blocks collapse rows onto one line).
    header: list[str] = [
        "cluster", "chrom", "n samples", "wallclock (s)", "peak RSS (GB)",
    ]
    aligns: list[str] = [":--------", ":------", "---------:", "------------:", "------------:"]
    for tool in tools:
        header.append(f"mean CCC vs {tool}")
        header.append(f"pass vs {tool}")
        aligns.append("--------------:")
        aligns.append(":------------")

    grid_rows: list[list[str]] = []
    for r in rows:
        cid = r[col["cluster_id"]]
        chrom = r[col["chrom"]]
        n_samples = r[col["n_samples"]] if "n_samples" in col else None
        wall = r[col["total_wallclock_seconds"]] if "total_wallclock_seconds" in col else None
        rss = r[col["peak_rss_gb"]] if "peak_rss_gb" in col else None

        t_metrics = tier_by_key.get((cid, chrom), {})
        row_cells: list[str] = [
            f"`{cid}`", f"`{chrom}`",
            fmt_int(n_samples), fmt_num(wall, 1), fmt_num(rss, 2),
        ]
        for tool in tools:
            ccc_v = t_metrics.get(f"popout_dx.mean_ccc_vs_{tool}")
            n_pass = to_float(t_metrics.get(f"popout_dx.n_pairs_passing_vs_{tool}"))
            n_fail = to_float(t_metrics.get(f"popout_dx.n_pairs_failing_vs_{tool}"))
            if n_pass is None and n_fail is None:
                pass_str = "—"
            else:
                p = int(n_pass or 0)
                f = int(n_fail or 0)
                pass_str = f"{p}/{p + f}" if (p + f) > 0 else "all μ·∅"
            row_cells.append(fmt_num(ccc_v))
            row_cells.append(pass_str)
        grid_rows.append(row_cells)

    return {
        "present": True,
        "tools": tools,
        "header": header,
        "aligns": aligns,
        "grid_rows": grid_rows,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:  # pragma: no cover
    # data-only chart; no figure. Renderer routes via `data:` option.
    raise RuntimeError("per_cluster_grid is data-only; do not call render()")
