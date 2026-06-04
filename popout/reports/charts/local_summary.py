"""Local-mode per-cluster summary table.

Reads ``cohort/local_per_sample.tsv`` (one row per sample × cluster ×
chrom with ``agree_pct`` and ``jaccard_tracts``). Aggregates to mean
per (cluster, chrom). No chart.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .._helpers import fmt_int, fmt_num, fmt_pct, read_tsv, to_float


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "local_per_sample.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    if "cluster_id" not in col or "chrom" not in col:
        return {"present": False}

    by_cluster: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}
    for r in rows:
        cid = r[col["cluster_id"]]
        chrom = r[col["chrom"]]
        agree = to_float(r[col["agree_pct"]]) if "agree_pct" in col else None
        jacc = to_float(r[col["jaccard_tracts"]]) if "jaccard_tracts" in col else None
        by_cluster.setdefault((cid, chrom), []).append((agree, jacc))

    table_rows: list[dict] = []
    for (cid, chrom), grp in sorted(by_cluster.items()):
        agree_vals = [v for v, _ in grp if v is not None]
        jacc_vals = [v for _, v in grp if v is not None]
        table_rows.append({
            "cluster_id": cid,
            "chrom": chrom,
            "n_samples": fmt_int(len(grp)),
            "mean_agree": fmt_pct(float(np.mean(agree_vals)))
                          if agree_vals else "—",
            "mean_jaccard": fmt_num(float(np.mean(jacc_vals)))
                             if jacc_vals else "—",
        })

    return {"present": True, "table_rows": table_rows}


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:  # pragma: no cover
    raise RuntimeError("local_summary is data-only; do not call render()")
