"""Label-correlation heatmap for the largest cluster.

Reads ``per_cluster/<cluster_id>/<chrom>/soft_correlation/labels.json``
for whichever cluster has the most samples in the cohort, and renders
the Pearson correlation matrix between inferred FLARE components and
the RF classifier's reference labels. Cells that the matcher
assigned (via ``popout_to_rf_label``) are highlighted with a thick
black border. This is the QC-side of the FLARE-vs-RF mapping —
matched cells should sit far from zero in the right direction.

Exemplar selector: cluster with the largest n_samples in
``cohort/manifest.tsv``. Analyst override goes via a
``cluster: "<cluster_id>"`` option in the YAML section.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from .._helpers import read_tsv, to_float


def _largest_cluster(bundle_dir) -> tuple[str, str] | None:
    """Pick (cluster_id, chrom) with the largest n_samples in manifest.tsv.

    Falls back to (None, None) if the manifest is missing or
    n_samples isn't a column.
    """
    path = bundle_dir / "cohort" / "manifest.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return None
    col = {h: i for i, h in enumerate(header)}
    if "cluster_id" not in col or "chrom" not in col:
        return None
    best_key = None
    best_n = -1.0
    for r in rows:
        cid = r[col["cluster_id"]]
        chrom = r[col["chrom"]]
        n = to_float(r[col["n_samples"]]) if "n_samples" in col else None
        if n is None:
            continue
        if n > best_n:
            best_n = n
            best_key = (cid, chrom)
    return best_key


def compute(ctx, section=None) -> dict:
    section_opts = section.options if section is not None else {}
    cluster_override = section_opts.get("cluster")
    chrom_override = section_opts.get("chrom")

    if cluster_override and chrom_override:
        cid, chrom = cluster_override, chrom_override
    else:
        pick = _largest_cluster(ctx.bundle_dir)
        if pick is None:
            return {"present": False, "reason": "no cluster manifest"}
        cid, chrom = pick

    labels_path = (
        ctx.bundle_dir / "per_cluster" / cid / chrom
        / "soft_correlation" / "labels.json"
    )
    if not labels_path.exists():
        return {
            "present": False,
            "reason": f"no labels.json at {labels_path.relative_to(ctx.bundle_dir)}",
        }
    labels = json.loads(labels_path.read_text())
    if "correlations" not in labels:
        return {
            "present": False,
            "reason": "labels.json has no correlations matrix",
        }
    corr = np.asarray(labels["correlations"], dtype=np.float64)
    ref_names = list(labels.get(
        "rf_ref_labels",
        [f"ref_{i}" for i in range(corr.shape[1])],
    ))
    n_inf = corr.shape[0]
    # FLARE component names come verbatim from the panel header
    # (``component_to_label`` is the by-name matcher's index -> name map;
    # ``method: name`` is the FLARE-verbatim path). Anonymous
    # ``flare_<i>`` only as a last resort if the matcher metadata is
    # absent.
    ctl = labels.get("component_to_label") or {}
    inf_names: list[str] = []
    for i in range(n_inf):
        nm = ctl.get(str(i)) or ctl.get(i)
        inf_names.append(nm if nm else f"flare_{i}")
    popout_to_rf = labels.get("popout_to_rf_label", {})

    # Honour the section's mid_rule. mid_rule="drop" on an SP6-targeted
    # section means RF's MID column is dropped from the comparison.
    # mid_rule="fold_to_eur" folds MID's correlation into EUR (not
    # summable, so degrade to drop and note it). When mid_rule is None
    # or "none", pass MID through.
    section_mid_rule = (section.mid_rule
                        if section is not None else None)
    if section_mid_rule in ("drop", "fold_to_eur") and "mid" in ref_names:
        mid_idx = ref_names.index("mid")
        keep = [j for j in range(len(ref_names)) if j != mid_idx]
        corr = corr[:, keep]
        ref_names = [ref_names[j] for j in keep]

    assigned: list[tuple[int, int]] = []
    for inf_idx, rf_name in popout_to_rf.items():
        try:
            i = int(inf_idx)
        except (TypeError, ValueError):
            continue
        if rf_name in ref_names:
            j = ref_names.index(rf_name)
            assigned.append((i, j))

    return {
        "present": True,
        "cluster_id": cid,
        "chrom": chrom,
        "corr": corr,
        "ref_names": ref_names,
        "inf_names": inf_names,
        "assigned": assigned,
        "n_overlapping": labels.get("n_overlapping_sites"),
        "mid_rule": section_mid_rule,
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.6))
        ax.text(
            0.5, 0.5,
            f"label correlation unavailable: {data.get('reason', '')}",
            ha="center", va="center", fontsize=10, color="#666",
        )
        ax.axis("off")
        return fig

    corr = data["corr"]
    ref_names = data["ref_names"]
    inf_names = data["inf_names"]
    assigned = set(data["assigned"])
    n_inf, n_ref = corr.shape

    fig, ax = plt.subplots(
        figsize=(0.85 * n_ref + 2.0, 0.55 * n_inf + 1.8),
    )
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n_ref))
    ax.set_yticks(range(n_inf))
    ax.set_xticklabels(ref_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(inf_names, fontsize=9)
    ax.set_xlabel("reference RF label ->")
    ax.set_ylabel("FLARE component ↓")

    from matplotlib.patches import Rectangle
    for i in range(n_inf):
        for j in range(n_ref):
            v = corr[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            weight = "bold" if (i, j) in assigned else "normal"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=8,
                    color=color, fontweight=weight)
            if (i, j) in assigned:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=2,
                ))

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    overlap = data.get("n_overlapping")
    suffix = f" — n={overlap:,} overlapping sites" if isinstance(overlap, int) else ""
    ax.set_title(
        f"Label correlation: {data['cluster_id']} / {data['chrom']}{suffix}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig
