"""Per-sample admixture stacked bar (cohort-pooled).

The classic ADMIXTURE figure: every cohort sample as one vertical
strip, sorted by primary FLARE ancestry → secondary proportion.
Per sample, the FLARE component vector is summed by its RF label
(from ``merged_groups_rf.tsv``), so colors map to the SP5 ancestry
the sample is being compared against.

Data: ``cohort/cohort_global.tsv`` + ``cohort/merged_groups_rf.tsv``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from popout.labelspace.registry import SP6

from .._helpers import read_tsv


def _read_flare_to_rf(bundle_dir: Path) -> dict[tuple[str, str], dict[int, str]]:
    """(cluster_id, chrom) → {FLARE component index → SP6 label}."""
    path = bundle_dir / "cohort" / "merged_groups_rf.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {}
    col = {h: i for i, h in enumerate(header)}
    out: dict[tuple[str, str], dict[int, str]] = {}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            rf = r[col["rf_label"]]
            idxs = r[col["component_indices"]]
        except (KeyError, IndexError):
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


def compute(ctx, section=None) -> dict:
    path = ctx.bundle_dir / "cohort" / "cohort_global.tsv"
    header, rows = read_tsv(path)
    if not rows:
        return {"present": False}
    col = {h: i for i, h in enumerate(header)}
    needed = ("cluster_id", "chrom", "sample_id")
    if any(c not in col for c in needed):
        return {"present": False}
    n_meta = col["sample_id"] + 1
    flare_to_rf = _read_flare_to_rf(ctx.bundle_dir)

    # Sum per-sample FLARE components into per-RF-label bins.
    sp6 = list(SP6.members)
    label_to_idx = {lab: i for i, lab in enumerate(sp6)}
    per_sample_props: list[np.ndarray] = []
    per_sample_primary: list[int] = []   # index into sp6
    for r in rows:
        cid = r[col["cluster_id"]]
        chrom = r[col["chrom"]]
        if len(r) <= n_meta:
            continue
        try:
            vals = [float(x) for x in r[n_meta:]]
        except ValueError:
            continue
        if not vals:
            continue
        bin_vec = np.zeros(len(sp6), dtype=np.float32)
        comp_to_rf = flare_to_rf.get((cid, chrom), {})
        for k, v in enumerate(vals):
            rf = comp_to_rf.get(k)
            if rf in label_to_idx and v > 0:
                bin_vec[label_to_idx[rf]] += v
        s = bin_vec.sum()
        if s <= 0:
            continue
        bin_vec /= s
        per_sample_props.append(bin_vec)
        per_sample_primary.append(int(np.argmax(bin_vec)))

    if not per_sample_props:
        return {"present": False}

    props = np.vstack(per_sample_props)
    primary = np.array(per_sample_primary, dtype=int)
    return {
        "present": True,
        "labels": sp6,
        "props": props,
        "primary": primary,
        "n_samples": props.shape[0],
    }


def render(data: dict, *, palette: dict[str, str]) -> plt.Figure:
    if not data.get("present"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "no cohort-composition data",
                ha="center", va="center")
        ax.axis("off")
        return fig

    labels = data["labels"]
    props = data["props"]
    primary = data["primary"]
    n_samples = props.shape[0]
    n_labels = props.shape[1]

    # Sort by primary ancestry, then descending proportion within group.
    primary_props = props[np.arange(n_samples), primary]
    order = np.lexsort((-primary_props, primary))
    props = props[order]

    # Subsample to MAX_DISPLAY for rendering speed.
    MAX_DISPLAY = 4000
    if n_samples > MAX_DISPLAY:
        step = n_samples / MAX_DISPLAY
        idx = np.clip(np.round(np.arange(MAX_DISPLAY) * step).astype(int),
                      0, n_samples - 1)
        props = props[idx]
    n_display = props.shape[0]

    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    bottom = np.zeros(n_display, dtype=np.float32)
    x = np.arange(n_display)
    for j, lab in enumerate(labels):
        h = props[:, j]
        if not h.any():
            continue
        color = palette.get(lab, "#888888")
        ax.bar(x, h, bottom=bottom, width=1.0,
               color=color, edgecolor="none", linewidth=0)
        bottom += h

    ax.set_xlim(0, n_display)
    ax.set_ylim(0, 1.0)
    if n_display < n_samples:
        ax.set_xlabel(
            f"Samples ({n_display:,} of {n_samples:,}, sorted by primary ancestry)",
            fontsize=10,
        )
    else:
        ax.set_xlabel(
            f"Samples (n={n_samples:,}, sorted by primary ancestry)",
            fontsize=10,
        )
    ax.set_ylabel("ancestry proportion", fontsize=10)
    ax.set_xticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    from matplotlib.patches import Rectangle
    handles = [Rectangle((0, 0), 1, 1, facecolor=palette.get(lab, "#888888"))
               for lab in labels]
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=min(len(labels), 6),
              fontsize=9, frameon=False)
    ax.set_title("Per-sample admixture (FLARE → SP6 by merged_groups_rf)",
                 fontsize=11, loc="left")
    fig.tight_layout()
    return fig
