"""Shared loaders + RF-basis projection for popout DX pairwise scripts.

All loaders subset to the cluster's sample roster (intersection with the
tool's universe). Missing samples are an error — silent drops would mask
upstream sample-id drift (CLAUDE.md: never silently drop exceptions).

Projection target is the canonical six-label RF basis
:data:`RF_LABELS_CANONICAL` = ``("afr", "amr", "eas", "eur", "mid", "sas")``.
Tools that lack a label (Rye and FLARE typically don't have ``mid``)
contribute a zero column for that label.

Popout components are not natively named; the popout-side ``labels.json``
(computed by ``step_align_labels`` in the DX collector) carries
``popout_to_rf_label`` (int → name) and ``rf_to_popout_components`` (name
→ list[int]). FLARE components are similarly unnamed in their
post-conversion ``global.tsv`` (``ancestry_0``..``K-1``); the per-cluster
``soft_correlation/labels.json`` from the FLARE-validate cohort bundle
carries the same mapping shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# popout is installed in the docker; loaders are pure-numpy.
from popout.viz._loaders import read_global_tsv, read_labels_json


RF_LABELS_CANONICAL: tuple[str, ...] = ("afr", "amr", "eas", "eur", "mid", "sas")
RYE_LABELS: tuple[str, ...] = ("eur", "eas", "amr", "afr", "sas")


# ── Errors ───────────────────────────────────────────────────────────────


class LoaderError(RuntimeError):
    pass


def _require_no_missing(label: str, requested: list[str], universe: set[str]) -> None:
    missing = [s for s in requested if s not in universe]
    if missing:
        raise LoaderError(
            f"{label}: {len(missing)} roster sample(s) absent from the tool's universe; "
            f"first: {missing[:5]}"
        )


# ── Per-tool loaders (roster-aligned) ────────────────────────────────────


def load_flare_global(global_tsv: Path) -> tuple[list[str], np.ndarray]:
    """Read FLARE's popout-format global.tsv (post flare_to_popout_format).

    Returns ``(sample_ids, q (n × K_flare))``. The sample_ids list is the
    cluster's roster (FLARE per-cluster ``global.tsv`` carries only the
    cluster's samples by construction).
    """
    data = read_global_tsv(global_tsv)
    sids = list(data.sample_names)
    q = np.asarray(data.proportions, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != len(sids):
        raise LoaderError(f"{global_tsv}: malformed proportions shape {q.shape}")
    return sids, q


def load_popout_for_roster(global_tsv: Path, roster: list[str]) -> np.ndarray:
    """Load popout's whole-cohort global.tsv; return rows in roster order."""
    data = read_global_tsv(global_tsv)
    name_to_idx = {s: i for i, s in enumerate(data.sample_names)}
    _require_no_missing("popout.global.tsv", roster, set(name_to_idx))
    idx = np.array([name_to_idx[s] for s in roster], dtype=np.int64)
    q = np.asarray(data.proportions, dtype=np.float64)[idx]
    return q


def load_rye_for_roster(rye_q_path: Path, roster: list[str]) -> np.ndarray:
    """Read a Rye Q TSV; return rows in roster order, columns in RYE_LABELS order."""
    id_aliases = ("research_id", "sample_id", "sample")
    with open(rye_q_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        lower = [h.lower() for h in header]
        try:
            id_col = next(i for i, h in enumerate(lower) if h in id_aliases)
        except StopIteration:
            raise LoaderError(
                f"{rye_q_path}: no sample-id column found in header. Expected one of "
                f"{id_aliases}; got {header}"
            )
        col_idx: list[int] = []
        for label in RYE_LABELS:
            try:
                col_idx.append(lower.index(label))
            except ValueError:
                raise LoaderError(
                    f"{rye_q_path}: missing required Rye column {label!r}; got {header}"
                )
        per_sample: dict[str, np.ndarray] = {}
        max_col = max(id_col, max(col_idx))
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max_col:
                continue
            sid = parts[id_col].strip()
            if not sid:
                continue
            per_sample[sid] = np.array([float(parts[i]) for i in col_idx], dtype=np.float64)
    _require_no_missing("rye_q", roster, set(per_sample))
    return np.stack([per_sample[s] for s in roster], axis=0)


def load_rf_for_roster(
    rf_path: Path, roster: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Read the foxtrot RF ancestry TSV; return ``(soft_q (n × 6), hard_labels (n,))``.

    Columns expected: ``research_id``, ``ancestry_pred``, ``probabilities``
    (whitespace- or JSON-list-encoded length-6 in RF_LABELS_CANONICAL order).
    """
    by_id: dict[str, tuple[np.ndarray, str]] = {}
    with open(rf_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        lower = [h.lower() for h in header]
        try:
            id_col = lower.index("research_id")
            hard_col = lower.index("ancestry_pred")
            prob_col = lower.index("probabilities")
        except ValueError:
            raise LoaderError(
                f"{rf_path}: header missing one of "
                f"(research_id, ancestry_pred, probabilities); got {header}"
            )
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_col, hard_col, prob_col):
                continue
            sid = parts[id_col].strip()
            if not sid:
                continue
            probs_raw = parts[prob_col].strip()
            try:
                if probs_raw.startswith("["):
                    vals = json.loads(probs_raw)
                else:
                    vals = [float(x) for x in probs_raw.split()]
            except (ValueError, json.JSONDecodeError) as e:
                raise LoaderError(f"{rf_path}: failed to parse probabilities for {sid!r}: {e}")
            if len(vals) != len(RF_LABELS_CANONICAL):
                raise LoaderError(
                    f"{rf_path}: probabilities length {len(vals)} != "
                    f"{len(RF_LABELS_CANONICAL)} for {sid!r}"
                )
            by_id[sid] = (np.asarray(vals, dtype=np.float64), parts[hard_col].strip().lower())
    _require_no_missing("rf_ancestry", roster, set(by_id))
    soft = np.stack([by_id[s][0] for s in roster], axis=0)
    hard = np.array([by_id[s][1] for s in roster], dtype=object)
    return soft, hard


# ── Labels.json helpers ──────────────────────────────────────────────────


def load_labels(path: Path) -> dict:
    return read_labels_json(path)


# ── RF-basis projection ──────────────────────────────────────────────────


def project_to_rf_basis(
    q: np.ndarray,
    source: str,
    labels: dict | None = None,
) -> np.ndarray:
    """Project ``q`` (n × K_source) onto :data:`RF_LABELS_CANONICAL` (n × 6).

    ``source`` is one of ``"popout"``, ``"flare"``, ``"rye"``, ``"rf"``.
    For ``popout`` / ``flare`` ``labels`` is required and must carry
    ``rf_to_popout_components``. For ``rye`` and ``rf`` the columns are
    named natively and ``labels`` is ignored.
    """
    n = q.shape[0]
    out = np.zeros((n, len(RF_LABELS_CANONICAL)), dtype=np.float64)

    if source in ("popout", "flare"):
        if labels is None:
            raise LoaderError(f"project_to_rf_basis: labels required for source={source!r}")
        mapping = labels.get("rf_to_popout_components")
        if not isinstance(mapping, dict) or not mapping:
            raise LoaderError(
                f"project_to_rf_basis: labels.rf_to_popout_components missing or empty "
                f"(source={source!r})"
            )
        for j, rf_label in enumerate(RF_LABELS_CANONICAL):
            comp_idxs = mapping.get(rf_label) or []
            if not comp_idxs:
                continue
            out[:, j] = q[:, list(comp_idxs)].sum(axis=1)
        return out

    if source == "rye":
        if q.shape[1] != len(RYE_LABELS):
            raise LoaderError(f"project_to_rf_basis: rye q shape {q.shape} != (*, {len(RYE_LABELS)})")
        for j_rf, rf_label in enumerate(RF_LABELS_CANONICAL):
            if rf_label in RYE_LABELS:
                j_rye = RYE_LABELS.index(rf_label)
                out[:, j_rf] = q[:, j_rye]
            # else: mid → 0 (Rye does not track mid)
        return out

    if source == "rf":
        if q.shape[1] != len(RF_LABELS_CANONICAL):
            raise LoaderError(
                f"project_to_rf_basis: rf q shape {q.shape} != (*, {len(RF_LABELS_CANONICAL)})"
            )
        return q.copy()

    raise LoaderError(f"project_to_rf_basis: unknown source {source!r}")


# ── Per-pair metrics ─────────────────────────────────────────────────────

JACCARD_THRESHOLDS: tuple[float, ...] = (0.10, 0.25, 0.50)
MU_GATE = 0.01
PEARSON_THRESHOLD = 0.95
CCC_THRESHOLD = 0.90


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's CCC. Pure numpy. Returns NaN if either has zero variance."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float((2 * cov) / denom)


def per_label_metrics(
    anchor: np.ndarray,   # popout column for one RF label
    other: np.ndarray,    # other-tool column for the same RF label
) -> dict:
    """Per-RF-label concordance metrics. Anchor is popout (μ-gate fires on
    popout's mean — popout is the anchor in DX).
    """
    cluster_mu = float(anchor.mean())
    n = int(len(anchor))

    if anchor.std() == 0 or other.std() == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(anchor, other)[0, 1])
    ccc = lin_ccc(anchor, other)

    err = np.abs(anchor - other)
    mae_mean = float(err.mean())
    mae_median = float(np.median(err))
    mae_p95 = float(np.percentile(err, 95))

    jaccards: dict[float, float] = {}
    for tau in JACCARD_THRESHOLDS:
        a_mask = anchor >= tau
        b_mask = other >= tau
        inter = int(np.sum(a_mask & b_mask))
        union = int(np.sum(a_mask | b_mask))
        jaccards[tau] = float(inter / union) if union > 0 else float("nan")

    if cluster_mu < MU_GATE:
        passed: bool | None = None
    else:
        passed = bool(
            (not np.isnan(pearson_r) and pearson_r >= PEARSON_THRESHOLD)
            and (not np.isnan(ccc) and ccc >= CCC_THRESHOLD)
        )

    return {
        "popout_mu": cluster_mu,
        "n_samples_compared": n,
        "pearson_r": pearson_r,
        "ccc": ccc,
        "mae_mean": mae_mean,
        "mae_median": mae_median,
        "mae_p95": mae_p95,
        "jaccard_0.10": jaccards[0.10],
        "jaccard_0.25": jaccards[0.25],
        "jaccard_0.50": jaccards[0.50],
        "pass": passed,
    }


# ── TSV writer ───────────────────────────────────────────────────────────


def fmt_tsv_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if np.isnan(v):
            return "NA"
        return f"{v:.6f}"
    return str(v)
