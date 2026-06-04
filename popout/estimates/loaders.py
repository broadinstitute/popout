"""Per-tool Estimate loaders.

Each loader trusts the tool's native naming convention:

================  ===========================================================
Loader            Source of label names
================  ===========================================================
read_flare_*      ``##ANCESTRY=`` header of the FLARE VCF (or supplied panel)
read_popout_*     external Assignment (popout components are unnamed)
read_rye_*        column header of the Rye Q TSV
read_rf_*         the canonical SP6 ordering (fixed by the foxtrot v4 model)
================  ===========================================================
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from popout.labelspace import Assignment, project_proportions
from popout.labelspace.registry import (
    SP6,
    LabelSpace,
    make_native_space,
)

from .record import Estimate


# ── FLARE ────────────────────────────────────────────────────────────────


def read_flare_panel_names(vcf_path: str | Path) -> list[str]:
    """Extract the panel population names from a FLARE VCF header.

    Reads only the leading header bytes (does not stream the body).
    The header line we expect:

        ##ANCESTRY=<afr,eas,amr,eur,sas>

    FLARE writes the names comma-separated inside angle brackets,
    space-separated without the brackets, or whitespace-separated —
    we accept all three. Casing is normalised to lower-case.
    """
    path = Path(vcf_path)
    opener = gzip.open if str(path).endswith(".gz") else open
    panel_line = None
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("##ANCESTRY="):
                panel_line = line.strip()
                break
            if not line.startswith("#"):
                break    # body started; we missed it
    if panel_line is None:
        raise ValueError(
            f"{vcf_path}: ##ANCESTRY= header not found "
            f"(FLARE VCFs are expected to declare the panel populations there)"
        )
    body = panel_line.removeprefix("##ANCESTRY=").strip()
    if body.startswith("<") and body.endswith(">"):
        body = body[1:-1]
    raw_names = [
        s.strip().lower()
        for s in body.replace(",", " ").split()
        if s.strip()
    ]
    if not raw_names:
        raise ValueError(
            f"{vcf_path}: ##ANCESTRY= header is present but empty: {panel_line!r}"
        )
    return raw_names


def read_flare_aggregated(
    global_tsv: str | Path,
    *,
    scope: tuple[str, ...],
    panel_names: Iterable[str] | None = None,
) -> Estimate:
    """Read a FLARE per-sample aggregated TSV into an Estimate.

    Accepted shapes:

    1. **Named columns** (post-Phase-6):
       ``sample_id<TAB>afr<TAB>amr<TAB>eas<TAB>eur<TAB>sas``
       — ``panel_names`` is optional; if supplied it must match.

    2. **Anonymous columns** (legacy popout-format):
       ``sample_id<TAB>ancestry_0<TAB>...<TAB>ancestry_{K-1}``
       — ``panel_names`` is **required** and must have length K. The
       loader renames the columns in-place.

    For the legacy shape, pass ``panel_names=read_flare_panel_names(vcf_path)``
    so the panel header is the single source of truth.
    """
    lines = Path(global_tsv).read_text().rstrip("\n").splitlines()
    if not lines:
        raise ValueError(f"{global_tsv}: empty file")
    header = lines[0].split("\t")
    if header[0] != "sample_id":
        raise ValueError(
            f"{global_tsv}: first column must be 'sample_id', got {header[0]!r}"
        )
    raw_cols = header[1:]
    if not raw_cols:
        raise ValueError(f"{global_tsv}: no ancestry columns")

    anonymous = all(
        c.startswith("ancestry_") and c[len("ancestry_"):].isdigit()
        for c in raw_cols
    )
    if anonymous:
        if panel_names is None:
            raise ValueError(
                f"{global_tsv}: anonymous columns ({raw_cols[:3]}...) require "
                f"panel_names. Pass read_flare_panel_names(vcf_path)."
            )
        members = tuple(str(n).lower() for n in panel_names)
        if len(members) != len(raw_cols):
            raise ValueError(
                f"{global_tsv}: {len(raw_cols)} columns but panel_names has "
                f"{len(members)}"
            )
    else:
        members = tuple(c.lower() for c in raw_cols)
        if panel_names is not None:
            supplied = tuple(str(n).lower() for n in panel_names)
            if supplied != members:
                raise ValueError(
                    f"{global_tsv}: header {members} disagrees with supplied "
                    f"panel_names {supplied}"
                )

    sample_ids: list[str] = []
    rows: list[list[float]] = []
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != len(header):
            raise ValueError(
                f"{global_tsv}:{n}: expected {len(header)} cols, got {len(parts)}"
            )
        sample_ids.append(parts[0])
        try:
            rows.append([float(v) for v in parts[1:]])
        except ValueError as e:
            raise ValueError(f"{global_tsv}:{n}: {e}")

    label_space = make_native_space("flare", members)
    return Estimate(
        tool="flare",
        scope=scope,
        sample_ids=tuple(sample_ids),
        label_space=label_space,
        proportions=np.asarray(rows, dtype=np.float64),
        hard_calls=None,
        provenance={
            "source": str(global_tsv),
            "input_format": "anonymous" if anonymous else "named",
            "panel_names": list(members),
        },
    )


# ── popout ───────────────────────────────────────────────────────────────


def read_popout_global(
    global_tsv: str | Path,
    *,
    scope: tuple[str, ...],
    assignment: Assignment | None = None,
    sample_ids: Iterable[str] | None = None,
) -> Estimate:
    """Read a popout ``.global.tsv`` and project into an Assignment's target.

    popout components are *unnamed* — they're emitted as
    ``ancestry_0..K-1``. Without an ``Assignment`` the resulting
    Estimate would violate the "named labels from the start"
    invariant, so the loader requires one.

    When ``assignment`` is supplied, the loader applies
    ``project_proportions`` and returns an Estimate in the
    assignment's ``target_space``. When ``sample_ids`` is supplied
    the rows are subset+reordered to that roster (error on missing).
    """
    if assignment is None:
        raise ValueError(
            "read_popout_global: an Assignment is required (popout "
            "components are unnamed; use labelspace.matching to build one)"
        )
    lines = Path(global_tsv).read_text().rstrip("\n").splitlines()
    if not lines:
        raise ValueError(f"{global_tsv}: empty file")
    header = lines[0].split("\t")
    if header[0] != "sample_id":
        raise ValueError(
            f"{global_tsv}: first column must be 'sample_id', got {header[0]!r}"
        )
    raw_cols = header[1:]
    n_components = len(raw_cols)

    file_sample_ids: list[str] = []
    rows: list[list[float]] = []
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != len(header):
            raise ValueError(
                f"{global_tsv}:{n}: expected {len(header)} cols, got {len(parts)}"
            )
        file_sample_ids.append(parts[0])
        try:
            rows.append([float(v) for v in parts[1:]])
        except ValueError as e:
            raise ValueError(f"{global_tsv}:{n}: {e}")
    q = np.asarray(rows, dtype=np.float64)

    if sample_ids is not None:
        requested = list(sample_ids)
        idx = {s: i for i, s in enumerate(file_sample_ids)}
        missing = [s for s in requested if s not in idx]
        if missing:
            raise ValueError(
                f"{global_tsv}: roster missing from popout global: {missing[:5]} "
                f"(+{len(missing) - 5} more)" if len(missing) > 5
                else f"{global_tsv}: roster missing from popout global: {missing}"
            )
        q = q[[idx[s] for s in requested]]
        out_sample_ids = tuple(requested)
    else:
        out_sample_ids = tuple(file_sample_ids)

    projected = project_proportions(q, assignment)
    return Estimate(
        tool="popout",
        scope=scope,
        sample_ids=out_sample_ids,
        label_space=assignment.target_space,
        proportions=projected,
        hard_calls=None,
        provenance={
            "source": str(global_tsv),
            "n_native_components": n_components,
            "assignment_method": assignment.method,
            "assignment_target": assignment.target_space.tag,
        },
    )


# ── Rye ──────────────────────────────────────────────────────────────────


_RYE_ID_ALIASES = ("research_id", "sample_id", "sample")


def read_rye_q(
    q_path: str | Path,
    *,
    scope: tuple[str, ...],
    sample_ids: Iterable[str] | None = None,
) -> Estimate:
    """Read a Rye Q TSV; column header is the label_space.

    Rye writes one named proportion column per ancestry (typically
    ``eur eas amr afr sas`` — note: *not* alphabetical). The sample-id
    column may live anywhere in the header and may be named
    ``research_id``, ``sample_id``, or ``sample``.
    """
    lines = Path(q_path).read_text().rstrip("\n").splitlines()
    if not lines:
        raise ValueError(f"{q_path}: empty file")
    header = lines[0].split("\t")
    lower = [h.lower() for h in header]
    try:
        id_col = next(i for i, h in enumerate(lower) if h in _RYE_ID_ALIASES)
    except StopIteration:
        raise ValueError(
            f"{q_path}: no sample-id column in header (looked for "
            f"{_RYE_ID_ALIASES}); got {header}"
        )
    members = tuple(
        h.lower() for i, h in enumerate(header) if i != id_col
    )
    label_cols = [i for i in range(len(header)) if i != id_col]
    if not members:
        raise ValueError(f"{q_path}: no label columns")

    file_sample_ids: list[str] = []
    rows: list[list[float]] = []
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != len(header):
            raise ValueError(
                f"{q_path}:{n}: expected {len(header)} cols, got {len(parts)}"
            )
        file_sample_ids.append(parts[id_col])
        try:
            rows.append([float(parts[i]) for i in label_cols])
        except ValueError as e:
            raise ValueError(f"{q_path}:{n}: {e}")
    q = np.asarray(rows, dtype=np.float64)

    if sample_ids is not None:
        requested = list(sample_ids)
        idx = {s: i for i, s in enumerate(file_sample_ids)}
        missing = [s for s in requested if s not in idx]
        if missing:
            raise ValueError(
                f"{q_path}: roster missing from rye Q: {missing[:5]}"
                + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
            )
        q = q[[idx[s] for s in requested]]
        out_sample_ids = tuple(requested)
    else:
        out_sample_ids = tuple(file_sample_ids)

    label_space = make_native_space("rye", members)
    return Estimate(
        tool="rye",
        scope=scope,
        sample_ids=out_sample_ids,
        label_space=label_space,
        proportions=q,
        hard_calls=None,
        provenance={"source": str(q_path), "native_order": list(members)},
    )


# ── RF classifier ────────────────────────────────────────────────────────


def read_rf_table(
    rf_path: str | Path,
    *,
    scope: tuple[str, ...],
    sample_ids: Iterable[str] | None = None,
) -> Estimate:
    """Read foxtrot RF predictions; emit Estimate in SP6 (with hard_calls).

    Columns required: ``research_id``, ``ancestry_pred``,
    ``probabilities`` (whitespace- or JSON-encoded length-6 vector in
    SP6 order).
    """
    lines = Path(rf_path).read_text().rstrip("\n").splitlines()
    if not lines:
        raise ValueError(f"{rf_path}: empty file")
    header = lines[0].split("\t")
    lower = [h.lower() for h in header]
    try:
        id_col = lower.index("research_id")
        hard_col = lower.index("ancestry_pred")
        prob_col = lower.index("probabilities")
    except ValueError:
        raise ValueError(
            f"{rf_path}: header missing one of (research_id, ancestry_pred, "
            f"probabilities); got {header}"
        )
    by_id: dict[str, tuple[np.ndarray, str]] = {}
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) <= max(id_col, hard_col, prob_col):
            continue
        sid = parts[id_col].strip()
        if not sid:
            continue
        raw = parts[prob_col].strip()
        if raw.startswith("["):
            vals = json.loads(raw)
        else:
            vals = [float(x) for x in raw.split()]
        if len(vals) != len(SP6.members):
            raise ValueError(
                f"{rf_path}:{n}: probabilities length {len(vals)} != "
                f"|SP6| = {len(SP6.members)}"
            )
        by_id[sid] = (np.asarray(vals, dtype=np.float64),
                      parts[hard_col].strip().lower())

    if sample_ids is None:
        out_sample_ids = tuple(sorted(by_id))
    else:
        out_sample_ids = tuple(sample_ids)
        missing = [s for s in out_sample_ids if s not in by_id]
        if missing:
            raise ValueError(
                f"{rf_path}: roster missing from RF table: {missing[:5]}"
                + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
            )

    proportions = np.stack(
        [by_id[s][0] for s in out_sample_ids], axis=0
    )
    hard_calls = np.array(
        [by_id[s][1] for s in out_sample_ids], dtype=object
    )
    return Estimate(
        tool="rf",
        scope=scope,
        sample_ids=out_sample_ids,
        label_space=SP6,
        proportions=proportions,
        hard_calls=hard_calls,
        provenance={"source": str(rf_path)},
    )
