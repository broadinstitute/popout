"""Identity signatures for emergent prior-to-component binding.

A signature is a structural description of what a matching ancestry
component should look like. At each EM iteration the framework scores
every prior's signature(s) against every component's current state and
soft-assigns priors via softmax over the scores. This replaces the
brittle component-index-based binding with one that re-resolves itself
as the model's understanding of each component sharpens.

This module is stage-agnostic on purpose: the same code is reused both
at EM time (live binding) and by downstream consumers that want to
compute identity score matrices for analysis.

Two starter signature types are provided:

  - ``AIMSignature``: variance-normalized weighted L2 against a panel of
    documented ancestry-informative markers.
  - ``FSTReferenceSignature``: negative Hudson F_ST against a reference
    allele-frequency vector (e.g. a 1KG superpop).

Add new signature types by writing a frozen dataclass that conforms to
the ``IdentitySignature`` protocol; the dispatcher needs no changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ComponentState:
    """Snapshot of one component's state at one EM iteration.

    Identity scoring is **per-chromosome** in the current EM loop —
    callers build one ComponentState per (component, chromosome) pair
    using ``model.allele_freq[k]`` and ``chrom_data.pos_bp``.

    Attributes
    ----------
    freq : (n_sites,) float ndarray
        Per-site allele frequency for this component on this chromosome.
    mu : float
        Component-wide mixture proportion (one entry of ``model.mu``).
    pos_bp : (n_sites,) int ndarray
        Genomic positions of the model's sites on this chromosome.
    chrom : str
        Chromosome name. Required so panels and reference vectors can be
        filtered to the right chromosome at score time.
    """

    freq: np.ndarray
    mu: float
    pos_bp: np.ndarray
    chrom: str


@runtime_checkable
class IdentitySignature(Protocol):
    """Score a component's current state against this signature.

    Returns a real number; higher = better match. Absolute scale does
    not matter — :func:`compose_scores` z-standardizes across components
    inside each signature before combining, so signatures with wider raw
    score ranges do not drown out narrower ones.
    """

    weight: float

    def score(self, cs: ComponentState) -> float:  # pragma: no cover - protocol
        ...


def _normalize_chrom(c: str) -> str:
    """Drop a leading ``chr`` for cross-source chromosome matching."""
    s = str(c)
    return s[3:] if s.startswith("chr") else s


# ---------------------------------------------------------------------------
# AIM panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AIMPanel:
    """A panel of ancestry-informative markers.

    All arrays are parallel (one entry per marker).
    """

    chrom: np.ndarray         # (M,) str
    pos_bp: np.ndarray        # (M,) int64
    expected_freq: np.ndarray # (M,) float64 — alt-allele freq in the target population
    marker_weight: np.ndarray # (M,) float64 — per-marker reliability in [0, 1]
    source: str = "<panel>"

    def __post_init__(self) -> None:
        n = len(self.chrom)
        for name, arr in (
            ("pos_bp", self.pos_bp),
            ("expected_freq", self.expected_freq),
            ("marker_weight", self.marker_weight),
        ):
            if len(arr) != n:
                raise ValueError(
                    f"AIMPanel arrays must be parallel, got "
                    f"chrom={n}, {name}={len(arr)}"
                )
        if n == 0:
            raise ValueError("AIMPanel must have at least one marker")
        # check (chrom, pos) uniqueness
        keys = list(zip([str(c) for c in self.chrom], self.pos_bp.tolist()))
        if len(set(keys)) != n:
            raise ValueError("AIMPanel has duplicate (chrom, pos_bp) entries")

    @classmethod
    def from_tsv(cls, path: str | Path, source: str | None = None) -> "AIMPanel":
        """Load a panel from a TSV with header columns
        ``chrom, pos_bp, expected_freq, weight`` (extra columns ignored).
        """
        path = Path(path)
        chrom_l: list[str] = []
        pos_l: list[int] = []
        exp_l: list[float] = []
        w_l: list[float] = []
        with open(path) as f:
            header_line = f.readline()
            if not header_line:
                raise ValueError(f"AIM panel {path} is empty")
            header = header_line.rstrip("\n").split("\t")
            need = {"chrom", "pos_bp", "expected_freq", "weight"}
            missing = need - set(header)
            if missing:
                raise ValueError(
                    f"AIM panel {path} missing columns: {sorted(missing)}"
                )
            ci = {h: i for i, h in enumerate(header)}
            for line in f:
                stripped = line.rstrip("\n")
                if not stripped or stripped.startswith("#"):
                    continue
                row = stripped.split("\t")
                chrom_l.append(str(row[ci["chrom"]]))
                pos_l.append(int(row[ci["pos_bp"]]))
                exp_l.append(float(row[ci["expected_freq"]]))
                w_l.append(float(row[ci["weight"]]))
        if not chrom_l:
            raise ValueError(f"AIM panel {path} has no markers")
        return cls(
            chrom=np.array(chrom_l, dtype=object),
            pos_bp=np.array(pos_l, dtype=np.int64),
            expected_freq=np.array(exp_l, dtype=np.float64),
            marker_weight=np.array(w_l, dtype=np.float64),
            source=source if source is not None else str(path),
        )


@dataclass(frozen=True)
class AIMSignature:
    """Variance-normalized weighted L2 fit to an AIM panel.

    For markers where the panel and the component's chromosome overlap,
    the score is::

        -sum_l weight_l * (freq_k[l] - expected_l)^2
                / (expected_l * (1 - expected_l) + eps)

    Markers whose expected frequency is near 0 or 1 carry little
    information and are de-weighted by the variance denominator anyway.
    Returns 0.0 when the panel and component share no markers on the
    component's chromosome (no information from this signature on this
    chromosome).
    """

    panel: AIMPanel
    weight: float = 1.0

    def score(self, cs: ComponentState) -> float:
        norm_chrom = _normalize_chrom(cs.chrom)
        norm_panel = np.array(
            [_normalize_chrom(c) for c in self.panel.chrom], dtype=object,
        )
        chrom_mask = norm_panel == norm_chrom
        if not chrom_mask.any():
            return 0.0

        panel_pos = self.panel.pos_bp[chrom_mask]
        panel_exp = self.panel.expected_freq[chrom_mask]
        panel_w = self.panel.marker_weight[chrom_mask]

        _common, p_idx, c_idx = np.intersect1d(
            panel_pos, cs.pos_bp, return_indices=True,
        )
        if len(_common) == 0:
            return 0.0

        exp = panel_exp[p_idx]
        w = panel_w[p_idx]
        freq = np.asarray(cs.freq)[c_idx]

        var = exp * (1.0 - exp) + 1e-9
        diff_sq = np.nan_to_num(
            (freq - exp) ** 2, nan=0.0, posinf=0.0, neginf=0.0,
        )
        contrib = w * diff_sq / var
        return float(-contrib.sum())


# ---------------------------------------------------------------------------
# F_ST reference signature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FSTReferenceSignature:
    """Negative Hudson F_ST against a reference allele-frequency vector.

    Hudson (1992) ratio-of-averages F_ST estimator (Bhatia 2013 form,
    bias-correction terms omitted because we have no per-site sample
    sizes for the model component)::

        F_ST = mean_l[(p1 - p2)^2]
             / mean_l[p1*(1 - p2) + p2*(1 - p1)]

    Lower F_ST = closer match; the score returns ``-F_ST`` so higher =
    better, consistent with other signatures.

    The reference vectors are typically loaded once per superpop via
    :func:`popout.fetch_ref.load_ref_frequencies` and filtered to a
    single chromosome before being passed to this constructor.
    """

    ref_freq: np.ndarray      # (n_ref_sites,) float
    ref_pos_bp: np.ndarray    # (n_ref_sites,) int
    ref_chrom: str
    ref_name: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.ref_freq) != len(self.ref_pos_bp):
            raise ValueError(
                f"ref_freq ({len(self.ref_freq)}) and ref_pos_bp "
                f"({len(self.ref_pos_bp)}) must have the same length"
            )

    def score(self, cs: ComponentState) -> float:
        if _normalize_chrom(cs.chrom) != _normalize_chrom(self.ref_chrom):
            return 0.0

        _common, ref_idx, c_idx = np.intersect1d(
            self.ref_pos_bp, cs.pos_bp, return_indices=True,
        )
        if len(_common) == 0:
            return 0.0

        p1 = np.asarray(cs.freq, dtype=np.float64)[c_idx]
        p2 = np.asarray(self.ref_freq, dtype=np.float64)[ref_idx]

        finite = np.isfinite(p1) & np.isfinite(p2)
        if not finite.any():
            return 0.0
        p1 = p1[finite]
        p2 = p2[finite]

        num = float(((p1 - p2) ** 2).mean())
        den = float((p1 * (1.0 - p2) + p2 * (1.0 - p1)).mean())
        if den < 1e-9:
            # Both populations fixed at the same allele across all sites:
            # no information, return 0 (neutral).
            return 0.0
        return float(-(num / den))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose_scores(
    signatures: Iterable[IdentitySignature],
    component_states: list[ComponentState],
) -> np.ndarray:
    """Combine signatures' per-component scores into one vector.

    For each signature: score all K components, z-standardize across the
    K, multiply by the signature's weight, and sum. Returns shape ``(K,)``.

    Z-standardization happens *within* a signature *across* components
    (not across signatures across components) so signatures with wider
    raw score ranges don't drown out narrower ones.

    Degenerate signatures (those whose K scores are all the same, or
    all non-finite) contribute zero to the combined score.
    """
    K = len(component_states)
    sigs = list(signatures)
    if K == 0:
        return np.zeros(0, dtype=np.float64)
    if not sigs:
        return np.zeros(K, dtype=np.float64)

    z_combined = np.zeros(K, dtype=np.float64)
    for sig in sigs:
        raw = np.array(
            [sig.score(cs) for cs in component_states], dtype=np.float64,
        )
        finite = np.isfinite(raw)
        if not finite.any():
            continue
        m = float(raw[finite].mean())
        s = float(raw[finite].std())
        if s < 1e-12:
            continue
        z = np.where(finite, (raw - m) / s, 0.0)
        z_combined += float(sig.weight) * z
    return z_combined
