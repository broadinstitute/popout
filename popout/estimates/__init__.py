"""Generic per-tool ancestry estimate record.

Every ancestry-emitting tool (FLARE, popout, Rye, RF) produces an
``Estimate`` via its loader. Estimates carry **named** labels from the
start — no ``ancestry_0..K-1`` intermediate, no ``labels.json``
indirection. Comparison between two Estimates goes through the
``compare`` layer, which projects both into a target ``LabelSpace`` and
returns a ``ConcordanceResult`` with per-label metrics, optional
confusion matrix, and label-permutation-invariant cluster metrics
(ARI / NMI / V-measure).

See ``my_notes/labels/LABEL_SPACE.md`` for the surrounding label-space
design. The companion :mod:`popout.labelspace` module owns vocabularies
(``SP6``, ``SP5``), matching strategies, projection, naming, and the
figure-tag shorthand.
"""

from __future__ import annotations

from .compare import ConcordanceResult, compare
from .loaders import (
    read_flare_aggregated,
    read_flare_panel_names,
    read_popout_global,
    read_rf_table,
    read_rye_q,
)
from .record import Estimate

__all__ = [
    "Estimate",
    "ConcordanceResult",
    "compare",
    "read_flare_aggregated",
    "read_flare_panel_names",
    "read_popout_global",
    "read_rf_table",
    "read_rye_q",
]
