"""Chart functions for report sections.

Each chart module exposes two functions:

  - ``compute(ctx, section=None) -> dict``
        Pure data prep. ``section`` is the active SectionSpec — gives
        access to target_space / mid_rule / pair so the chart can
        apply per-section transformations (e.g. fold MID into EUR
        for FLARE vs RF). The returned dict is also threaded into
        the section's Jinja2 template under ``data``.

  - ``render(data, *, palette) -> matplotlib.Figure``
        Pure matplotlib; no file I/O.

The renderer (``popout.reports.render.render_report``) calls
``compute`` once per section, saves the figure to the assets dir,
stamps the figure-tag footer, and threads the data dict into the
template.
"""

from __future__ import annotations

from typing import Any

from . import (
    cohort_composition,
    concordance_strip,
    confusion_heatmap,
    hap_disagreement,
    regional_manhattan,
    switch_rate,
    tract_length,
)


# Registry of available chart modules. A section in the YAML manifest
# references a chart by its registry key (``chart: cohort_composition``).
CHARTS: dict[str, Any] = {
    "cohort_composition": cohort_composition,
    "concordance_strip": concordance_strip,
    "confusion_heatmap": confusion_heatmap,
    "hap_disagreement": hap_disagreement,
    "regional_manhattan": regional_manhattan,
    "switch_rate": switch_rate,
    "tract_length": tract_length,
}


def get(name: str):
    """Look up a chart module by name; raise KeyError on miss."""
    try:
        return CHARTS[name]
    except KeyError:
        raise KeyError(
            f"unknown chart {name!r}; registered: {sorted(CHARTS)}"
        )
