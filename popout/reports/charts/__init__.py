"""Chart functions for report sections.

Each chart module exposes two functions:

  - ``compute(ctx) -> dict``   — pure data prep; the dict is passed back
                                  into the section's Jinja2 template so
                                  prose / tables / callouts share the
                                  same numbers as the figure.
  - ``render(data, *, palette) -> matplotlib.Figure``
                                  — pure matplotlib; no file I/O.

The renderer (``popout.reports.render.render_report``) calls
``compute`` once per section, saves the figure to the assets dir,
stamps the figure-tag footer, and threads the data dict into the
template under the key ``data``.
"""

from __future__ import annotations

from typing import Any

from . import cohort_composition


# Registry of available chart modules. A section in the YAML manifest
# references a chart by its registry key (``chart: cohort_composition``).
CHARTS: dict[str, Any] = {
    "cohort_composition": cohort_composition,
}


def get(name: str):
    """Look up a chart module by name; raise KeyError on miss."""
    try:
        return CHARTS[name]
    except KeyError:
        raise KeyError(
            f"unknown chart {name!r}; registered: {sorted(CHARTS)}"
        )
