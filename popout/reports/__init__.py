"""Report rendering infrastructure.

Two parallel deliverables — the FLARE validation report (NIH
deliverable) and the popout DX report (experimental) — share this
package. Each report is described by a YAML manifest listing its
sections in order; sections combine chart functions (pure
matplotlib renderers in :mod:`popout.reports.charts`) with Jinja2
templates that own the surrounding prose.

The render pipeline:

1. ``config.load_report_config(yaml_path)`` → list of ``SectionSpec``.
2. ``context.ReportContext`` is built per run with the bundle dir,
   ``Estimate`` registry, palette, and an assets-dir for chart PNGs.
3. ``render.render_report(ctx)`` walks sections in order, calls each
   section's chart function (saving PNGs to the assets dir), runs
   Jinja2 against the section template, concatenates the markdown,
   and shells out to pandoc.
"""

from __future__ import annotations

from .config import ReportConfig, SectionSpec, load_report_config
from .context import ReportContext
from .render import render_report, run_pandoc

__all__ = [
    "ReportConfig",
    "ReportContext",
    "SectionSpec",
    "load_report_config",
    "render_report",
    "run_pandoc",
]
