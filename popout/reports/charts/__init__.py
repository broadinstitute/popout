"""Chart functions for report sections.

Each chart is a pure function: ``render(ctx, **opts) -> matplotlib.Figure``.
No file I/O inside the chart; ``render_report`` is responsible for
calling ``fig.savefig(assets_dir / f"{section.id}.png")`` and stamping
the figure-tag footer via the Jinja2 ``figure`` macro.

Phase 3 onwards populates this module with one file per chart kind.
"""

from __future__ import annotations
