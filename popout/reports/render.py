"""Section renderer + pandoc driver.

``render_report(ctx)`` walks the section list in order, evaluates each
section's ``when:`` clause, runs the section's chart (if any), then
renders its Jinja2 template with ``ctx``, the chart path, and the
computed data dict. ``run_pandoc(md, pdf)`` invokes pandoc + xelatex.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from . import charts as _charts
from .context import ReportContext


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_DPI = 220


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.globals["now"] = lambda: dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    env.globals["page_break"] = "\n\n\\newpage\n\n"
    return env


def _run_charts_for_section(ctx: ReportContext, sec) -> tuple[Path | None, dict]:
    """Compute + render a section's chart (if any). Returns (png_path, data_dict).

    A section can declare either ``chart: <name>`` (compute + render +
    save PNG) or ``data: <name>`` (compute only, no figure — for
    table-only sections that still need a data dict).

    The label-space figure tag is emitted only via the markdown
    ``figure`` macro (see ``_macros.j2``) — no matplotlib stamp.
    """
    chart_name = sec.options.get("chart")
    data_name = sec.options.get("data")
    if chart_name:
        mod = _charts.get(chart_name)
        data = mod.compute(ctx, sec)
        fig = mod.render(data, palette=ctx.palette)
        path = ctx.assets_dir / f"{sec.id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path, data
    if data_name:
        mod = _charts.get(data_name)
        return None, mod.compute(ctx, sec)
    return None, {}


def render_report(ctx: ReportContext) -> str:
    """Return the assembled markdown for the entire report."""
    env = _env()
    parts: list[str] = []
    for sec in ctx.config.sections:
        if not ctx.when_passes(sec):
            continue
        chart_path, data = _run_charts_for_section(ctx, sec)
        template = env.get_template(sec.template)
        rendered = template.render(
            ctx=ctx, section=sec,
            chart=str(chart_path) if chart_path else None,
            data=data,
            **{k: v for k, v in sec.options.items()
               if k not in ("chart", "data")},
        )
        parts.append(rendered)
        parts.append("\n\n\\newpage\n\n")
    if parts:
        parts.pop()                              # drop trailing page break
    return "".join(parts)


_DRAFT_HEADER = r"""\usepackage{fancyhdr}
\usepackage{xcolor}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\color{red!70!black}\textbf{DRAFT}\ \textbar\ working draft, not a finalized document}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \fancyfoot[C]{\color{red!70!black}\textbf{DRAFT}\ \textbar\ working draft, not a finalized document}%
  \fancyfoot[R]{\thepage}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0.4pt}%
}
"""


def run_pandoc(md_path: Path, out_pdf: Path, *,
               style=None, draft: bool = False) -> None:
    """Render a markdown file → PDF via pandoc + xelatex.

    ``draft=True`` injects a ``DRAFT`` footer on every page via fancyhdr.
    """
    if style is None:
        # Sensible defaults; tests pass a real ReportStyle here.
        margin = "0.75in"
        fontsize = "10pt"
        mainfont = "Helvetica"
        monofont = "Menlo"
        engine = "xelatex"
        highlight = "tango"
    else:
        margin = style.margin
        fontsize = style.fontsize
        mainfont = style.mainfont
        monofont = style.monofont
        engine = style.pdf_engine
        highlight = style.highlight_style
    cmd = [
        "pandoc", str(md_path), "-o", str(out_pdf),
        f"--pdf-engine={engine}",
        "-V", f"geometry:margin={margin}",
        "-V", f"fontsize={fontsize}",
        "-V", f"mainfont={mainfont}",
        "-V", f"monofont={monofont}",
        f"--highlight-style={highlight}",
    ]
    if draft:
        cmd += ["-V", f"header-includes={_DRAFT_HEADER}"]
    print(f"[reports] pandoc → {out_pdf}", file=sys.stderr, flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"pandoc exit {res.returncode}")
