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


def _stamp_tag(fig, tag: str) -> None:
    """Inject the figure-tag shorthand as a bottom-strip footer."""
    if not tag:
        return
    fig.text(
        0.5, 0.005, tag,
        ha="center", va="bottom",
        fontsize=6.5, color="#666",
        family="monospace", alpha=0.85,
    )


def _run_charts_for_section(ctx: ReportContext, sec) -> tuple[Path | None, dict]:
    """Compute + render a section's chart (if any). Returns (png_path, data_dict)."""
    chart_name = sec.options.get("chart")
    if not chart_name:
        return None, {}
    mod = _charts.get(chart_name)
    data = mod.compute(ctx)
    fig = mod.render(data, palette=ctx.palette)
    _stamp_tag(fig, ctx.tag(sec.id))
    path = ctx.assets_dir / f"{sec.id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path, data


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
            **{k: v for k, v in sec.options.items() if k != "chart"},
        )
        parts.append(rendered)
        parts.append("\n\n\\newpage\n\n")
    if parts:
        parts.pop()                              # drop trailing page break
    return "".join(parts)


def run_pandoc(md_path: Path, out_pdf: Path, *, style=None) -> None:
    """Render a markdown file → PDF via pandoc + xelatex."""
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
    print(f"[reports] pandoc → {out_pdf}", file=sys.stderr, flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"pandoc exit {res.returncode}")
