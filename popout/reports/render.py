"""Section renderer + pandoc driver.

``render_report(ctx)`` walks the section list in order, evaluates each
section's ``when:`` clause, runs its Jinja2 template with ``ctx`` and
section-scoped helpers, and concatenates the rendered markdown into
one document. ``run_pandoc(md, pdf)`` invokes pandoc + xelatex.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .context import ReportContext


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


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


def render_report(ctx: ReportContext) -> str:
    """Return the assembled markdown for the entire report."""
    env = _env()
    parts: list[str] = []
    for sec in ctx.config.sections:
        if not ctx.when_passes(sec):
            continue
        template = env.get_template(sec.template)
        rendered = template.render(ctx=ctx, section=sec, **sec.options)
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
