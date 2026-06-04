"""YAML manifest → ``ReportConfig`` + per-section ``SectionSpec`` list.

Sample manifest::

    title: FLARE validation report
    style:
      pdf_engine: xelatex
      margin: 0.75in
      fontsize: 10pt
      mainfont: Helvetica
      monofont: Menlo
    defaults:
      target_space: SP5
      mid_rule: drop
    sections:
      - id: cover
        template: flare_validation/cover.j2
      - id: flare_vs_rf_confusion
        template: flare_validation/confusion.j2
        pair: [flare, rf]
        mid_rule: fold_to_eur
      - id: bp_confusion
        template: popout_dx/bp_confusion.j2
        when: bundle.mode == "global_local"

A ``SectionSpec`` is the parsed dict: every section's ``target_space``,
``mid_rule``, and ``pair`` defaults are filled in from the report-level
``defaults`` if absent.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


# ── Style ───────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ReportStyle:
    pdf_engine: str = "xelatex"
    margin: str = "0.75in"
    fontsize: str = "10pt"
    mainfont: str = "Helvetica"
    monofont: str = "Menlo"
    highlight_style: str = "tango"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ReportStyle":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


# ── Section ─────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SectionSpec:
    """One section in the report."""

    id: str
    template: str
    pair: tuple[str, ...] = ()
    target_space: str = "SP5"
    mid_rule: str | None = None         # "drop" / "fold_to_eur" / None
    when: str | None = None             # Python expression on ctx
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any], defaults: dict[str, Any]) -> "SectionSpec":
        d = {**defaults, **d}
        return cls(
            id=d["id"],
            template=d["template"],
            pair=tuple(d.get("pair") or ()),
            target_space=d.get("target_space", "SP5"),
            mid_rule=d.get("mid_rule"),
            when=d.get("when"),
            options={
                k: v for k, v in d.items()
                if k not in {"id", "template", "pair",
                              "target_space", "mid_rule", "when"}
            },
        )


# ── Report ─────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ReportConfig:
    title: str
    style: ReportStyle
    sections: tuple[SectionSpec, ...]
    raw: dict[str, Any]                 # the original parsed YAML dict


def load_report_config(yaml_path: str | Path) -> ReportConfig:
    """Read a report-manifest YAML and return a parsed ``ReportConfig``."""
    raw = yaml.safe_load(Path(yaml_path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: top-level YAML must be a mapping")
    title = raw.get("title", "")
    style = ReportStyle.from_dict(raw.get("style"))
    defaults = raw.get("defaults", {}) or {}
    sec_dicts = raw.get("sections", []) or []
    if not isinstance(sec_dicts, list):
        raise ValueError(f"{yaml_path}: 'sections' must be a list")
    sections: list[SectionSpec] = []
    for n, s in enumerate(sec_dicts, start=1):
        if not isinstance(s, dict):
            raise ValueError(f"{yaml_path}: section #{n} must be a mapping")
        if "id" not in s or "template" not in s:
            raise ValueError(
                f"{yaml_path}: section #{n} missing required keys (id, template); got {s}"
            )
        sections.append(SectionSpec.from_dict(s, defaults))
    return ReportConfig(title=title, style=style,
                         sections=tuple(sections), raw=raw)
