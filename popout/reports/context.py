"""Per-run state: bundle, estimates, palette, assets dir, tag resolver.

The same ``ReportContext`` instance is threaded through every section
template. Templates access only:

  - ``ctx.bundle`` — the cohort bundle dict / dataclass for the run
  - ``ctx.estimates`` — ``dict[str, Estimate]`` (one per tool)
  - ``ctx.palette`` — ``dict[label_name, hex]`` for matplotlib + table colors
  - ``ctx.tag(section_id)`` — figure-tag shorthand for that section
  - ``ctx.assets_dir`` — where chart functions save PNGs
  - ``ctx.results`` — per-section ConcordanceResult cache (lazy)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from popout.estimates import ConcordanceResult, Estimate, compare
from popout.labelspace import by_name, get
from popout.labelspace.registry import LabelSpace
from popout.labelspace.shorthand import format as format_tag

from .config import ReportConfig, SectionSpec


# Canonical SP6 / SP5 colors (Paul Tol palette via popout.viz._style).
# Falling back to a sane default if popout.viz isn't importable.
try:
    from popout.viz._style import ANCESTRY_PALETTE as _PALETTE
except Exception:                       # pragma: no cover
    _PALETTE = [
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB", "#EE8866",
    ]


_LABEL_COLOR: dict[str, str] = {
    "afr": _PALETTE[0],
    "amr": _PALETTE[1],
    "eas": _PALETTE[2],
    "eur": _PALETTE[3],
    "mid": _PALETTE[4],
    "sas": _PALETTE[5],
}


@dataclasses.dataclass
class ReportContext:
    """Per-run context object passed into every template."""

    bundle: dict[str, Any]
    bundle_dir: Path
    config: ReportConfig
    estimates: dict[str, Estimate]
    assets_dir: Path
    palette: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(_LABEL_COLOR)
    )
    results: dict[str, ConcordanceResult] = dataclasses.field(default_factory=dict)

    # ── Section-aware helpers ────────────────────────────────────────

    def section(self, section_id: str) -> SectionSpec:
        for s in self.config.sections:
            if s.id == section_id:
                return s
        raise KeyError(f"no section with id {section_id!r}")

    def tag(self, section_id: str) -> str:
        """Per-section figure-tag shorthand.

        The report's ``tag_policy`` (set in the YAML ``defaults`` block)
        controls verbosity:

        - ``"verbose"`` (default; popout_dx): always emit the full tag
          ``L=<target>/MID± | tool=>method ... | v=<hash>``.
        - ``"minimal"`` (FLARE validation): when every parameter of
          this section matches the report's defaults and every
          per-tool matcher is ``by_name`` (the silent default in a
          verbatim-FLARE report), suppress the tag entirely. Otherwise
          emit a slimmed tag with ``=>name`` clauses removed and the
          ``/MID±`` qualifier dropped when it carries no information.
        """
        sec = self.section(section_id)
        if not sec.pair:
            return ""
        target = get(sec.target_space)
        assignments = []
        for tool in sec.pair:
            est = self.estimates.get(tool)
            members = est.label_space.members if est is not None else target.members
            a = by_name(members, target, source={"tool": tool})
            assignments.append(a)

        policy = getattr(self.config, "tag_policy", "verbose")
        if policy == "minimal":
            default_target = self.config.defaults.get("target_space", "SP5")
            default_mid = self.config.defaults.get("mid_rule")
            if (sec.target_space == default_target
                    and sec.mid_rule == default_mid
                    and all(a.method == "name" for a in assignments)):
                return ""
            return format_tag(
                target, assignments, mid_rule=sec.mid_rule,
                suppress_default_mid=True,
                suppress_name_clauses=True,
            )
        return format_tag(target, assignments, mid_rule=sec.mid_rule)

    def verbose_tag(self, section_id: str) -> str:
        """Full verbose tag for a section, ignoring the report policy.

        Used by the provenance appendix so the audit trail records the
        complete label-space coordinates even when the body suppresses
        them.
        """
        sec = self.section(section_id)
        if not sec.pair:
            return ""
        target = get(sec.target_space)
        assignments = []
        for tool in sec.pair:
            est = self.estimates.get(tool)
            members = est.label_space.members if est is not None else target.members
            a = by_name(members, target, source={"tool": tool})
            assignments.append(a)
        return format_tag(target, assignments, mid_rule=sec.mid_rule)

    def result(self, section_id: str) -> ConcordanceResult:
        """Lazy-compute and cache a ConcordanceResult for the section.

        Requires the section to declare a two-tool ``pair``.
        """
        if section_id in self.results:
            return self.results[section_id]
        sec = self.section(section_id)
        if len(sec.pair) != 2:
            raise ValueError(
                f"section {section_id!r}: result() requires a 2-tool pair, "
                f"got {sec.pair}"
            )
        left = self.estimates[sec.pair[0]]
        right = self.estimates[sec.pair[1]]
        out = compare(
            left, right,
            target_space=sec.target_space,
            mid_rule=sec.mid_rule,
        )
        self.results[section_id] = out
        return out

    def when_passes(self, sec: SectionSpec) -> bool:
        """Evaluate the section's ``when:`` clause (Python expression)."""
        if not sec.when:
            return True
        env = {"bundle": self.bundle, "estimates": self.estimates,
               "ctx": self}
        try:
            return bool(eval(sec.when, {"__builtins__": {}}, env))   # noqa: S307
        except Exception as e:
            raise ValueError(
                f"section {sec.id!r}: when-clause {sec.when!r} failed: {e}"
            )
