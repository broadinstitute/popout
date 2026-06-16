"""Figure-tag shorthand: ``L=SP6/MID+ | popout=>corrH | v=ab12cd``.

See ``my_notes/labels/LABEL_SPACE.md`` §6. The tag is a deterministic
function of (target_space, every per-tool ``Assignment``, params). Two
figures with the same tag are guaranteed to be in the same mapped label
space.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

from .assignment import Assignment
from .registry import LabelSpace


_CLAUSE_RE = re.compile(r"^\s*(?P<tool>[^=\s]+)\s*=>\s*(?P<method>\w+)(?:\s*=>\s*(?P<extra>\w+))?\s*$")


def format(
    target: LabelSpace,
    assignments: Iterable[Assignment] | dict[str, Assignment],
    params: dict | None = None,
    *,
    hash_len: int = 6,
    mid_rule: str | None = None,
    suppress_default_mid: bool = False,
    suppress_name_clauses: bool = False,
) -> str:
    """Build the canonical figure tag.

    ``assignments`` may be either a list (tool name pulled from each
    ``Assignment.source['tool']``) or a ``{tool: Assignment}`` mapping.
    The output is sorted by tool name so identical inputs ⇒ identical
    string.

    ``mid_rule`` overrides the default MID flag rendering:

    - ``None`` (default): MID flag is ``MID+`` if target has MID,
      ``MID-`` otherwise.
    - ``"fold_to_eur"``: target is MID-less but the source's MID mass
      was redistributed into EUR before collapse. Tag renders ``MID->eur``.
    - ``"drop"``: equivalent to ``MID-`` (source MID dropped; rendered
      explicitly).

    Optional display flags (both default ``False`` — popout_dx and any
    other report that wants the verbose form is unaffected):

    - ``suppress_default_mid=True``: drop the ``/MID±`` qualifier when
      the target inherently has no MID and ``mid_rule`` would just
      restate that (``None`` on a MID-less target). Explicit
      overrides (``"drop"``, ``"fold_to_eur"``) and MID-bearing
      targets still render the qualifier.
    - ``suppress_name_clauses=True``: drop per-tool ``tool=>name``
      clauses (the silent default in a verbatim-FLARE report). Other
      methods (``postS``, ``confH``, ``manual`` …) still render.
    """
    if isinstance(assignments, dict):
        items = sorted(assignments.items())
    else:
        items = sorted(((a.source.get("tool", "?"), a) for a in assignments),
                       key=lambda kv: kv[0])
    if mid_rule == "fold_to_eur":
        mid_flag: str | None = "MID->eur"
    elif mid_rule == "drop":
        mid_flag = "MID-"
    else:
        mid_flag = "MID+" if target.has_mid else "MID-"
    if (suppress_default_mid and mid_rule is None
            and not target.has_mid):
        mid_flag = None
    target_str = target.tag if mid_flag is None else f"{target.tag}/{mid_flag}"
    if suppress_name_clauses:
        clauses = [f"{tool}=>{a.method}" for tool, a in items
                   if a.method != "name"]
    else:
        clauses = [f"{tool}=>{a.method}" for tool, a in items]
    h = _hash(target, [a for _, a in items], params or {})[:hash_len]
    return " | ".join([f"L={target_str}"] + clauses + [f"v={h}"])


def parse(tag: str) -> dict:
    """Return ``{target, mid, clauses, version}`` from a tag string."""
    parts = [p.strip() for p in tag.split("|")]
    out: dict = {"clauses": []}
    if not parts or not parts[0].startswith("L="):
        raise ValueError(f"tag must start with 'L=<target>/MID±': {tag!r}")
    head = parts[0][2:]
    if "/" in head:
        target, mid = head.split("/", 1)
    else:
        target, mid = head, "MID?"
    out["target"] = target.strip()
    out["mid"] = mid.strip()
    for p in parts[1:]:
        if p.startswith("v="):
            out["version"] = p[2:].strip()
            continue
        m = _CLAUSE_RE.match(p)
        if not m:
            raise ValueError(f"unparseable clause in tag {tag!r}: {p!r}")
        out["clauses"].append({
            "tool": m.group("tool"),
            "method": m.group("method"),
            "extra": m.group("extra"),
        })
    return out


def version_hash(assignment_or_assignments) -> str:
    """Short hex hash of an Assignment (or iterable thereof).

    Stable across Python sessions; identical maps + params ⇒ identical
    hash. Any change to target, per-tool map, or params ⇒ different.
    """
    if isinstance(assignment_or_assignments, Assignment):
        items = [assignment_or_assignments]
    else:
        items = list(assignment_or_assignments)
    target = items[0].target_space if items else None
    return _hash(target, items, {})[:6]


def _hash(target: LabelSpace | None, assignments: list[Assignment], params: dict) -> str:
    payload = {
        "target": target.tag if target else None,
        "members": list(target.members) if target else None,
        "maps": [
            {
                "tool": a.source.get("tool", "?"),
                "method": a.method,
                "component_to_label": {str(k): v for k, v
                                       in sorted(a.component_to_label.items())},
            }
            for a in assignments
        ],
        "params": _canonicalize(params),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _canonicalize(obj):
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    return obj
