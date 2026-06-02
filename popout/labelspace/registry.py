"""Named label spaces — the registry that replaces the >=8 hardcoded copies.

A ``LabelSpace`` is a finite, ordered set of named categories together
with an optional parent map to a coarser space. Members are accessed by
name; ordering is fixed at construction. See
``my_notes/labels/LABEL_SPACE.md`` §2 for the design.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping


@dataclasses.dataclass(frozen=True)
class LabelSpace:
    """A named, ordered set of ancestry labels.

    ``tag`` is the canonical short name used in the registry and the
    figure-tag shorthand. ``members`` is the ordered tuple of label
    names. ``parent`` and ``parent_of`` express coarsening relations:
    ``SP6.sub.parent_of["afr.1"] == "afr"`` and ``SP6.sub.parent is SP6``.
    """

    tag: str
    members: tuple[str, ...]
    parent: "LabelSpace | None" = None
    parent_of: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError(f"LabelSpace {self.tag!r} requires at least one member")
        if len(set(self.members)) != len(self.members):
            raise ValueError(
                f"LabelSpace {self.tag!r} has duplicate members: {self.members}"
            )
        if self.parent_of is not None:
            unknown_parents = {v for v in self.parent_of.values()
                               if self.parent is None or v not in self.parent.members}
            if unknown_parents:
                raise ValueError(
                    f"LabelSpace {self.tag!r}: parent_of references unknown parent "
                    f"labels {sorted(unknown_parents)}"
                )

    def index(self, label: str) -> int:
        """Return the ordinal of ``label``. Raises ``KeyError`` if absent."""
        try:
            return self.members.index(label)
        except ValueError:
            raise KeyError(
                f"{label!r} is not a member of {self.tag} (members={list(self.members)})"
            )

    def __contains__(self, label: str) -> bool:
        return label in self.members

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    # ── MID convenience ─────────────────────────────────────────────────

    @property
    def has_mid(self) -> bool:
        return "mid" in self.members

    def with_mid(self) -> "LabelSpace":
        """Return the MID-having sibling of this space (raises if not one of SP5/SP6)."""
        if self.tag == "SP6":
            return self
        if self.tag == "SP5":
            return SP6
        raise ValueError(f"{self.tag} has no canonical with-MID sibling")

    def without_mid(self) -> "LabelSpace":
        """Return the MID-less sibling (raises if not one of SP5/SP6)."""
        if self.tag == "SP5":
            return self
        if self.tag == "SP6":
            return SP5
        raise ValueError(f"{self.tag} has no canonical without-MID sibling")


# ── Canonical spaces ─────────────────────────────────────────────────────


SP6 = LabelSpace(tag="SP6", members=("afr", "amr", "eas", "eur", "mid", "sas"))
"""Six-way continental superpop. The shared target most figures live in."""

SP5 = LabelSpace(tag="SP5", members=("afr", "amr", "eas", "eur", "sas"))
"""SP6 with MID removed. RYE's native target."""


def make_sub_space(parent: LabelSpace, members_per_parent: Mapping[str, int]) -> LabelSpace:
    """Build a subcontinental refinement of ``parent``.

    ``members_per_parent`` maps each parent label to the number of
    subcomponents it received in a particular run. Singletons keep their
    parent name unchanged; splits add 1-based dense suffixes
    (``afr.1, afr.2``). The resulting space is a deterministic function
    of the per-parent counts, with the parent map populated.
    """
    if not all(v >= 0 for v in members_per_parent.values()):
        raise ValueError(f"member counts must be non-negative: {members_per_parent}")
    if any(p not in parent.members for p in members_per_parent):
        unknown = [p for p in members_per_parent if p not in parent.members]
        raise ValueError(f"parents {unknown} not in {parent.tag}")

    members: list[str] = []
    parent_of: dict[str, str] = {}
    for p in parent.members:
        n = members_per_parent.get(p, 0)
        if n == 0:
            continue
        if n == 1:
            members.append(p)
            parent_of[p] = p
        else:
            for i in range(1, n + 1):
                name = f"{p}.{i}"
                members.append(name)
                parent_of[name] = p
    return LabelSpace(
        tag=f"{parent.tag}.sub",
        members=tuple(members),
        parent=parent,
        parent_of=parent_of,
    )


# ── Truth / tool-native registry ────────────────────────────────────────


def make_truth_space(k: int) -> LabelSpace:
    """The simulator's ``anc_0..anc_{k-1}`` synthetic classes."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return LabelSpace(tag="TRUTH", members=tuple(f"anc_{i}" for i in range(k)))


def make_native_space(tool: str, members: tuple[str, ...]) -> LabelSpace:
    """A tool-native label space (``<tool>.native``)."""
    return LabelSpace(tag=f"{tool}.native", members=tuple(members))


# ── Lookup ───────────────────────────────────────────────────────────────


_CANONICAL: dict[str, LabelSpace] = {sp.tag: sp for sp in (SP6, SP5)}


def get(tag: str) -> LabelSpace:
    """Resolve a canonical label-space tag (``SP6`` / ``SP5``).

    Raises ``KeyError`` for unknown tags. Run-specific spaces
    (``SP6.sub``, ``TRUTH``, ``<tool>.native``) are constructed via
    ``make_sub_space``, ``make_truth_space``, or ``make_native_space``.
    """
    try:
        return _CANONICAL[tag]
    except KeyError:
        raise KeyError(
            f"no canonical label space tagged {tag!r}; "
            f"known tags: {sorted(_CANONICAL)}"
        )
