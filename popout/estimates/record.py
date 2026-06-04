"""``Estimate`` — one per (tool, run scope) ancestry record.

Carries the per-sample proportion matrix in the tool's *native* label
space; the label_space's members are the column names in column order.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from popout.labelspace.registry import LabelSpace, get, make_native_space


SCHEMA_TAG = "estimate/1"


@dataclasses.dataclass(frozen=True)
class Estimate:
    """Per-tool ancestry estimate.

    Attributes
    ----------
    tool
        Canonical tool name (``"flare"``, ``"popout"``, ``"rye"``,
        ``"rf"``, …).
    scope
        Tuple of identifiers that pin the scope of the estimate, e.g.
        ``("cluster_000", "chr1")`` for a per-cluster-per-chrom record
        or ``("cohort",)`` for a whole-cohort record.
    sample_ids
        Per-row sample identifiers (canonical ordering).
    label_space
        The native label space — ``label_space.members`` are the column
        names in column order. Members are *named*; no
        ``ancestry_0..K-1`` permitted.
    proportions
        ``(n_samples, |label_space|)`` non-negative array; each row
        should sum to ≈ 1, but the loader is not required to enforce
        this (some tools emit unnormalised counts).
    hard_calls
        Optional per-sample categorical call (``(n_samples,)`` array of
        member names). Populated for RF; ``None`` for tools that only
        emit soft proportions.
    provenance
        Free-form dict capturing source paths, header extracts, hashes,
        and any other reproducibility metadata.
    """

    tool: str
    scope: tuple[str, ...]
    sample_ids: tuple[str, ...]
    label_space: LabelSpace
    proportions: np.ndarray
    hard_calls: np.ndarray | None = None
    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        n_rows, n_cols = self.proportions.shape
        if n_rows != len(self.sample_ids):
            raise ValueError(
                f"Estimate({self.tool!r}): proportions has {n_rows} rows "
                f"but {len(self.sample_ids)} sample_ids"
            )
        if n_cols != len(self.label_space.members):
            raise ValueError(
                f"Estimate({self.tool!r}): proportions has {n_cols} columns "
                f"but label_space {self.label_space.tag} has "
                f"{len(self.label_space.members)} members"
            )
        if any(m.startswith("ancestry_") and m[len("ancestry_"):].isdigit()
               for m in self.label_space.members):
            raise ValueError(
                f"Estimate({self.tool!r}): label_space.members must be named "
                f"(got {list(self.label_space.members)}). "
                f"Use the loader's panel_names argument to rename."
            )
        if self.hard_calls is not None:
            if self.hard_calls.shape != (n_rows,):
                raise ValueError(
                    f"Estimate({self.tool!r}): hard_calls shape "
                    f"{self.hard_calls.shape} != ({n_rows},)"
                )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return self.proportions.shape[0]

    @property
    def members(self) -> tuple[str, ...]:
        return self.label_space.members

    def column(self, label: str) -> np.ndarray:
        """Return the proportion column for ``label`` (raises if absent)."""
        return self.proportions[:, self.label_space.index(label)]

    # ── Equivalence ────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Estimate):
            return NotImplemented
        if (self.tool, self.scope, self.sample_ids) != (
            other.tool, other.scope, other.sample_ids
        ):
            return False
        if self.label_space.tag != other.label_space.tag:
            return False
        if self.label_space.members != other.label_space.members:
            return False
        if not np.array_equal(self.proportions, other.proportions):
            return False
        if (self.hard_calls is None) != (other.hard_calls is None):
            return False
        if self.hard_calls is not None and not np.array_equal(
            self.hard_calls, other.hard_calls
        ):
            return False
        return self.provenance == other.provenance

    def __hash__(self) -> int:
        return hash((self.tool, self.scope, self.sample_ids,
                     self.label_space.tag))

    # ── (de)serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": SCHEMA_TAG,
            "tool": self.tool,
            "scope": list(self.scope),
            "sample_ids": list(self.sample_ids),
            "label_space": {
                "tag": self.label_space.tag,
                "members": list(self.label_space.members),
            },
            "proportions": self.proportions.tolist(),
            "hard_calls": (
                self.hard_calls.tolist() if self.hard_calls is not None else None
            ),
            "provenance": self.provenance,
        }
        return d

    def dump(self, path: str | Path) -> None:
        """Write the Estimate as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Estimate":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Estimate":
        schema = d.get("schema")
        if schema != SCHEMA_TAG:
            raise ValueError(
                f"Estimate.from_dict: unsupported schema {schema!r}; "
                f"expected {SCHEMA_TAG!r}"
            )
        ls_d = d["label_space"]
        tag = ls_d["tag"]
        members = tuple(ls_d["members"])
        if tag in ("SP6", "SP5"):
            label_space = get(tag)
            if label_space.members != members:
                raise ValueError(
                    f"Estimate.from_dict: serialized members {members} "
                    f"disagree with canonical {tag} {label_space.members}"
                )
        else:
            label_space = make_native_space(tag.split(".", 1)[0], members) \
                if "." in tag else make_native_space(tag, members)
            # Override the constructed tag if it didn't match (e.g. ".native"
            # suffix vs bare tool name).
            if label_space.tag != tag:
                label_space = dataclasses.replace(label_space, tag=tag)
        hard = d.get("hard_calls")
        hard_arr = np.asarray(hard, dtype=object) if hard is not None else None
        return cls(
            tool=d["tool"],
            scope=tuple(d["scope"]),
            sample_ids=tuple(d["sample_ids"]),
            label_space=label_space,
            proportions=np.asarray(d["proportions"], dtype=np.float64),
            hard_calls=hard_arr,
            provenance=dict(d.get("provenance", {})),
        )

    # ── Named-column TSV (human-inspectable proportions) ──────────────

    def to_named_tsv(self, path: str | Path) -> None:
        """Emit ``sample_id<TAB>label1<TAB>label2<TAB>...`` (no provenance).

        Companion to :meth:`dump`. The TSV is what downstream consumers
        read; the JSON carries provenance + label-space metadata.
        """
        header = "sample_id\t" + "\t".join(self.label_space.members) + "\n"
        with open(path, "w") as f:
            f.write(header)
            for sid, row in zip(self.sample_ids, self.proportions):
                f.write(sid + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")

    @classmethod
    def from_named_tsv(
        cls,
        path: str | Path,
        *,
        tool: str,
        scope: tuple[str, ...],
        label_space_tag: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> "Estimate":
        """Read a ``sample_id<TAB>label1<TAB>...`` TSV back into an Estimate.

        ``label_space_tag`` is optional — defaults to a constructed
        native space named after ``tool``.
        """
        lines = Path(path).read_text().rstrip("\n").splitlines()
        if not lines:
            raise ValueError(f"{path}: empty")
        header = lines[0].split("\t")
        if header[0] != "sample_id":
            raise ValueError(
                f"{path}: first column must be 'sample_id', got {header[0]!r}"
            )
        members = tuple(header[1:])
        if not members:
            raise ValueError(f"{path}: no label columns")
        if any(m.startswith("ancestry_") and m[len("ancestry_"):].isdigit()
               for m in members):
            raise ValueError(
                f"{path}: refusing to load anonymous columns {members}. "
                f"The TSV must carry named labels; rename upstream."
            )
        if label_space_tag is None or label_space_tag == f"{tool}.native":
            label_space = make_native_space(tool, members)
        elif label_space_tag in ("SP6", "SP5"):
            label_space = get(label_space_tag)
            if label_space.members != members:
                raise ValueError(
                    f"{path}: header {members} != {label_space_tag} "
                    f"members {label_space.members}"
                )
        else:
            label_space = make_native_space(
                label_space_tag.split(".", 1)[0], members
            )
            if label_space.tag != label_space_tag:
                label_space = dataclasses.replace(
                    label_space, tag=label_space_tag,
                )
        sample_ids: list[str] = []
        rows: list[list[float]] = []
        for n, line in enumerate(lines[1:], start=2):
            parts = line.split("\t")
            if len(parts) != len(header):
                raise ValueError(
                    f"{path}:{n}: expected {len(header)} cols, got {len(parts)}"
                )
            sample_ids.append(parts[0])
            try:
                rows.append([float(v) for v in parts[1:]])
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}")
        return cls(
            tool=tool,
            scope=tuple(scope),
            sample_ids=tuple(sample_ids),
            label_space=label_space,
            proportions=np.asarray(rows, dtype=np.float64),
            hard_calls=None,
            provenance=dict(provenance or {}),
        )
