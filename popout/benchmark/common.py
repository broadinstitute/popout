"""Common data structures for the LAI benchmark tool."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MISSING_LABEL: int = 65535


@dataclass
class TractSet:
    """Per-haplotype local ancestry calls for one chromosome.

    Stored as per-site dense arrays. Tract representation can be derived
    via run-length encoding when needed.
    """

    tool_name: str
    chrom: str
    hap_ids: np.ndarray  # (H,) string
    site_positions: np.ndarray  # (T,) int64 — bp positions
    calls: np.ndarray  # (H, T) uint16 — integer ancestry labels
    label_map: dict[int, str]  # int code -> population name
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_haps(self) -> int:
        return self.calls.shape[0]

    @property
    def n_sites(self) -> int:
        return self.calls.shape[1]

    def to_tracts(self) -> list[tuple[int, int, int, int]]:
        """Run-length encode calls into tracts.

        Returns list of (hap_idx, start_site_idx, end_site_idx, label) tuples.
        end_site_idx is exclusive.
        """
        tracts = []
        for h in range(self.n_haps):
            row = self.calls[h]
            if len(row) == 0:
                continue
            start = 0
            current_label = row[0]
            for t in range(1, len(row)):
                if row[t] != current_label:
                    tracts.append((h, start, t, int(current_label)))
                    start = t
                    current_label = row[t]
            tracts.append((h, start, len(row), int(current_label)))
        return tracts

    def global_fractions(self) -> np.ndarray:
        """Per-haplotype ancestry fractions. Returns (H, K) array."""
        labels = sorted(k for k in self.label_map if k != MISSING_LABEL)
        K = len(labels)
        fracs = np.zeros((self.n_haps, K), dtype=np.float64)
        for idx, k in enumerate(labels):
            fracs[:, idx] = (self.calls == k).mean(axis=1)
        return fracs

    def validate(self) -> None:
        """Check internal consistency. Raises ValueError on problems."""
        if self.hap_ids.shape[0] != self.calls.shape[0]:
            raise ValueError(
                f"hap_ids length {self.hap_ids.shape[0]} != "
                f"calls rows {self.calls.shape[0]}"
            )
        if self.site_positions.shape[0] != self.calls.shape[1]:
            raise ValueError(
                f"site_positions length {self.site_positions.shape[0]} != "
                f"calls columns {self.calls.shape[1]}"
            )
        if len(self.site_positions) > 1:
            diffs = np.diff(self.site_positions)
            if np.any(diffs <= 0):
                raise ValueError("site_positions must be strictly monotonically increasing")
        valid_labels = set(self.label_map.keys()) | {MISSING_LABEL}
        unique_calls = set(np.unique(self.calls).tolist())
        invalid = unique_calls - valid_labels
        if invalid:
            raise ValueError(f"calls contain labels not in label_map: {invalid}")


def load_ancestry_header(line: str) -> dict[int, str]:
    """Parse an ANCESTRY header line into {int: name} dict.

    Handles formats like:
        ##ANCESTRY=<eas=0,amr=1,eur=2,afr=3,sas=4>
    """
    match = re.search(r"<(.+)>", line)
    if not match:
        raise ValueError(f"Cannot parse ancestry header: {line!r}")
    label_map = {}
    for pair in match.group(1).split(","):
        name, code = pair.strip().split("=")
        label_map[int(code)] = name.strip()
    return label_map
