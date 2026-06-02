"""Phase 4: hardcoded canonical-superpop tuples must live in the registry only.

The audit (LABEL_AUDIT.md §3c) identified eight independent redeclarations
of ``("afr", "amr", "eas", "eur", "mid", "sas")``. After Phase 4 every
producer reads it from ``popout.labelspace.registry.SP6.members``. This
test sweeps the source tree to make sure no new copy sneaks in.

Allow-list: the registry itself, the LABEL_* design docs, and a handful
of small/ad-hoc inline uses (test fixtures, in-function tuples inside
``run_cluster_validation`` — out of scope for Phase 4, slated for the
delete-duplicates pass).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TARGET_DIRS = ("popout", "validation", "diagnostics")

PATTERNS = (
    re.compile(r'"afr"\s*,\s*"amr"\s*,\s*"eas"\s*,\s*"eur"\s*,\s*"mid"\s*,\s*"sas"'),
)

# Sites we deliberately leave un-routed (test fixtures, doc strings,
# in-function tuples that are clearly local SP5 uses, third-party tests).
ALLOWLIST = {
    "popout/labelspace/registry.py",        # the *one* declaration
    "validation/popout_dx/tests/test_e2e_fixture.py",  # synthetic fixture
    "tests/labelspace/test_constants_pruned.py",
    "tests/labelspace/conftest.py",          # synthetic SP6 for goldens
    "popout/labelspace/assignment.py",       # default tuple in v1 upgrade
}


def test_no_redundant_sp6_tuple():
    bad: list[tuple[str, int, str]] = []
    for d in TARGET_DIRS:
        for path in (REPO / d).rglob("*.py"):
            rel = path.relative_to(REPO).as_posix()
            if rel in ALLOWLIST:
                continue
            if "/.claude/" in rel or "/__pycache__/" in rel or "/build/" in rel:
                continue
            text = path.read_text()
            for n, line in enumerate(text.splitlines(), start=1):
                for pat in PATTERNS:
                    if pat.search(line):
                        bad.append((rel, n, line.strip()))
    assert not bad, (
        "Phase 4 retrofit incomplete — these files redeclare the SP6 tuple "
        "instead of importing it from popout.labelspace.registry:\n  "
        + "\n  ".join(f"{f}:{n}: {ln}" for f, n, ln in bad)
    )
