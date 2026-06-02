"""Unified label-space module.

One module owns the four (five) ancestry label spaces in play across
popout, FLARE, the RF classifier, RYE, and the simulator. Every other
site that previously redeclared a canonical superpop tuple, ran its own
matching algorithm, or carried its own projection logic now imports from
here.

See ``my_notes/labels/LABEL_SPACE.md`` for the design document and
``my_notes/labels/LABEL_SPACE_RETROFIT.md`` for the migration plan.

Public API
----------
The submodules export the operational primitives:

``registry``      named label spaces (``SP6``, ``SP5``, ``SP6.sub``, ``TRUTH``).
``assignment``    the ``Assignment`` dataclass + ``labels.json`` v1↔v2 serdes.
``shorthand``     the figure-tag grammar (``L=SP6/MID+ | popout=>corrH | v=…``).
``matching``      named matching strategies (Phase 2).
``project``       the single proportions + tract-code projector (Phase 3).
``naming``        the stable correlation-rank subcomponent namer (Phase 3).
``metrics``       ARI / NMI / V-measure for honest-floor accuracy (Phase 5).
"""

from __future__ import annotations

from . import assignment, matching, naming, project, registry, shorthand  # noqa: F401
from .assignment import Assignment
from .matching import (
    by_name,
    confusion_hungarian,
    corr_hungarian,
    manual,
    posterior_slope,
)
from .naming import name_components, ordered_subcomponent_names
from .project import collapse, project_proportions, project_tracts
from .registry import LabelSpace, get

__all__ = [
    "Assignment", "LabelSpace",
    "assignment", "matching", "naming", "project", "registry", "shorthand",
    "by_name", "confusion_hungarian", "corr_hungarian", "manual",
    "posterior_slope", "get",
    "name_components", "ordered_subcomponent_names",
    "collapse", "project_proportions", "project_tracts",
]
