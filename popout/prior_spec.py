"""Prior specification — schema-v2 YAML loader for emergent-identity priors.

A prior carries:
  * an **identity** (one or more :class:`popout.identity.IdentitySignature`
    instances) describing what a matching component should look like;
  * a **parameter claim** on generations since admixture (Beta(α, β) on
    the per-step transition probability ``r = 1 - exp(-T * mps)``,
    matched at load time to the documented [gen_lo, gen_hi] band).

The container also carries the annealing schedule for soft assignment
and a content-aware fingerprint covering the YAML *and* every
referenced data file (AIM panel TSVs, 1KG superpop-frequency TSV bytes
for the superpops requested by the priors). Changing any of these
invalidates cached EM/decode stages — the previous v1 fingerprint
covered only the YAML and silently accepted stale panel content.

YAML schema (v2)::

    schema_version: 2
    morgans_per_step: 0.0001

    priors:
      - name: AFR
        identity:
          aims:
            panel: aim_panels/african.tsv     # path relative to YAML dir,
                                              # absolute, or 'bundled:<name>'
            weight: 1.0                       # optional, default 1.0
          fst_reference:
            superpop: AFR                     # name in the 1KG superpop-freqs TSV
            weight: 1.0
        parameters:
          gen:
            mean: 7
            range: [4, 12]
        source: "Atlantic slave trade primary phase, Bryc 2015"

    annealing:
      schedule: linear
      tau_start: 1.0
      tau_end: 0.1
      ramp_iters: 10

Schema v1 (the previous index-based ``component_idx`` format) is
explicitly rejected at load time; see ``docs/PRIORS.md`` for the
migration path.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.stats import beta as scipy_beta

from .fetch_superpop_freqs import resolve_superpop_freqs_path
from .identity import (
    AIMPanel,
    AIMSignature,
    FSTReferenceSignature,
    IdentitySignature,
)


# ---------------------------------------------------------------------------
# Beta solver (salvaged from the deleted v1 popout/priors.py)
# ---------------------------------------------------------------------------


def _r_of_T(T: float, morgans_per_step: float) -> float:
    return 1.0 - math.exp(-T * morgans_per_step)


def prior_to_beta(
    gen_mean: float,
    gen_lo: float,
    gen_hi: float,
    morgans_per_step: float,
) -> tuple[float, float]:
    """Solve Beta(α, β) so that the 5th/95th percentiles match
    ``r(gen_lo)`` / ``r(gen_hi)``.
    """
    if morgans_per_step <= 0:
        raise ValueError(
            f"morgans_per_step must be positive, got {morgans_per_step}"
        )

    r_lo = _r_of_T(gen_lo, morgans_per_step)
    r_hi = _r_of_T(gen_hi, morgans_per_step)
    r_mean = _r_of_T(gen_mean, morgans_per_step)

    width = max(r_hi - r_lo, 1e-9)
    concentration = max(min(1.0 / width, 1e6), 4.0)
    alpha0 = max(r_mean * concentration, 1.01)
    beta0 = max((1.0 - r_mean) * concentration, 1.01)

    def residual(log_ab):
        a, b = math.exp(log_ab[0]), math.exp(log_ab[1])
        p05 = scipy_beta.ppf(0.05, a, b)
        p95 = scipy_beta.ppf(0.95, a, b)
        return [p05 - r_lo, p95 - r_hi]

    sol = least_squares(
        residual,
        x0=[math.log(alpha0), math.log(beta0)],
        method="lm",
        xtol=1e-10,
        ftol=1e-10,
    )
    return math.exp(sol.x[0]), math.exp(sol.x[1])


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearAnnealingSchedule:
    """Linear ramp from ``tau_start`` to ``tau_end`` over ``ramp_iters``,
    constant at ``tau_end`` thereafter.
    """

    tau_start: float = 1.0
    tau_end: float = 0.1
    ramp_iters: int = 10

    def __post_init__(self) -> None:
        if self.tau_start <= 0 or self.tau_end <= 0:
            raise ValueError(
                f"tau_start and tau_end must be > 0; "
                f"got {self.tau_start}, {self.tau_end}"
            )
        if self.ramp_iters <= 0:
            raise ValueError(
                f"ramp_iters must be > 0; got {self.ramp_iters}"
            )

    def tau_at(self, iteration: int) -> float:
        if iteration <= 0:
            return float(self.tau_start)
        if iteration >= self.ramp_iters:
            return float(self.tau_end)
        frac = iteration / self.ramp_iters
        return float(self.tau_start + frac * (self.tau_end - self.tau_start))


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    """One prior: identity signatures + a parameter claim on T."""

    name: str
    identity_signatures: tuple[IdentitySignature, ...]
    gen_mean: float
    gen_lo: float
    gen_hi: float
    alpha: float
    beta: float
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("prior name must be non-empty")
        if not self.identity_signatures:
            raise ValueError(
                f"prior {self.name!r} has no identity signatures; "
                f"a prior with only parameter claims and no identity is "
                f"meaningless. Add at least one of: aims, fst_reference"
            )
        if not (0 < self.gen_lo < self.gen_mean < self.gen_hi):
            raise ValueError(
                f"prior {self.name!r}: need 0 < gen_lo < gen_mean < gen_hi, "
                f"got ({self.gen_lo}, {self.gen_mean}, {self.gen_hi})"
            )
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"prior {self.name!r}: Beta parameters must be positive, "
                f"got alpha={self.alpha}, beta={self.beta}"
            )


@dataclass(frozen=True)
class Priors:
    """Top-level priors container loaded from a v2 YAML file."""

    priors: tuple[Prior, ...]
    morgans_per_step: float
    annealing: LinearAnnealingSchedule
    fingerprint: str             # sha256 hex of YAML + every referenced data file
    source_path: str

    _by_name: dict[str, Prior] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        seen: dict[str, Prior] = {}
        for p in self.priors:
            if p.name in seen:
                raise ValueError(f"duplicate prior name: {p.name!r}")
            seen[p.name] = p
        object.__setattr__(self, "_by_name", seen)

    def get(self, name: str) -> Optional[Prior]:
        return self._by_name.get(name)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


_BUNDLED_AIM_DIR = Path(__file__).parent / "data" / "aim_panels"


def _resolve_panel_path(panel: str, yaml_dir: Path) -> Path:
    """Resolve an AIM panel path.

    Accepts:
      * absolute path
      * ``bundled:<name>`` → ``popout/data/aim_panels/<name>``
      * any other string → relative to the YAML's parent directory
    """
    if panel.startswith("bundled:"):
        name = panel[len("bundled:"):]
        return _BUNDLED_AIM_DIR / name
    p = Path(panel)
    if p.is_absolute():
        return p
    return (yaml_dir / panel).resolve()


# ---------------------------------------------------------------------------
# 1KG superpop-frequencies loading (multi-chrom for FSTReferenceSignature)
# ---------------------------------------------------------------------------


def _load_superpop_freqs(
    superpop_freqs_path: Path,
    superpop: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the 1KG TSV and return per-site (chrom, pos_bp, freq) arrays
    for one superpopulation.
    """
    import csv
    import gzip

    chroms: list[str] = []
    positions: list[int] = []
    freqs: list[float] = []
    pop_names: list[str] | None = None

    opener = gzip.open if superpop_freqs_path.suffix == ".gz" else open
    with opener(superpop_freqs_path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                pop_names = row[4:]
                continue
            if pop_names is None:
                raise ValueError(
                    f"superpop frequencies TSV {superpop_freqs_path} is missing "
                    f"a header row (expected: '#chrom  pos  ref  alt  POP1  POP2 ...')"
                )
            if superpop not in pop_names:
                raise ValueError(
                    f"superpop {superpop!r} not in superpop frequencies TSV "
                    f"{superpop_freqs_path}; available: {pop_names}"
                )
            col = pop_names.index(superpop) + 4  # +4 for the chrom/pos/ref/alt prefix
            chroms.append(str(row[0]))
            positions.append(int(row[1]))
            freqs.append(float(row[col]))

    if not chroms:
        raise ValueError(f"no sites loaded from {superpop_freqs_path}")
    return (
        np.array(chroms, dtype=object),
        np.array(positions, dtype=np.int64),
        np.array(freqs, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_SCHEMA_V1_MIGRATION = (
    "Schema v1 (component_idx-based priors) is no longer supported. "
    "The new schema is keyed by structural identity (AIM panels + 1KG "
    "F_ST reference) rather than recursive-seeding leaf indices. "
    "See docs/PRIORS.md for the migration path."
)


def load_priors(
    path: str | Path,
    *,
    superpop_freqs: str | Path | None = None,
) -> Priors:
    """Load and validate a v2 priors YAML; build identity signatures and
    Beta parameters; return a :class:`Priors` container with a content-
    aware fingerprint.

    Parameters
    ----------
    path : YAML file path.
    superpop_freqs : optional explicit path to the 1KG superpop allele-
        frequency TSV. When given, every ``fst_reference: superpop: ...``
        block in the YAML resolves through this file instead of the
        cached ``~/.popout/superpop_freqs/GRCh38/1kg_superpop_freq.tsv.gz``.
        Used by the WDL/Terra pipeline where the cache is not pre-
        populated; users can localize the TSV via Terra and pass its
        container path here.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"priors file not found: {p}")

    yaml_bytes = p.read_bytes()
    raw = yaml.safe_load(yaml_bytes)
    if not isinstance(raw, dict):
        raise ValueError(f"priors file {p} must be a YAML mapping at top level")

    schema_version = raw.get("schema_version")
    if schema_version != 2:
        if schema_version is None or schema_version == 1:
            raise ValueError(_SCHEMA_V1_MIGRATION)
        raise ValueError(
            f"priors file {p}: unknown schema_version {schema_version!r}; "
            f"only schema_version: 2 is supported"
        )

    morgans_per_step = raw.get("morgans_per_step")
    if morgans_per_step is None:
        raise ValueError(
            f"priors file {p}: missing required 'morgans_per_step'"
        )
    morgans_per_step = float(morgans_per_step)
    if morgans_per_step <= 0:
        raise ValueError(
            f"morgans_per_step must be positive, got {morgans_per_step}"
        )

    priors_raw = raw.get("priors")
    if not isinstance(priors_raw, list) or not priors_raw:
        raise ValueError(
            f"priors file {p}: 'priors' must be a non-empty list"
        )

    annealing_raw = raw.get("annealing", {}) or {}
    schedule = annealing_raw.get("schedule", "linear")
    if schedule != "linear":
        raise ValueError(
            f"priors file {p}: unknown annealing schedule {schedule!r}; "
            f"only 'linear' is supported"
        )
    annealing = LinearAnnealingSchedule(
        tau_start=float(annealing_raw.get("tau_start", 1.0)),
        tau_end=float(annealing_raw.get("tau_end", 0.1)),
        ramp_iters=int(annealing_raw.get("ramp_iters", 10)),
    )

    yaml_dir = p.parent.resolve()

    # Collect referenced files for fingerprinting.
    referenced_aim_paths: list[Path] = []
    referenced_superpop_freqs_paths: list[Path] = []

    priors_list: list[Prior] = []
    for entry in priors_raw:
        if not isinstance(entry, dict):
            raise ValueError(
                f"priors file {p}: each prior must be a mapping, got {type(entry)}"
            )
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(
                f"priors file {p}: each prior requires a 'name' string"
            )

        identity_raw = entry.get("identity") or {}
        if not isinstance(identity_raw, dict):
            raise ValueError(
                f"prior {name!r}: 'identity' must be a mapping"
            )

        signatures: list[IdentitySignature] = []

        if "aims" in identity_raw:
            aims_block = identity_raw["aims"]
            if not isinstance(aims_block, dict):
                raise ValueError(
                    f"prior {name!r}: 'identity.aims' must be a mapping"
                )
            panel_str = aims_block.get("panel")
            if not panel_str:
                raise ValueError(
                    f"prior {name!r}: 'identity.aims.panel' is required"
                )
            panel_path = _resolve_panel_path(str(panel_str), yaml_dir)
            if not panel_path.exists():
                raise FileNotFoundError(
                    f"prior {name!r}: AIM panel not found: {panel_path}"
                )
            referenced_aim_paths.append(panel_path)
            panel = AIMPanel.from_tsv(panel_path)
            signatures.append(
                AIMSignature(
                    panel=panel,
                    weight=float(aims_block.get("weight", 1.0)),
                )
            )

        if "fst_reference" in identity_raw:
            fst_block = identity_raw["fst_reference"]
            if not isinstance(fst_block, dict):
                raise ValueError(
                    f"prior {name!r}: 'identity.fst_reference' must be a mapping"
                )
            superpop = fst_block.get("superpop")
            if not superpop:
                raise ValueError(
                    f"prior {name!r}: 'identity.fst_reference.superpop' "
                    f"is required (e.g. 'AFR', 'EUR')"
                )
            superpop_freqs_arg = fst_block.get("superpop_freqs_path")
            superpop_freqs_path = (
                _resolve_panel_path(str(superpop_freqs_arg), yaml_dir)
                if superpop_freqs_arg
                else resolve_superpop_freqs_path("GRCh38", arg=superpop_freqs)
            )
            if not superpop_freqs_path.exists():
                raise FileNotFoundError(
                    f"prior {name!r}: 1KG superpop frequencies TSV not found: "
                    f"{superpop_freqs_path}. Provide one via 'superpop_freqs_path' "
                    f"or run 'popout fetch-superpop-freqs' to populate the cache."
                )
            referenced_superpop_freqs_paths.append(superpop_freqs_path)
            chrom_arr, pos_arr, freq_arr = _load_superpop_freqs(
                superpop_freqs_path, str(superpop),
            )
            signatures.append(
                FSTReferenceSignature(
                    ref_freq=freq_arr,
                    ref_pos_bp=pos_arr,
                    ref_chrom=chrom_arr,
                    ref_name=f"1KG_{superpop}",
                    weight=float(fst_block.get("weight", 1.0)),
                )
            )

        if not signatures:
            raise ValueError(
                f"prior {name!r}: 'identity' must contain at least one of "
                f"'aims', 'fst_reference'"
            )

        params = entry.get("parameters") or {}
        if not isinstance(params, dict) or "gen" not in params:
            raise ValueError(
                f"prior {name!r}: 'parameters.gen' is required"
            )
        gen = params["gen"]
        if not isinstance(gen, dict):
            raise ValueError(
                f"prior {name!r}: 'parameters.gen' must be a mapping"
            )
        gen_mean = float(gen["mean"])
        gen_range = gen.get("range")
        if not (isinstance(gen_range, list) and len(gen_range) == 2):
            raise ValueError(
                f"prior {name!r}: 'parameters.gen.range' must be a "
                f"two-element list [lo, hi]"
            )
        gen_lo, gen_hi = float(gen_range[0]), float(gen_range[1])
        alpha, beta = prior_to_beta(gen_mean, gen_lo, gen_hi, morgans_per_step)

        priors_list.append(
            Prior(
                name=name,
                identity_signatures=tuple(signatures),
                gen_mean=gen_mean,
                gen_lo=gen_lo,
                gen_hi=gen_hi,
                alpha=alpha,
                beta=beta,
                source=str(entry.get("source", "")),
            )
        )

    fingerprint = _compute_fingerprint(
        yaml_bytes=yaml_bytes,
        aim_paths=sorted(set(referenced_aim_paths)),
        superpop_freqs_paths=sorted(set(referenced_superpop_freqs_paths)),
    )

    return Priors(
        priors=tuple(priors_list),
        morgans_per_step=morgans_per_step,
        annealing=annealing,
        fingerprint=fingerprint,
        source_path=str(p.resolve()),
    )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _compute_fingerprint(
    yaml_bytes: bytes,
    aim_paths: list[Path],
    superpop_freqs_paths: list[Path],
) -> str:
    """SHA-256 over the YAML bytes plus every referenced data file's SHA-256.

    Order is deterministic (sorted paths in the loader). Path strings are
    not hashed — only file *contents* — so renaming/moving a panel
    without changing its content does not invalidate the fingerprint.
    """
    h = hashlib.sha256()
    h.update(b"yaml:")
    h.update(yaml_bytes)
    for ap in aim_paths:
        h.update(b"aim:")
        h.update(hashlib.sha256(ap.read_bytes()).digest())
    for sp in superpop_freqs_paths:
        h.update(b"superpop_freqs:")
        h.update(hashlib.sha256(sp.read_bytes()).digest())
    return h.hexdigest()
