"""Unified checkpoint system for popout.

Manages a work directory (``{out}.work/``) containing stage artifacts
and a ``manifest.json`` that tracks which stages are complete.  Resume
is automatic: if the work dir exists and its manifest matches the
current invocation, the pipeline picks up where it left off.

Stages
------
seed   → seed.npz              (leaf_labels + initial model)
em     → em.npz                (converged model)
decode → decode/chr{N}.parquet  (per-chromosome dense decode)
tracts → (final output file)   (tracts TSV, lives outside work dir)

Each stage declares which fingerprint/args keys it depends on.  A
mismatch on resume invalidates that stage and everything after it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_ORDER = ("seed", "em", "decode", "tracts")

# Per-stage dependency declarations.
# "fingerprint" keys come from the input data identity.
# "args" keys come from user-facing algorithm parameters.
# A mismatch in any declared dep invalidates that stage + all later stages.
STAGE_DEPS: dict[str, dict[str, list[str]]] = {
    "seed": {
        "fingerprint": [
            "pgen_sha_prefix", "n_haps", "thin_cm",
            "seeding_exclusion_sha_prefix",
        ],
        "args": [
            "seed_method", "n_ancestries", "max_ancestries",
            "ancestry_detection", "recursive_kwargs_hash", "seed",
        ],
    },
    "em": {
        "fingerprint": [],  # transitively covered by seed
        "args": [
            "gen_since_admix", "n_em_iter", "block_emissions", "block_size",
            "freeze_anchors_iters", "per_hap_T", "n_T_buckets",
        ],
    },
    "decode": {
        "fingerprint": [],
        # bucketed decode is a different code path from standard decode,
        # and parquet output presence depends on probs
        "args": ["probs", "per_hap_T", "n_T_buckets"],
    },
    "tracts": {
        "fingerprint": [],
        "args": [],
    },
}


def _stage_index(stage: str) -> int:
    """Return the position of *stage* in STAGE_ORDER, or raise ValueError."""
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        raise ValueError(
            f"Unknown stage {stage!r}; valid stages: {STAGE_ORDER}"
        ) from None


# ---------------------------------------------------------------------------
# WorkDir
# ---------------------------------------------------------------------------

class WorkDir:
    """Manages the ``{out}.work/`` directory and ``manifest.json``.

    Parameters
    ----------
    path : Path
        Work directory path (e.g. ``Path("results/aou_v9.work")``).
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._manifest: dict[str, Any] | None = None  # lazily loaded

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    @property
    def _manifest_path(self) -> Path:
        return self._path / "manifest.json"

    def _read_manifest(self) -> dict[str, Any] | None:
        """Read manifest from disk. Returns None if missing."""
        if not self._manifest_path.exists():
            return None
        with open(self._manifest_path) as f:
            return json.load(f)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """Atomically write *manifest* to disk (tmpfile + rename)."""
        self._path.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=self._path, suffix=".manifest.tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
                f.write("\n")
            os.rename(tmp, self._manifest_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._manifest = manifest

    def _ensure_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = self._read_manifest()
        if self._manifest is None:
            raise RuntimeError(
                f"No manifest in {self._path}; call open_or_create() first"
            )
        return self._manifest

    # ------------------------------------------------------------------
    # open_or_create
    # ------------------------------------------------------------------

    def open_or_create(
        self,
        *,
        popout_version: str,
        input_fingerprint: dict[str, Any],
        args: dict[str, Any],
        restart_stage: str | None = None,
    ) -> None:
        """Open an existing work dir or create a fresh one.

        If a manifest already exists, validate *input_fingerprint* and
        *args* against it.  The first mismatch invalidates that stage
        and all subsequent stages.

        Parameters
        ----------
        popout_version : str
            Current popout version string (for soft version warning).
        input_fingerprint : dict
            Keys identifying the input data (pgen hash, n_haps, etc.).
        args : dict
            User-facing algorithm parameters.
        restart_stage : str or None
            If set, force-invalidate from this stage forward.
        """
        existing = self._read_manifest()

        if existing is None:
            # Fresh work dir
            manifest = self._make_fresh_manifest(
                popout_version, input_fingerprint, args,
            )
            self._write_manifest(manifest)
            log.info("Created work dir: %s", self._path)
            self._log_status()
            return

        # Existing manifest — validate
        self._manifest = existing

        # Soft version warning
        old_ver = existing.get("popout_version", "unknown")
        if old_ver != popout_version:
            log.warning(
                "Work dir was written by popout %s, current is %s",
                old_ver, popout_version,
            )

        # Check fingerprint + args, invalidate from first mismatch
        invalidate_from: str | None = None
        for stage in STAGE_ORDER:
            deps = STAGE_DEPS[stage]
            for key in deps["fingerprint"]:
                old_val = existing.get("input_fingerprint", {}).get(key)
                new_val = input_fingerprint.get(key)
                if old_val != new_val:
                    log.info(
                        "Fingerprint mismatch on %r: %r -> %r "
                        "(invalidating %s and later)",
                        key, old_val, new_val, stage,
                    )
                    invalidate_from = stage
                    break
            if invalidate_from is not None:
                break
            for key in deps["args"]:
                old_val = existing.get("args", {}).get(key)
                new_val = args.get(key)
                if old_val != new_val:
                    log.info(
                        "Args mismatch on %r: %r -> %r "
                        "(invalidating %s and later)",
                        key, old_val, new_val, stage,
                    )
                    invalidate_from = stage
                    break
            if invalidate_from is not None:
                break

        # Apply invalidation
        if invalidate_from is not None:
            self._invalidate_from(invalidate_from)

        # --restart-stage overrides
        if restart_stage is not None:
            if restart_stage == "all":
                self._invalidate_from("seed")
            else:
                self._invalidate_from(restart_stage)

        # Update manifest with current fingerprint/args/version
        existing["popout_version"] = popout_version
        existing["input_fingerprint"] = input_fingerprint
        existing["args"] = args
        self._write_manifest(existing)
        self._log_status()

    # ------------------------------------------------------------------
    # Stage queries
    # ------------------------------------------------------------------

    def stage_done(self, stage: str, *, chrom: str | None = None) -> bool:
        """Check whether *stage* (optionally for *chrom*) is complete."""
        m = self._ensure_manifest()
        stages = m.get("stages", {})

        if stage == "decode":
            if chrom is None:
                raise ValueError("stage_done('decode') requires chrom=")
            chroms = stages.get("decode", {}).get("chroms", {})
            return chroms.get(str(chrom), {}).get("done", False)

        return stages.get(stage, {}).get("done", False)

    def all_decode_done(self, chroms: list[str]) -> bool:
        """True if decode is done for every chromosome in *chroms*."""
        return all(self.stage_done("decode", chrom=c) for c in chroms)

    def pending_decode_chroms(self, all_chroms: list[str]) -> list[str]:
        """Return chromosomes where decode is not yet done."""
        return [c for c in all_chroms
                if not self.stage_done("decode", chrom=c)]

    def stage_path(self, stage: str, *, chrom: str | None = None) -> Path:
        """Canonical path for a stage artifact."""
        if stage == "seed":
            return self._path / "seed.npz"
        elif stage == "em":
            return self._path / "em.npz"
        elif stage == "decode":
            if chrom is None:
                raise ValueError("stage_path('decode') requires chrom=")
            return self._path / "decode" / f"chr{chrom}.parquet"
        elif stage == "tracts":
            return self._path / "tracts.tsv.gz"
        else:
            raise ValueError(f"Unknown stage: {stage!r}")

    # ------------------------------------------------------------------
    # Stage completion
    # ------------------------------------------------------------------

    def mark_done(
        self,
        stage: str,
        *,
        chrom: str | None = None,
        wall_s: float = 0.0,
    ) -> None:
        """Mark *stage* as complete and update the manifest."""
        m = self._ensure_manifest()
        stages = m.setdefault("stages", {})
        now = datetime.now(timezone.utc).isoformat()

        if stage == "decode":
            if chrom is None:
                raise ValueError("mark_done('decode') requires chrom=")
            decode = stages.setdefault("decode", {"chroms": {}})
            decode["chroms"][str(chrom)] = {
                "done": True,
                "wall_s": round(wall_s, 1),
                "timestamp": now,
            }
        else:
            stages[stage] = {
                "done": True,
                "wall_s": round(wall_s, 1),
                "timestamp": now,
            }

        self._write_manifest(m)

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def _invalidate_from(self, stage: str) -> None:
        """Mark *stage* and all subsequent stages as not done.

        Also removes the stage artifact files from disk.
        """
        idx = _stage_index(stage)
        m = self._ensure_manifest()
        stages = m.setdefault("stages", {})

        for s in STAGE_ORDER[idx:]:
            if s == "decode":
                decode_info = stages.get("decode", {})
                chroms = decode_info.get("chroms", {})
                for c, cinfo in chroms.items():
                    if cinfo.get("done"):
                        # Remove parquet and companion files
                        pq = self._path / "decode" / f"chr{c}.parquet"
                        gs = pq.with_suffix(".global_sums.npy")
                        for p in (pq, gs):
                            if p.exists():
                                p.unlink()
                                log.debug("Removed %s", p)
                stages["decode"] = {"chroms": {}}
            else:
                if stages.get(s, {}).get("done"):
                    # Remove artifact file
                    try:
                        artifact = self.stage_path(s)
                        if artifact.exists():
                            artifact.unlink()
                            log.debug("Removed %s", artifact)
                    except ValueError:
                        pass
                stages[s] = {"done": False}

        self._write_manifest(m)
        log.info("Invalidated stages: %s",
                 ", ".join(STAGE_ORDER[idx:]))

    # ------------------------------------------------------------------
    # Atomic file writing
    # ------------------------------------------------------------------

    def atomic_write_npz(
        self,
        dest: Path,
        save_dict: dict[str, Any],
    ) -> None:
        """Write a compressed .npz atomically (tmpfile + rename).

        Uses a ``.tmp.npz`` suffix so ``np.savez_compressed`` does not
        append an extra ``.npz``.
        """
        import numpy as np

        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp.npz")
        os.close(fd)
        try:
            np.savez_compressed(tmp, **save_dict)
            os.rename(tmp, dest)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def atomic_write_npy(
        self,
        dest: Path,
        array,
    ) -> None:
        """Write a single array as .npy atomically.

        Uses a ``.tmp.npy`` suffix so ``np.save`` does not append an
        extra ``.npy``.
        """
        import numpy as np

        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp.npy")
        os.close(fd)
        try:
            np.save(tmp, array)
            os.rename(tmp, dest)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Fingerprint computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pgen_fingerprint(pgen_path: str | Path) -> str:
        """SHA-256 of the first 1 MiB of each .pgen file.

        For a directory, hashes all .pgen files in sorted order.
        Returns a 16-character hex prefix (64 bits — ample for
        distinguishing distinct inputs).
        """
        path = Path(pgen_path)
        h = hashlib.sha256()

        if path.is_dir():
            pgen_files = sorted(path.glob("*.pgen"))
        elif path.suffix == ".pgen":
            pgen_files = [path]
        else:
            # Treat as a prefix: look for {prefix}*.pgen
            pgen_files = sorted(path.parent.glob(f"{path.name}*.pgen"))

        if not pgen_files:
            log.warning("No .pgen files found at %s", pgen_path)
            return "0" * 16

        for pf in pgen_files:
            with open(pf, "rb") as f:
                h.update(f.read(1024 * 1024))  # first 1 MiB

        return h.hexdigest()[:16]

    @staticmethod
    def compute_vcf_fingerprint(vcf_path: str | Path) -> str:
        """SHA-256 of the first 1 MiB of a VCF/BCF file."""
        path = Path(vcf_path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(1024 * 1024))
        return h.hexdigest()[:16]

    @staticmethod
    def hash_exclusion_file(path: str | Path | None) -> str | None:
        """SHA-256 prefix of the seeding exclusion TSV, or None."""
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        return h.hexdigest()[:16]

    @staticmethod
    def hash_recursive_kwargs(kwargs: dict | None) -> str | None:
        """Deterministic hash of recursive seeding kwargs."""
        if kwargs is None:
            return None
        # Sort keys for determinism, then hash the JSON repr
        canonical = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_fresh_manifest(
        self,
        popout_version: str,
        input_fingerprint: dict,
        args: dict,
    ) -> dict[str, Any]:
        return {
            "popout_version": popout_version,
            "created": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": input_fingerprint,
            "args": args,
            "stages": {
                "seed": {"done": False},
                "em": {"done": False},
                "decode": {"chroms": {}},
                "tracts": {"done": False},
            },
        }

    def _log_status(self) -> None:
        """Log a human-readable summary of work dir status."""
        m = self._ensure_manifest()
        stages = m.get("stages", {})
        lines = [f"Work dir: {self._path}"]

        for s in STAGE_ORDER:
            if s == "decode":
                decode = stages.get("decode", {})
                chroms = decode.get("chroms", {})
                n_done = sum(1 for c in chroms.values() if c.get("done"))
                n_total = len(chroms)
                if n_done == 0 and n_total == 0:
                    lines.append("  decode: pending")
                else:
                    lines.append(f"  decode: {n_done}/{n_total} chroms done")
            else:
                info = stages.get(s, {})
                if info.get("done"):
                    wall = info.get("wall_s", 0)
                    ts = info.get("timestamp", "")
                    lines.append(f"  {s:6s}: done ({wall:.0f}s, {ts})")
                else:
                    lines.append(f"  {s:6s}: pending")

        log.info("\n".join(lines))

    def __repr__(self) -> str:
        return f"WorkDir({self._path!r})"
