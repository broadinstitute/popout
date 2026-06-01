#!/usr/bin/env python3
"""Discover popout DX scatter inputs from a JSON config.

The data model is asymmetric:

- **popout** is whole-cohort. One ``run_dir`` contains a single
  ``*.global.tsv`` (anchor), a single ``*.tracts.tsv.gz``, one
  ``*.model``, etc. There is no popout-side cluster dimension.
- **FLARE** is per-(cluster, chrom). The FLARE-validate cohort bundle
  is the source of truth for the ``(cluster_id, chrom)`` universe; its
  ``per_cluster/<cid>/<chrom>/`` tree defines what shards exist.

The scatter unit is ``(cluster_id, chrom)``. Each shard pulls its
per-cluster FLARE slice plus the (constant) popout files, then subsets
the popout data in-process to the cluster's sample roster (derived from
the FLARE per-cluster ``global.tsv``).

Outputs:
  ``runs_manifest.json``  — rich nested structure for Python consumers
  ``runs_manifest.tsv``   — flat tabular form consumable from WDL via
                            ``read_tsv()``; one row per scatter shard.
                            Singleton paths (popout, rye, rf) repeat
                            in every row for self-containment.

When ``flare.cohort_bundle`` is a localised file (path starts with
``/``), this script also extracts the per-cluster FLARE slices into
``<out-dir>/flare_slices/<cluster_id>.<chrom>.global.tsv`` so each
scatter shard does not re-download the multi-GB bundle.

Fails fast (no fallbacks, no silent drops). Empty discovery, missing
required keys, and disagreement between the cohort bundle and the
requested cluster/chrom globs are all hard errors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable


ANCHOR_TOOL = "popout"
COMPARISON_TOOLS = ("flare", "rye", "rf")
ALL_TOOLS = (ANCHOR_TOOL,) + COMPARISON_TOOLS

POPOUT_DISCOVERY_SUFFIX = ".global.tsv"

# Filenames inside a popout run_dir, keyed by their basename suffix
# relative to the popout output prefix.
POPOUT_FILE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("global_tsv", ".global.tsv"),
    ("tracts", ".tracts.tsv.gz"),
    ("model", ".model"),
    ("model_npz", ".model.npz"),
    ("summary", ".summary.json"),
    ("stats_jsonl", ".stats.jsonl"),
    ("spectral_npz", ".spectral.npz"),
)


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"discover_runs: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── GCS / local listing ──────────────────────────────────────────────────


def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def list_uris(root: str) -> list[str]:
    """Recursively list every URI under ``root`` (files only, sorted)."""
    if _is_gcs(root):
        if not root.endswith("/"):
            root = root + "/"
        cmd = ["gcloud", "storage", "ls", "--recursive", root]
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            die("gcloud not found in PATH; required to list GCS roots")
        except subprocess.CalledProcessError as e:
            die(f"gcloud storage ls failed for {root!r}: {e.stderr.strip()}")
        out: list[str] = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line or line.endswith(":") or line.endswith("/"):
                continue   # section headers and dir placeholders
            out.append(line)
        return sorted(out)

    p = Path(root)
    if not p.is_dir():
        die(f"local root {root!r} is not a directory")
    return sorted(str(q) for q in p.rglob("*") if q.is_file())


# ── popout discovery — per-chrom, walks a Cromwell rundir ──────────────


_POPOUT_CHROM_RE = re.compile(r"\.(?P<chrom>chr[0-9XYM]+)\.global\.tsv$")


def discover_popout(popout_outputs: str, requested_chroms: set[str]) -> dict[str, dict[str, str]]:
    """Walk ``popout_outputs`` recursively; return ``{chrom: {label: uri}}``.

    ``popout_outputs`` is the GCS or local path containing one popout
    run's emitted files — typically a Cromwell ``call-popout_task/``
    directory. Two valid layouts:

      * per-chrom shards — each ``*.global.tsv`` basename embeds the
        chrom (``*.chrN.global.tsv``); one anchor per chrom.
      * whole-genome single anchor — exactly one ``*.global.tsv`` with
        no chrom in the name; same paths repeat under every requested chrom.

    Mixed naming, duplicate chrom anchors, or chroms requested but not
    found are hard errors.
    """
    uris = list_uris(popout_outputs)
    anchors = [u for u in uris if u.endswith(POPOUT_DISCOVERY_SUFFIX)]
    if not anchors:
        die(f"popout_outputs {popout_outputs!r} contained no *.global.tsv anchor "
            f"({len(uris)} files listed total)")

    by_chrom: dict[str, str] = {}
    chromless: list[str] = []
    for anchor in anchors:
        m = _POPOUT_CHROM_RE.search(anchor.rsplit("/", 1)[-1])
        if m:
            chrom = m.group("chrom")
            if chrom in by_chrom:
                die(f"popout_outputs {popout_outputs!r} has multiple *.global.tsv anchors "
                    f"for {chrom!r}: {by_chrom[chrom]!r} and {anchor!r}")
            by_chrom[chrom] = anchor
        else:
            chromless.append(anchor)

    if by_chrom and chromless:
        die(f"popout_outputs {popout_outputs!r}: mixed naming "
            f"({len(by_chrom)} chrom-named, {len(chromless)} chromless anchors)")

    by_basename = set(uris)

    def _siblings_for(anchor: str) -> dict[str, str]:
        prefix = anchor[: -len(POPOUT_DISCOVERY_SUFFIX)]
        return {
            label: prefix + suffix
            for label, suffix in POPOUT_FILE_SUFFIXES
            if prefix + suffix in by_basename
        }

    out: dict[str, dict[str, str]] = {}
    if by_chrom:
        missing = sorted(requested_chroms - set(by_chrom))
        if missing:
            die(f"popout_outputs {popout_outputs!r} missing per-chrom anchor(s) for "
                f"{missing}; present: {sorted(by_chrom)[:10]}")
        for chrom in requested_chroms:
            out[chrom] = _siblings_for(by_chrom[chrom])
    else:
        if len(chromless) > 1:
            die(f"popout_outputs {popout_outputs!r}: {len(chromless)} chromless anchors "
                f"but exactly one is required for the whole-genome layout")
        siblings = _siblings_for(chromless[0])
        for chrom in requested_chroms:
            out[chrom] = siblings
    return out


# ── FLARE discovery via cohort bundle ────────────────────────────────────


_FLARE_MEMBER_RE = re.compile(
    r"(?:^|/)per_cluster/(?P<cluster_id>[^/]+)/(?P<chrom>chr[^/]+)/(?P<rest>.+)$"
)


FLARE_BUNDLE_WANTED_RESTS: dict[str, str] = {
    "global.tsv": "global_tsv",
    # FLARE's per-cluster labels.json carries the FLARE-component → RF-label
    # mapping derived from correlations with RF (NOT from column names —
    # flare_to_popout_format.py strips the names). Required for projecting
    # FLARE q matrices onto the RF basis in pairwise soft-call metrics.
    "soft_correlation/labels.json": "labels_json",
    "provenance/flare_command_line.txt": "flare_command_line",
    "provenance/input_vcf_header.txt": "flare_input_vcf_header",
}


def list_flare_in_cohort_bundle(bundle_uri: str) -> set[tuple[str, str]]:
    """Enumerate every ``(cluster_id, chrom)`` pair in the bundle without
    leaving a local copy. Streams ``gcloud storage cat`` (or a local file)
    into Python's tarfile in r|* mode; reads member names only.
    """
    if _is_gcs(bundle_uri):
        proc = subprocess.Popen(
            ["gcloud", "storage", "cat", bundle_uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        pairs: set[tuple[str, str]] = set()
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|*") as tar:
                for m in tar:
                    if not m.isfile():
                        continue
                    mm = _FLARE_MEMBER_RE.search(m.name)
                    if mm and mm.group("rest") == "global.tsv":
                        pairs.add((mm.group("cluster_id"), mm.group("chrom")))
        finally:
            # Drain & close so gcloud storage cat can exit even if tar bailed early.
            if proc.stdout is not None:
                proc.stdout.close()
            proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            die(f"gcloud storage cat {bundle_uri!r} failed: {stderr.strip()}")
        return pairs

    path = Path(bundle_uri)
    if not path.is_file():
        die(f"flare.cohort_bundle {bundle_uri!r} is not a local file")
    pairs = set()
    with tarfile.open(path, "r|*") as tar:
        for m in tar:
            if not m.isfile():
                continue
            mm = _FLARE_MEMBER_RE.search(m.name)
            if mm and mm.group("rest") == "global.tsv":
                pairs.add((mm.group("cluster_id"), mm.group("chrom")))
    return pairs


def extract_flare_slices_from_cohort_bundle(
    bundle_path: Path,
    selected_pairs: set[tuple[str, str]],
    out_slices_dir: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    """Stream the bundle once; extract only the files for
    ``selected_pairs`` whose basename appears in
    :data:`FLARE_BUNDLE_WANTED_RESTS`.
    """
    out_slices_dir.mkdir(parents=True, exist_ok=True)
    found: dict[tuple[str, str], dict[str, str]] = {}
    with tarfile.open(bundle_path, "r|*") as tar:   # streaming
        for m in tar:
            if not m.isfile():
                continue
            mm = _FLARE_MEMBER_RE.search(m.name)
            if not mm:
                continue
            cid = mm.group("cluster_id")
            chrom = mm.group("chrom")
            if (cid, chrom) not in selected_pairs:
                continue
            rest = mm.group("rest")
            label = FLARE_BUNDLE_WANTED_RESTS.get(rest)
            if label is None:
                continue
            dest = out_slices_dir / cid / chrom / rest
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                die(f"failed to read {m.name!r} from {bundle_path}")
            with open(dest, "wb") as dst:
                dst.write(src.read())
            found.setdefault((cid, chrom), {})[label] = str(dest.resolve())

    missing = sorted(p for p in selected_pairs if p not in found)
    if missing:
        die(
            f"flare.cohort_bundle {bundle_path} missing global.tsv for {len(missing)} "
            f"selected pairs; first: {missing[:5]}"
        )
    return found


# ── Config validation ───────────────────────────────────────────────────


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        die("config root must be a mapping")
    for key in ("run_name", "schema_version", "tools"):
        if key not in cfg:
            die(f"config missing required key {key!r}")
    tools = cfg["tools"]
    if not isinstance(tools, list) or ANCHOR_TOOL not in tools:
        die(f"config.tools must be a list containing {ANCHOR_TOOL!r}")
    extras = [t for t in tools if t != ANCHOR_TOOL]
    if not extras:
        die("config.tools must include at least one comparison tool besides popout")
    for t in tools:
        if t not in ALL_TOOLS:
            die(f"config.tools contains unknown tool {t!r}; allowed: {ALL_TOOLS}")
    for t in extras:
        if t not in cfg:
            die(f"config.tools includes {t!r} but config.{t} block is missing")

    if "flare" in tools:
        flare = cfg["flare"]
        if not isinstance(flare, dict) or not flare.get("cohort_bundle"):
            die("config.flare.cohort_bundle is required when flare is in tools (v1.0.0)")
        # anc_vcf_root is optional; required only for local mode. The validator
        # here just type-checks the field; the build_manifest step enforces
        # the local-mode requirement.
        if "anc_vcf_root" in flare and not isinstance(flare["anc_vcf_root"], str):
            die("config.flare.anc_vcf_root must be a string (GCS prefix)")

    if "rye" in tools:
        if not isinstance(cfg.get("rye"), dict) or not cfg["rye"].get("q_path"):
            die("config.rye.q_path is required when rye is in tools")
    if "rf" in tools:
        if not isinstance(cfg.get("rf"), dict) or not cfg["rf"].get("ancestry_path"):
            die("config.rf.ancestry_path is required when rf is in tools")

    # Glob filters: must be lists of strings if present.
    for fld in ("clusters", "chroms"):
        if fld in cfg:
            v = cfg[fld]
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                die(f"config.{fld} must be a list of glob strings")


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


# ── Manifest assembly ───────────────────────────────────────────────────


def build_manifest(
    cfg: dict,
    mode: str,
    popout_outputs: str,
    out_dir: Path,
) -> tuple[dict, list[dict]]:
    """Return ``(manifest_dict, rows)``. One row per scatter shard."""
    if mode not in ("global", "global_local"):
        die(f"mode {mode!r} not in (global, global_local)")
    if not popout_outputs:
        die("popout_outputs is required")

    tools = list(cfg["tools"])
    cluster_globs: list[str] = cfg.get("clusters", ["cluster_*"])
    chrom_globs: list[str] = cfg.get("chroms", ["chr*"])

    if "flare" not in tools:
        die(
            "no (cluster_id, chrom) shards discoverable. popout DX scatters on the "
            "FLARE cohort bundle's per_cluster tree; include 'flare' in tools to "
            "make the run meaningful."
        )

    flare_cohort_bundle = cfg["flare"]["cohort_bundle"]
    # Enumerate by streaming; never materialize the bundle locally. Cromwell
    # localizes it once per shard from the URI we expose to the WDL.
    all_pairs = list_flare_in_cohort_bundle(flare_cohort_bundle)
    if not all_pairs:
        die(
            f"flare.cohort_bundle {flare_cohort_bundle!r} contained no "
            "per_cluster/<cid>/<chrom>/global.tsv members; not a v2.x "
            "FLARE-validate cohort bundle?"
        )
    selected = {
        (cid, chrom) for (cid, chrom) in all_pairs
        if _matches_any(cid, cluster_globs) and _matches_any(chrom, chrom_globs)
    }
    if not selected:
        die(
            f"clusters={cluster_globs} × chroms={chrom_globs} matched no entries in the "
            f"cohort bundle (bundle has {len(all_pairs)} (cluster, chrom) pairs; first "
            f"few: {sorted(all_pairs)[:5]})"
        )
    rejected = len(all_pairs) - len(selected)
    if rejected:
        print(
            f"discover_runs: glob filter selected {len(selected)}/{len(all_pairs)} "
            f"(cluster, chrom) pairs ({rejected} rejected)",
            file=sys.stderr,
        )
    # Per-cluster slice extraction is deferred to the per-shard task: each
    # shard receives the cohort bundle Cromwell-localized from its gs:// URI
    # and tars out its own global.tsv + labels.json. discover never makes
    # a copy of the bundle — it just streams to enumerate members.

    # Local mode requires FLARE per-cluster .anc.vcf.gz too. Discover those
    # alongside the cohort bundle slices when the user supplies anc_vcf_root.
    flare_anc_vcf: dict[tuple[str, str], str] = {}
    anc_vcf_root = cfg["flare"].get("anc_vcf_root", "") if "flare" in tools else ""
    if mode == "global_local":
        if not anc_vcf_root:
            die(
                "mode=global_local requires config.flare.anc_vcf_root (the GCS prefix "
                "where FLARE pipeline emits per-cluster <cluster_id>.<chrom>.anc.vcf.gz). "
                "The cohort bundle does not carry the raw VCFs."
            )
        anc_uris = list_uris(anc_vcf_root)
        idx = _index_by_id_chrom(anc_uris, "anc.vcf.gz")
        if not idx:
            die(
                f"flare.anc_vcf_root {anc_vcf_root!r} contained no files matching "
                f"<cluster_id>.<chrom>.anc.vcf.gz (listed {len(anc_uris)} files total)"
            )
        for key, siblings in idx.items():
            anc = siblings.get("anc.vcf.gz")
            if anc is not None:
                flare_anc_vcf[key] = anc
        missing = sorted(k for k in selected if k not in flare_anc_vcf)
        if missing:
            die(
                f"flare.anc_vcf_root missing .anc.vcf.gz for {len(missing)} selected "
                f"(cluster, chrom) pairs; first: {missing[:5]}"
            )

    rye_q_path = cfg["rye"]["q_path"] if "rye" in tools else ""
    rf_ancestry_path = cfg["rf"]["ancestry_path"] if "rf" in tools else ""

    # Popout discovery happens AFTER the cluster/chrom filter so we only
    # walk for chroms we'll actually use.
    selected_chroms = {chrom for (_, chrom) in selected}
    popout_by_chrom = discover_popout(popout_outputs, selected_chroms)

    selected_keys = sorted(selected)

    # Per-shard rows. Popout / rye / rf singletons repeat in every row
    # for self-containment (WDL read_tsv passes one row per shard).
    rows: list[dict] = []
    for cid, chrom in selected_keys:
        p = popout_by_chrom[chrom]
        rows.append({
            "cluster_id": cid,
            "chrom": chrom,
            # flare_global_tsv / flare_labels_json are emitted empty here —
            # the per-shard WDL task extracts them from the cohort bundle
            # and populates the orchestrator's shard-local TSV directly.
            "flare_global_tsv": "",
            "flare_labels_json": "",
            "flare_anc_vcf": flare_anc_vcf.get((cid, chrom), ""),
            "popout_global_tsv": p.get("global_tsv", ""),
            "popout_tracts": p.get("tracts", ""),
            "popout_model": p.get("model", ""),
            "popout_model_npz": p.get("model_npz", ""),
            "popout_summary": p.get("summary", ""),
            "rye_q_path": rye_q_path,
            "rf_ancestry_path": rf_ancestry_path,
        })

    manifest = {
        "schema_version": cfg["schema_version"],
        "run_name": cfg["run_name"],
        "mode": mode,
        "tools": tools,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cluster_globs": cluster_globs,
        "chrom_globs": chrom_globs,
        "n_runs": len(rows),
        "cluster_ids": sorted({r["cluster_id"] for r in rows}),
        "chroms": sorted({r["chrom"] for r in rows}),
        "popout_by_chrom": popout_by_chrom,
        "rye_q_path": rye_q_path,
        "rf_ancestry_path": rf_ancestry_path,
        "flare_cohort_bundle": flare_cohort_bundle,
        "runs": rows,
    }
    return manifest, rows


# ── TSV emission ────────────────────────────────────────────────────────

TSV_COLUMNS: tuple[str, ...] = (
    "cluster_id", "chrom",
    "flare_global_tsv", "flare_labels_json", "flare_anc_vcf",
    "popout_global_tsv", "popout_tracts", "popout_model", "popout_model_npz", "popout_summary",
    "rye_q_path", "rf_ancestry_path",
)


def write_tsv(rows: Iterable[dict], path: Path) -> None:
    """Write the scatter-row TSV.

    Two files are produced next to ``path``:
      ``<path>``                  no header, machine-consumable by WDL
                                  ``read_tsv()`` (each row → one scatter shard)
      ``<path>.with_header.tsv``  same data with a header line for humans

    The column order is :data:`TSV_COLUMNS`; the JSON manifest documents it.
    """
    with open(path, "w") as f:
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in TSV_COLUMNS) + "\n")
    with open(str(path) + ".with_header.tsv", "w") as f:
        f.write("\t".join(TSV_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in TSV_COLUMNS) + "\n")


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON config (localised by WDL)")
    ap.add_argument("--popout-outputs", required=True,
                    help="GCS or local path containing one popout run's emitted files "
                         "(typically a Cromwell call-popout_task/ directory); walked recursively")
    ap.add_argument("--mode", required=True, choices=("global", "global_local"))
    ap.add_argument("--out-dir", required=True, help="Output dir; manifests + flare_slices/ written here")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = json.load(f)

    validate_config(cfg)
    manifest, rows = build_manifest(cfg, args.mode, args.popout_outputs, out_dir)

    json_path = out_dir / "runs_manifest.json"
    tsv_path = out_dir / "runs_manifest.tsv"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    write_tsv(rows, tsv_path)

    # Expose the cohort bundle URI to the WDL as a one-line String output
    # (no copy is made; Cromwell localizes per shard from this URI).
    (out_dir / "cohort_bundle_uri.txt").write_text(
        manifest["flare_cohort_bundle"] + "\n"
    )

    print(
        f"discover_runs: wrote {json_path} and {tsv_path} ({len(rows)} runs, "
        f"tools={manifest['tools']}, mode={manifest['mode']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
