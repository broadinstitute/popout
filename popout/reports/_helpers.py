"""Small TSV / formatting helpers used by chart functions and sections."""

from __future__ import annotations

import gzip
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from popout.labelspace.registry import SP5, SP6, LabelSpace


def read_tsv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Return ``(header, rows)`` from a TSV (or gzipped TSV).

    Empty / missing file returns ``([], [])`` — sections gate on this.
    """
    p = Path(path)
    if not p.exists():
        return [], []
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as f:
        text = f.read().rstrip("\n")
    if not text:
        return [], []
    lines = text.split("\n")
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


def to_float(s: Any) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        f = float(s)
        return None if f != f else f
    s = str(s).strip()
    if s in ("", "NA", "nan", "NaN"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if f != f else f


def fmt_num(v: Any, places: int = 3) -> str:
    f = to_float(v)
    if f is None:
        return "—"
    return f"{f:.{places}f}"


def fmt_int(v: Any) -> str:
    f = to_float(v)
    if f is None:
        return "—"
    return f"{int(f):,}"


def fmt_pct(v: Any, places: int = 1) -> str:
    """Render a 0-1 fraction as a percentage string."""
    f = to_float(v)
    if f is None:
        return "—"
    return f"{(f * 100):.{places}f}%"


def md_escape(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("_", "\\_")


def n_weighted_mean(pairs: Iterable[tuple[float, float]]) -> float | None:
    """``[(weight, value), ...]`` → n-weighted mean. Drops NaN values / zero
    weights. Returns None if no usable data."""
    num = 0.0
    den = 0.0
    for w, v in pairs:
        if v != v or w != w or w <= 0:
            continue
        num += w * v
        den += w
    return num / den if den > 0 else None


def overlay_ticks(
    ax, y_center: float, values: Iterable[float], *,
    color: str = "#222", tick_height: float = 0.32,
    alpha: float = 0.95, lw: float = 1.5,
) -> None:
    """Draw short vertical tick marks at each value on a horizontal-bar
    chart — used to overlay per-cluster values on top of a cohort-pooled
    bar."""
    import numpy as np

    arr = np.array([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return
    y0 = y_center - tick_height / 2
    y1 = y_center + tick_height / 2
    ax.vlines(arr, y0, y1, colors=color, alpha=alpha, linewidth=lw, zorder=5)


def topn(rows: Iterable[tuple[str, float]], n: int = 3,
         *, reverse: bool = True) -> list[tuple[str, float]]:
    """Top-n (lab, value) by descending value (ascending when reverse=False)."""
    clean = [(lab, v) for lab, v in rows if v is not None and v == v]
    return sorted(clean, key=lambda r: r[1], reverse=reverse)[:n]


# ── ancestry stratification primitives ──────────────────────────────────


def classify_sample_regime(
    props: np.ndarray,
    members: list[str],
    *,
    threshold_dominant: float = 0.95,
    threshold_leaning: float = 0.85,
    threshold_admix: float = 0.10,
) -> str:
    """Per-sample regime label from a single-chrom proportion vector.

    Returns ``"X≥{thr_dom:.2f}"`` if top-1 ≥ ``threshold_dominant``;
    ``"X-leaning"`` if top-1 ≥ ``threshold_leaning`` and top-2 <
    ``threshold_admix``; ``"X+Y+Z"`` (alpha-sorted) if top-3 ≥
    ``threshold_admix``; ``"X+Y"`` (alpha-sorted) if top-2 ≥
    ``threshold_admix``; otherwise ``"X-leaning"``.

    The canonical regime classifier; callers should prefer this over
    inlined argsort logic.
    """
    order = np.argsort(props)[::-1]
    p1, p2, p3 = props[order[0]], props[order[1]], props[order[2]]
    a1, a2, a3 = members[order[0]], members[order[1]], members[order[2]]
    if p1 >= threshold_dominant:
        return f"{a1}≥{threshold_dominant:.2f}"
    if p1 >= threshold_leaning and p2 < threshold_admix:
        return f"{a1}-leaning"
    if p3 >= threshold_admix:
        return "+".join(sorted([a1, a2, a3]))
    if p2 >= threshold_admix:
        return "+".join(sorted([a1, a2]))
    return f"{a1}-leaning"


# Strict per-row lane partition for K2-style rainclouds. Each label row
# occupies [i-0.5, i+0.5]: violin band above, cohort bar in the middle,
# sina rain band below.
RAINCLOUD_LANE = dict(
    violin_top=-0.46, violin_base=-0.14,   # band height 0.32
    bar_top=-0.08, bar_bot=0.08,            # band height 0.16
    rain_top=0.14, rain_bot=0.46,           # band height 0.32
)


def _kde_1d(values: np.ndarray, x_grid: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian KDE of ``values`` evaluated on ``x_grid``. Returns zeros
    if fewer than 2 finite values."""
    import math as _math
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.zeros_like(x_grid)
    diff = (x_grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * diff ** 2).sum(axis=1) / (
        v.size * bw * _math.sqrt(2 * _math.pi)
    )


# Filled marker shapes used to disambiguate clusters that happen to land
# on perceptually-close colors. Cycled at a different period than the
# golden-ratio hue increment so two clusters with similar colors get
# different shapes by construction.
_CLUSTER_MARKERS: tuple[str, ...] = ("o", "s", "D", "^", "*", "P", "v")


def cluster_palette(cluster_ids: Iterable[str]) -> dict[str, str]:
    """Deterministic ``cluster_id -> hex`` map. Kept for callers that only
    need the color channel. ``cluster_styles`` returns the same hex plus
    a per-cluster marker shape; prefer it when the rendering surface
    benefits from a second visual channel.
    """
    return {cid: style["color"]
            for cid, style in cluster_styles(cluster_ids).items()}


def cluster_styles(
    cluster_ids: Iterable[str],
) -> dict[str, dict[str, str]]:
    """Deterministic ``cluster_id -> {'color': hex, 'marker': str}`` map.

    Color: golden-ratio hue spacing at fixed lightness/saturation,
    started in a warm range so the swatches stay clear of the Paul Tol
    SP5 ancestry palette. Marker: cycle through a curated list of
    filled shapes (``o, s, D, ^, *, P, v``). Color cycles fast and
    marker cycles slowly relative to it, so two clusters with similar
    colors are guaranteed to differ in marker shape.

    Input order is sorted so two charts that see the same cluster set
    assign the same (color, marker) pair regardless of discovery order.
    """
    import colorsys
    golden = 0.6180339887498949
    h = 0.07  # warm orange-yellow start
    out: dict[str, dict[str, str]] = {}
    for i, cid in enumerate(sorted({c for c in cluster_ids if c})):
        r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.55)
        out[cid] = {
            "color": f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}",
            "marker": _CLUSTER_MARKERS[i % len(_CLUSTER_MARKERS)],
        }
        h = (h + golden) % 1.0
    return out


def raincloud_panel(
    ax,
    labels: list[str],
    pooled: dict,
    values_by_label: dict,
    *,
    palette: dict[str, str],
    x_lo: float,
    x_hi: float,
    title: str,
    xlabel: str,
    threshold: float | None = None,
    threshold_label: str | None = None,
    pooled_fmt: str = "{:.3f}",
    log: bool = False,
    bw: float | None = None,
    seed: int = 0xCAFE,
    pooled_marker: bool = True,
    clusters_by_label: dict | None = None,
    cluster_palette_map: dict[str, str] | None = None,
    cluster_style_map: dict[str, dict[str, str]] | None = None,
) -> None:
    """Draw a K2 raincloud + sina rain panel on ``ax``.

    Layout: for each row ``i``, the [i-0.5, i+0.5] band splits into
    a half-violin (KDE) above the bar, a thin cohort-pooled bar in the
    middle, and a sina-jittered rain band below — strict, no overlap.

    Args:
      labels: row labels (top → bottom on the panel).
      pooled: ``{label: pooled_value | None}``.
      values_by_label: ``{label: list[float]}`` per-(cluster, chrom) values
        that become raindrops.
      palette: row colour lookup.
      x_lo, x_hi: x-axis range. With ``log=True``, both must be positive.
      title, xlabel: panel decoration.
      threshold: optional vertical reference line (e.g. pass criterion).
      threshold_label: optional label printed next to ``threshold``.
      pooled_fmt: format string for the pooled value annotation.
      log: True → KDE in log10 space (used by tract length).
      bw: KDE bandwidth; default is span * 0.012 (or 0.18 in log space).
      clusters_by_label: optional ``{label: list[str]}`` aligned to
        ``values_by_label`` — when provided alongside a cluster style
        map, raindrops are coloured (and shaped, if a style map is
        supplied) by cluster identity instead of by row ancestry. KDE
        and pooled bar stay row-coloured.
      cluster_palette_map: ``{cluster_id: hex}`` (see ``cluster_palette``).
        Used for color only.
      cluster_style_map: ``{cluster_id: {'color': hex, 'marker': str}}``
        (see ``cluster_styles``). Takes precedence over
        ``cluster_palette_map`` and gives each cluster its own marker
        shape, which is the readable choice when there are enough
        clusters that color alone risks muddy pairs.
    """
    n = len(labels)
    rng_master = np.random.default_rng(seed)
    LANE = RAINCLOUD_LANE
    violin_h = LANE["violin_base"] - LANE["violin_top"]
    rain_h = LANE["rain_bot"] - LANE["rain_top"]
    rain_cy = (LANE["rain_top"] + LANE["rain_bot"]) / 2
    rain_half = rain_h / 2
    bar_h = LANE["bar_bot"] - LANE["bar_top"]

    if log:
        if x_lo <= 0 or x_hi <= 0:
            raise ValueError("raincloud_panel(log=True) needs positive x_lo/x_hi")
        xs_disp = np.logspace(np.log10(x_lo), np.log10(x_hi), 400)
        xs_kde = np.log10(xs_disp)
        eff_bw = 0.18 if bw is None else bw
    else:
        xs_disp = np.linspace(x_lo, x_hi, 400)
        xs_kde = xs_disp
        eff_bw = ((x_hi - x_lo) * 0.012) if bw is None else bw

    for i, lab in enumerate(labels):
        color = palette.get(lab.split(".")[0], "#888888")
        raw = values_by_label.get(lab, [])
        raw = np.array([v for v in raw if v == v], dtype=float)
        v_pool = pooled.get(lab)
        if v_pool is not None and v_pool == v_pool:
            ax.barh(i, v_pool, color=color, edgecolor="white",
                    height=bar_h, alpha=0.85, zorder=3)
            ax.text(v_pool * (1.005 if log else 1) + (
                        (x_hi - x_lo) * 0.005 if not log else 0),
                    i, pooled_fmt.format(v_pool),
                    va="center", fontsize=9, zorder=6)
            if pooled_marker:
                ax.vlines(v_pool, i + LANE["bar_top"], i + LANE["bar_bot"],
                          color="#222", linewidth=1.4, zorder=7)
        if raw.size < 2:
            continue
        # half-violin in upper band
        kde_vals = np.log10(raw) if log else raw
        d = _kde_1d(kde_vals, xs_kde, bw=eff_bw)
        d_max = d.max() or 1.0
        d_scaled = d / d_max * violin_h
        y_top = i + LANE["violin_base"]
        y_bot = y_top - d_scaled
        ax.fill_between(xs_disp, y_top, y_bot, color=color, alpha=0.42,
                        linewidth=0, zorder=2)
        ax.plot(xs_disp, y_bot, color=color, linewidth=0.8, zorder=2)
        # sina rain in lower band
        dens_at_pt = _kde_1d(kde_vals, kde_vals, bw=eff_bw)
        dens_max = dens_at_pt.max()
        dens_norm = (dens_at_pt / dens_max) if dens_max > 0 \
                    else np.zeros_like(dens_at_pt)
        rng = np.random.default_rng(int(rng_master.integers(0, 2 ** 31 - 1)))
        jitter = rng.uniform(-1, 1, size=raw.size) * dens_norm * rain_half
        ys = i + rain_cy + jitter
        # Per-drop colors and (optionally) shapes when cluster context
        # is threaded through. The aligned cluster_ids list must match
        # the order of the unfiltered values; values_by_label[lab] may
        # have NaN entries that get filtered for ``raw`` but the
        # alignment with clusters_by_label must survive that filter.
        drop_colors: list[str] | None = None
        drop_markers: list[str] | None = None
        if clusters_by_label is not None:
            raw_full = list(values_by_label.get(lab, []))
            cids_full = list(clusters_by_label.get(lab, []))
            if len(cids_full) == len(raw_full):
                kept = [
                    (v, cid) for v, cid in zip(raw_full, cids_full)
                    if v == v
                ]
                if cluster_style_map is not None:
                    drop_colors = [
                        cluster_style_map.get(cid, {}).get("color", color)
                        for _v, cid in kept
                    ]
                    drop_markers = [
                        cluster_style_map.get(cid, {}).get("marker", "o")
                        for _v, cid in kept
                    ]
                elif cluster_palette_map is not None:
                    drop_colors = [
                        cluster_palette_map.get(cid, color)
                        for _v, cid in kept
                    ]

        if drop_markers is not None:
            # matplotlib's scatter accepts a single marker per call, so
            # group points by shape and emit one call per group.
            by_marker: dict[str, tuple[list, list, list]] = {}
            for v, y, c, m in zip(raw, ys, drop_colors, drop_markers):
                b = by_marker.setdefault(m, ([], [], []))
                b[0].append(v); b[1].append(y); b[2].append(c)
            for m, (xs, yvs, cs) in by_marker.items():
                ax.scatter(xs, yvs, s=18, c=cs, marker=m,
                           edgecolor="white", linewidth=0.4,
                           alpha=0.85, zorder=5)
        else:
            single_colors = drop_colors if drop_colors is not None else [color]
            ax.scatter(raw, ys, s=14, c=single_colors,
                       edgecolor="white", linewidth=0.4,
                       alpha=0.85, zorder=5)

    if threshold is not None:
        ax.axvline(threshold, color="#666", linestyle="--", linewidth=0.9,
                   zorder=1)
        if threshold_label:
            ax.text(threshold, -0.55, f"  {threshold_label}",
                    fontsize=8, color="#555", ha="left", va="bottom")
    if log:
        ax.set_xscale("log")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(-0.55, n - 0.45)
    ax.invert_yaxis()
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def top1_strata(
    cube: np.ndarray,
    members: list[str],
    *,
    chrom_idx: int = 0,
) -> dict[str, np.ndarray]:
    """``{ancestry_label: bool mask over N_samples}`` keyed by top-1 ancestry.

    Each sample lands in exactly one stratum (the argmax of its
    proportion vector at ``chrom_idx``). Default chrom is chr1 — matches
    the convention used by the CCC suite and cohort-structure charts.
    """
    primary = np.argmax(cube[:, chrom_idx, :], axis=1)
    out: dict[str, np.ndarray] = {}
    for ai, lab in enumerate(members):
        out[lab] = primary == ai
    return out


# ── cohort_global.tsv → per-sample cube ──────────────────────────────────

def _read_flare_to_rf(bundle_dir: Path) -> dict[tuple[str, str], dict[int, str]]:
    """``(cluster_id, chrom) → {FLARE component index → SP6 label}``.

    Returns an empty dict if ``merged_groups_rf.tsv`` is missing or empty.
    """
    path = Path(bundle_dir) / "cohort" / "merged_groups_rf.tsv"
    header, rows = read_tsv(path)
    out: dict[tuple[str, str], dict[int, str]] = {}
    if not rows:
        return out
    col = {h: i for i, h in enumerate(header)}
    for r in rows:
        try:
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
            rf = r[col["rf_label"]]
            idxs = r[col["component_indices"]]
        except (IndexError, KeyError):
            continue
        d = out.setdefault((cid, chrom), {})
        for token in idxs.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                d[int(token)] = rf
            except ValueError:
                continue
    return out


@lru_cache(maxsize=8)
def _load_cohort_cube_cached(
    bundle_dir_str: str,
    label_space_tag: str,
    mid_rule: str,
    chroms: tuple[str, ...] | None,
) -> dict[str, Any]:
    from popout.labelspace.registry import get as get_space
    return _load_cohort_cube_impl(
        Path(bundle_dir_str), label_space=get_space(label_space_tag),
        mid_rule=mid_rule, chroms=chroms)


def load_cohort_cube(
    bundle_dir: Path,
    *,
    label_space: LabelSpace = SP5,
    mid_rule: str = "drop",
    chroms: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Cached entrypoint. See ``_load_cohort_cube_impl`` for details."""
    return _load_cohort_cube_cached(
        str(Path(bundle_dir).resolve()), label_space.tag,
        mid_rule, tuple(chroms) if chroms is not None else None)


def _load_cohort_cube_impl(
    bundle_dir: Path,
    *,
    label_space: LabelSpace = SP5,
    mid_rule: str = "drop",
    chroms: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load ``cohort_global.tsv`` into a per-sample 3-D array.

    Returns a dict with:

    - ``cube``: ndarray ``(N_samples, N_chroms, len(label_space))`` of
      proportions, normalised so every (sample, chrom) row sums to 1.
    - ``sample_ids``: list of sample IDs (axis 0).
    - ``chroms``: list of chrom labels (axis 1).
    - ``label_space``: the label space used (axis 2).
    - ``cluster_of``: ``(sample_id, chrom) → cluster_id`` mapping used.
    - ``n_dropped_samples``: count of samples filtered out because they
      were missing on at least one chrom.

    Per (cluster, chrom) FLARE component indices are mapped to SP6
    labels via ``merged_groups_rf.tsv``. With ``label_space=SP5`` and
    ``mid_rule="drop"`` the MID column is dropped and remaining rows
    renormalised. With ``mid_rule="fold_to_eur"`` the MID component is
    added to the eur column before dropping. Returns an empty dict if
    either source file is missing.
    """
    bundle_dir = Path(bundle_dir)
    flare_to_rf = _read_flare_to_rf(bundle_dir)
    if not flare_to_rf:
        return {}
    header, rows = read_tsv(bundle_dir / "cohort" / "cohort_global.tsv")
    if not rows:
        return {}
    col = {h: i for i, h in enumerate(header)}
    required = ("sample_id", "cluster_id", "chrom")
    if any(c not in col for c in required):
        return {}
    n_meta = col["sample_id"] + 1

    chroms_seen: set[str] = set()
    for r in rows:
        try:
            chroms_seen.add(r[col["chrom"]])
        except IndexError:
            continue
    if chroms is None:
        chroms = tuple(f"chr{i}" for i in range(1, 23)
                       if f"chr{i}" in chroms_seen)
    chrom_idx = {c: i for i, c in enumerate(chroms)}
    n_chroms = len(chroms)
    members = label_space.members
    n_anc = len(members)
    anc_idx = {a: i for i, a in enumerate(members)}

    per_sample: dict[str, np.ndarray] = defaultdict(
        lambda: np.full((n_chroms, n_anc), np.nan, dtype=float))
    cluster_of: dict[tuple[str, str], str] = {}

    for r in rows:
        try:
            sid = r[col["sample_id"]]
            cid = r[col["cluster_id"]]
            chrom = r[col["chrom"]]
        except IndexError:
            continue
        ci = chrom_idx.get(chrom)
        if ci is None:
            continue
        if len(r) <= n_meta:
            continue
        try:
            vals = [float(x) for x in r[n_meta:]]
        except ValueError:
            continue
        mapping = flare_to_rf.get((cid, chrom))
        if not mapping:
            continue
        row = np.zeros(n_anc, dtype=float)
        mid_extra = 0.0
        for fi, v in enumerate(vals):
            lab = mapping.get(fi)
            if lab is None:
                continue
            if lab == "mid":
                if mid_rule == "fold_to_eur" and "eur" in anc_idx:
                    row[anc_idx["eur"]] += v
                # mid_rule == "drop" or fold is impossible: skip
                mid_extra += v
                continue
            ai = anc_idx.get(lab)
            if ai is None:
                continue
            row[ai] += v
        total = row.sum()
        if total <= 0:
            continue
        row /= total
        per_sample[sid][ci, :] = row
        cluster_of[(sid, chrom)] = cid

    sample_ids: list[str] = []
    arrays: list[np.ndarray] = []
    n_dropped = 0
    for sid, arr in per_sample.items():
        if np.isnan(arr).any():
            n_dropped += 1
            continue
        sample_ids.append(sid)
        arrays.append(arr)
    if not arrays:
        return {}
    cube = np.stack(arrays, axis=0)
    return {
        "cube": cube,
        "sample_ids": sample_ids,
        "chroms": list(chroms),
        "label_space": label_space,
        "cluster_of": cluster_of,
        "n_dropped_samples": n_dropped,
    }
