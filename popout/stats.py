"""Runtime statistics collection and monitoring for popout.

Always-on: writes JSONL event log and summary JSON.  Zero required
dependencies beyond numpy.

Optional: forwards metrics to W&B or TensorBoard if installed and
requested via --monitor flag.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _jsonify(value: Any) -> Any:
    """Convert value to JSON-serializable form."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    try:
        import jax.numpy as jnp
        if isinstance(value, jnp.ndarray):
            return np.array(value).tolist()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Monitor backends
# ---------------------------------------------------------------------------

class _NullBackend:
    """No-op backend. Always available."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


class _WandbBackend:
    """Forwards metrics to Weights & Biases."""

    def __init__(self, config: dict[str, Any] | None = None):
        import wandb
        self._run = wandb.init(project="popout", config=config or {})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        import wandb
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        import wandb
        wandb.finish()


class _TensorBoardBackend:
    """Forwards metrics to TensorBoard."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        s = step or 0
        for key, val in metrics.items():
            self._writer.add_scalar(key, val, global_step=s)

    def finish(self) -> None:
        self._writer.flush()
        self._writer.close()


def _make_backend(name: str | None, prefix: str, config: dict | None = None):
    """Factory: create the requested monitoring backend."""
    if name is None:
        return _NullBackend()
    if name == "wandb":
        try:
            return _WandbBackend(config=config)
        except ImportError:
            raise ImportError(
                "wandb is required for --monitor wandb. "
                "Install with: pip install wandb"
            )
    if name == "tensorboard":
        try:
            return _TensorBoardBackend(log_dir=f"{prefix}_tb")
        except ImportError:
            raise ImportError(
                "torch.utils.tensorboard is required for --monitor tensorboard. "
                "Install with: pip install tensorboard"
            )
    raise ValueError(f"Unknown monitor backend: {name!r}. Use 'wandb' or 'tensorboard'.")


# ---------------------------------------------------------------------------
# StatsCollector
# ---------------------------------------------------------------------------

class StatsCollector:
    """Central metrics collector.  One per pipeline run.

    Usage::

        stats = StatsCollector("output/cohort")
        stats.emit("io/sites_biallelic", 45000, chrom="1")
        stats.timer_start("e_step")
        ...
        stats.timer_stop("e_step", chrom="1", iteration=0)
        ...
        summary = stats.finalize()

    Writes:
      - ``{prefix}.stats.jsonl`` — timestamped event log (always)
      - ``{prefix}.summary.json`` — aggregated QC stats (at finalize)
    """

    def __init__(
        self,
        prefix: str,
        monitor: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._prefix = prefix
        self._t0 = time.time()
        self._t0_perf = time.perf_counter()
        self._jsonl_path = Path(f"{prefix}.stats.jsonl")
        self._summary_path = Path(f"{prefix}.summary.json")
        self._jsonl_fh = open(self._jsonl_path, "w")
        self._backend = _make_backend(monitor, prefix, config)
        self._events: list[dict] = []
        self._timers: dict[str, float] = {}
        self._config = config or {}
        self._step = 0

    # --- Core API ---

    def emit(
        self,
        key: str,
        value: Any,
        *,
        step: int | None = None,
        chrom: str | None = None,
        iteration: int | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """Record a single metric event."""
        if step is None:
            step = self._step
            self._step += 1

        event: dict[str, Any] = {
            "t": round(time.time() - self._t0, 4),
            "step": step,
            "key": key,
            "value": _jsonify(value),
        }
        if chrom is not None:
            event["chrom"] = chrom
        if iteration is not None:
            event["iteration"] = iteration
        if tags:
            event["tags"] = _jsonify(tags)

        # Write to JSONL
        self._jsonl_fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._jsonl_fh.flush()

        # Keep for summary
        self._events.append(event)

        # Forward scalars to live backend
        if isinstance(value, (int, float)):
            prefix = f"chr{chrom}/" if chrom is not None else ""
            self._backend.log_metrics({f"{prefix}{key}": value}, step=step)

    def timer_start(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.perf_counter()

    def timer_stop(self, name: str, **emit_kwargs) -> float:
        """Stop a named timer and emit the elapsed time.  Returns seconds."""
        start = self._timers.pop(name, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        self.emit(f"timing/{name}", round(elapsed, 4), **emit_kwargs)
        return elapsed

    def emit_device_info(self) -> None:
        """Record JAX device information."""
        try:
            import jax
            devices = jax.devices()
            info: dict[str, Any] = {
                "platform": jax.default_backend(),
                "device_count": len(devices),
                "devices": [
                    {"kind": d.device_kind, "id": d.id}
                    for d in devices
                ],
                "python_platform": platform.platform(),
            }
        except Exception:
            info = {"platform": "unknown", "python_platform": platform.platform()}
        self.emit("runtime/device_info", info)

    def finalize(self) -> dict[str, Any]:
        """Aggregate events into summary, write summary JSON, and close."""
        summary = self._build_summary()
        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Wrote stats JSONL to %s", self._jsonl_path)
        log.info("Wrote summary JSON to %s", self._summary_path)
        self._jsonl_fh.close()
        self._backend.finish()
        return summary

    # --- Summary aggregation ---

    def _build_summary(self) -> dict[str, Any]:
        """Build summary dict from accumulated events."""
        s: dict[str, Any] = {
            "popout_version": _get_version(),
            "config": self._config,
            "total_wall_clock_sec": round(time.perf_counter() - self._t0_perf, 2),
        }

        by_key = self._group_by_key()

        s["timing"] = self._agg_timing(by_key)
        s["site_filter_funnel"] = self._agg_funnel(by_key)
        s["spectral"] = self._agg_spectral(by_key)
        s["em_convergence"] = self._agg_em(by_key)
        s["final_model"] = self._agg_final_model(by_key)
        s["output"] = self._agg_output(by_key)
        s["runtime"] = self._agg_runtime(by_key)

        return s

    def _group_by_key(self) -> dict[str, list[dict]]:
        by_key: dict[str, list[dict]] = {}
        for ev in self._events:
            by_key.setdefault(ev["key"], []).append(ev)
        return by_key

    def _agg_timing(self, by_key: dict) -> dict:
        timing: dict[str, float] = {}
        for key, evts in by_key.items():
            if key.startswith("timing/"):
                name = key[len("timing/"):]
                timing[name] = round(sum(e["value"] for e in evts), 4)
        return timing

    def _agg_funnel(self, by_key: dict) -> dict:
        funnel: dict[str, dict] = {}
        for key, evts in by_key.items():
            if not key.startswith("io/"):
                continue
            metric = key[len("io/"):]
            for ev in evts:
                chrom = ev.get("chrom", "unknown")
                funnel.setdefault(chrom, {})[metric] = ev["value"]
        return funnel

    def _agg_spectral(self, by_key: dict) -> dict:
        spec: dict[str, Any] = {}
        if "spectral/singular_values" in by_key:
            spec["singular_values"] = by_key["spectral/singular_values"][-1]["value"]
        if "spectral/gap_ratios" in by_key:
            spec["gap_ratios"] = by_key["spectral/gap_ratios"][-1]["value"]
        if "spectral/n_ancestries" in by_key:
            spec["n_ancestries"] = by_key["spectral/n_ancestries"][-1]["value"]
        if "spectral/gmm_ll" in by_key:
            spec["gmm_lls"] = [e["value"] for e in by_key["spectral/gmm_ll"]]
        if "spectral/gmm_best_ll" in by_key:
            spec["gmm_best_ll"] = by_key["spectral/gmm_best_ll"][-1]["value"]
        return spec

    def _agg_em(self, by_key: dict) -> list[dict]:
        """Build per-iteration convergence records."""
        # Collect iteration-tagged events
        iters: dict[tuple, dict] = {}  # (chrom, iteration) -> record
        for key in ["em/max_delta_freq", "em/mean_delta_freq", "em/mu", "em/T"]:
            if key not in by_key:
                continue
            metric = key.split("/")[1]
            for ev in by_key[key]:
                chrom = ev.get("chrom", "unknown")
                it = ev.get("iteration", 0)
                rec = iters.setdefault((chrom, it), {"chrom": chrom, "iteration": it})
                rec[metric] = ev["value"]

        # Add timing
        for key in ["timing/e_step", "timing/m_step"]:
            if key not in by_key:
                continue
            name = key.split("/")[1]
            for ev in by_key[key]:
                chrom = ev.get("chrom", "unknown")
                it = ev.get("iteration", 0)
                rec = iters.setdefault((chrom, it), {"chrom": chrom, "iteration": it})
                rec[f"{name}_sec"] = ev["value"]

        return sorted(iters.values(), key=lambda r: (r["chrom"], r["iteration"]))

    def _agg_final_model(self, by_key: dict) -> dict:
        model: dict[str, Any] = {}
        if "em/mu" in by_key:
            model["mu"] = by_key["em/mu"][-1]["value"]
        if "em/T" in by_key:
            model["T"] = by_key["em/T"][-1]["value"]
        if "spectral/n_ancestries" in by_key:
            model["n_ancestries"] = by_key["spectral/n_ancestries"][-1]["value"]
        if "em/ancestry_proportion" in by_key:
            props = {}
            for ev in by_key["em/ancestry_proportion"]:
                anc = ev.get("tags", {}).get("ancestry", "?")
                props[str(anc)] = ev["value"]
            model["ancestry_proportions"] = props
        return model

    def _agg_output(self, by_key: dict) -> dict:
        out: dict[str, Any] = {}
        for key in ["output/n_tracts", "output/tract_stats_by_ancestry",
                     "output/switching_rate_per_chrom",
                     "output/mean_posterior_confidence",
                     "output/genome_wide_ancestry_proportions"]:
            if key in by_key:
                metric = key.split("/", 1)[1]
                out[metric] = by_key[key][-1]["value"]
        return out

    def _agg_runtime(self, by_key: dict) -> dict:
        rt: dict[str, Any] = {}
        if "runtime/device_info" in by_key:
            rt["device_info"] = by_key["runtime/device_info"][-1]["value"]
        if "runtime/t_compute" in by_key:
            rt["t_compute_sec"] = by_key["runtime/t_compute"][-1]["value"]
        if "runtime/t_total" in by_key:
            rt["t_total_sec"] = by_key["runtime/t_total"][-1]["value"]
        throughput: dict[str, float] = {}
        if "runtime/throughput" in by_key:
            for ev in by_key["runtime/throughput"]:
                chrom = ev.get("chrom", "unknown")
                throughput[chrom] = ev["value"]
        if throughput:
            rt["throughput_per_chrom"] = throughput
        return rt


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"
