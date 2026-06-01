# popout DX collector — performance contract

The FLARE validation collector (`validation/scripts/validate_per_site_metrics.py`) went from **2h45m → ~6 min** on the largest production cluster through a specific set of techniques. Every collector in this package — global mode and local mode — must satisfy the contracts below on day one. Violations are perf regressions, not stylistic preferences.

| # | Contract | Why | Evidence |
|---|---|---|---|
| 1 | **Stream VCFs via `bcftools query` subprocess, not pysam per-cell.** Use `-f '%CHROM\t%POS[\t%AN1\t%AN2]\n'`; parse each line with `parts = line.split('\t')` and `np.array(parts[2:], dtype=np.int8)`. | pysam wraps every (sample,field) cell in a Python dict — ~1 µs/cell × 30k samples × 587k sites = ~5h. bcftools is C-speed; numpy parses the whole row in one C call. | `validate_per_site_metrics.py:346, 360-370` |
| 2 | **No per-record Python object construction in the inner loop.** No dataclasses, namedtuples, DataFrame rows. Unpack `parts` inline, slice numpy with `flat[0::2]` / `flat[1::2]` (these are views, not copies). | The inner loop runs ~587k times per chr1; per-record allocations dominate. | `validate_per_site_metrics.py:357-370` |
| 3 | **Fancy-index scatter-add `arr[idx, val] += delta`, not `np.add.at`,** when indices are guaranteed unique (e.g. `idx = np.arange(n_samples)`). Document the no-duplicates contract at the call site. | `np.add.at` is a serial atomic loop; ~100× slower than fancy-index `+=`. At 34k samples it was the bottleneck (2h45m production hit). | `validate_per_site_metrics.py:311, 386-387`; commit `4e8f813` |
| 4 | **Parallelise by per-worker `bcftools query -S samples.txt`,** not by reading the whole VCF in Python and slicing. Each worker writes its sample list, spawns its own bcftools, sees only its samples on stdout. | The C-layer FORMAT filter is free; in-Python slicing pays full I/O and parse cost per worker. | `validate_per_site_metrics.py:335-348, 486-496, 577-594`; commit `de28713` |
| 5 | **Cap workers by `MIN_SAMPLES_PER_SLICE` (300).** `workers = min(--workers, max(1, n_samples // 300))`. | ProcessPoolExecutor spawn + per-process numpy import is ~100 ms. Below 300 samples per worker, overhead exceeds gain — over-parallelising small clusters is a net loss. | `validate_per_site_metrics.py:556-559` |
| 6 | **Attribute bp to regional windows at tract-close time,** not per-site. On close, `searchsorted` to find overlapping windows → intersect bp → scatter-add. | Per-site attribution double-counts bp in overlapping windows and produces `mean_anc > 1.0` (contract violation). Tract-close attribution is exact. | `validate_per_site_metrics.py:313-332, 395-463` |
| 7 | **One pass over the VCF, multiple accumulators in lock-step. No scratch intermediates on disk.** | The old pipeline wrote a multi-GB `tracts.tsv.gz` between passes and re-read it three times. The fused pass eliminates the file and the redundant I/O. | `validate_per_site_metrics.py:1-29, 641-652` |
| 8 | **Worker functions must be module-level (picklable).** Anything sent to `ProcessPoolExecutor.submit` cannot be a nested function, an instance method, or a closure. Pass dependencies as explicit args. | ProcessPoolExecutor pickles the worker by qualified name. Closures and bound methods don't pickle. | `validate_per_site_metrics.py:270, 313, 579-594` |
| 9 | **Orchestrator uses `Step` dataclass + `ThreadPoolExecutor`** so independent DAG steps (different subprocess invocations) run concurrently. Step granularity is one subprocess per step — do not collapse independent steps to look tidier. | The per-cluster DAG has independent branches; serialising them throws away the easy wall-clock win. | `run_cluster_validation.py:94-109, 751-874` |
| 10 | **Pass the popout-format `.model`, not the raw FLARE `.model`,** to anything that calls `read_model_text`. The orchestrator runs `flare_to_popout_format` before any consumer and threads `ws.intermediates["popout_model"]` through. | The two formats are not interchangeable. Wrong file → `ValueError` deep in parsing. Easy bug to hit when wiring a new consumer naively. | `run_cluster_validation.py:217` (and the bug-history that motivated this) |
| 11 | **Emit `step.<name>.wallclock_seconds` to `tier1_metrics.tsv`** per step. The WDL replay pipes these to W&B as a per-step time-series; before/after deltas surface immediately in the dashboard. | A single total wallclock hides regressions. Per-step is what reveals them. | `run_cluster_validation.py:_write_tier1_metrics` |

## Where each contract bites in this package

- **Global-mode collector** (`run_dx_cluster.py` global steps): contracts 9, 11.
- **Local-mode FLARE parser** (`dx_local_parse_flare.py`): contracts 1, 2, 3, 4, 5, 8.
- **Local-mode views** (`dx_local_views.py`): contract 6 if any sliding-window accounting is added; contract 8 if parallelised.
- **Orchestrator** (`run_dx_cluster.py`): contracts 9, 10, 11.

If you change any of these, benchmark before and after on the largest available cluster fixture, and record the wallclock delta in your PR description.
