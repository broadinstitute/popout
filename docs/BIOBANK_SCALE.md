# Biobank-scale memory notes

Tracks known places where tensor shapes proportional to H (haplotype
count, 500k–1M+) must be batched rather than materialized whole.

Last audited: 2026-04-26.

## Fixed

| Location | What | Fix |
|---|---|---|
| `spectral.py` `window_init_allele_freq` | Lazy JAX graph across 506 windows retained full `(H, T)` geno dependency; forced materialization via `np.array()` uploaded 27 GB to device | Double-loop (windows x hap batches) with numpy host accumulators; `jnp.asarray()` on return severs lazy graph |
| `em.py` `init_model_soft` | `resp.T @ geno` as single `(A, T)` GEMM on `(H, T)` geno | Batched in 20k-hap chunks (`_INIT_BATCH`) |
| `hmm.py` `forward_backward_em` | Per-site HMM posteriors `(H, T, A)` | Streaming checkpointed scan in `batch_size`-hap batches |
| `hmm.py` `forward_backward_decode` | Decode outputs `(H, T, A)` | Streaming decode in `batch_size`-hap batches |
| `spectral.py` `_genotypes_to_pca_projection` | SVD on full `(H, T)` | Subsamples `max_haps_svd` for SVD; projects all H in `projection_batch` chunks |
| `output.py` `write_decode_parquet` | `.tobytes()` on 10 GB calls + 20 GB fp16 max_post; full-H row group | Zero-copy `pa.py_buffer(ndarray)`; streaming 50k-hap row groups via `ParquetWriter` |
| `output.py` `read_decode_parquet` | `combine_chunks().to_pylist()` + `b"".join()` on multi-GB binary column | Iterate row groups; `buffers()[2]` + `np.frombuffer` into preallocated output |
| `output.py` `write_ancestry_tracts` | `np.array(result.calls)` redundant 10 GB copy | `np.asarray()` — zero-copy when input is already ndarray |
| `em.py` ancestry proportion logging | `(result.calls == a).mean()` creates 10 GB bool × 20 ancestries | Chunked `np.bincount` in 50k-hap slices |
| `em.py` `run_em` line 542 | `jnp.array(geno_np)` unconditionally uploads 27 GB to 40 GB A100; OOMs at T≥25k | `fits_on_device` guard; host-resident with batched transfers when geno exceeds device budget |
| `em.py` `run_em_genome` warm-start | `jnp.array(chrom_data.geno)` on chroms 2+; same unconditional upload | `fits_on_device` guard (mirroring the chrom-1 fix) |
| `recursive_seed.py` seeding_mask | `geno[bool_mask]` copies 25 GB contiguously | `_MaskedGeno` wrapper — 4 MB index array, zero geno copy |
| `hmm.py` `forward_backward_decode` | `max_post` allocated as `(H, T) float32` — 108 GB at T=25k | Changed to `float16`; matches the block-emissions decode path |
| `hmm.py` `forward_backward_bucketed_decode` | Same `float32` `max_post` | Changed to `float16` |
| `em.py` `init_model_from_labels` | `geno.astype(jnp.float32)` on full geno — 108 GB | Deleted (uncalled dead code) |
| `post_em_consolidation.py` `consolidate` | `(calls == a)` and `(calls == a) & (mp > 0.8)` create 25 GB bool masks per ancestry in a loop | Chunked `np.bincount` in 50k-hap slices; parquet branch already chunked |
| `recursive_seed.py` `_run_k2_em_split` + main loop | `geno[node.indices]` on `_MaskedGeno` triggers advanced-index copy (30+ GB) | `_sub_geno()` helper composes index arrays through `_MaskedGeno` — zero geno copy |
| `post_em_consolidation.py` `consolidate` line 293 | `remap[res.calls]` — int32 remap fancy-indexes int8 calls, producing int32 (H,T) = 108 GB; even with int8 remap, allocates a second 27 GB array while old is still alive (54 GB peak) | Cast `remap` to int8 + chunked in-place remap (`res.calls[s:e] = remap_i8[res.calls[s:e]]`); peak transient ~1.3 GB |
| `panel.py` `extract_whole_haplotypes` / `extract_segments` | `np.array(result.calls)` copies 27 GB; no parquet-streaming branch for max_post; `np.ones((H,T), bool)` fallback = 27 GB | `np.asarray` zero-copy; added `_iter_max_post_groups` streaming; removed all-True fallback |
| `em.py` `decode_chromosome` | `calls = np.empty((H, T), int8)` allocates 41.6 GB pinned VM at T=38k; chunked writes fault ~1.7 GB/chunk and exhaust the 100 GB AoU VM by chunk 9 | `np.memmap` into sibling of `decode_parquet_path` when streaming; kernel manages residency; downstream consumers unchanged (memmap is an ndarray subclass) |
| `em.py` `decode_chromosome` block-emissions branch | When `bucket_assignments` is set on the model, the chunk loop ran once with the global model, ignoring per-hap-T entirely; no streaming, no merge | Bucket dispatch wrapper around the chunk loop; per-bucket `AncestryModel` carries `bucket_centers[b]` as `gen_since_admix`; per-bucket parquet writers + `_merge_bucket_parquets` heap merge produce hap-ordered output; calls memmap supports fancy-index writes per bucket |
| `em.py` `decode_chromosome` non-block branch + `hmm.py` `forward_backward_decode` / `forward_backward_bucketed_decode` | Both FB primitives `np.zeros((H, T), int8)` for calls + `(H, T) float16` for max_post — 81 GB pinned RSS at AoU scale; reachable via `--per-hap-T` without `--block-emissions`; OOMs the 100 GB VM | `calls_out` / `max_post_writer` keyword-only params on both FB primitives; `decode_chromosome` opens memmap for calls + `DecodeParquetWriter` (single, when not bucketed; per-bucket + merge when bucket_assignments is set) and passes them through. `forward_backward_bucketed_decode` delegates per-bucket work to `forward_backward_decode(..., hap_idx_map=hap_idx)` for fancy-index writes into a single global buffer |

## Known remaining

- **`cnn/refine.py` lines 255, 421**: `update_allele_freq(geno, gamma)`
  materializes full `(H, T, A)` gamma. Not on the main HMM EM path —
  only the CNN backend. Risk at biobank scale if CNN is used.

- **`hmm.py` `forward_backward_bucketed` line 1055**: Calls
  `model.log_emission(geno)` once for all H, producing `(H, T, A)`.
  Currently guarded by batch-size checks, but the function itself
  doesn't batch.

## Audit checklist for new code

Any function that:

1. Accepts `geno` at full cohort scale, AND
2. Contains tensors with an `H` dimension where the other dimension is
   non-trivial (more than a constant factor), AND
3. Returns a JAX array

must either:

- Call `jax.device_get()` + `jnp.asarray()` on its return value to
  sever the lazy graph, OR
- Guarantee that the caller materializes the result before `geno` goes
  out of scope.

Similarly, any output function that converts a large numpy array to
bytes for serialization should use zero-copy buffer wrapping (e.g.
`pa.py_buffer(ndarray)`) rather than `.tobytes()`.

### Numpy reductions on integer arrays

`np.bincount`, `np.sum`, `np.cumsum` and similar reductions on
int8/int16/int32 arrays internally promote to int64 working buffers of
the same element count as the input. At 10 GB of int8, that's 80 GB of
scratch. Always chunk these reductions when the input exceeds ~1 GB.

## Memory traps to watch for

The following idioms look innocuous but silently allocate tensors
proportional to H or H*T at biobank scale. Each has bitten us at least once.

### 1. `jnp.array(big_numpy_array)`
Unconditional device upload. Always gate with `fits_on_device(arr.nbytes)`.
Fixed sites: `em.py:544`, `em.py:1300`, `recursive_seed.py:301`,
`recursive_seed.py:786`.

### 2. `arr[bool_mask]`
Boolean-mask indexing always copies the selected rows contiguously. At
(H, T) uint8 with H~1M, T~25k, that's 25 GB. Use integer indices and a
wrapper like `_MaskedGeno` instead.

### 3. `np.bincount(int8_array.ravel())`
Internally upcasts to int64, producing a scratch buffer 8x the input size.
At 10 GB int8 input, that's 80 GB. Chunk the bincount: call it in batches
of ~50k haplotypes and sum results.

### 4. `np.array(lazy_jax_expression)`
Forces materialization of any pending JAX graph. If that graph depends on
an (H, T, A) intermediate, materialization peaks at `H*T*A*4` bytes.
Always call `jax.device_get` immediately after producing a result you
intend to return from a function, to sever lazy dependencies.

### 5. `(H, T)` output arrays at T >= 25k
Even at fp16, `(1M, 25k)` is 54 GB. Never preallocate full-H output
tensors when an incremental write path exists. Stream to disk via
`ParquetWriter.write_batch` inside the decode loop.

### 6. `ndarray.tobytes()` for pa.py_buffer
Always copies. pyarrow accepts the buffer protocol directly: use
`pa.py_buffer(ndarray)` for zero-copy.

### 7. `row_group_size=H` in `pq.write_table`
Forces the whole column to one row group; pyarrow holds all buffers alive
until the group is finalized. Use a `ParquetWriter` loop with ~50k haps
per row group.

### 8. Wrappers with `__array__` fallback
A `_MaskedGeno`-style wrapper that overrides `__getitem__` but inherits
numpy's `__array__` protocol gets silently materialized by any
`np.asarray(wrapper)` or `jnp.asarray(wrapper)` call. Make sure the
wrapper's `__array__` is intentional and the code path that triggers it
is gated by a size check.

### 9. Python loops over blocks/ancestries with JAX ops
A Python `for` loop that does `x.at[idx].add(...)` on a JAX array creates
one XLA trace entry per iteration. They merge into a single graph whose
compiled-kernel argument size exceeds device memory. Replace with
`jax.vmap` or `jax.lax.scan`.

### 10. `(arr == value)` in a per-ancestry loop
`for a in range(A): mask = (calls == a)` allocates one full (H, T) bool
mask per iteration. At H=1M, T=25k, that's 25 GB × A iterations. Use
`np.bincount` on the flattened or filtered array in haplotype chunks
instead.

### 11. Fancy-indexing dtype promotion and double-hold
`remap[calls]` where `remap` is int32 and `calls` is int8 returns int32
— 4× the expected size. Cast the lookup table to the result dtype before
indexing. Even with matching dtypes, `result = remap[arr]` allocates a
full copy while `arr` is still alive in the enclosing object — peak is
2× the array size. Remap in place with row-chunk iteration:
`arr[s:e] = remap[arr[s:e]]`, which limits peak transient to one chunk.
Document the mutation in the function contract.

### 12. On-disk artifacts stale after in-memory mutations
Consolidation, renaming, and merging update data structures in memory
but can leave stale files on disk pointed at by `parquet_path`-style
fields. Every mutation that invalidates on-disk state must either
rewrite the file or null out the path so downstream code re-writes it.
Skipping the rewrite silently corrupts the output.

### 13. Preallocated `(H_total, T)` arrays even when output is streamed
Streaming the *writes* of a per-(H, T) output to disk does not
automatically relieve the in-memory companion. `calls = np.empty((H, T),
int8)` is 27 GB at T=25k and 42 GB at T=38k. Even though `np.empty`
doesn't touch pages, every `calls[bs:be] = ...` write faults in the
chunk's pages and they stay resident for the rest of the run. When the
array is logically write-once-then-read (like decode `calls`), back it
with `np.memmap` instead. Sequential writes flush dirty pages under
pressure; downstream consumers that already iterate in 50k-hap chunks
see good locality with no API change.

## Pattern recurrence policy

Every pattern on this list has recurred at least once in a new call site
after its first fix. Before landing any new feature that touches `calls`,
`max_post`, `geno`, or `results[*]`, grep the codebase for all existing
call sites of the same pattern and verify they are already guarded. The
cost of re-auditing is hours; the cost of a crashed biobank run is two
days.
