# Biobank-scale memory notes

Tracks known places where tensor shapes proportional to H (haplotype
count, 500k–1M+) must be batched rather than materialized whole.

Last audited: 2026-04-22.

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
| `em.py` ancestry proportion logging | `(result.calls == a).mean()` creates 10 GB bool × 20 ancestries | Single `np.bincount` pass |

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
