"""Write ancestry results to output files.

Produces:
  - Ancestry tract BED/TSV (compact interval-based local ancestry)
  - Global ancestry proportions TSV
  - Model parameters file
  - Per-chromosome dense decode arrays (for 'popout convert')
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .datatypes import AncestryResult, ChromData

log = logging.getLogger(__name__)


def _load_max_post_from_parquet(path: str, n_haps: int) -> np.ndarray:
    """Load max_post from a decode parquet file into a single array.

    Only safe at small scale. At biobank scale (H*T*2 > 10 GB), use
    ``_iter_max_post_groups`` to stream per row group instead.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = pf.schema_arrow.metadata or {}
    T = int(meta[b"T"])
    H = pf.metadata.num_rows
    assert H == n_haps, f"Parquet has {H} rows but expected {n_haps}"

    alloc_bytes = H * T * 2
    if alloc_bytes > 10 * 1024**3:
        raise RuntimeError(
            f"_load_max_post_from_parquet would allocate {alloc_bytes / 1e9:.0f} GB "
            f"({H} haps × {T} sites × fp16). Use the streaming row-group "
            "path in write_ancestry_tracts instead."
        )

    max_post = np.empty((H, T), dtype=np.float16)
    row = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=["max_post"])
        mp_col = rg.column("max_post")
        for chunk in mp_col.chunks:
            n_rows = len(chunk)
            mp_buf = chunk.buffers()[2]
            mp_flat = np.frombuffer(mp_buf, dtype=np.float16, count=n_rows * T)
            max_post[row:row + n_rows] = mp_flat.reshape(n_rows, T)
            row += n_rows
    assert row == H, f"read {row} rows from parquet, expected {H}"
    log.info("Loaded max_post from %s for tract posteriors", path)
    return max_post


def _iter_max_post_groups(path: str):
    """Yield ``(start_row, end_row, mp_chunk)`` per parquet row group.

    Peak transient: one row group's worth of max_post (~2.5 GB at 50k
    haps × 25k sites × fp16).  The previous group is freed before the
    next is read.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    T = int(pf.schema_arrow.metadata[b"T"])
    row = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=["max_post"])
        mp_col = rg.column("max_post")
        parts = []
        for chunk in mp_col.chunks:
            n = len(chunk)
            buf = chunk.buffers()[2]
            arr = np.frombuffer(buf, dtype=np.float16, count=n * T)
            parts.append(arr.reshape(n, T).copy())
        mp_block = np.concatenate(parts) if len(parts) > 1 else parts[0]
        n_block = mp_block.shape[0]
        yield row, row + n_block, mp_block
        row += n_block


def write_ancestry_tracts(
    results: list[AncestryResult],
    chrom_data_list: list[ChromData],
    n_samples: int,
    sample_names: list[str],
    out_path: str,
    write_posteriors: bool = False,
    stats=None,
) -> None:
    """Write local ancestry as tract intervals (BED-like TSV).

    Converts per-site hard ancestry calls into contiguous tracts where
    ancestry is constant.  One row per tract per haplotype.  This is
    orders of magnitude more compact than per-site output for large
    cohorts.

    Output columns:
        chrom, start_bp, end_bp, sample, haplotype (0/1), ancestry,
        n_sites, [mean_posterior if --probs]

    Compatible with downstream analysis in R/pandas and similar to
    RFMix .msp.tsv format.

    Parameters
    ----------
    results : list of AncestryResult, one per chromosome
    chrom_data_list : list of ChromData, matching results
    n_samples : number of diploid samples
    sample_names : sample IDs
    out_path : output file path
    write_posteriors : if True, include mean posterior per tract
    """
    import gzip

    compress = out_path.endswith(".gz")
    opener = gzip.open if compress else open

    n_haps = 2 * n_samples
    n_tracts = 0
    # Accumulate tract lengths per ancestry for stats
    tract_lengths_by_anc: dict[int, list[int]] = {}
    confidence_sum = 0.0
    confidence_count = 0

    with opener(out_path, "wt") as f:
        # Header
        cols = ["#chrom", "start_bp", "end_bp", "sample", "haplotype",
                "ancestry", "n_sites"]
        if write_posteriors:
            cols.append("mean_posterior")
        f.write("\t".join(cols) + "\n")

        from ._memcheck import check_no_copy

        for result, cdata in zip(results, chrom_data_list):
            calls = np.asarray(result.calls)  # (n_haps, n_sites) int8
            check_no_copy("write_ancestry_tracts:calls", result.calls, calls)
            pos_bp = cdata.pos_bp           # (n_sites,) int64
            chrom = cdata.chrom
            n_sites = cdata.n_sites

            if n_sites == 0:
                continue

            # Resolve max_post for mean_posterior column.
            # Within a tract where hap_calls[s..e] == anc, by construction
            # hap_calls[t] = argmax_a posteriors[h,t,a], so
            # posteriors[h,t,anc] = max_a posteriors[h,t,a] = max_post[h,t].
            # Therefore posteriors[hi,s:e+1,anc].mean() == max_post[hi,s:e+1].mean().
            max_post = None
            parquet_path = None
            if write_posteriors:
                if result.decode is not None and result.decode.max_post is not None:
                    max_post = result.decode.max_post          # (H, T) float16
                elif (result.decode is not None
                      and result.decode.parquet_path is not None):
                    parquet_path = result.decode.parquet_path
                elif result.posteriors is not None:
                    # Legacy fallback for test fixtures that still populate posteriors.
                    max_post = np.asarray(result.posteriors).max(axis=2)  # (H, T)
                else:
                    log.warning(
                        "--probs requested but chrom %s has no max_post or posteriors; "
                        "mean_posterior column will be blank for this chromosome",
                        result.chrom,
                    )

            # Per-haplotype tract writer (shared by both paths)
            def _write_hap_tracts(hi, mp_row):
                """Write tracts for one haplotype. mp_row is (T,) or None."""
                nonlocal n_tracts
                si = hi // 2
                hap = hi % 2
                sample = sample_names[si]
                hap_calls = calls[hi]

                switches = np.where(hap_calls[1:] != hap_calls[:-1])[0] + 1
                starts = np.concatenate([[0], switches])
                ends = np.concatenate([switches - 1, [n_sites - 1]])

                for k in range(len(starts)):
                    s, e = int(starts[k]), int(ends[k])
                    anc = int(hap_calls[s])
                    n_sites_tract = e - s + 1
                    line = f"{chrom}\t{pos_bp[s]}\t{pos_bp[e]}\t{sample}\t{hap}\t{anc}\t{n_sites_tract}"
                    if write_posteriors:
                        if mp_row is not None:
                            mean_post = float(mp_row[s:e + 1].mean())
                            line += f"\t{mean_post:.4f}"
                        else:
                            line += "\t."
                    f.write(line + "\n")
                    n_tracts += 1
                    tract_lengths_by_anc.setdefault(anc, []).append(n_sites_tract)

            if parquet_path is not None:
                # Stream max_post per row group (~2.5 GB each) instead of
                # loading the full (H, T) fp16 array (54 GB at AoU scale).
                log.info("Streaming max_post from %s for tract posteriors",
                         parquet_path)
                for rg_start, rg_end, mp_chunk in _iter_max_post_groups(parquet_path):
                    for hi in range(rg_start, rg_end):
                        _write_hap_tracts(hi, mp_chunk[hi - rg_start])
                    confidence_sum += float(mp_chunk.sum(dtype=np.float32))
                    confidence_count += mp_chunk.size
            else:
                for hi in range(n_haps):
                    mp_row = max_post[hi] if max_post is not None else None
                    _write_hap_tracts(hi, mp_row)
                if max_post is not None:
                    confidence_sum += float(max_post.sum(dtype=np.float32))
                    confidence_count += max_post.size

    log.info("Wrote %d ancestry tracts to %s", n_tracts, out_path)

    # Emit stats
    if stats is not None:
        stats.emit("output/n_tracts", n_tracts)
        tract_summary = {}
        for anc, lengths in sorted(tract_lengths_by_anc.items()):
            arr = np.array(lengths)
            tract_summary[str(anc)] = {
                "count": len(arr),
                "mean_sites": round(float(arr.mean()), 1),
                "median_sites": int(np.median(arr)),
                "p5_sites": int(np.percentile(arr, 5)),
                "p95_sites": int(np.percentile(arr, 95)),
            }
        stats.emit("output/tract_stats_by_ancestry", tract_summary)
        if confidence_count > 0:
            stats.emit("output/mean_posterior_confidence",
                       round(confidence_sum / confidence_count, 4))


def write_global_ancestry(
    results: list[AncestryResult],
    n_samples: int,
    sample_names: list[str],
    out_path: str,
    stats=None,
) -> None:
    """Write per-sample global ancestry proportions to TSV.

    Global ancestry = mean posterior across all sites and chromosomes.
    """
    A = results[0].model.n_ancestries

    # Accumulate posteriors across chromosomes
    # posteriors are (n_haps, n_sites, A) — need per-sample (diploid) averages
    sample_sums = np.zeros((n_samples, A))
    total_sites = 0

    for result in results:
        # Use pre-computed global_sums from DecodeResult when available
        if result.decode is not None and result.decode.global_sums is not None:
            hap_sums = result.decode.global_sums  # (n_haps, A) float64
            T = result.calls.shape[1]
        elif result.posteriors is not None:
            post_shape = getattr(result.posteriors, "shape", None)
            if post_shape is not None and post_shape[0] > 100_000:
                raise RuntimeError(
                    f"write_global_ancestry posteriors fallback triggered with "
                    f"{post_shape[0]} haplotypes — would allocate "
                    f"{post_shape[0] * post_shape[1] * post_shape[2] * 4 / 1e9:.0f} GB. "
                    "At biobank scale, decode.global_sums must be populated. "
                    "Check forward_backward_decode's compute_max_post path."
                )
            gamma = np.array(result.posteriors)  # (n_haps, T, A)
            T = gamma.shape[1]
            hap_sums = gamma.sum(axis=1)  # (n_haps, A)
        else:
            log.warning("No posteriors or decode for chrom %s, skipping", result.chrom)
            continue
        # Average paired haplotypes: even indices = hap1, odd = hap2
        sample_sums += (hap_sums[0::2] + hap_sums[1::2]) / 2
        total_sites += T

    sample_props = sample_sums / total_sites

    with open(out_path, "w") as f:
        header = "sample\t" + "\t".join(f"ancestry_{a}" for a in range(A))
        f.write(header + "\n")
        for si, name in enumerate(sample_names):
            vals = "\t".join(f"{v:.4f}" for v in sample_props[si])
            f.write(f"{name}\t{vals}\n")

    log.info("Wrote global ancestry to %s", out_path)

    if stats is not None:
        mean_props = sample_props.mean(axis=0).tolist()
        stats.emit("output/genome_wide_ancestry_proportions", mean_props)


def write_model(
    result: AncestryResult,
    out_path: str,
    chrom_data: ChromData | None = None,
    ancestry_names: list[str] | None = None,
) -> None:
    """Write model parameters to a human-readable file.

    Also saves a companion ``.npz`` archive with full-precision arrays
    (allele frequencies, mu) for potential re-inference.
    """
    model = result.model
    with open(out_path, "w") as f:
        f.write(f"n_ancestries\t{model.n_ancestries}\n")
        f.write(f"gen_since_admix\t{model.gen_since_admix:.2f}\n")
        f.write(f"mu\t{','.join(f'{x:.4f}' for x in np.array(model.mu))}\n")

    save_dict = dict(
        allele_freq=np.array(model.allele_freq),
        mu=np.array(model.mu),
        n_ancestries=np.array(model.n_ancestries),
        gen_since_admix=np.array(model.gen_since_admix),
    )
    if chrom_data is not None:
        save_dict["pos_bp"] = np.array(chrom_data.pos_bp)
        save_dict["pos_cm"] = np.array(chrom_data.pos_cm)
        save_dict["chrom"] = np.array(chrom_data.chrom)
    if getattr(model, "gen_per_hap", None) is not None:
        save_dict["gen_per_hap"] = np.array(model.gen_per_hap)
    if getattr(model, "bucket_centers", None) is not None:
        save_dict["bucket_centers"] = np.array(model.bucket_centers)
    if ancestry_names is not None:
        if len(ancestry_names) != model.n_ancestries:
            raise ValueError(
                f"ancestry_names has {len(ancestry_names)} entries but "
                f"model has {model.n_ancestries} ancestries"
            )
        save_dict["ancestry_names"] = np.array(ancestry_names, dtype=object)
    np.savez_compressed(f"{out_path}.npz", **save_dict)
    log.info("Wrote model to %s (+ .npz)", out_path)


class DecodeParquetWriter:
    """Streaming writer for decode parquet files.

    Opens a ParquetWriter on enter and writes chunks via write_batch.
    Used by the decode loop in em.py to stream max_post to disk
    instead of holding a full (H, T) float16 array in memory.
    """

    def __init__(
        self,
        path: str,
        T: int,
        K: int,
        chrom: str,
        pos_bp: np.ndarray,
        include_max_post: bool = True,
    ):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._path = path
        self._T = T
        self._include_max_post = include_max_post

        pos_bp_c = np.ascontiguousarray(np.asarray(pos_bp, dtype=np.int64))
        metadata = {
            b"T": str(T).encode("ascii"),
            b"K": str(K).encode("ascii"),
            b"chrom": str(chrom).encode("ascii"),
            b"calls_dtype": b"uint8",
            b"pos_bp": pos_bp_c.tobytes(),
        }
        if include_max_post:
            metadata[b"max_post_dtype"] = b"float16"

        schema_fields = [("calls", pa.large_binary())]
        if include_max_post:
            schema_fields.append(("max_post", pa.large_binary()))
        self._schema = pa.schema(schema_fields, metadata=metadata)

        self._writer = pq.ParquetWriter(
            path, self._schema,
            compression="zstd", compression_level=1,
            use_dictionary=False,
        )

    def write_batch(
        self,
        calls_chunk: np.ndarray,
        max_post_chunk: np.ndarray | None = None,
    ) -> None:
        """Write one chunk of calls (and optionally max_post) as a row group."""
        import pyarrow as pa

        chunk_H = calls_chunk.shape[0]
        T = self._T

        # calls column
        c_arr = np.ascontiguousarray(calls_chunk.view(np.uint8))
        c_offsets = np.arange(0, chunk_H * T + 1, T, dtype=np.int64)
        c_col = pa.Array.from_buffers(
            pa.large_binary(), chunk_H,
            [None, pa.py_buffer(c_offsets), pa.py_buffer(c_arr)],
        )
        batch_cols = {"calls": c_col}

        if self._include_max_post and max_post_chunk is not None:
            mp_arr = np.ascontiguousarray(max_post_chunk.astype(np.float16))
            mp_offsets = np.arange(
                0, chunk_H * T * 2 + 1, T * 2, dtype=np.int64,
            )
            mp_col = pa.Array.from_buffers(
                pa.large_binary(), chunk_H,
                [None, pa.py_buffer(mp_offsets), pa.py_buffer(mp_arr)],
            )
            batch_cols["max_post"] = mp_col

        batch = pa.record_batch(batch_cols, schema=self._schema)
        self._writer.write_batch(batch)

    def close(self) -> None:
        self._writer.close()
        log.info("Wrote streaming decode parquet to %s", self._path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def write_decode_parquet(
    result: AncestryResult,
    chrom_data: ChromData,
    out_path: str,
    include_max_post: bool = True,
    hap_chunk: int = 50_000,
) -> None:
    """Write per-chromosome dense decode arrays as a Parquet file.

    Uses binary-blob-per-haplotype column layout with ZSTD-1 compression.
    On-disk size matches np.savez_compressed; write is ~14x faster at
    biobank scale because ZSTD releases the GIL and Parquet compresses
    columns in parallel.

    Streams in ``hap_chunk``-sized row groups so peak transient memory is
    O(hap_chunk * T * 2) rather than O(H * T).  The only full-H arrays
    held are the input ``result.calls`` and ``result.decode.max_post``
    (which already exist as part of the result).

    File contents:
        Column ``calls``    : large_binary, H rows x T bytes (uint8)
        Column ``max_post`` : large_binary, H rows x 2T bytes (float16)  [optional]
    Metadata: T, K, chrom, dtypes, pos_bp serialized as int64 bytes.

    Uses pa.large_binary() (int64 offsets) because H*T exceeds 2^31 bytes
    at biobank scale (e.g. H=1.07M, T=9471 -> 10.1 GB).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ._memcheck import check_no_copy

    # Zero-copy uint8 view of int8 calls — same byte width, no allocation
    calls_arr = np.asarray(result.calls)
    if calls_arr.dtype != np.uint8:
        calls_arr = calls_arr.view(np.uint8)
    calls_arr = np.ascontiguousarray(calls_arr)
    check_no_copy("write_decode_parquet:calls", result.calls, calls_arr)
    H, T = calls_arr.shape
    K = int(result.model.n_ancestries)
    pos_bp = np.ascontiguousarray(np.asarray(chrom_data.pos_bp, dtype=np.int64))
    assert pos_bp.shape == (T,), f"pos_bp length {pos_bp.shape[0]} != T={T}"

    has_mp = (include_max_post
              and result.decode is not None
              and result.decode.max_post is not None)

    metadata = {
        b"T": str(T).encode("ascii"),
        b"K": str(K).encode("ascii"),
        b"chrom": str(chrom_data.chrom).encode("ascii"),
        b"calls_dtype": b"uint8",
        b"pos_bp": pos_bp.tobytes(),
    }
    if has_mp:
        metadata[b"max_post_dtype"] = b"float16"

    schema_fields = [("calls", pa.large_binary())]
    if has_mp:
        schema_fields.append(("max_post", pa.large_binary()))
    schema = pa.schema(schema_fields, metadata=metadata)

    with pq.ParquetWriter(
        out_path, schema,
        compression="zstd", compression_level=1,
        use_dictionary=False,
    ) as writer:
        for cs in range(0, H, hap_chunk):
            ce = min(cs + hap_chunk, H)
            chunk_H = ce - cs

            # calls column — zero-copy slice of the contiguous view
            c_chunk = calls_arr[cs:ce]
            c_offsets = np.arange(0, chunk_H * T + 1, T, dtype=np.int64)
            c_col = pa.Array.from_buffers(
                pa.large_binary(), chunk_H,
                [None, pa.py_buffer(c_offsets), pa.py_buffer(c_chunk)],
            )
            batch_cols = {"calls": c_col}

            if has_mp:
                mp_chunk = np.ascontiguousarray(
                    result.decode.max_post[cs:ce].astype(np.float16)
                )
                mp_offsets = np.arange(
                    0, chunk_H * T * 2 + 1, T * 2, dtype=np.int64,
                )
                mp_col = pa.Array.from_buffers(
                    pa.large_binary(), chunk_H,
                    [None, pa.py_buffer(mp_offsets), pa.py_buffer(mp_chunk)],
                )
                batch_cols["max_post"] = mp_col

            batch = pa.record_batch(batch_cols, schema=schema)
            writer.write_batch(batch)

    log.info("Wrote decode parquet to %s", out_path)


def read_decode_parquet(path: str) -> dict:
    """Read a decode parquet file.

    Streams row groups so peak memory is the preallocated output arrays
    plus one row group's decompressed buffers — not the full file twice.

    Returns dict with keys:
        calls    : (H, T) uint8
        max_post : (H, T) float16  [present only if the file has it]
        pos_bp   : (T,)  int64
        chrom    : str
        T, K     : int
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = pf.schema_arrow.metadata or {}
    T = int(meta[b"T"])
    K = int(meta[b"K"])
    chrom = meta[b"chrom"].decode("ascii")
    pos_bp = np.frombuffer(meta[b"pos_bp"], dtype=np.int64).copy()
    assert pos_bp.shape == (T,)

    H = pf.metadata.num_rows
    has_mp = "max_post" in pf.schema_arrow.names

    # Preallocate output arrays — one O(H·T) allocation each
    calls = np.empty((H, T), dtype=np.uint8)
    max_post = np.empty((H, T), dtype=np.float16) if has_mp else None

    row = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        c_col = rg.column("calls")

        if has_mp:
            mp_col = rg.column("max_post")
            assert len(mp_col.chunks) == len(c_col.chunks), (
                f"Chunk count mismatch in row group {rg_idx}: "
                f"calls has {len(c_col.chunks)}, max_post has "
                f"{len(mp_col.chunks)}"
            )

        for chunk_idx, chunk in enumerate(c_col.chunks):
            n_rows = len(chunk)
            # buffers(): [validity, offsets, data] for large_binary
            data_buf = chunk.buffers()[2]
            flat = np.frombuffer(data_buf, dtype=np.uint8, count=n_rows * T)
            calls[row:row + n_rows] = flat.reshape(n_rows, T)

            if has_mp:
                mp_chunk = mp_col.chunks[chunk_idx]
                mp_buf = mp_chunk.buffers()[2]
                mp_flat = np.frombuffer(
                    mp_buf, dtype=np.float16, count=n_rows * T,
                )
                max_post[row:row + n_rows] = mp_flat.reshape(n_rows, T)

            row += n_rows

    assert row == H, f"read {row} rows, expected {H}"

    out = {
        "calls": calls,
        "pos_bp": pos_bp,
        "chrom": chrom,
        "T": T,
        "K": K,
    }
    if has_mp:
        out["max_post"] = max_post

    return out
