"""Subcommand entry points for the train/infer scatter shape.

Three subcommands:

``popout seed``
    Run seeding on one chromosome and write a portable seeding artifact
    (``<out>.seed.npz``) containing per-haplotype assignments and the
    initial allele frequencies. No EM is run.

``popout train``
    Run seeding + EM on one chromosome and write a portable model
    (``<out>.model.npz``). Decode + tracts are NOT written. Optionally
    consumes a pre-computed seeding artifact via ``--seed-input``.

``popout infer``
    Load a trained model and infer ancestry on one chromosome. Uses the
    saved ``leaf_labels`` to re-fit allele frequencies for this chrom's
    sites (does NOT re-cluster, unlike the legacy warm-start path).
    Writes per-chrom global TSV, tracts, and optionally a dense decode
    parquet.

Each subcommand is intentionally self-contained: it parses its own
arguments and reuses the seeding/EM/decode primitives directly. There is
no shared resume machinery (one chrom in, outputs out -- retry = re-run).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _common_input_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--vcf", help="Phased VCF/BCF file (indexed)")
    g.add_argument(
        "--pgen",
        help="Per-chromosome PGEN directory or single file prefix",
    )
    p.add_argument(
        "--map", default=None,
        help="HapMap-format genetic map (file or per-chrom directory). "
             "Auto-downloads from Beagle if omitted.",
    )
    p.add_argument(
        "--genome", choices=["GRCh38", "GRCh37", "GRCh36"], default="GRCh38",
    )
    p.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Restrict to these chromosomes. For seed/train/infer this "
             "MUST be exactly one chromosome.",
    )
    p.add_argument("--thin-cm", type=float, default=0.0)
    p.add_argument("--maf", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")


def _setup_logging(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("popout")


def _load_single_chrom(args) -> "ChromData":
    """Load exactly one chromosome of haplotype data.

    All three subcommands operate on a single chrom. Reject inputs that
    contain or are restricted to more than one.
    """
    from .fetch_map import resolve_map_dir
    from .gmap import load_genetic_map, load_genetic_map_per_chrom

    if args.chromosomes is not None and len(args.chromosomes) != 1:
        raise SystemExit(
            "seed/train/infer expect exactly one chromosome. "
            f"Got --chromosomes={args.chromosomes}."
        )

    map_resolved = resolve_map_dir(genome=args.genome, map_arg=args.map)
    map_path = Path(map_resolved)
    if map_path.is_dir():
        gmap = load_genetic_map_per_chrom(map_path)
    else:
        gmap = load_genetic_map(map_path)
    log.info("Loaded genetic map: %d chromosomes", len(gmap))

    if args.vcf:
        from .vcf_io import iter_chromosomes
        chrom_iter = iter_chromosomes(
            args.vcf, gmap,
            min_maf=args.maf,
            chromosomes=args.chromosomes,
        )
    else:
        from .pgen_io import iter_chromosomes as pgen_iter
        chrom_iter = pgen_iter(
            args.pgen, gmap,
            min_maf=args.maf,
            chromosomes=args.chromosomes,
            thin_cm=args.thin_cm,
        )

    chroms = list(chrom_iter)
    if len(chroms) != 1:
        raise SystemExit(
            "seed/train/infer expect exactly one chromosome in the input; "
            f"got {len(chroms)}. Use --chromosomes to restrict.",
        )
    return chroms[0]


def _build_seeding_mask(
    exclude_path: str,
    sample_names: list[str],
    n_haps: int,
) -> "np.ndarray":
    """Build a per-haplotype boolean mask from a sample-id exclusion TSV.

    Matches the cli.py implementation: True = keep, False = exclude.
    Both haplotypes of an excluded sample are masked out.
    """
    from .cli import load_seeding_exclusion_list

    exclude_set = load_seeding_exclusion_list(exclude_path)
    log.info(
        "Loaded %d exclusion samples for seeding from %s",
        len(exclude_set), exclude_path,
    )

    sample_set = {str(s) for s in sample_names}
    unknown = exclude_set - sample_set
    if unknown:
        log.warning(
            "%d sample IDs in --exclude-seeding-samples not found in input: "
            "%s%s",
            len(unknown), sorted(unknown)[:5],
            " ..." if len(unknown) > 5 else "",
        )

    seeding_mask = np.ones(n_haps, dtype=bool)
    n_excluded = 0
    for i, name in enumerate(sample_names):
        if str(name) in exclude_set:
            seeding_mask[2 * i] = False
            seeding_mask[2 * i + 1] = False
            n_excluded += 1
    H_kept = int(seeding_mask.sum())
    log.info(
        "Seeding on %d / %d haplotypes after excluding %d samples",
        H_kept, n_haps, n_excluded,
    )
    if H_kept < 1000:
        raise SystemExit(
            f"Only {H_kept} haplotypes remain after excluding {n_excluded} "
            "samples; minimum 1000 required for seeding.",
        )
    return seeding_mask


def _get_sample_names(args) -> list[str]:
    if args.vcf:
        import pysam
        vcf = pysam.VariantFile(args.vcf)
        names = list(vcf.header.samples)
        vcf.close()
        return names
    from .pgen_io import get_sample_names
    pgen_path = Path(args.pgen)
    if pgen_path.is_dir():
        psams = sorted(pgen_path.glob("*.psam"))
        if not psams:
            raise SystemExit(f"No .psam files in {pgen_path}")
        return get_sample_names(psams[0])
    psam = (
        pgen_path.with_suffix(".psam")
        if pgen_path.suffix != ".psam" else pgen_path
    )
    return get_sample_names(psam)


# ---------------------------------------------------------------------------
# popout seed
# ---------------------------------------------------------------------------

def _build_parser_seed() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="popout seed",
        description="Run seeding on one chromosome and write a seeding artifact.",
    )
    _common_input_args(p)
    p.add_argument("--out", required=True, help="Output prefix (writes <out>.seed.npz)")
    p.add_argument(
        "--seed-method", choices=["gmm", "recursive"], default="recursive",
    )
    p.add_argument("--n-ancestries", type=int, default=None)
    p.add_argument("--max-ancestries", type=int, default=20)
    p.add_argument(
        "--ancestry-detection",
        choices=["marchenko-pastur", "recursive", "eigenvalue-gap"],
        default="marchenko-pastur",
    )
    p.add_argument("--gen-since-admix", type=float, default=10.0)
    p.add_argument("--exclude-seeding-samples", default=None)
    # Recursive seeding params
    p.add_argument("--recursive-max-leaves", type=int, default=20)
    p.add_argument("--recursive-min-leaf-size", type=int, default=1000)
    p.add_argument("--recursive-min-cluster-size", type=int, default=1000)
    p.add_argument("--recursive-max-depth", type=int, default=6)
    p.add_argument("--recursive-merge-hellinger", type=float, default=0.008)
    return p


def cmd_seed(argv: list[str]) -> None:
    args = _build_parser_seed().parse_args(argv)
    if args.seed_method == "recursive" and args.n_ancestries is not None:
        raise SystemExit(
            "--seed-method recursive and --n-ancestries are incompatible.",
        )
    _setup_logging(args.verbose)

    chrom_data = _load_single_chrom(args)
    log.info(
        "Seeding chr%s: H=%d, T=%d, method=%s",
        chrom_data.chrom, chrom_data.n_haps, chrom_data.n_sites,
        args.seed_method,
    )

    import jax.numpy as jnp

    from .em import _build_seed_resp, init_model_soft
    from .output import write_seed
    from .recursive_seed import LeafInfo

    seeding_mask = None
    if args.exclude_seeding_samples is not None:
        if args.seed_method != "recursive":
            raise SystemExit(
                "--exclude-seeding-samples requires --seed-method recursive",
            )
        seeding_mask = _build_seeding_mask(
            args.exclude_seeding_samples,
            _get_sample_names(args),
            chrom_data.n_haps,
        )

    t0 = time.perf_counter()
    if args.seed_method == "recursive":
        from .recursive_seed import recursive_split_seed
        leaf_labels, leaf_info = recursive_split_seed(
            chrom_data.geno,
            chrom_data=chrom_data,
            gen_since_admix=args.gen_since_admix,
            rng_seed=args.seed,
            seeding_mask=seeding_mask,
            max_leaves=args.recursive_max_leaves,
            min_leaf_size=args.recursive_min_leaf_size,
            min_cluster_size=args.recursive_min_cluster_size,
            max_depth=args.recursive_max_depth,
            merge_hellinger_threshold=args.recursive_merge_hellinger,
        )
        K = len(leaf_info)
        seed_resp = _build_seed_resp(
            chrom_data.geno, leaf_labels, K, seeding_mask=seeding_mask,
        )
        if seeding_mask is not None:
            full = np.full(chrom_data.n_haps, -1, dtype=np.int32)
            full[np.where(seeding_mask)[0]] = leaf_labels
            leaf_labels_full = full
        else:
            leaf_labels_full = np.array(leaf_labels, dtype=np.int32)
        leaf_paths = np.array([li.path for li in leaf_info])
        leaf_meta = {
            "bic_scores": np.array(
                [li.bic_score for li in leaf_info], dtype=np.float32,
            ),
            "depths": np.array(
                [li.depth for li in leaf_info], dtype=np.int32,
            ),
            "n_haps": np.array(
                [li.n_haps for li in leaf_info], dtype=np.int32,
            ),
        }
    else:
        from .spectral import seed_ancestry_soft
        labels, resp, K, _proj = seed_ancestry_soft(
            chrom_data.geno,
            n_ancestries=args.n_ancestries,
            rng_seed=args.seed,
            detection_method=args.ancestry_detection,
            max_ancestries=args.max_ancestries,
        )
        seed_resp = resp
        leaf_labels_full = np.array(labels, dtype=np.int32)
        leaf_paths = np.array([f"L{i}" for i in range(K)])
        leaf_meta = {
            "n_haps": np.array(
                [int((leaf_labels_full == i).sum()) for i in range(K)],
                dtype=np.int32,
            ),
        }

    # Initial model (provides allele_freq for seed.npz)
    model = init_model_soft(
        chrom_data.geno, seed_resp, K, args.gen_since_admix,
        window_refine=(args.seed_method == "gmm"),
    )

    out_path = (
        args.out if args.out.endswith(".seed.npz")
        else f"{args.out}.seed.npz"
    )
    write_seed(
        out_path=out_path,
        seed_method=args.seed_method,
        n_ancestries=K,
        leaf_labels=leaf_labels_full,
        leaf_paths=leaf_paths,
        responsibilities=np.array(seed_resp, dtype=np.float32),
        allele_freq=np.array(model.allele_freq, dtype=np.float32),
        chrom_data=chrom_data,
        leaf_meta=leaf_meta,
        seed_kwargs={
            "seed_method": args.seed_method,
            "n_ancestries": args.n_ancestries,
            "max_ancestries": args.max_ancestries,
            "ancestry_detection": args.ancestry_detection,
            "gen_since_admix": args.gen_since_admix,
            "rng_seed": args.seed,
            "recursive_max_leaves": args.recursive_max_leaves,
            "recursive_min_leaf_size": args.recursive_min_leaf_size,
            "recursive_min_cluster_size": args.recursive_min_cluster_size,
            "recursive_max_depth": args.recursive_max_depth,
            "recursive_merge_hellinger": args.recursive_merge_hellinger,
        },
    )
    log.info(
        "popout seed done in %.1fs (K=%d, chrom=%s)",
        time.perf_counter() - t0, K, chrom_data.chrom,
    )


# ---------------------------------------------------------------------------
# popout train
# ---------------------------------------------------------------------------

def _build_parser_train() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="popout train",
        description="Train an ancestry model on one chromosome.",
    )
    _common_input_args(p)
    p.add_argument("--out", required=True, help="Output prefix")
    p.add_argument(
        "--seed-input", default=None,
        help="Seeding artifact (.seed.npz). If given, skip seeding.",
    )
    # Seeding (only used if --seed-input not given)
    p.add_argument(
        "--seed-method", choices=["gmm", "recursive"], default="recursive",
    )
    p.add_argument("--n-ancestries", type=int, default=None)
    p.add_argument("--max-ancestries", type=int, default=20)
    p.add_argument(
        "--ancestry-detection",
        choices=["marchenko-pastur", "recursive", "eigenvalue-gap"],
        default="marchenko-pastur",
    )
    p.add_argument("--exclude-seeding-samples", default=None)
    p.add_argument("--recursive-max-leaves", type=int, default=20)
    p.add_argument("--recursive-min-leaf-size", type=int, default=1000)
    p.add_argument("--recursive-min-cluster-size", type=int, default=1000)
    p.add_argument("--recursive-max-depth", type=int, default=6)
    p.add_argument("--recursive-merge-hellinger", type=float, default=0.008)
    # EM
    p.add_argument("--n-em-iter", type=int, default=20)
    p.add_argument("--gen-since-admix", type=float, default=10.0)
    p.add_argument(
        "--em-t-policy", choices=["hold", "gated", "every-iter"], default="gated",
    )
    p.add_argument("--freeze-anchors-iters", type=int, default=None)
    p.add_argument(
        "--held-out-init", choices=["uniform", "soft"], default="soft",
    )
    p.add_argument("--batch-size", type=int, default=None)
    # Block emissions / per-hap T
    p.add_argument(
        "--block-emissions", dest="block_emissions",
        action="store_true", default=True,
    )
    p.add_argument(
        "--no-block-emissions", dest="block_emissions", action="store_false",
    )
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--per-hap-T", dest="per_hap_T", action="store_true", default=False,
    )
    p.add_argument(
        "--no-per-hap-T", dest="per_hap_T", action="store_false",
    )
    p.add_argument("--n-T-buckets", type=int, default=20)
    # Decode (train decodes its own chrom too -- the EM-fitted allele_freq
    # IS the correct per-site freq for the train chrom; running a separate
    # infer scatter on the train chrom would refit it with a worse 1-iter
    # estimate).
    p.add_argument("--probs", action="store_true")
    p.add_argument("--write-dense-decode", action="store_true")
    p.add_argument("--ancestry-names", default=None)
    return p


def _resolve_freeze_anchors(value: Optional[int], seed_method: str) -> int:
    if value is not None:
        return value
    return 5 if seed_method == "recursive" else 0


def cmd_train(argv: list[str]) -> None:
    args = _build_parser_train().parse_args(argv)
    args.freeze_anchors_iters = _resolve_freeze_anchors(
        args.freeze_anchors_iters, args.seed_method,
    )
    _setup_logging(args.verbose)

    chrom_data = _load_single_chrom(args)
    log.info(
        "Training on chr%s: H=%d, T=%d",
        chrom_data.chrom, chrom_data.n_haps, chrom_data.n_sites,
    )

    import jax.numpy as jnp
    from dataclasses import replace

    from .datatypes import AncestryResult
    from .em import run_em
    from .output import read_seed, write_model

    seed_resp = None
    seed_method_used = args.seed_method
    leaf_labels_for_model = None
    leaf_paths_for_model = None
    K_override = args.n_ancestries

    if args.seed_input is not None:
        seed = read_seed(args.seed_input)
        if seed["n_haps"] != chrom_data.n_haps:
            raise SystemExit(
                f"seed.npz H={seed['n_haps']} != chrom H={chrom_data.n_haps}",
            )
        seed_resp = jnp.array(seed["responsibilities"])
        K_override = seed["n_ancestries"]
        seed_method_used = seed["seed_method"]
        leaf_labels_for_model = seed["leaf_labels"]
        leaf_paths_for_model = seed["leaf_paths"]
        log.info(
            "Loaded seed.npz: method=%s, K=%d", seed_method_used, K_override,
        )

    t0 = time.perf_counter()
    if args.seed_input is None and args.seed_method == "recursive":
        # Recursive seeding requires the pre-EM dispatch (run_em only does
        # GMM internally). Run it here and pass responsibilities through.
        from .em import _build_seed_resp
        from .recursive_seed import recursive_split_seed
        seeding_mask = None
        if args.exclude_seeding_samples is not None:
            seeding_mask = _build_seeding_mask(
                args.exclude_seeding_samples,
                _get_sample_names(args),
                chrom_data.n_haps,
            )
        leaf_labels, leaf_info = recursive_split_seed(
            chrom_data.geno,
            chrom_data=chrom_data,
            gen_since_admix=args.gen_since_admix,
            rng_seed=args.seed,
            seeding_mask=seeding_mask,
            max_leaves=args.recursive_max_leaves,
            min_leaf_size=args.recursive_min_leaf_size,
            min_cluster_size=args.recursive_min_cluster_size,
            max_depth=args.recursive_max_depth,
            merge_hellinger_threshold=args.recursive_merge_hellinger,
        )
        K_override = len(leaf_info)
        seed_resp = _build_seed_resp(
            chrom_data.geno, leaf_labels, K_override,
            seeding_mask=seeding_mask, held_out_init=args.held_out_init,
        )
        if seeding_mask is not None:
            full = np.full(chrom_data.n_haps, -1, dtype=np.int32)
            full[np.where(seeding_mask)[0]] = leaf_labels
            leaf_labels_for_model = full
        else:
            leaf_labels_for_model = np.array(leaf_labels, dtype=np.int32)
        leaf_paths_for_model = np.array([li.path for li in leaf_info])

    # Decode the train chrom in this same task. The EM-fitted allele_freq
    # IS the right per-site freq for the train chrom; running the infer
    # scatter on it would refit with a worse 1-iter estimate. Parquet
    # routing mirrors cmd_infer (temp subdir for --probs, first-class
    # path for --write-dense-decode).
    decode_pq = None
    parquet_is_temp = False
    if args.write_dense_decode:
        decode_pq = f"{args.out}.chr{chrom_data.chrom}.decode.parquet"
    elif args.probs:
        from pathlib import Path as _Path
        tmpdir = _Path(f"{args.out}.decode_tmp")
        tmpdir.mkdir(parents=True, exist_ok=True)
        decode_pq = str(tmpdir / f"chr{chrom_data.chrom}.decode.parquet")
        parquet_is_temp = True

    result = run_em(
        chrom_data,
        n_ancestries=K_override,
        n_em_iter=args.n_em_iter,
        gen_since_admix=args.gen_since_admix,
        batch_size=args.batch_size,
        rng_seed=args.seed,
        per_hap_T=args.per_hap_T,
        n_T_buckets=args.n_T_buckets,
        use_block_emissions=args.block_emissions,
        block_size=args.block_size,
        detection_method=args.ancestry_detection,
        max_ancestries=args.max_ancestries,
        seed_responsibilities=seed_resp,
        freeze_anchors_iters=args.freeze_anchors_iters,
        em_t_policy=args.em_t_policy,
        write_dense_decode=(args.write_dense_decode or args.probs),
        decode_parquet_path=decode_pq,
    )

    # For GMM no-seed-input path, derive labels/paths from final responsibilities
    if leaf_labels_for_model is None:
        # GMM seeding inside run_em: we don't have access to its labels;
        # recompute via argmax over allele_freq emission for the trained
        # model on the training chrom.
        import jax.numpy as jnp
        log_emit = result.model.log_emission(jnp.array(chrom_data.geno))
        marg = log_emit.sum(axis=1)  # (H, A) sum over sites
        leaf_labels_for_model = np.array(
            jnp.argmax(marg, axis=1), dtype=np.int32,
        )
        leaf_paths_for_model = np.array(
            [f"L{i}" for i in range(result.model.n_ancestries)],
        )

    model_with_id = replace(
        result.model,
        seed_method=seed_method_used,
        leaf_labels=np.array(leaf_labels_for_model, dtype=np.int32),
        leaf_paths=np.array(leaf_paths_for_model),
    )
    result = AncestryResult(
        calls=result.calls, model=model_with_id, chrom=result.chrom,
        decode=result.decode, posteriors=result.posteriors,
        spectral=result.spectral,
    )

    ancestry_names = None
    if args.ancestry_names is not None:
        from .names import parse_ancestry_names
        ancestry_names = parse_ancestry_names(
            args.ancestry_names, result.model.n_ancestries,
        )

    write_model(
        result, f"{args.out}.model",
        chrom_data=chrom_data, ancestry_names=ancestry_names,
    )

    # Per-chrom outputs for the train chrom. Same shape as cmd_infer.
    from .output import write_global_ancestry, write_ancestry_tracts
    sample_names = _get_sample_names(args)
    n_samples = len(sample_names)
    write_global_ancestry(
        [result], n_samples, sample_names, f"{args.out}.global.tsv",
    )
    write_ancestry_tracts(
        [result], [chrom_data], n_samples, sample_names,
        f"{args.out}.tracts.tsv.gz",
        write_posteriors=args.probs or args.write_dense_decode,
    )

    if parquet_is_temp and decode_pq is not None:
        from pathlib import Path as _Path
        _Path(decode_pq).unlink(missing_ok=True)
        try:
            _Path(decode_pq).parent.rmdir()
        except OSError:
            pass
        log.info("Removed temp decode parquet at %s", decode_pq)

    # Minimal summary.json
    import json
    summary = {
        "subcommand": "train",
        "chrom": str(chrom_data.chrom),
        "n_haps": int(chrom_data.n_haps),
        "n_sites": int(chrom_data.n_sites),
        "n_ancestries": int(result.model.n_ancestries),
        "seed_method": str(seed_method_used),
        "n_em_iter": int(args.n_em_iter),
        "gen_since_admix": float(result.model.gen_since_admix),
        "use_block_emissions": bool(args.block_emissions),
        "block_size": int(args.block_size),
        "per_hap_T": bool(args.per_hap_T),
        "wall_s": time.perf_counter() - t0,
    }
    with open(f"{args.out}.summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        "popout train done in %.1fs (K=%d)",
        summary["wall_s"], result.model.n_ancestries,
    )


# ---------------------------------------------------------------------------
# popout infer
# ---------------------------------------------------------------------------

def _build_parser_infer() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="popout infer",
        description="Infer ancestry on one chromosome using a trained model.",
    )
    _common_input_args(p)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--model", required=True,
        help="Path to .model.npz produced by `popout train`",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--probs", action="store_true")
    p.add_argument("--write-dense-decode", action="store_true")
    p.add_argument("--ancestry-names", default=None)
    return p


def cmd_infer(argv: list[str]) -> None:
    args = _build_parser_infer().parse_args(argv)
    _setup_logging(args.verbose)

    chrom_data = _load_single_chrom(args)
    log.info(
        "Inferring chr%s: H=%d, T=%d", chrom_data.chrom,
        chrom_data.n_haps, chrom_data.n_sites,
    )

    import jax.numpy as jnp
    from dataclasses import replace

    from .datatypes import AncestryResult
    from .output import (
        read_model, write_global_ancestry, write_ancestry_tracts,
    )

    t0 = time.perf_counter()
    loaded = read_model(args.model)
    trained = loaded["model"]
    if trained.leaf_labels is None:
        raise SystemExit(
            "Model NPZ is missing leaf_labels; cannot run popout infer.",
        )
    if trained.leaf_labels.shape[0] != chrom_data.n_haps:
        raise SystemExit(
            f"Model leaf_labels H={trained.leaf_labels.shape[0]} != "
            f"chrom H={chrom_data.n_haps}. The haplotype set must match the "
            "training cohort.",
        )

    K = trained.n_ancestries
    # Validation hooks (Task #7 surface): check model self-consistency
    if trained.allele_freq.shape[0] != K:
        raise SystemExit(
            f"Model allele_freq has {trained.allele_freq.shape[0]} rows "
            f"but n_ancestries={K}",
        )
    log.info(
        "Loaded model: K=%d, method=%s, train_chrom=%s, version=%s",
        K, trained.seed_method, loaded["train_chrom"], loaded["popout_version"],
    )

    # Build one-hot responsibilities from training leaf_labels
    H = chrom_data.n_haps
    valid = trained.leaf_labels >= 0
    resp = np.zeros((H, K), dtype=np.float32)
    idx_valid = np.where(valid)[0]
    resp[idx_valid, trained.leaf_labels[idx_valid]] = 1.0
    # Excluded haps (label -1) get uniform priors
    excluded = np.where(~valid)[0]
    if len(excluded) > 0:
        resp[excluded, :] = 1.0 / K
    resp_j = jnp.array(resp)

    # Detect the training-time emission/T modes from the loaded model. The
    # trained model carries block_data / bucket_assignments when those
    # modes were used; their PRESENCE on the loaded NPZ is the source of
    # truth. We rebuild block_data fresh for THIS chrom's sites (the
    # trained block_data was sized to the train chrom), but reuse the
    # training block_size and per-hap-T config.
    use_block_em = trained.block_data is not None
    train_block_size = (
        trained.block_data.block_size if use_block_em else 8
    )
    train_per_hap_T = trained.bucket_assignments is not None
    log.info(
        "Inference modes: block_emissions=%s (block_size=%d), per_hap_T=%s",
        use_block_em, train_block_size, train_per_hap_T,
    )

    # Decode parquet routing.
    #
    #   --write-dense-decode : parquet written at <out>.chr<N>.decode.parquet
    #                          (first-class WDL output, glob-visible).
    #   --probs only         : parquet written under <out>.decode_tmp/
    #                          so the WDL glob does NOT match, deleted
    #                          after tract write. mean_posterior streamed
    #                          into tracts.tsv.gz at near-zero cost.
    #   neither              : no parquet; tracts have no mean_posterior.
    decode_pq = None
    parquet_is_temp = False
    if args.write_dense_decode:
        decode_pq = f"{args.out}.chr{chrom_data.chrom}.decode.parquet"
    elif args.probs:
        from pathlib import Path as _Path
        tmpdir = _Path(f"{args.out}.decode_tmp")
        tmpdir.mkdir(parents=True, exist_ok=True)
        decode_pq = str(tmpdir / f"chr{chrom_data.chrom}.decode.parquet")
        parquet_is_temp = True

    # One EM iteration + decode via run_em. Passing seed_responsibilities
    # built from the trained leaf_labels bypasses run_em's own seeding
    # stage. With use_block_emissions=True the function pack_blocks() on
    # this chrom's geno and init_pattern_freq() on the per-chrom
    # allele_freq, so FB and decode both run via the block code paths
    # (workspace O(H, n_blocks, max_patterns, A) instead of single-site
    # O(H, T, A) which OOMs at biobank scale).
    #
    # em_t_policy='hold' freezes T (we don't refit at infer time).
    # freeze_anchors_iters=0 disables anchor-freeze blending (only matters
    # over multiple iters anyway).
    from .em import run_em
    # force_host_geno=True: the block-aware E-step in run_em batches
    # host→device per chunk anyway, and fits_on_device cannot see the
    # JAX preallocated pool. On large chroms (chr3: 21.7 GB int8 geno)
    # the lazy `jnp.array(geno_np)` transfer triggered by the first
    # slice of a "device-resident" geno OOMs the device. Keep geno on
    # host throughout infer; the E-step transfers per chunk.
    result = run_em(
        chrom_data,
        n_ancestries=K,
        n_em_iter=1,
        gen_since_admix=trained.gen_since_admix,
        batch_size=args.batch_size,
        rng_seed=args.seed,
        seed_responsibilities=resp_j,
        use_block_emissions=use_block_em,
        block_size=train_block_size,
        per_hap_T=train_per_hap_T,
        em_t_policy="hold",
        freeze_anchors_iters=0,
        write_dense_decode=(args.write_dense_decode or args.probs),
        decode_parquet_path=decode_pq,
        force_host_geno=True,
    )

    # Use the trained mu (more reliable than 1-iter refit). allele_freq
    # is the per-chrom-refit value from run_em. Attach the trained
    # seeding identity so the result.model is consistent with the
    # trained model identity downstream.
    result_model = replace(
        result.model,
        mu=trained.mu,
        seed_method=trained.seed_method,
        leaf_labels=trained.leaf_labels,
        leaf_paths=trained.leaf_paths,
    )
    result = AncestryResult(
        calls=result.calls, model=result_model, chrom=result.chrom,
        decode=result.decode, posteriors=result.posteriors,
        spectral=result.spectral,
    )

    # Outputs
    sample_names = _get_sample_names(args)
    n_samples = len(sample_names)
    ancestry_names = None
    if args.ancestry_names is not None:
        from .names import parse_ancestry_names
        ancestry_names = parse_ancestry_names(args.ancestry_names, K)

    write_global_ancestry(
        [result], n_samples, sample_names, f"{args.out}.global.tsv",
    )
    write_ancestry_tracts(
        [result], [chrom_data], n_samples, sample_names,
        f"{args.out}.tracts.tsv.gz",
        write_posteriors=args.probs or args.write_dense_decode,
    )

    if parquet_is_temp and decode_pq is not None:
        from pathlib import Path as _Path
        _Path(decode_pq).unlink(missing_ok=True)
        try:
            _Path(decode_pq).parent.rmdir()
        except OSError:
            pass
        log.info("Removed temp decode parquet at %s", decode_pq)

    import json
    summary = {
        "subcommand": "infer",
        "chrom": str(chrom_data.chrom),
        "n_haps": int(chrom_data.n_haps),
        "n_sites": int(chrom_data.n_sites),
        "n_ancestries": int(K),
        "model_train_chrom": loaded["train_chrom"],
        "model_seed_method": trained.seed_method,
        "wall_s": time.perf_counter() - t0,
    }
    with open(f"{args.out}.summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        "popout infer done in %.1fs (chr%s, K=%d)",
        summary["wall_s"], chrom_data.chrom, K,
    )
