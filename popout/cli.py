"""Command-line interface for popout.

Usage:
    popout --vcf input.vcf.gz --map genetic_map.txt --out output_prefix

    popout --pgen /path/to/pgen_dir/ \\
           --map plink.GRCh38.map \\
           --out results/cohort \\
           --thin-cm 0.02 \\
           --n-ancestries 6 \\
           --batch-size 100000

    popout report --stats results/cohort.summary.json --out report/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from . import __version__
    print(f"popout {__version__}", file=sys.stderr)

    # Dispatch subcommands before parsing main args
    raw_args = argv if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] == "report":
        from .report import report_main
        report_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "fetch-map":
        from .fetch_map import fetch_map_main
        fetch_map_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "viz":
        from .viz.gallery import viz_main
        viz_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "label":
        from .label import label_main
        label_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "fetch-ref":
        from .fetch_ref import fetch_ref_main
        fetch_ref_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "build-ref":
        from .build_ref import build_ref_main
        build_ref_main(raw_args[1:])
        return

    parser = argparse.ArgumentParser(
        description="GPU-accelerated self-bootstrapping local ancestry inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input format (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--vcf",
        help="Phased VCF/BCF file (indexed)",
    )
    input_group.add_argument(
        "--pgen",
        help="Per-chromosome PGEN directory or single file prefix "
             "(expects .pgen/.pvar/.psam files)",
    )

    parser.add_argument(
        "--map", default=None,
        help="HapMap-format genetic map (single file or directory of per-chrom files). "
             "If omitted, auto-downloads from Beagle project based on --genome.",
    )
    parser.add_argument(
        "--genome", choices=["GRCh38", "GRCh37", "GRCh36"], default="GRCh38",
        help="Reference genome build for auto-downloading genetic map (default: GRCh38)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output prefix (will produce .global.tsv, .model, .tracts.tsv.gz, .summary.json)",
    )
    parser.add_argument(
        "--n-ancestries", type=int, default=None,
        help="Number of ancestries (default: auto-detect from PCA)",
    )
    parser.add_argument(
        "--n-em-iter", type=int, default=5,
        help="Maximum EM iterations on seed chromosome (default: 5; "
             "stops early on convergence)",
    )
    parser.add_argument(
        "--gen-since-admix", type=float, default=20.0,
        help="Initial guess for generations since admixture (default: 20)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Haplotypes per forward-backward batch (default: auto-tuned from GPU memory)",
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Restrict to these chromosomes (default: all autosomes)",
    )
    parser.add_argument(
        "--thin-cm", type=float, default=None,
        help="Minimum cM spacing for site thinning, e.g. 0.02 for WGS "
             "(default: no thinning)",
    )
    parser.add_argument(
        "--method",
        choices=["hmm", "cnn", "cnn-crf"],
        default="hmm",
        help="Refinement backend: hmm (default), cnn, or cnn-crf",
    )
    parser.add_argument(
        "--ancestry-detection",
        choices=["marchenko-pastur", "recursive", "eigenvalue-gap"],
        default="marchenko-pastur",
        help="Method for auto-detecting number of ancestries "
             "(default: marchenko-pastur)",
    )
    parser.add_argument(
        "--max-ancestries", type=int, default=20,
        help="Upper bound for auto-detected ancestry count (default: 20)",
    )
    parser.add_argument(
        "--per-hap-T", action=argparse.BooleanOptionalAction, default=False,
        help="Estimate per-haplotype admixture time (default: disabled; "
             "pass --per-hap-T to enable)",
    )
    parser.add_argument(
        "--n-T-buckets", type=int, default=20,
        help="Number of transition-matrix buckets for per-haplotype T (default: 20)",
    )
    parser.add_argument(
        "--block-emissions", action="store_true",
        help="Use block-level haplotype pattern emissions instead of single-site Bernoulli",
    )
    parser.add_argument(
        "--block-size", type=int, default=8,
        help="SNPs per block for block emissions (default: 8)",
    )
    # --- Recursive seeding ---
    parser.add_argument(
        "--seed-method",
        choices=["gmm", "recursive"],
        default="gmm",
        help="Seeding strategy (default: gmm). "
             "Use 'recursive' for cohorts with imbalanced ancestry proportions "
             "or low-FST populations that flat GMM misses; it is more robust "
             "but slower. On imbalanced K=4 simulations, recursive achieved "
             "92.7%% accuracy vs 64.6%% for GMM.",
    )
    parser.add_argument(
        "--recursive-min-cluster-size", type=int, default=1000,
        help="Minimum cluster size to consider for further splitting "
             "(default: 1000)",
    )
    parser.add_argument(
        "--recursive-max-depth", type=int, default=6,
        help="Maximum recursion depth for K=2 splitting (default: 6)",
    )
    parser.add_argument(
        "--recursive-max-leaves", type=int, default=12,
        help="Maximum number of leaf populations from recursive splitting "
             "(default: 12). Caps both the recursion and the resulting K.",
    )
    parser.add_argument(
        "--recursive-bic-per-sample", type=float, default=0.01,
        help="Per-sample BIC-improvement floor for K=2 splits (default: 0.01). "
             "A split is accepted when BIC(1)-BIC(2) > this * cluster_size. "
             "Not the operational gate — the Hellinger merge (see "
             "--recursive-merge-hellinger) is the primary mechanism for "
             "controlling K. This threshold prevents gross over-splitting.",
    )
    parser.add_argument(
        "--recursive-em-iter", type=int, default=3,
        help="EM iterations per K=2 split (default: 3). Cheap; small values fine.",
    )
    parser.add_argument(
        "--freeze-anchors-iters", type=int, default=0,
        help="Freeze seed responsibilities for the first N EM iterations "
             "(default: 0 = no freezing). With recursive seeding, try 3-5 "
             "if small leaves are getting absorbed.",
    )
    parser.add_argument(
        "--recursive-split-restarts", type=int, default=5,
        help="K-means++ restarts per K=2 split for balance selection (default: 5). "
             "Set to 1 to disable balance preference and use the first spectral split.",
    )
    parser.add_argument(
        "--recursive-balance-tolerance", type=float, default=0.10,
        help="BIC tolerance for balance selection: candidates within this "
             "fraction of the best BIC are eligible for balance-based selection "
             "(default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--recursive-merge-hellinger", type=float, default=0.08,
        help="Merge sibling leaves whose allele-frequency Hellinger "
             "distance is below this threshold (default: 0.08). "
             "Only sibling/near-sibling pairs in the recursion tree are "
             "candidates. Safe up to ~0.10 (below continental F_ST). "
             "Set to 0 to disable.",
    )

    parser.add_argument(
        "--probs", action="store_true",
        help="Write posterior probabilities to output files",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--monitor", choices=["wandb", "tensorboard"], default=None,
        help="Live monitoring backend (requires wandb or tensorboard package)",
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Disable stats file generation",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )

    # --- Panel export ---
    panel_group = parser.add_argument_group("Panel export")
    panel_group.add_argument(
        "--export-panel", action="store_true",
        help="Export reference panel outputs alongside standard results",
    )
    panel_group.add_argument(
        "--panel-threshold", type=float, default=0.95,
        help="Minimum posterior for whole-haplotype extraction (default: 0.95)",
    )
    panel_group.add_argument(
        "--panel-segment-threshold", type=float, default=0.99,
        help="Minimum posterior for segment extraction (default: 0.99)",
    )
    panel_group.add_argument(
        "--panel-min-segment-cm", type=float, default=1.0,
        help="Minimum segment length in cM (default: 1.0)",
    )
    panel_group.add_argument(
        "--panel-max-per-ancestry", type=int, default=None,
        help="Maximum haplotypes per ancestry in panel (default: all passing)",
    )

    # --- CNN backend ---
    cnn_group = parser.add_argument_group("CNN backend (--method cnn or cnn-crf)")
    cnn_group.add_argument(
        "--cnn-layers", type=int, default=12,
        help="Number of dilated conv layers (default: 12)",
    )
    cnn_group.add_argument(
        "--cnn-channels", type=int, default=64,
        help="Hidden channel dimension (default: 64)",
    )
    cnn_group.add_argument(
        "--cnn-epochs", type=int, default=5,
        help="Training epochs per pseudo-label round (default: 5)",
    )
    cnn_group.add_argument(
        "--cnn-pseudo-rounds", type=int, default=2,
        help="Number of pseudo-label self-training rounds (default: 2)",
    )
    cnn_group.add_argument(
        "--cnn-lr", type=float, default=1e-3,
        help="CNN learning rate (default: 1e-3)",
    )
    cnn_group.add_argument(
        "--cnn-batch-size", type=int, default=512,
        help="Haplotypes per CNN training/inference batch (default: 512)",
    )

    args = parser.parse_args(argv)

    # --- Validate flag combinations ---
    if args.seed_method == "recursive" and args.n_ancestries is not None:
        print(
            "ERROR: --seed-method recursive and --n-ancestries are incompatible: "
            "recursive seeding discovers K from the data. Remove --n-ancestries "
            "to let the recursion determine the number of populations.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Logging ---
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("popout")

    # --- Check JAX backend ---
    import jax
    devices = jax.devices()
    log.info("JAX devices: %s", devices)
    if devices[0].device_kind == "cpu":
        log.warning("Running on CPU — this will be slow for large datasets.")
        log.warning("Install jax[cuda12] for GPU acceleration.")

    t0 = time.perf_counter()

    # --- Stats collector ---
    stats = None
    if not args.no_stats:
        from .stats import StatsCollector
        config = {
            "method": args.method,
            "n_ancestries": args.n_ancestries,
            "n_em_iter": args.n_em_iter,
            "gen_since_admix": args.gen_since_admix,
            "batch_size": args.batch_size,
            "thin_cm": args.thin_cm,
            "seed": args.seed,
        }
        stats = StatsCollector(args.out, monitor=args.monitor, config=config)
        stats.emit_device_info()

    # --- Load genetic map ---
    from .gmap import load_genetic_map, load_genetic_map_per_chrom
    from .fetch_map import resolve_map_dir

    map_resolved = resolve_map_dir(genome=args.genome, map_arg=args.map)
    map_path = Path(map_resolved)
    if map_path.is_dir():
        gmap = load_genetic_map_per_chrom(map_path)
    else:
        gmap = load_genetic_map(map_path)
    log.info("Loaded genetic map: %d chromosomes", len(gmap))

    # --- Set up input reader (format-aware) ---
    if args.vcf:
        import pysam
        vcf = pysam.VariantFile(args.vcf)
        sample_names = list(vcf.header.samples)
        vcf.close()

        from .vcf_io import iter_chromosomes
        chrom_iter = iter_chromosomes(
            args.vcf, gmap,
            chromosomes=args.chromosomes,
            stats=stats,
        )
    else:
        from .pgen_io import iter_chromosomes as pgen_iter_chromosomes, get_sample_names
        pgen_path = Path(args.pgen)

        # Find a .psam file for sample names
        if pgen_path.is_dir():
            psam_files = sorted(pgen_path.glob("*.psam"))
            if not psam_files:
                log.error("No .psam files found in %s", pgen_path)
                sys.exit(1)
            sample_names = get_sample_names(psam_files[0])
        else:
            psam_path = pgen_path.with_suffix(".psam") if pgen_path.suffix != ".psam" else pgen_path
            if pgen_path.suffix == ".pgen":
                psam_path = pgen_path.with_suffix(".psam")
            else:
                psam_path = Path(str(pgen_path) + ".psam")
            sample_names = get_sample_names(psam_path)

        chrom_iter = pgen_iter_chromosomes(
            args.pgen, gmap,
            chromosomes=args.chromosomes,
            thin_cm=args.thin_cm,
            stats=stats,
        )

    n_samples = len(sample_names)
    log.info("Input: %d samples (%d haplotypes)", n_samples, 2 * n_samples)

    # --- Stream chromosomes and run pipeline ---
    # We need to keep ChromData for output writing, so collect them
    chrom_data_list = []
    def chrom_iter_with_save():
        for cd in chrom_iter:
            chrom_data_list.append(cd)
            yield cd

    if args.method in ("cnn", "cnn-crf"):
        from .cnn.refine import run_cnn_genome
        results = run_cnn_genome(
            chrom_iter_with_save(),
            n_ancestries=args.n_ancestries,
            gen_since_admix=args.gen_since_admix,
            hmm_batch_size=args.batch_size,
            rng_seed=args.seed,
            stats=stats,
            n_layers=args.cnn_layers,
            hidden_dim=args.cnn_channels,
            n_epochs=args.cnn_epochs,
            n_pseudo_rounds=args.cnn_pseudo_rounds,
            cnn_lr=args.cnn_lr,
            cnn_batch_size=args.cnn_batch_size,
            use_crf=(args.method == "cnn-crf"),
        )
    else:
        from .em import run_em_genome
        recursive_kwargs = None
        if args.seed_method == "recursive":
            recursive_kwargs = dict(
                min_cluster_size=args.recursive_min_cluster_size,
                max_depth=args.recursive_max_depth,
                max_leaves=args.recursive_max_leaves,
                bic_per_sample=args.recursive_bic_per_sample,
                em_iter_per_split=args.recursive_em_iter,
                merge_hellinger_threshold=args.recursive_merge_hellinger,
                split_restarts=args.recursive_split_restarts,
                balance_bic_tolerance=args.recursive_balance_tolerance,
            )

        results = run_em_genome(
            chrom_iter_with_save(),
            n_ancestries=args.n_ancestries,
            n_em_iter=args.n_em_iter,
            gen_since_admix=args.gen_since_admix,
            batch_size=args.batch_size,
            rng_seed=args.seed,
            stats=stats,
            per_hap_T=args.per_hap_T,
            n_T_buckets=args.n_T_buckets,
            use_block_emissions=args.block_emissions,
            detection_method=args.ancestry_detection,
            max_ancestries=args.max_ancestries,
            block_size=args.block_size,
            seed_method=args.seed_method,
            recursive_kwargs=recursive_kwargs,
            freeze_anchors_iters=args.freeze_anchors_iters,
        )

    t_compute = time.perf_counter() - t0
    log.info("Computation complete in %.1f seconds", t_compute)

    # --- Write outputs ---
    from .output import write_global_ancestry, write_model, write_ancestry_tracts

    out_prefix = args.out

    write_global_ancestry(
        results, n_samples, sample_names,
        f"{out_prefix}.global.tsv",
        stats=stats,
    )

    write_model(results[0], f"{out_prefix}.model", chrom_data=chrom_data_list[0])

    if results[0].spectral is not None:
        import numpy as np
        np.savez_compressed(f"{out_prefix}.spectral.npz", **results[0].spectral)
        log.info("Wrote spectral data to %s.spectral.npz", out_prefix)

    write_ancestry_tracts(
        results, chrom_data_list, n_samples, sample_names,
        f"{out_prefix}.tracts.tsv.gz",
        write_posteriors=args.probs,
        stats=stats,
    )

    # --- Optional panel export ---
    if args.export_panel:
        from .panel import PanelConfig, export_panel

        panel_cfg = PanelConfig(
            whole_hap_threshold=args.panel_threshold,
            segment_threshold=args.panel_segment_threshold,
            min_segment_cm=args.panel_min_segment_cm,
            max_per_ancestry=args.panel_max_per_ancestry,
        )
        export_panel(
            results, chrom_data_list, n_samples, sample_names,
            out_prefix, panel_cfg, stats=stats,
        )

    t_total = time.perf_counter() - t0
    log.info("Total wall clock: %.1f seconds", t_total)

    # --- Finalize stats ---
    if stats is not None:
        stats.emit("runtime/t_compute", round(t_compute, 2))
        stats.emit("runtime/t_total", round(t_total, 2))
        stats.finalize()


if __name__ == "__main__":
    main()
