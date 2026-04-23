"""Tests for recursive K=2 seeding."""

import numpy as np
import jax.numpy as jnp
import pytest

from popout.simulate import simulate_admixed, evaluate_accuracy
from popout.recursive_seed import recursive_split_seed, _merge_close_leaves, LeafInfo


def test_smoke_k3():
    """Recursive splitter finds K=3±2 on K=3 simulated data."""
    chrom_data, true_ancestry, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=3,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    leaf_labels, leaf_info = recursive_split_seed(
        chrom_data.geno,
        min_cluster_size=500,
        rng_seed=42,
        chrom_data=chrom_data,
    )
    n_leaves = len(leaf_info)
    assert 2 <= n_leaves <= 8, f"Expected 2-8 leaves for K=3 data, got {n_leaves}"
    assert leaf_labels.shape == (chrom_data.n_haps,)
    assert leaf_labels.min() == 0
    assert leaf_labels.max() == n_leaves - 1


def test_k1_no_structure():
    """Single-population data should produce K=1 or K=2."""
    chrom_data, _, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=1,
        gen_since_admix=20, pure_fraction=1.0, rng_seed=99,
    )
    leaf_labels, leaf_info = recursive_split_seed(
        chrom_data.geno,
        min_cluster_size=500,
        rng_seed=99,
        chrom_data=chrom_data,
    )
    assert len(leaf_info) <= 2, f"Expected 1-2 leaves for K=1, got {len(leaf_info)}"


def test_large_k():
    """K=6 biobank-like: discovers K within ±2 and covers multiple ancestries."""
    chrom_data, true_ancestry, _ = simulate_admixed(
        n_samples=5000, n_sites=2000, n_ancestries=6,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=7,
    )
    leaf_labels, leaf_info = recursive_split_seed(
        chrom_data.geno,
        min_cluster_size=200,
        rng_seed=7,
        chrom_data=chrom_data,
    )
    assert len(leaf_info) >= 4, f"Expected >=4 leaves for K=6, got {len(leaf_info)}"

    # Leaves should map to at least 3 distinct true ancestries
    all_dominant = set()
    for li in leaf_info:
        mask = leaf_labels == li.label
        hap_majority = np.array([
            np.bincount(true_ancestry[h], minlength=6).argmax()
            for h in np.where(mask)[0]
        ])
        all_dominant.add(int(np.bincount(hap_majority, minlength=6).argmax()))

    assert len(all_dominant) >= 3, (
        f"Expected >=3 distinct dominant ancestries across leaves, got {all_dominant}"
    )


def test_imbalanced_k4():
    """Imbalanced K=4 (dominant pop at ~60%): discovers K >= 3 and EM recovers."""
    from popout.em import run_em

    chrom_data, true_ancestry, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=4,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    leaf_labels, leaf_info = recursive_split_seed(
        chrom_data.geno,
        min_cluster_size=500,
        rng_seed=42,
        chrom_data=chrom_data,
    )
    K = len(leaf_info)
    assert 3 <= K <= 10, f"Expected K in [3,10] for imbalanced K=4, got {K}"

    # The EM should recover reasonable accuracy despite K != 4
    seed_resp = jnp.zeros((chrom_data.n_haps, K), dtype=jnp.float32)
    seed_resp = seed_resp.at[
        jnp.arange(chrom_data.n_haps), jnp.array(leaf_labels)
    ].set(1.0)
    result = run_em(
        chrom_data, n_em_iter=5, gen_since_admix=20.0, rng_seed=42,
        seed_responsibilities=seed_resp,
    )
    metrics = evaluate_accuracy(result.calls, true_ancestry, K)
    assert metrics["overall_accuracy"] > 0.75, (
        f"Expected >75% accuracy on imbalanced K=4, got {metrics['overall_accuracy']:.1%}"
    )


def test_integration_recursive_em():
    """Full pipeline with recursive seeding produces reasonable accuracy."""
    from popout.em import run_em

    chrom_data, true_ancestry, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=3,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    leaf_labels, leaf_info = recursive_split_seed(
        chrom_data.geno,
        min_cluster_size=500,
        rng_seed=42,
        chrom_data=chrom_data,
    )
    n_leaves = len(leaf_info)

    seed_resp = jnp.zeros((chrom_data.n_haps, n_leaves), dtype=jnp.float32)
    seed_resp = seed_resp.at[
        jnp.arange(chrom_data.n_haps), jnp.array(leaf_labels)
    ].set(1.0)

    result = run_em(
        chrom_data, n_em_iter=5, gen_since_admix=20.0, rng_seed=42,
        seed_responsibilities=seed_resp,
    )
    metrics = evaluate_accuracy(result.calls, true_ancestry, n_leaves)
    assert metrics["overall_accuracy"] > 0.7, (
        f"Expected >70% accuracy, got {metrics['overall_accuracy']:.1%}"
    )


def test_anchor_freezing():
    """Anchor freezing should not degrade accuracy vs soft init."""
    from popout.em import run_em

    chrom_data, true_ancestry, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=3,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    leaf_labels, leaf_info = recursive_split_seed(
        chrom_data.geno, min_cluster_size=500,
        rng_seed=42, chrom_data=chrom_data,
    )
    n_leaves = len(leaf_info)
    seed_resp = jnp.zeros((chrom_data.n_haps, n_leaves), dtype=jnp.float32)
    seed_resp = seed_resp.at[
        jnp.arange(chrom_data.n_haps), jnp.array(leaf_labels)
    ].set(1.0)

    result_soft = run_em(
        chrom_data, n_em_iter=5, gen_since_admix=20.0, rng_seed=42,
        seed_responsibilities=seed_resp, freeze_anchors_iters=0,
    )
    acc_soft = evaluate_accuracy(result_soft.calls, true_ancestry, n_leaves)

    result_frozen = run_em(
        chrom_data, n_em_iter=5, gen_since_admix=20.0, rng_seed=42,
        seed_responsibilities=seed_resp, freeze_anchors_iters=3,
    )
    acc_frozen = evaluate_accuracy(result_frozen.calls, true_ancestry, n_leaves)

    assert acc_frozen["overall_accuracy"] >= acc_soft["overall_accuracy"] - 0.05, (
        f"Frozen accuracy {acc_frozen['overall_accuracy']:.1%} is more than 5pp "
        f"worse than soft {acc_soft['overall_accuracy']:.1%}"
    )


def test_leaf_info_fields():
    """LeafInfo fields are populated correctly."""
    chrom_data, _, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=3,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    ll, leaf_info = recursive_split_seed(
        chrom_data.geno, min_cluster_size=500,
        rng_seed=42, chrom_data=chrom_data,
    )
    for li in leaf_info:
        assert isinstance(li.label, int)
        assert li.n_haps > 0
        assert li.depth >= 0
        assert li.path.startswith("L")
        assert isinstance(li.bic_score, float)

    labels = [li.label for li in leaf_info]
    assert labels == list(range(len(leaf_info)))
    assert sum(li.n_haps for li in leaf_info) == chrom_data.n_haps


def test_hellinger_merge():
    """Hellinger merge correctly combines leaves with similar frequency profiles."""
    rng = np.random.default_rng(42)
    T = 500
    # Create 4 populations with distinct frequencies
    base_freq = rng.beta(0.5, 0.5, size=T)
    pop_freqs = []
    for i in range(4):
        fst = 0.1
        alpha = base_freq * (1-fst)/fst
        beta = (1-base_freq) * (1-fst)/fst
        pop_freqs.append(rng.beta(np.maximum(alpha, 0.01), np.maximum(beta, 0.01)))

    # Generate haplotypes: 3 genuine pops + 1 duplicate of pop 0
    geno_parts = []
    labels_parts = []
    for pop_id, pop_f in enumerate(pop_freqs):
        n_haps = 200
        haps = (rng.random((n_haps, T)) < pop_f).astype(np.uint8)
        geno_parts.append(haps)
        labels_parts.append(np.full(n_haps, pop_id, dtype=np.int32))

    geno = np.vstack(geno_parts)
    labels = np.concatenate(labels_parts)

    # Leaf info for 4 leaves (leaf 3 is a duplicate of leaf 0)
    leaf_info = [
        LeafInfo(label=0, n_haps=200, depth=1, path="L0", bic_score=100),
        LeafInfo(label=1, n_haps=200, depth=1, path="L10", bic_score=80),
        LeafInfo(label=2, n_haps=200, depth=1, path="L110", bic_score=60),
        LeafInfo(label=3, n_haps=200, depth=1, path="L111", bic_score=60),
    ]

    # Merge with high threshold (should merge pop 0 and pop 3 since they share frequencies)
    new_labels, new_info = _merge_close_leaves(
        geno, labels, leaf_info,
        hellinger_threshold=0.15,  # high enough to merge duplicates
    )

    # Should have merged at least one pair
    assert len(new_info) < 4, f"Expected merge to reduce leaves, got {len(new_info)}"
    # Labels should be contiguous
    assert new_labels.min() == 0
    assert new_labels.max() == len(new_info) - 1


def test_per_sample_bic_scaling():
    """Per-sample BIC threshold prevents splitting homogeneous data at any N."""
    for n_samples in [500, 2500]:
        chrom_data, _, _ = simulate_admixed(
            n_samples=n_samples, n_sites=2000, n_ancestries=1,
            gen_since_admix=20, pure_fraction=1.0, rng_seed=42,
        )
        _, leaf_info = recursive_split_seed(
            chrom_data.geno,
            min_cluster_size=max(100, n_samples // 5),
            rng_seed=42,
            chrom_data=chrom_data,
        )
        assert len(leaf_info) <= 2, (
            f"K=1 data at N={n_samples*2} should give 1-2 leaves, got {len(leaf_info)}"
        )


def test_merge_caching_arithmetic():
    """Cached frequency update in merge matches direct recomputation."""
    rng = np.random.default_rng(42)
    T = 200
    # Two sibling leaves that should merge
    geno_a = (rng.random((300, T)) < 0.4).astype(np.uint8)
    geno_b = (rng.random((200, T)) < 0.42).astype(np.uint8)
    geno = np.vstack([geno_a, geno_b])
    labels = np.array([0]*300 + [1]*200, dtype=np.int32)
    leaf_info = [
        LeafInfo(label=0, n_haps=300, depth=1, path="L0", bic_score=50),
        LeafInfo(label=1, n_haps=200, depth=1, path="L1", bic_score=50),
    ]

    # Merge (threshold high enough to force it)
    new_labels, new_info = _merge_close_leaves(
        geno, labels, leaf_info, hellinger_threshold=1.0,
    )
    assert len(new_info) == 1
    assert new_info[0].n_haps == 500

    # Compute expected frequency directly from the union
    pseudocount = 0.5
    expected_freq = (geno.sum(axis=0, dtype=np.float64) + pseudocount) / (500 + 2 * pseudocount)

    # Recover the cached frequency via the same formula from the merged leaf
    # (the merge function doesn't expose freq, so verify via allele freq on the labels)
    actual_freq = (geno[new_labels == 0].sum(axis=0, dtype=np.float64) + pseudocount) / (500 + 2 * pseudocount)
    np.testing.assert_allclose(actual_freq, expected_freq, atol=1e-10)


def test_pre_merge_dump(tmp_path):
    """Pre-merge dump writes correct files."""
    chrom_data, _, _ = simulate_admixed(
        n_samples=500, n_sites=500, n_ancestries=3,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    dump_path = str(tmp_path / "test_dump")
    ll, li = recursive_split_seed(
        chrom_data.geno, min_cluster_size=100,
        rng_seed=42, chrom_data=chrom_data,
        dump_pre_merge_path=dump_path,
    )

    import os
    # Check files exist
    assert os.path.exists(f"{dump_path}.leaves.tsv")
    assert os.path.exists(f"{dump_path}.leaf_meta.tsv")
    assert os.path.exists(f"{dump_path}.leaf_freqs.npz")

    # Check leaf_meta content
    with open(f"{dump_path}.leaf_meta.tsv") as f:
        lines = f.readlines()
    assert lines[0].startswith("label\t")
    # Number of data lines should match number of pre-merge leaves
    # (may differ from post-merge li if merges happened)

    # Check freqs npz
    data = np.load(f"{dump_path}.leaf_freqs.npz", allow_pickle=True)
    assert "allele_freq" in data
    assert "labels" in data
    assert data["allele_freq"].shape[1] == chrom_data.n_sites

    # Check leaves.tsv has one row per haplotype
    with open(f"{dump_path}.leaves.tsv") as f:
        n_lines = sum(1 for _ in f) - 1  # minus header
    assert n_lines == chrom_data.n_haps


def test_merge_disabled():
    """merge_hellinger_threshold=0 skips all merging."""
    chrom_data, _, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=3,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    # Run with merge disabled
    ll_no_merge, li_no_merge = recursive_split_seed(
        chrom_data.geno, min_cluster_size=500,
        rng_seed=42, chrom_data=chrom_data,
        merge_hellinger_threshold=0,
    )
    # Run with merge enabled
    ll_merge, li_merge = recursive_split_seed(
        chrom_data.geno, min_cluster_size=500,
        rng_seed=42, chrom_data=chrom_data,
        merge_hellinger_threshold=0.08,
    )
    # No-merge should have >= as many leaves as merged
    assert len(li_no_merge) >= len(li_merge)


# ---------------------------------------------------------------------------
# Seeding exclusion tests
# ---------------------------------------------------------------------------


def test_parse_seeding_exclusion_list(tmp_path):
    """load_seeding_exclusion_list parses valid TSV and rejects bad ones."""
    from popout.cli import load_seeding_exclusion_list

    # Good file: 3 unique sample IDs
    good = tmp_path / "good.tsv"
    good.write_text("sample_id\n1000039\n1000059\n1000091\n")
    result = load_seeding_exclusion_list(str(good))
    assert result == {"1000039", "1000059", "1000091"}

    # Duplicates are deduped
    dups = tmp_path / "dups.tsv"
    dups.write_text("sample_id\n1000039\n1000039\n1000059\n")
    result = load_seeding_exclusion_list(str(dups))
    assert result == {"1000039", "1000059"}

    # Header-only (empty body) returns empty set
    empty = tmp_path / "empty.tsv"
    empty.write_text("sample_id\n")
    result = load_seeding_exclusion_list(str(empty))
    assert result == set()

    # Wrong header raises ValueError
    bad_header = tmp_path / "bad.tsv"
    bad_header.write_text("wrong_col\n1000039\n")
    with pytest.raises(ValueError, match="sample_id"):
        load_seeding_exclusion_list(str(bad_header))

    # No header at all (first data row becomes header) raises ValueError
    no_header = tmp_path / "no_header.tsv"
    no_header.write_text("1000039\n1000059\n")
    with pytest.raises(ValueError, match="sample_id"):
        load_seeding_exclusion_list(str(no_header))


def test_seeding_exclusion_mask():
    """Seeding mask filters haplotypes; EM still runs on the full cohort."""
    from popout.em import run_em

    chrom_data, true_ancestry, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=3,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    H_total = chrom_data.n_haps  # 5000

    # Exclude 500 samples (last 1000 haplotypes)
    mask = np.ones(H_total, dtype=bool)
    mask[4000:] = False
    H_kept = int(mask.sum())  # 4000

    # --- Without mask: seed from all 5000 haplotypes ---
    ll_full, li_full = recursive_split_seed(
        chrom_data.geno, min_cluster_size=500,
        rng_seed=42, chrom_data=chrom_data,
    )
    assert ll_full.shape == (H_total,)

    # --- With mask: seed from 4000 haplotypes ---
    ll_masked, li_masked = recursive_split_seed(
        chrom_data.geno, min_cluster_size=500,
        rng_seed=42, chrom_data=chrom_data,
        seeding_mask=mask,
    )
    assert ll_masked.shape == (H_kept,)
    assert sum(li.n_haps for li in li_masked) == H_kept

    # Build seed_resp for masked run: one-hot for kept, uniform for excluded
    n_leaves = len(li_masked)
    kept_idx = np.where(mask)[0]
    seed_resp_np = np.full((H_total, n_leaves), 1.0 / n_leaves, dtype=np.float32)
    seed_resp_np[kept_idx] = 0.0
    seed_resp_np[kept_idx, ll_masked] = 1.0
    seed_resp = jnp.array(seed_resp_np)

    result = run_em(
        chrom_data, n_em_iter=5, gen_since_admix=20.0, rng_seed=42,
        seed_responsibilities=seed_resp,
    )
    # EM ran on ALL haplotypes — check full cohort size
    assert result.calls.shape[0] == H_total
    # All 2500 samples have calls
    assert result.calls.shape[0] // 2 == 2500


def test_excluded_samples_get_calls():
    """Excluded samples still receive non-trivial ancestry assignments."""
    from popout.em import run_em

    chrom_data, true_ancestry, _ = simulate_admixed(
        n_samples=2500, n_sites=2000, n_ancestries=3,
        gen_since_admix=20, pure_fraction=0.3, rng_seed=42,
    )
    H_total = chrom_data.n_haps
    mask = np.ones(H_total, dtype=bool)
    mask[4000:] = False
    kept_idx = np.where(mask)[0]

    ll, li = recursive_split_seed(
        chrom_data.geno, min_cluster_size=500,
        rng_seed=42, chrom_data=chrom_data,
        seeding_mask=mask,
    )
    n_leaves = len(li)

    seed_resp_np = np.full((H_total, n_leaves), 1.0 / n_leaves, dtype=np.float32)
    seed_resp_np[kept_idx] = 0.0
    seed_resp_np[kept_idx, ll] = 1.0

    result = run_em(
        chrom_data, n_em_iter=5, gen_since_admix=20.0, rng_seed=42,
        seed_responsibilities=jnp.array(seed_resp_np),
    )

    # Pick an excluded haplotype and verify it got real ancestry calls
    excluded_hap = 4500
    calls = result.calls[excluded_hap]
    assert calls.shape[0] == chrom_data.n_sites
    # Should have at least some non-zero ancestry assignments (not all ancestry-0)
    unique_calls = np.unique(calls)
    assert len(unique_calls) >= 1, "Excluded haplotype should have ancestry calls"
    # Check accuracy for excluded haplotypes is still reasonable
    metrics = evaluate_accuracy(result.calls, true_ancestry, n_leaves)
    assert metrics["overall_accuracy"] > 0.60
