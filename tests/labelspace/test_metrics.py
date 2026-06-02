"""Phase 5: ARI / NMI / V-measure on canonical edge cases."""

from __future__ import annotations

import math

import numpy as np

from popout.labelspace.metrics import (
    adjusted_rand_index,
    cluster_eval,
    mutual_information,
    normalized_mutual_info,
    v_measure,
)


def test_ari_identical_clustering():
    a = [0, 0, 1, 1, 2, 2]
    assert math.isclose(adjusted_rand_index(a, a), 1.0)


def test_ari_random_clustering_near_zero():
    rng = np.random.default_rng(0)
    n = 600
    a = rng.integers(0, 4, size=n)
    b = rng.integers(0, 4, size=n)
    assert abs(adjusted_rand_index(a, b)) < 0.05


def test_ari_label_permutation_invariant():
    a = [0, 0, 1, 1, 2, 2]
    b = ["X", "X", "Y", "Y", "Z", "Z"]
    assert math.isclose(adjusted_rand_index(a, b), 1.0)


def test_nmi_identical_clustering_is_one():
    a = [0, 0, 1, 1, 2, 2]
    assert math.isclose(normalized_mutual_info(a, a), 1.0)


def test_nmi_uncorrelated_is_zero():
    a = [0] * 10
    b = list(range(10))
    # H(A) = 0, mi = 0
    assert normalized_mutual_info(a, b) == 0.0


def test_v_measure_identical_clustering_is_one():
    a = [0, 0, 1, 1, 2, 2]
    assert math.isclose(v_measure(a, a), 1.0)


def test_mutual_information_nonneg():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, size=200)
    b = rng.integers(0, 3, size=200)
    assert mutual_information(a, b) >= 0.0


def test_cluster_eval_dict_shape():
    a = [0, 0, 1, 1, 2, 2]
    b = [1, 1, 2, 2, 0, 0]
    out = cluster_eval(a, b)
    assert set(out) == {"ari", "nmi", "v_measure", "n"}
    assert out["n"] == 6
    # b is a label-permutation of a → ARI = 1, NMI = 1.
    assert math.isclose(out["ari"], 1.0)
    assert math.isclose(out["nmi"], 1.0)
    assert math.isclose(out["v_measure"], 1.0)
