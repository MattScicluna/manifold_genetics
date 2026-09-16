"""Pair sampling for the preservation metrics (metrics/pairs.py).

The metrics compare ``num_samples`` sample pairs. They must draw those pairs
first and compute distances for them alone: computing every pairwise distance
and subsampling afterwards needs O(n^2) memory, which for a biobank-scale
cohort is hundreds of GiB. The last two tests here call each metric on
500,000 samples, which would attempt that allocation before the fix.
"""

import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from manifold_genetics.metrics.admixture import compute_admixture_preservation
from manifold_genetics.metrics.geographic import (
    _haversine_distances,
    _haversine_pairs,
    compute_geographic_preservation,
)
from manifold_genetics.metrics.pairs import (
    _linear_to_pair,
    n_pairs,
    pair_distances,
    sample_pairs,
    sampled_pair_distances,
)

# ---------------------------------------------------------------------------
# sample_pairs
# ---------------------------------------------------------------------------


def _assert_valid_pairs(i, j, n, expected):
    assert len(i) == len(j) == expected
    assert (i < j).all()
    assert (i >= 0).all() and (j < n).all()
    assert len(set(zip(i.tolist(), j.tolist()))) == expected, "pairs not distinct"


@pytest.mark.parametrize("n,num_samples", [(40, 50), (200, 1000), (1000, 50000)])
def test_sample_pairs_subset_is_distinct_ordered_and_in_range(n, num_samples):
    i, j = sample_pairs(n, num_samples, np.random.default_rng(0))
    assert n_pairs(n) > num_samples
    _assert_valid_pairs(i, j, n, num_samples)


@pytest.mark.parametrize("n,num_samples", [(2, 1), (10, 45), (10, 1000)])
def test_sample_pairs_returns_every_pair_when_they_fit(n, num_samples):
    i, j = sample_pairs(n, num_samples, np.random.default_rng(0))
    _assert_valid_pairs(i, j, n, n_pairs(n))
    ti, tj = np.triu_indices(n, k=1)
    np.testing.assert_array_equal(i, ti)
    np.testing.assert_array_equal(j, tj)


def test_sample_pairs_count_is_min_of_requested_and_available():
    for n, k in [(5, 3), (5, 10), (5, 20), (60, 1770), (60, 1769)]:
        i, _ = sample_pairs(n, k, np.random.default_rng(1))
        assert len(i) == min(k, n_pairs(n))


def test_sample_pairs_rejection_branch_is_distinct_and_in_range():
    # n large enough that n(n-1)/2 exceeds the direct-choice limit, so the
    # rejection-and-top-up path is what runs.
    n = 100_000
    assert n_pairs(n) > 2**24
    i, j = sample_pairs(n, 20_000, np.random.default_rng(2))
    _assert_valid_pairs(i, j, n, 20_000)


def test_sample_pairs_is_seeded():
    a = sample_pairs(300, 500, np.random.default_rng(7))
    b = sample_pairs(300, 500, np.random.default_rng(7))
    c = sample_pairs(300, 500, np.random.default_rng(8))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not (np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))


@pytest.mark.parametrize("branch", ["direct_choice", "rejection"])
def test_sample_pairs_is_uniform_over_pairs(branch, monkeypatch):
    # Every pair should be drawn about equally often; with 12 items (66
    # pairs) and draws of 6 pairs over many seeds, no pair should be far
    # from its expected count. Run through both drawing paths: the
    # rejection path is forced by making the direct-choice limit zero.
    if branch == "rejection":
        monkeypatch.setattr("manifold_genetics.metrics.pairs._DIRECT_CHOICE_LIMIT", 0)
    n, k, reps = 12, 6, 2000
    counts = np.zeros(n_pairs(n))
    for seed in range(reps):
        i, j = sample_pairs(n, k, np.random.default_rng(seed))
        lin = i * n - i * (i + 1) // 2 + (j - i - 1)
        counts[lin] += 1
    expected = reps * k / n_pairs(n)
    assert counts.min() > expected * 0.6 and counts.max() < expected * 1.4


# ---------------------------------------------------------------------------
# linear index -> (i, j)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 7, 50, 333])
def test_linear_to_pair_matches_pdist_order(n):
    i, j = _linear_to_pair(np.arange(n_pairs(n)), n)
    ti, tj = np.triu_indices(n, k=1)
    np.testing.assert_array_equal(i, ti)
    np.testing.assert_array_equal(j, tj)


def test_linear_to_pair_is_exact_at_biobank_scale():
    n = 486_748
    total = n_pairs(n)
    rng = np.random.default_rng(0)
    lin = np.concatenate([[0, 1, total - 2, total - 1], rng.integers(0, total, 50_000)])
    i, j = _linear_to_pair(lin, n)
    assert (i < j).all() and (i >= 0).all() and (j < n).all()
    # Round trip through the condensed-index formula.
    back = i * n - i * (i + 1) // 2 + (j - i - 1)
    np.testing.assert_array_equal(back, lin)


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------


def test_pair_distances_match_pdist_for_the_same_pairs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    full = pdist(x, metric="euclidean")
    i, j = sample_pairs(30, 100, rng)
    lin = i * 30 - i * (i + 1) // 2 + (j - i - 1)
    np.testing.assert_allclose(pair_distances(x, i, j), full[lin], rtol=1e-12)


def test_sampled_pair_distances_every_pair_branch_equals_pdist_in_order():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(25, 3))
    b = rng.normal(size=(25, 2))
    da, db = sampled_pair_distances(a, b, num_samples=n_pairs(25), rng=rng)
    np.testing.assert_array_equal(da, pdist(a, metric="euclidean"))
    np.testing.assert_array_equal(db, pdist(b, metric="euclidean"))


def test_sampled_pair_distances_subsample_branch_uses_the_same_pairs_in_both():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(60, 3))
    b = a * 2.0  # distances in b are exactly twice those in a, pair for pair
    da, db = sampled_pair_distances(a, b, num_samples=200, rng=rng)
    assert len(da) == len(db) == 200
    np.testing.assert_allclose(db, 2.0 * da, rtol=1e-12)


def test_sampled_pair_distances_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="same number of rows"):
        sampled_pair_distances(np.zeros((3, 2)), np.zeros((4, 2)), 10, np.random.default_rng(0))


def test_haversine_pairs_matches_loop_implementation():
    rng = np.random.default_rng(0)
    coords = np.column_stack([rng.uniform(-180, 180, 40), rng.uniform(-90, 90, 40)])
    i, j = np.triu_indices(40, k=1)
    np.testing.assert_allclose(_haversine_pairs(coords, i, j), _haversine_distances(coords))


# ---------------------------------------------------------------------------
# the metrics at biobank scale: never compute every pairwise distance
# ---------------------------------------------------------------------------

N_BIG = 500_000


def _big_embedding():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(N_BIG, 2)).astype(np.float32), columns=["dim_1", "dim_2"])


def test_geographic_metric_never_materialises_all_pairs():
    emb = _big_embedding()
    geo = pd.DataFrame(
        {"latitude": emb["dim_1"].to_numpy() * 10, "longitude": emb["dim_2"].to_numpy() * 10},
        index=emb.index,
    )
    with (
        patch("manifold_genetics.metrics.geographic.pdist") as geo_pdist,
        patch("manifold_genetics.metrics.pairs.pdist") as pairs_pdist,
    ):
        start = time.perf_counter()
        result = compute_geographic_preservation(emb, geo, num_samples=1000)
        elapsed = time.perf_counter() - start
    geo_pdist.assert_not_called()
    pairs_pdist.assert_not_called()
    assert result["n_pairs"] == 1000
    assert result["n_samples"] == N_BIG
    assert result["correlation"] > 0.9  # geo == scaled embedding
    assert elapsed < 5.0, f"took {elapsed:.2f}s"


def test_admixture_metric_never_materialises_all_pairs():
    emb = _big_embedding()
    q = pd.DataFrame(
        {"component_1": emb["dim_1"].to_numpy(), "component_2": emb["dim_2"].to_numpy()},
        index=emb.index,
    )
    with (
        patch("manifold_genetics.metrics.admixture._load_q_matrix", return_value=q),
        patch("manifold_genetics.metrics.pairs.pdist") as pairs_pdist,
    ):
        start = time.perf_counter()
        result = compute_admixture_preservation(emb, {2: "unused.csv"}, num_samples=1000)
        elapsed = time.perf_counter() - start
    pairs_pdist.assert_not_called()
    assert result[2]["n_pairs"] == 1000
    assert result[2]["n_samples"] == N_BIG
    assert result[2]["correlation"] > 0.9  # Q == embedding
    assert elapsed < 5.0, f"took {elapsed:.2f}s"


def test_metrics_use_pdist_when_every_pair_fits():
    emb = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 2)), columns=["dim_1", "dim_2"])
    geo = pd.DataFrame(
        {"latitude": emb["dim_1"].to_numpy(), "longitude": emb["dim_2"].to_numpy()},
        index=emb.index,
    )
    with patch("manifold_genetics.metrics.geographic.pdist", wraps=pdist) as geo_pdist:
        result = compute_geographic_preservation(emb, geo, num_samples=1000)
    geo_pdist.assert_called_once()
    assert result["n_pairs"] == n_pairs(20)
