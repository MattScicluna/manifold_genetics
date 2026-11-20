"""Tests for k-NN preservation."""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.metrics import compute_knn_preservation


def _frame(X, ids):
    df = pd.DataFrame(X, columns=[f"dim_{i + 1}" for i in range(X.shape[1])])
    df.insert(0, "sample_id", ids)
    return df


@pytest.fixture
def points():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    return X, [f"s{i}" for i in range(300)]


def test_identical_embedding_keeps_every_neighbour(points):
    X, ids = points
    r = compute_knn_preservation(_frame(X, ids), _frame(X, ids))
    assert r == {"knn_preservation": 1.0, "k": 10, "n_samples": 300}


def test_uniform_rescaling_keeps_every_neighbour(points):
    X, ids = points
    assert compute_knn_preservation(_frame(X * 7, ids), _frame(X, ids))["knn_preservation"] == 1.0


def test_shuffled_embedding_scores_near_chance(points):
    X, ids = points
    shuffled = X[np.random.default_rng(1).permutation(len(X))]
    r = compute_knn_preservation(_frame(shuffled, ids), _frame(X, ids), k=10)
    assert r["knn_preservation"] < 0.15  # chance is k / n, about 0.03


def test_hand_computed_overlap():
    """Points on a line with no distance ties; the embedding swaps the last two.

    Nearest neighbour in the reference (0, 1, 3, 6, 10, 15): a-b, b-a, c-b, d-c, e-d, f-e.
    In the embedding (0, 1, 3, 6, 15, 10): a-b, b-a, c-b, d-c, e-f, f-d. Four of six agree.
    """
    ids = list("abcdef")
    ref = np.array([0, 1, 3, 6, 10, 15], dtype=float)[:, None]
    emb = np.array([0, 1, 3, 6, 15, 10], dtype=float)[:, None]
    r = compute_knn_preservation(_frame(emb, ids), _frame(ref, ids), k=1)
    assert r["knn_preservation"] == pytest.approx(4 / 6)


def test_missing_samples_are_left_out(points):
    X, ids = points
    r = compute_knn_preservation(_frame(X[:-20], ids[:-20]), _frame(X, ids))
    assert r["n_samples"] == 280
    assert r["knn_preservation"] == 1.0


def test_too_few_samples(points):
    X, ids = points
    with pytest.raises(ValueError, match="more than k"):
        compute_knn_preservation(_frame(X[:5], ids[:5]), _frame(X[:5], ids[:5]), k=10)
