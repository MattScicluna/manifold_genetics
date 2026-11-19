"""Tests for separation inflation."""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.metrics import compute_separation_inflation


def _frame(X, ids):
    df = pd.DataFrame(X, columns=[f"dim_{i + 1}" for i in range(X.shape[1])])
    df.insert(0, "sample_id", ids)
    return df


@pytest.fixture
def two_groups():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [6.0, 0.0]])
    groups = np.repeat(["a", "b"], 60)
    X = centres[(groups == "b").astype(int)] + rng.normal(size=(120, 2))
    ids = [f"s{i}" for i in range(120)]
    labels = pd.DataFrame({"sample_id": ids, "group": groups})
    return X, groups, ids, labels, centres


def test_identical_embedding_scores_zero(two_groups):
    X, _, ids, labels, _ = two_groups
    r = compute_separation_inflation(_frame(X, ids), _frame(X, ids), labels, "group")
    assert r["inflation"] == pytest.approx(0.0)
    assert (r["n_samples"], r["n_groups"]) == (120, 2)


def test_collapsed_groups_score_positive(two_groups):
    """Shrinking each group 10x around its centre inflates separation by about log(10)."""
    X, groups, ids, labels, centres = two_groups
    c = centres[(groups == "b").astype(int)]
    collapsed = c + (X - c) / 10
    r = compute_separation_inflation(_frame(collapsed, ids), _frame(X, ids), labels, "group")
    assert r["inflation"] > 1.5
    assert set(r["per_group"]) == {"a", "b"}


def test_merged_groups_score_negative(two_groups):
    X, groups, ids, labels, centres = two_groups
    merged = X - centres[(groups == "b").astype(int)] * 0.8  # pull group b's centre towards a's
    r = compute_separation_inflation(_frame(merged, ids), _frame(X, ids), labels, "group")
    assert r["inflation"] < 0


def test_uniform_rescaling_does_not_matter(two_groups):
    X, _, ids, labels, _ = two_groups
    r = compute_separation_inflation(_frame(X * 50, ids), _frame(X, ids), labels, "group")
    assert r["inflation"] == pytest.approx(0.0)


def test_small_groups_and_missing_samples_are_left_out(two_groups):
    X, _, ids, labels, _ = two_groups
    labels = labels.copy()
    labels.loc[:2, "group"] = "tiny"  # 3 samples: at most k_own, so left out
    r = compute_separation_inflation(_frame(X[:-5], ids[:-5]), _frame(X, ids), labels, "group")
    assert r["n_samples"] == 120 - 3 - 5
    assert r["n_groups"] == 2


def test_duplicate_coordinates_give_a_finite_score(two_groups):
    """Identical points across groups (as in admixture proportions) must not produce log(0)."""
    X, _, ids, labels, _ = two_groups
    ref = X.copy()
    ref[60] = ref[0]  # a group-b sample sits exactly on a group-a sample
    with np.errstate(all="raise"):
        r = compute_separation_inflation(_frame(X, ids), _frame(ref, ids), labels, "group")
    assert np.isfinite(r["inflation"])
    assert all(np.isfinite(v) for v in r["per_group"].values())


def test_errors(two_groups):
    X, _, ids, labels, _ = two_groups
    with pytest.raises(ValueError, match="not in labels"):
        compute_separation_inflation(_frame(X, ids), _frame(X, ids), labels, "nope")
    one = labels.assign(group="a")
    with pytest.raises(ValueError, match="two groups"):
        compute_separation_inflation(_frame(X, ids), _frame(X, ids), one, "group")
