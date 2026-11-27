"""Tests for ancestry resolution."""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.metrics import compute_ancestry_resolution


def _frame(X, ids, prefix="dim"):
    df = pd.DataFrame(X, columns=[f"{prefix}_{i + 1}" for i in range(X.shape[1])])
    df.insert(0, "sample_id", ids)
    return df


@pytest.fixture
def cohort():
    """Three ancestries of 100 samples; Q puts most weight on the sample's own component."""
    rng = np.random.default_rng(0)
    n, k = 100, 3
    labels = np.repeat(np.arange(k), n)
    q = rng.dirichlet(np.ones(k), size=n * k) * 0.3
    q[np.arange(n * k), labels] += 0.7
    ids = [f"s{i}" for i in range(n * k)]
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return ids, labels, _frame(q, ids, "component"), centres


def test_separated_groups_score_one(cohort):
    ids, labels, q, centres = cohort
    emb = centres[labels] + np.random.default_rng(1).normal(size=(len(ids), 2))
    r = compute_ancestry_resolution(_frame(emb, ids), q)
    assert r["ancestry_resolution"] == 1.0
    assert r["n_groups"] == 3 and r["n_samples"] == 300
    assert r["chance"] == pytest.approx(1 / 3)


def test_unrelated_embedding_scores_near_chance(cohort):
    ids, labels, q, _ = cohort
    emb = np.random.default_rng(2).normal(size=(len(ids), 2))
    r = compute_ancestry_resolution(_frame(emb, ids), q)
    assert abs(r["ancestry_resolution"] - r["chance"]) < 0.12


def test_merging_two_groups_lowers_the_score(cohort):
    ids, labels, q, centres = cohort
    centres = centres.copy()
    centres[2] = centres[1]  # groups 1 and 2 drawn on top of each other
    emb = centres[labels] + np.random.default_rng(3).normal(size=(len(ids), 2))
    r = compute_ancestry_resolution(_frame(emb, ids), q)
    assert 0.55 < r["ancestry_resolution"] < 0.85  # group 0 kept apart, the other two mixed


def test_small_groups_are_left_out(cohort):
    ids, labels, q, centres = cohort
    q = q.copy()
    q["component_4"] = 0.0
    q.loc[:9, "component_4"] = 2.0  # 10 samples whose largest component is a fourth one
    emb = centres[labels] + np.random.default_rng(4).normal(size=(len(ids), 2))
    r = compute_ancestry_resolution(_frame(emb, ids), q, min_group_size=30)
    assert r["n_groups"] == 3 and r["n_samples"] == 290


def test_missing_samples_and_subsample(cohort):
    ids, labels, q, centres = cohort
    emb = centres[labels] + np.random.default_rng(5).normal(size=(len(ids), 2))
    r = compute_ancestry_resolution(_frame(emb[:-30], ids[:-30]), q)
    assert r["n_samples"] == 270
    r = compute_ancestry_resolution(_frame(emb, ids), q, subsample=150, seed=7)
    assert r["n_samples"] <= 150 and r["ancestry_resolution"] == 1.0


def test_reads_files_and_is_deterministic(cohort, tmp_path):
    ids, labels, q, centres = cohort
    emb = centres[labels] + np.random.default_rng(6).normal(size=(len(ids), 2)) * 4
    emb_path, q_path = tmp_path / "emb.csv", tmp_path / "q.csv"
    _frame(emb, ids).to_csv(emb_path, index=False)
    q.to_csv(q_path, index=False)
    a = compute_ancestry_resolution(emb_path, q_path, seed=3)
    b = compute_ancestry_resolution(_frame(emb, ids), q, seed=3)
    assert a == b


def test_needs_two_groups_and_two_components(cohort):
    ids, labels, q, centres = cohort
    emb = centres[labels]
    with pytest.raises(ValueError, match="at least 2 groups"):
        compute_ancestry_resolution(_frame(emb, ids), q, min_group_size=1000)
    with pytest.raises(ValueError, match="at least 2 admixture components"):
        compute_ancestry_resolution(_frame(emb, ids), q[["sample_id", "component_1"]])
