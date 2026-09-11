"""The measurements the real-cohort tests assert on.

These are the part of an expensive test that can be got wrong cheaply. A
separation statistic that reads high on structureless data, or a neighbourhood
score that is high for any two point clouds, turns a six-hour cohort run into a
six-hour way of learning nothing -- so they are pinned here, on synthetic data
where the right answer is known by construction.
"""

import numpy as np
import pytest

from tests.science import (
    neighbourhood_preservation,
    separation_over_chance,
    variance_explained_by_group,
)


def clustered(n_per_group=60, n_groups=3, spread=0.1, seed=0):
    """Tight, well-separated blobs: group membership explains nearly everything."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_groups, 5)) * 10
    X = np.repeat(centres, n_per_group, axis=0) + rng.normal(
        scale=spread, size=(n_per_group * n_groups, 5)
    )
    y = np.repeat([f"g{i}" for i in range(n_groups)], n_per_group)
    return X, y


class TestVarianceExplainedByGroup:
    def test_separated_groups_explain_almost_all_the_variance(self):
        X, y = clustered()

        assert variance_explained_by_group(X, y) > 0.99

    def test_labels_unrelated_to_the_data_explain_almost_none(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 5))
        y = rng.choice(["a", "b", "c"], size=300)

        assert variance_explained_by_group(X, y) < 0.05

    def test_one_group_containing_everything_explains_nothing(self):
        # The degenerate case: no between-group variance exists to find.
        X, _ = clustered()

        assert variance_explained_by_group(X, np.array(["only"] * len(X))) == pytest.approx(0.0)

    def test_it_is_a_fraction(self):
        X, y = clustered(spread=5.0)
        value = variance_explained_by_group(X, y)

        assert 0.0 <= value <= 1.0

    def test_imbalance_does_not_inflate_it(self):
        """UK Biobank is ~80% one ancestry label; a statistic that rewards that
        would pass on data with no structure at all."""
        rng = np.random.default_rng(2)
        X = rng.normal(size=(1000, 5))
        y = np.array(["major"] * 900 + ["minor"] * 100)
        rng.shuffle(y)

        assert variance_explained_by_group(X, y) < 0.05


class TestSeparationOverChance:
    def test_structure_scores_far_above_one(self):
        X, y = clustered()

        assert separation_over_chance(X, y) > 20

    def test_unrelated_labels_score_about_one(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(400, 5))
        y = rng.choice(["a", "b", "c", "d"], size=400)

        assert 0.2 < separation_over_chance(X, y) < 5

    def test_many_rare_groups_still_score_about_one_without_structure(self):
        # eta^2 grows with the number of groups regardless of structure, which is
        # exactly what dividing by the permuted value is for.
        rng = np.random.default_rng(4)
        X = rng.normal(size=(300, 5))
        y = rng.choice([f"g{i}" for i in range(40)], size=300)

        assert separation_over_chance(X, y) < 5


class TestNeighbourhoodPreservation:
    def test_a_space_preserves_itself_perfectly(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(200, 4))

        assert neighbourhood_preservation(X, X, k=10) == pytest.approx(1.0)

    def test_a_rotation_and_scaling_preserves_everything(self):
        # Embeddings are only defined up to rotation; the score must not care.
        rng = np.random.default_rng(6)
        X = rng.normal(size=(200, 4))
        Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))

        assert neighbourhood_preservation(X, (X @ Q) * 3.0, k=10) == pytest.approx(1.0)

    def test_unrelated_spaces_score_near_chance(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(400, 4))
        Y = rng.normal(size=(400, 2))

        # Chance is roughly k/n = 0.025.
        assert neighbourhood_preservation(X, Y, k=10) < 0.15

    def test_a_two_dimensional_view_of_clusters_keeps_most_neighbours(self):
        X, _ = clustered(n_per_group=80, spread=1.0)

        assert neighbourhood_preservation(X, X[:, :2], k=10) > 0.3

    def test_probing_a_subsample_gives_the_same_answer_within_noise(self):
        # Cohort runs cannot afford all-pairs distances on 486k samples.
        rng = np.random.default_rng(8)
        X = rng.normal(size=(600, 4))
        Y = X + rng.normal(scale=0.3, size=X.shape)

        full = neighbourhood_preservation(X, Y, k=10)
        probed = neighbourhood_preservation(X, Y, k=10, n_probe=200, random_state=0)

        assert probed == pytest.approx(full, abs=0.1)

    def test_it_rejects_mismatched_sample_counts(self):
        with pytest.raises(ValueError):
            neighbourhood_preservation(np.zeros((10, 2)), np.zeros((9, 2)), k=3)
