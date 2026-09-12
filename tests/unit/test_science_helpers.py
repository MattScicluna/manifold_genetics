"""Correctness guards for the statistics the real-cohort suite asserts on.

These pin ``_nearest``'s output so it can be rewritten for memory without
changing what it measures. ``_nearest`` built a ``(n_probe, n, d)`` intermediate
-- 155.8 GB when probing 2,000 points against UK Biobank's 486,748 at 20 PCs --
which OOM-killed the run that was checking it. The rewrite must return exactly
the same neighbours.
"""

import numpy as np
import pytest

from tests.science import (
    _nearest,
    neighbourhood_preservation,
    neighbourhood_preservation_over_chance,
)


def brute_force_nearest(query, reference, k):
    """The definition, written the slowest and most obvious way."""
    out = []
    for q in query:
        d = ((reference - q) ** 2).sum(axis=1)
        out.append(np.argsort(d, kind="stable")[1 : k + 1])
    return np.array(out)


def test_nearest_matches_the_brute_force_definition():
    rng = np.random.default_rng(0)
    points = rng.standard_normal((200, 5))

    got = _nearest(points[:20], points, k=7)

    np.testing.assert_array_equal(got, brute_force_nearest(points[:20], points, k=7))


def test_nearest_excludes_the_query_point_itself():
    """Every point is its own nearest neighbour at distance zero; it must not count."""
    rng = np.random.default_rng(1)
    points = rng.standard_normal((50, 3))

    got = _nearest(points, points, k=4)

    assert not any(i in row for i, row in enumerate(got))


def test_nearest_handles_duplicate_points():
    """Duplicated rows make distance ties, where an unstable rule would differ."""
    points = np.repeat(np.arange(10.0)[:, None], 2, axis=1)
    points = np.vstack([points, points])

    got = _nearest(points[:5], points, k=3)

    np.testing.assert_array_equal(got, brute_force_nearest(points[:5], points, k=3))


def test_a_perfect_embedding_preserves_every_neighbourhood():
    """A rotation changes coordinates and no neighbour sets at all."""
    rng = np.random.default_rng(2)
    reference = rng.standard_normal((300, 4))
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    assert neighbourhood_preservation(reference[:, :2], reference[:, :2] @ rotation.T, k=10) == 1.0


def test_an_unrelated_embedding_preserves_almost_nothing():
    rng = np.random.default_rng(3)

    score = neighbourhood_preservation(
        rng.standard_normal((400, 4)), rng.standard_normal((400, 2)), k=10
    )

    assert score < 0.1


# ---------------------------------------------------------------------------
# Chance correction
# ---------------------------------------------------------------------------


def test_an_unrelated_embedding_scores_about_one_times_chance():
    """The whole point of the correction: no signal reads as 1, at any cohort size."""
    rng = np.random.default_rng(4)

    score = neighbourhood_preservation_over_chance(
        rng.standard_normal((2000, 4)), rng.standard_normal((2000, 2)), k=30
    )

    assert 0.2 < score < 3.0, f"unrelated data should sit near 1x chance, got {score:.2f}"


def test_a_perfect_embedding_scores_the_maximum_possible():
    """Preserving every neighbour is ``1 / chance`` times chance, by definition."""
    rng = np.random.default_rng(5)
    points = rng.standard_normal((500, 2))

    score = neighbourhood_preservation_over_chance(points, points.copy(), k=25)

    assert score == pytest.approx((len(points) - 1) / 25)


def test_the_same_quality_scores_the_same_at_two_cohort_sizes():
    """A fixed floor on the raw fraction cannot transfer between cohort sizes.

    Measured on the real cohorts: HGDP scored 0.3328 at n=4,094 and UK Biobank
    0.01122 at n=486,748 -- a 30x gap on the raw number, but 45x versus 182x
    chance, so the larger cohort is the better embedding. A raw floor of 0.05
    passed the worse one and failed the better one.
    """
    rng = np.random.default_rng(6)
    scores = []
    for n in (1000, 20000):
        reference = rng.standard_normal((n, 3))
        embedding = reference[:, :2] + 0.01 * rng.standard_normal((n, 2))
        scores.append(neighbourhood_preservation_over_chance(reference, embedding, k=10))

    small, large = scores
    assert large > 0.3 * small, f"correction did not transfer across n: {small:.0f} vs {large:.0f}"
