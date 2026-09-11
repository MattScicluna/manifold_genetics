"""Measurements the real-cohort tests assert on.

An expensive test is only worth its runtime if what it measures would actually
move when the pipeline breaks. These are chance-corrected for that reason: the
cohorts differ by two orders of magnitude in size and by a factor of twenty in
how many label groups they have, and raw statistics that look reassuring on one
would be meaningless on another.
"""

import numpy as np


def variance_explained_by_group(X, y) -> float:
    """One-way eta^2: the fraction of total variance in ``X`` lying between groups.

    Weighted by group size, so a large group cannot be ignored, and bounded in
    [0, 1]. On its own it rises with the number of groups even when there is no
    structure -- see :func:`separation_over_chance`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    total = ((X - X.mean(axis=0)) ** 2).sum()
    if total == 0:
        return 0.0

    between = 0.0
    grand = X.mean(axis=0)
    for group in np.unique(y):
        members = X[y == group]
        between += len(members) * ((members.mean(axis=0) - grand) ** 2).sum()

    return float(between / total)


def separation_over_chance(X, y, n_permutations: int = 5, random_state: int = 0) -> float:
    """How many times more variance the real labels explain than shuffled ones.

    The denominator is what this grouping would explain by accident given its
    number of groups and its imbalance, so a value near 1 means the labels carry
    no information about the space -- whatever the raw eta^2 happens to be.
    """
    observed = variance_explained_by_group(X, y)

    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    permuted = np.mean(
        [variance_explained_by_group(X, rng.permutation(y)) for _ in range(n_permutations)]
    )

    if permuted == 0:
        return float("inf") if observed > 0 else 1.0
    return float(observed / permuted)


def _nearest(query, reference, k):
    """Indices of the k nearest rows of ``reference`` to each row of ``query``.

    Excludes the query point itself, which is at distance zero from its own row.
    """
    distances = ((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=-1)
    order = np.argsort(distances, axis=1, kind="stable")
    return order[:, 1 : k + 1]


def neighbourhood_preservation(
    reference,
    embedding,
    k: int = 30,
    n_probe: int = 2000,
    random_state: int = 0,
) -> float:
    """Mean fraction of each point's k nearest neighbours that survive the embedding.

    Invariant to rotation, reflection and uniform scaling, because it compares
    neighbour *sets* rather than coordinates -- an embedding is only defined up
    to those transformations.

    ``n_probe`` caps the number of points measured; all-pairs distances on a
    486,748-sample cohort are not affordable, and the estimate converges long
    before then.
    """
    reference = np.asarray(reference, dtype=float)
    embedding = np.asarray(embedding, dtype=float)

    if len(reference) != len(embedding):
        raise ValueError(f"reference has {len(reference)} samples, embedding has {len(embedding)}")

    n = len(reference)
    k = min(k, n - 1)

    if n_probe >= n:
        probe = np.arange(n)
    else:
        probe = np.random.default_rng(random_state).choice(n, size=n_probe, replace=False)

    before = _nearest(reference[probe], reference, k)
    after = _nearest(embedding[probe], embedding, k)

    kept = [len(set(a) & set(b)) for a, b in zip(before, after)]
    return float(np.mean(kept) / k)
