"""Sampling pairwise distances without materialising every pair.

Both preservation metrics correlate the distance between sample pairs in two
spaces. On a cohort of ``n`` samples there are ``n(n-1)/2`` pairs, which for a
biobank-scale run is ~10^11 -- far too many to compute, let alone hold in
memory. Only ``num_samples`` of them are ever used, so this module draws the
pairs first and computes distances only for those. Memory is O(num_samples)
whatever ``n`` is.

When every pair fits within ``num_samples`` the condensed ``pdist`` output is
returned unchanged, so small-cohort results are identical to computing all
pairs directly.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

__all__ = ["n_pairs", "pair_distances", "sample_pairs", "sampled_pair_distances"]

# ``Generator.choice(total, k, replace=False)`` is O(k) memory only when
# ``k`` is small relative to ``total``; otherwise numpy permutes an array of
# ``total`` indices. Below this many candidate pairs that array is cheap, so
# the exact draw is used; above it, rejection sampling keeps memory O(k).
_DIRECT_CHOICE_LIMIT = 2**24


def n_pairs(n: int) -> int:
    """Number of unordered pairs among ``n`` items."""
    return n * (n - 1) // 2


def _linear_to_pair(lin: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Map linear indices in condensed (``pdist``) order to ``(i, j)`` with ``i < j``.

    Row ``i`` of the upper triangle starts at ``S(i) = i*n - i*(i+1)/2``. The
    row is solved for in floating point, then corrected with exact integer
    arithmetic so the mapping is right for every index up to the int64 limit.
    """
    lin = np.asarray(lin, dtype=np.int64)
    n = int(n)

    def row_start(i):
        return i * n - i * (i + 1) // 2

    # Largest i with S(i) <= lin: root of i^2 - (2n-1) i + 2 lin = 0.
    disc = np.sqrt((2 * n - 1) ** 2 - 8.0 * lin)
    i = np.floor(((2 * n - 1) - disc) / 2).astype(np.int64)
    i = np.clip(i, 0, n - 2)
    i = np.where(row_start(i) > lin, i - 1, i)
    i = np.where((i < n - 2) & (row_start(i + 1) <= lin), i + 1, i)
    j = lin - row_start(i) + i + 1
    return i, j


def sample_pairs(
    n: int, num_samples: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw distinct pairs ``(i, j)``, ``i < j``, uniformly from ``n`` items.

    Returns ``min(num_samples, n(n-1)/2)`` pairs. When every pair fits, they are
    returned in condensed (``pdist``) order; otherwise they are a uniform random
    subset, drawn without ever building the full set of pairs.

    Args:
        n: Number of items.
        num_samples: Maximum number of pairs to return.
        rng: Random generator used for the draw.

    Returns:
        Tuple ``(i, j)`` of int64 index arrays of equal length.
    """
    total = n_pairs(n)
    if total <= num_samples:
        i, j = np.triu_indices(n, k=1)
        return i.astype(np.int64), j.astype(np.int64)

    logger.info(f"Subsampling {num_samples} of {total} pairwise distances...")
    if total <= _DIRECT_CHOICE_LIMIT or 4 * num_samples >= total:
        lin = rng.choice(total, num_samples, replace=False)
    else:
        # Rejection sampling: draw with replacement, keep the distinct ones,
        # top up until enough. num_samples < total/4 here, so each round
        # collides on well under a quarter of its draws.
        lin = np.unique(rng.integers(0, total, num_samples, dtype=np.int64))
        while len(lin) < num_samples:
            shortfall = num_samples - len(lin)
            extra = rng.integers(0, total, shortfall + shortfall // 4 + 1, dtype=np.int64)
            lin = np.unique(np.concatenate([lin, extra]))
        # np.unique sorts, so truncating would keep the smallest indices;
        # pick the surplus off at random instead to stay uniform.
        if len(lin) > num_samples:
            lin = rng.choice(lin, num_samples, replace=False)
    lin = np.sort(lin)
    return _linear_to_pair(lin, n)


def pair_distances(x: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Euclidean distance between rows ``x[i]`` and ``x[j]``, computed row-wise."""
    x = np.asarray(x)
    return np.linalg.norm(x[i] - x[j], axis=1)


def sampled_pair_distances(
    a: np.ndarray, b: np.ndarray, num_samples: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Euclidean distances between the same sampled pairs of rows in ``a`` and ``b``.

    ``a`` and ``b`` must have the same number of rows (one per sample). When
    all ``n(n-1)/2`` pairs fit within ``num_samples`` this is exactly
    ``(pdist(a), pdist(b))``; otherwise ``num_samples`` distinct pairs are drawn
    with :func:`sample_pairs` and only their distances are computed, keeping
    memory O(num_samples) regardless of ``n``.

    Args:
        a: Array of shape (n, d_a).
        b: Array of shape (n, d_b).
        num_samples: Maximum number of pairs to compare.
        rng: Random generator used when subsampling.

    Returns:
        Tuple ``(dist_a, dist_b)`` of equal-length 1-D arrays.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError(f"a and b must have the same number of rows: {len(a)} != {len(b)}")

    if n_pairs(len(a)) <= num_samples:
        return pdist(a, metric="euclidean"), pdist(b, metric="euclidean")

    i, j = sample_pairs(len(a), num_samples, rng)
    return pair_distances(a, i, j), pair_distances(b, i, j)
