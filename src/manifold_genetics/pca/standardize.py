"""flashpca-compatible genotype standardisation.

flashpca is invoked with no ``--stand`` flag, so it applies its default. Its
``.meansd`` file is what makes a projection reproducible: projecting a second
cohort means applying the *reference* cohort's per-variant mean and SD, never
statistics recomputed on the new cohort.

The convention was reverse-engineered from real flashpca output over all 172,152
HGDP variants rather than taken from documentation:

* ``Mean`` is the mean A1 dosage over non-missing genotypes (agreement 5e-7)
* ``SD`` is ``sqrt(Mean * (1 - Mean/2))``, i.e. the binomial SD ``sqrt(2p(1-p))``
  with ``p = Mean/2`` -- *not* the empirical standard deviation (agreement 1.7e-6)

See tests/unit/test_pca_standardize.py, which pins both.
"""

from typing import Tuple

import numpy as np

__all__ = ["binom2_stats", "standardize_dosages"]


# Variants per block when accumulating per-variant statistics. Chosen so the
# temporary is a few hundred MB at realistic cohort sizes rather than a copy of
# the whole matrix.
_STATS_BLOCK_VARIANTS = 8192


def binom2_stats(dosages: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-variant mean and binomial SD, flashpca's default convention.

    Args:
        dosages: ``(n_samples, n_variants)`` A1 dosages, NaN where missing.

    Returns:
        ``(mean, sd)``, each ``(n_variants,)``. A variant with no observed
        genotypes gets mean 0 and SD 0 rather than NaN, so it contributes
        nothing instead of poisoning every sample's coordinates.
    """
    dosages = np.asarray(dosages, dtype=np.float64)
    # Computed in column blocks, because every whole-array route allocates a
    # second copy of the cohort: np.where(observed, dosages, 0) does it
    # explicitly, and np.nansum does it internally -- _replace_nan copies when
    # NaNs are present. On 3,400 x 172,152 that is 4.4 GB on top of the 4.4 GB
    # already held, and it OOM-killed a fit on an 8 GB node, 2026-09-13.
    #
    # A block holds n_samples x BLOCK, which is ~220 MB at this cohort size.
    n_samples, n_variants = dosages.shape
    counts = np.empty(n_variants, dtype=np.int64)
    totals = np.empty(n_variants, dtype=np.float64)
    for start in range(0, n_variants, _STATS_BLOCK_VARIANTS):
        stop = min(start + _STATS_BLOCK_VARIANTS, n_variants)
        block = dosages[:, start:stop]
        missing = np.isnan(block)
        counts[start:stop] = n_samples - missing.sum(axis=0)
        totals[start:stop] = np.where(missing, 0.0, block).sum(axis=0)

    mean = np.divide(totals, counts, out=np.zeros(counts.shape, dtype=np.float64), where=counts > 0)

    # sqrt(2p(1-p)) with p = mean/2, written so it cannot go negative on
    # floating-point noise when mean is at the 0 or 2 boundary.
    var = np.clip(mean * (1.0 - mean / 2.0), 0.0, None)
    return mean, np.sqrt(var)


def standardize_dosages(
    dosages: np.ndarray, mean: np.ndarray, sd: np.ndarray, copy: bool = True
) -> np.ndarray:
    """Centre and scale dosages, imputing missing genotypes to the variant mean.

    Args:
        dosages: ``(n_samples, n_variants)`` A1 dosages, NaN where missing.
        mean: ``(n_variants,)`` reference means.
        sd: ``(n_variants,)`` reference SDs.
        copy: False standardises into ``dosages`` and returns it, which is the
            difference between holding one array of the cohort and holding
            several. Only safe when the caller owns the array; the PCA backend
            does, having just read it.

    Returns:
        ``(n_samples, n_variants)`` standardised float64 array, always finite.
        Missing genotypes become 0 (the centred mean) and zero-SD variants become
        0 rather than infinite.

    Raises:
        ValueError: ``mean`` or ``sd`` does not have one entry per variant.
    """
    dosages = np.asarray(dosages, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)

    n_variants = dosages.shape[1]
    if mean.shape != (n_variants,) or sd.shape != (n_variants,):
        raise ValueError(
            f"mean and sd must have one entry per variant ({n_variants}); "
            f"got mean{mean.shape} and sd{sd.shape}"
        )

    if copy:
        out = dosages - mean
    else:
        # Four float64 arrays of the input's size were allocated here -- the
        # input, the centred copy, a zeros_like for the divide, and
        # nan_to_num's copy. At ~32 bytes per element a 3,400 x 172,152 fit
        # wants 17 GB, while the budget deciding whether to hold it counts 8
        # bytes per element. A fit was OOM-killed on that gap, 2026-09-13.
        out = dosages

    # Column blocks, because the whole-array form still allocated: nan_to_num
    # builds a full boolean mask per non-finite kind even under copy=False,
    # which measured +2.1 GB on that cohort. A block's masks are ~220 MB.
    for start in range(0, n_variants, _STATS_BLOCK_VARIANTS):
        stop = min(start + _STATS_BLOCK_VARIANTS, n_variants)
        block = out[:, start:stop]
        block_sd = sd[start:stop]

        if copy:
            # Already centred above; only the scaling remains.
            pass
        else:
            np.subtract(block, mean[start:stop], out=block)

        np.divide(block, block_sd, out=block, where=block_sd > 0)
        # `where` leaves those entries untouched rather than zeroed, so the
        # zero-variance columns still carry their centred values and must be set.
        zero_variance = block_sd <= 0
        if zero_variance.any():
            block[:, zero_variance] = 0.0
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return out
