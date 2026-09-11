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
    observed = ~np.isnan(dosages)
    counts = observed.sum(axis=0)

    totals = np.nansum(np.where(observed, dosages, 0.0), axis=0)
    mean = np.divide(totals, counts, out=np.zeros(counts.shape, dtype=np.float64), where=counts > 0)

    # sqrt(2p(1-p)) with p = mean/2, written so it cannot go negative on
    # floating-point noise when mean is at the 0 or 2 boundary.
    var = np.clip(mean * (1.0 - mean / 2.0), 0.0, None)
    return mean, np.sqrt(var)


def standardize_dosages(dosages: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Centre and scale dosages, imputing missing genotypes to the variant mean.

    Args:
        dosages: ``(n_samples, n_variants)`` A1 dosages, NaN where missing.
        mean: ``(n_variants,)`` reference means.
        sd: ``(n_variants,)`` reference SDs.

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

    centred = dosages - mean
    out = np.divide(centred, sd, out=np.zeros_like(centred), where=sd > 0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
