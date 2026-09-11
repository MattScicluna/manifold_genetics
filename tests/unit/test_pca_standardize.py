"""Unit tests for flashpca-compatible genotype standardisation.

flashpca is invoked with no ``--stand`` flag, so it applies its default, and its
``.meansd`` output is the definition of how a second cohort gets projected onto a
reference PCA. Reverse-engineered from real output over all 172,152 HGDP
variants:

* ``Mean`` is the mean A1 dosage over **non-missing** genotypes (matched to 5e-7)
* ``SD`` is ``sqrt(Mean * (1 - Mean/2))`` -- i.e. binom2, with ``p = Mean/2``
  (matched to 1.7e-6, the precision of flashpca's own text output)

Getting this wrong produces projections that look plausible and are wrong, so it
is pinned here rather than left implicit in the backend.
"""

import numpy as np
import pytest

from manifold_genetics.pca.standardize import binom2_stats, standardize_dosages


class TestBinom2Stats:
    def test_mean_ignores_missing_genotypes(self):
        # Mean over the three observed values is 1.0. Treating NaN as 0 gives 0.75.
        dosages = np.array([[2.0], [1.0], [0.0], [np.nan]])

        mean, _ = binom2_stats(dosages)

        np.testing.assert_allclose(mean, [1.0])

    def test_sd_is_the_binomial_sd_not_the_empirical_sd(self):
        # Empirical SD of [2, 1, 0] is 0.8165; binom2 SD at p=0.5 is sqrt(2*.5*.5)
        # = 0.7071. They differ, so this distinguishes the two conventions.
        dosages = np.array([[2.0], [1.0], [0.0]])

        _, sd = binom2_stats(dosages)

        np.testing.assert_allclose(sd, [np.sqrt(0.5)])
        assert not np.isclose(sd[0], np.std(dosages, ddof=1))

    def test_sd_follows_sqrt_mean_times_one_minus_half_mean(self):
        dosages = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        mean, sd = binom2_stats(dosages)

        np.testing.assert_allclose(sd, np.sqrt(mean * (1 - mean / 2)))

    def test_returns_one_statistic_per_variant(self):
        dosages = np.zeros((7, 3))

        mean, sd = binom2_stats(dosages)

        assert mean.shape == (3,)
        assert sd.shape == (3,)

    def test_variant_with_no_observed_genotypes_gets_zero_stats(self):
        # An all-missing variant has no defined frequency. It must not propagate
        # NaN into every sample's PCA coordinates.
        dosages = np.array([[np.nan], [np.nan]])

        mean, sd = binom2_stats(dosages)

        assert np.isfinite(mean).all()
        assert np.isfinite(sd).all()


class TestStandardizeDosages:
    def test_centres_and_scales_by_the_given_statistics(self):
        dosages = np.array([[2.0], [1.0], [0.0]])
        mean, sd = np.array([1.0]), np.array([0.5])

        out = standardize_dosages(dosages, mean, sd)

        np.testing.assert_allclose(out, [[2.0], [0.0], [-2.0]])

    def test_missing_genotypes_become_zero(self):
        # Zero is the variant's mean after centring, i.e. mean imputation, which
        # is what contributes nothing to the covariance.
        dosages = np.array([[np.nan], [1.0]])
        mean, sd = np.array([1.0]), np.array([0.5])

        out = standardize_dosages(dosages, mean, sd)

        np.testing.assert_allclose(out, [[0.0], [0.0]])
        assert np.isfinite(out).all()

    def test_monomorphic_variant_yields_zeros_not_infinities(self):
        # Every sample homozygous A1 -> mean 2, binom2 sd 0. Dividing by that sd
        # gives inf/NaN and poisons the SVD.
        dosages = np.array([[2.0], [2.0], [2.0]])
        mean, sd = binom2_stats(dosages)

        out = standardize_dosages(dosages, mean, sd)

        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, np.zeros((3, 1)))

    def test_does_not_mutate_the_input(self):
        dosages = np.array([[np.nan], [1.0]])
        before = dosages.copy()

        standardize_dosages(dosages, np.array([1.0]), np.array([0.5]))

        np.testing.assert_array_equal(dosages, before, err_msg="input was modified in place")

    @pytest.mark.parametrize("bad", [np.array([1.0, 2.0]), np.array([])])
    def test_rejects_statistics_that_do_not_match_the_variant_count(self, bad):
        dosages = np.ones((3, 1))

        with pytest.raises(ValueError, match="variant"):
            standardize_dosages(dosages, bad, bad)
