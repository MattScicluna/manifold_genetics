"""The Python PCA backend must checkpoint like the flashpca one does.

flashpca's backend is idempotent because ``_run_flashpca_fit`` sees its own
output files and skips. Without an equivalent, switching the default to the
Python backend would silently make every pipeline re-run recompute PCA from
scratch -- on a 486k-sample cohort that is the difference between minutes and
most of an hour.

The Python backend checkpoints on the *same* artefact set flashpca writes, not a
private format, so the output contract does not depend on the backend and a
model fitted by either is readable by the other.

Reuse is asserted by changing the genotypes underneath and checking the *stale*
answer comes back. That is the only way to prove the cache was read rather than
the work quietly redone.
"""

import numpy as np
import pytest

from manifold_genetics.pca.flashpca import PCA

from .test_pca_backend_sklearn import write_plink


@pytest.fixture
def cohort_a():
    rng = np.random.default_rng(0)
    return np.vstack(
        [
            rng.binomial(2, 0.2, size=(12, 40)).astype(float),
            rng.binomial(2, 0.7, size=(12, 40)).astype(float),
        ]
    )


@pytest.fixture
def cohort_b():
    rng = np.random.default_rng(99)
    return np.vstack(
        [
            rng.binomial(2, 0.05, size=(12, 40)).astype(float),
            rng.binomial(2, 0.9, size=(12, 40)).astype(float),
        ]
    )


class TestCheckpointing:
    def test_fit_writes_the_flashpca_artefact_set(self, tmp_path, cohort_a):
        prefix = write_plink(tmp_path / "c", cohort_a)
        out = tmp_path / "outputs"

        PCA(n_components=3, backend="python").fit(prefix, output_dir=out)

        for name in ("fit.meansd", "fit.loadings", "fit.eigenval"):
            assert (out / name).exists(), f"missing {name}"

    def test_second_fit_reuses_the_checkpoint_instead_of_recomputing(
        self, tmp_path, cohort_a, cohort_b
    ):
        prefix = write_plink(tmp_path / "c", cohort_a)
        out = tmp_path / "outputs"
        first = PCA(n_components=3, backend="python").fit(prefix, output_dir=out)
        before = first._model.eigenvalues.copy()

        write_plink(tmp_path / "c", cohort_b)  # genotypes change underneath
        second = PCA(n_components=3, backend="python").fit(prefix, output_dir=out)

        # Not exact: the checkpoint is flashpca's text format, lossy at the 10th
        # significant digit. The tolerance is still four orders of magnitude
        # tighter than the gap to a genuine recompute, which lands near 33.8.
        np.testing.assert_allclose(second._model.eigenvalues, before, rtol=1e-6)

    def test_force_recomputes_and_overwrites_the_checkpoint(self, tmp_path, cohort_a, cohort_b):
        prefix = write_plink(tmp_path / "c", cohort_a)
        out = tmp_path / "outputs"
        first = PCA(n_components=3, backend="python").fit(prefix, output_dir=out)
        before = first._model.eigenvalues.copy()

        write_plink(tmp_path / "c", cohort_b)
        second = PCA(n_components=3, backend="python", force=True).fit(prefix, output_dir=out)

        assert not np.allclose(second._model.eigenvalues, before)

    def test_checkpoint_for_a_different_component_count_is_not_reused(self, tmp_path, cohort_a):
        prefix = write_plink(tmp_path / "c", cohort_a)
        out = tmp_path / "outputs"
        PCA(n_components=3, backend="python").fit(prefix, output_dir=out)

        wider = PCA(n_components=5, backend="python").fit(prefix, output_dir=out)

        assert wider._model.n_components == 5

    def test_checkpoint_for_a_different_variant_count_is_not_reused(self, tmp_path, cohort_a):
        # A stale checkpoint from another dataset must not be applied to this one;
        # its loadings would be the wrong length and the projection meaningless.
        prefix = write_plink(tmp_path / "c", cohort_a)
        out = tmp_path / "outputs"
        PCA(n_components=3, backend="python").fit(prefix, output_dir=out)

        narrower = write_plink(tmp_path / "narrow", cohort_a[:, :20])
        model = PCA(n_components=3, backend="python").fit(narrower, output_dir=out)._model

        assert model.n_variants == 20

    def test_a_reloaded_checkpoint_projects_identically(self, tmp_path, cohort_a):
        prefix = write_plink(tmp_path / "c", cohort_a)
        out = tmp_path / "outputs"
        fresh = PCA(n_components=3, backend="python")
        fresh.fit(prefix, output_dir=out)
        expected = fresh.project(prefix)

        reloaded = PCA(n_components=3, backend="python")
        reloaded.fit(prefix, output_dir=out)
        got = reloaded.project(prefix)

        np.testing.assert_allclose(
            got.filter(like="dim_").to_numpy(),
            expected.filter(like="dim_").to_numpy(),
            atol=1e-12,
        )

    def test_a_corrupt_checkpoint_is_ignored_rather_than_crashing(self, tmp_path, cohort_a):
        prefix = write_plink(tmp_path / "c", cohort_a)
        out = tmp_path / "outputs"
        out.mkdir(parents=True)
        for name in ("fit.meansd", "fit.loadings", "fit.eigenval"):
            (out / name).write_text("this is not a flashpca artefact\n")

        model = PCA(n_components=3, backend="python").fit(prefix, output_dir=out)._model

        assert model.n_components == 3
