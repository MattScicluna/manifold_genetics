"""The PCA facade must be usable without the flashpca binary.

``PCA.__init__`` resolved flashpca eagerly, so constructing the object at all
failed on a machine without the binary -- before any decision about which
backend to use. On a pip-installed package that is the difference between "works
everywhere" and "Linux x86-64 only", so it is pinned here.
"""

import numpy as np
import pytest

from manifold_genetics.pca.flashpca import PCA
from manifold_genetics.utils.tools import ToolNotFoundError

from .test_pca_backend_sklearn import write_plink


@pytest.fixture
def no_flashpca(monkeypatch):
    """Simulate a machine where flashpca cannot be found."""

    def boom(self, *a, **kw):
        raise ToolNotFoundError("flashpca not found (simulated)")

    monkeypatch.setattr("manifold_genetics.utils.tools.ToolResolver.resolve_flashpca", boom)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """fit_transform/project default their output dir to the CWD.

    Without this the suite writes pca_outputs/ into whatever directory pytest
    was started from -- the repo root, in practice.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def cohort(tmp_path):
    rng = np.random.default_rng(0)
    dosages = np.vstack(
        [
            rng.binomial(2, 0.2, size=(12, 40)).astype(float),
            rng.binomial(2, 0.7, size=(12, 40)).astype(float),
        ]
    )
    return write_plink(tmp_path / "cohort", dosages)


class TestBackendSelection:
    def test_python_backend_constructs_without_the_flashpca_binary(self, no_flashpca):
        PCA(n_components=3, backend="python")

    def test_flashpca_backend_still_resolves_the_binary_eagerly(self, no_flashpca):
        with pytest.raises(ToolNotFoundError):
            PCA(n_components=3, backend="flashpca")

    def test_unknown_backend_is_rejected_by_name(self, no_flashpca):
        with pytest.raises(ValueError, match="backend"):
            PCA(n_components=3, backend="magic")


class TestPythonBackendEndToEnd:
    def test_fit_transform_returns_the_standard_dataframe_layout(self, cohort, no_flashpca):
        pca = PCA(n_components=3, backend="python")

        coords = pca.fit_transform(cohort)

        assert list(coords.columns) == ["sample_id", "dim_1", "dim_2", "dim_3"]
        assert len(coords) == 24

    def test_fit_transform_writes_the_requested_csv(self, cohort, no_flashpca, tmp_path):
        out = tmp_path / "out" / "fit_pca_3.csv"
        pca = PCA(n_components=3, backend="python")

        pca.fit_transform(cohort, output_path=out)

        assert out.exists()

    def test_project_after_fit_reproduces_the_fitted_coordinates(self, cohort, no_flashpca):
        pca = PCA(n_components=3, backend="python")

        fitted = pca.fit_transform(cohort)
        projected = pca.project(cohort)

        np.testing.assert_allclose(
            projected.filter(like="dim_").to_numpy(),
            fitted.filter(like="dim_").to_numpy(),
            atol=1e-8,
        )

    def test_projecting_before_fitting_is_an_error(self, cohort, no_flashpca):
        pca = PCA(n_components=3, backend="python")

        with pytest.raises(RuntimeError, match="fit"):
            pca.project(cohort)

    def test_sample_ids_come_from_the_fam_file_in_order(self, cohort, no_flashpca):
        pca = PCA(n_components=3, backend="python")

        coords = pca.fit_transform(cohort)

        assert coords.sample_id.tolist()[:3] == ["S0", "S1", "S2"]
