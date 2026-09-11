"""One output contract, whichever backend produced it.

The Python backend writes the same artefact set flashpca does -- .meansd,
.loadings, .eigenval, .eigenvec, .PC, pve.txt -- rather than a private format.
That keeps the documented output layout true regardless of backend, and makes
the two genuinely interchangeable: a model fitted by one can be projected by the
other, which a .npz could never offer.

Formats are taken from real flashpca output:

    fit.meansd    SNP  RefAllele  Mean  SD                 (tab, header)
    fit.loadings  SNP  RefAllele  V1..Vd                   (tab, header)
    fit.eigenval  one value per line                       (no header)
    fit.eigenvec  FID  IID  U1..Ud                         (tab, header)
    fit.PC        FID  IID  PC1..PCd                       (tab, header)
    pve.txt       one proportion per line                  (no header)
"""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.pca.backends.base import PCAModel
from manifold_genetics.pca.flashpca_format import read_model, write_model


@pytest.fixture
def model():
    """A self-consistent model: fit_coords really are U * sqrt(eigenvalues).

    Building it any other way would make the eigenvec assertion a statement
    about the fixture rather than about write_model.
    """
    rng = np.random.default_rng(0)
    loadings = np.linalg.qr(rng.standard_normal((6, 3)))[0]
    eigenvalues = np.array([9.0, 4.0, 1.0])
    U = np.linalg.qr(rng.standard_normal((4, 3)))[0]
    return PCAModel(
        mean=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
        sd=np.sqrt(
            np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
            * (1 - np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]) / 2)
        ),
        loadings=loadings,
        eigenvalues=eigenvalues,
        variant_ids=[f"rs{i}" for i in range(6)],
        ref_alleles=list("ACGTAC"),
        fit_coords=U * np.sqrt(eigenvalues),
        fit_sample_ids=[f"IID{i}" for i in range(4)],
        fit_family_ids=[f"FAM{i}" for i in range(4)],
    )


class TestWriteModel:
    def test_writes_the_full_flashpca_artefact_set(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        for name in ("fit.meansd", "fit.loadings", "fit.eigenval", "fit.eigenvec", "fit.PC"):
            assert (tmp_path / name).exists(), f"missing {name}"
        assert (tmp_path / "pve.txt").exists()

    def test_meansd_has_flashpca_columns(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        df = pd.read_csv(tmp_path / "fit.meansd", sep="\t")
        assert list(df.columns) == ["SNP", "RefAllele", "Mean", "SD"]
        assert df.SNP.tolist() == list(model.variant_ids)
        assert df.RefAllele.tolist() == list(model.ref_alleles)

    def test_loadings_columns_are_v_indexed_from_one(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        df = pd.read_csv(tmp_path / "fit.loadings", sep="\t")
        assert list(df.columns) == ["SNP", "RefAllele", "V1", "V2", "V3"]

    def test_pc_columns_carry_family_and_individual_ids(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        df = pd.read_csv(tmp_path / "fit.PC", sep="\t")
        assert list(df.columns) == ["FID", "IID", "PC1", "PC2", "PC3"]
        assert df.FID.tolist() == model.fit_family_ids
        assert df.IID.tolist() == model.fit_sample_ids

    def test_eigenvalues_are_written_one_per_line_without_a_header(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        values = np.loadtxt(tmp_path / "fit.eigenval")
        np.testing.assert_allclose(values, model.eigenvalues, rtol=1e-6)

    def test_eigenvectors_are_the_unit_norm_vectors_not_the_scaled_coordinates(
        self, tmp_path, model
    ):
        # flashpca's .eigenvec holds U (unit norm); .PC holds U * sqrt(eigenvalues).
        write_model(model, tmp_path / "fit")

        U = pd.read_csv(tmp_path / "fit.eigenvec", sep="\t").filter(regex=r"^U\d+$").to_numpy()
        np.testing.assert_allclose(np.linalg.norm(U, axis=0), 1.0, rtol=1e-6)

    def test_pve_sums_to_at_most_one(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        pve = np.loadtxt(tmp_path / "pve.txt", ndmin=1)
        assert pve.shape == (3,)
        assert 0 < pve.sum() <= 1.0 + 1e-9


class TestRoundTrip:
    def test_reading_back_recovers_the_projection_inputs(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        back = read_model(tmp_path / "fit")

        np.testing.assert_allclose(back.mean, model.mean, rtol=1e-6)
        np.testing.assert_allclose(back.sd, model.sd, rtol=1e-6)
        np.testing.assert_allclose(back.loadings, model.loadings, rtol=1e-6)
        np.testing.assert_allclose(back.eigenvalues, model.eigenvalues, rtol=1e-6)

    def test_reading_back_recovers_variant_identity(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        back = read_model(tmp_path / "fit")

        assert list(back.variant_ids) == list(model.variant_ids)
        assert list(back.ref_alleles) == list(model.ref_alleles)

    def test_reading_back_recovers_the_fit_coordinates(self, tmp_path, model):
        write_model(model, tmp_path / "fit")

        back = read_model(tmp_path / "fit")

        np.testing.assert_allclose(back.fit_coords, model.fit_coords, rtol=1e-5)
        assert back.fit_sample_ids == model.fit_sample_ids

    def test_a_model_written_without_coordinates_still_round_trips(self, tmp_path, model):
        model.fit_coords = None
        model.fit_sample_ids = None
        model.fit_family_ids = None

        write_model(model, tmp_path / "fit")
        back = read_model(tmp_path / "fit")

        assert back.fit_coords is None
        np.testing.assert_allclose(back.loadings, model.loadings, rtol=1e-6)

    def test_reading_a_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_model(tmp_path / "absent")
