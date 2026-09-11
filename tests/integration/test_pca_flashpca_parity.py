"""Parity between the pure-Python PCA backend and the flashpca binary.

This is the gate that lets the pure-Python backend become the default. A subtle
divergence here shifts every downstream result -- embeddings, metrics, figures --
while still looking entirely plausible, so it is checked against the real binary
on real data rather than against a reimplementation of the same idea.

flashpca is run with no ``--stand`` flag, exactly as ``pca/flashpca.py`` invokes
it, so this pins its *default* behaviour. The conventions asserted here were
reverse-engineered from its output over all 172,152 HGDP variants:

    dosage       = count of A1 (.bim column 5)
    mean, sd     = mean over non-missing, sqrt(mean * (1 - mean/2))
    eigenvalues  = S**2 / n_variants
    loadings     = right singular vectors, unit-norm columns
    PC (fit)     = U * sqrt(eigenvalues)
    PC (project) = X_new @ loadings / sqrt(n_variants), using REFERENCE mean/sd

Singular vector signs are arbitrary, so every comparison is sign-aligned from the
loadings; a flipped PC is correct, not a failure.

Run it:

    pytest tests/integration/test_pca_flashpca_parity.py -m "slow and integration" -v
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.pca.backends import SklearnPCABackend

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "examples" / "hgdp_1kgp" / "data"
FLASHPCA = REPO / "bin" / "flashpca"
FIT = DATA / "fit_subset"
PROJECT = DATA / "project_subset"

pytestmark = [pytest.mark.slow, pytest.mark.integration]

if not (DATA / "fit_subset.bed").exists() or not FLASHPCA.exists():
    pytest.skip(
        "HGDP+1KGP example data or bin/flashpca not available",
        allow_module_level=True,
    )

N_PCS = 20
N_VARIANTS = 172152


@pytest.fixture(scope="module")
def flashpca_outputs(tmp_path_factory):
    """Run the real flashpca binary: fit on fit_subset, project project_subset."""
    out = tmp_path_factory.mktemp("flashpca")
    subprocess.run(
        [
            str(FLASHPCA),
            "--bfile",
            str(FIT),
            "--outpc",
            str(out / "fit.PC"),
            "--outload",
            str(out / "fit.loadings"),
            "--outmeansd",
            str(out / "fit.meansd"),
            "--outvec",
            str(out / "fit.eigenvec"),
            "--outval",
            str(out / "fit.eigenval"),
            "-d",
            str(N_PCS),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=out,
    )
    subprocess.run(
        [
            str(FLASHPCA),
            "--bfile",
            str(PROJECT),
            "--project",
            "--inload",
            str(out / "fit.loadings"),
            "--inmeansd",
            str(out / "fit.meansd"),
            "--outproj",
            str(out / "project.PC"),
            "-d",
            str(N_PCS),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=out,
    )
    return {
        "meansd": pd.read_csv(out / "fit.meansd", sep="\t"),
        "loadings": pd.read_csv(out / "fit.loadings", sep="\t").filter(regex=r"^V\d+$").to_numpy(),
        "eigenval": np.loadtxt(out / "fit.eigenval"),
        "fit_pc": pd.read_csv(out / "fit.PC", sep="\t").filter(like="PC").to_numpy(),
        "project_pc": pd.read_csv(out / "project.PC", sep="\t").filter(like="PC").to_numpy(),
    }


@pytest.fixture(scope="module")
def model():
    return SklearnPCABackend(n_components=N_PCS, random_state=42).fit(FIT)


def sign_alignment(ours, theirs):
    """+1/-1 per component, from the dominant-magnitude entry of each column."""
    idx = np.argmax(np.abs(theirs), axis=0)
    ref = np.sign(theirs[idx, np.arange(theirs.shape[1])])
    mine = np.sign(ours[idx, np.arange(ours.shape[1])])
    return np.where(ref * mine < 0, -1.0, 1.0)


class TestStandardisationParity:
    def test_per_variant_means_match_flashpca(self, model, flashpca_outputs):
        np.testing.assert_allclose(model.mean, flashpca_outputs["meansd"].Mean.values, atol=1e-5)

    def test_per_variant_sds_match_flashpca(self, model, flashpca_outputs):
        np.testing.assert_allclose(model.sd, flashpca_outputs["meansd"].SD.values, atol=1e-5)

    def test_reference_allele_is_the_bim_a1_column(self, model, flashpca_outputs):
        assert list(model.ref_alleles) == list(flashpca_outputs["meansd"].RefAllele.values)


class TestDecompositionParity:
    def test_eigenvalues_match_flashpca(self, model, flashpca_outputs):
        np.testing.assert_allclose(
            model.eigenvalues, flashpca_outputs["eigenval"][:N_PCS], rtol=1e-5
        )

    def test_loadings_match_flashpca_up_to_sign(self, model, flashpca_outputs):
        theirs = flashpca_outputs["loadings"][:, :N_PCS]
        ours = model.loadings * sign_alignment(model.loadings, theirs)
        corr = [abs(np.corrcoef(ours[:, i], theirs[:, i])[0, 1]) for i in range(N_PCS)]
        assert min(corr) > 0.9999, f"worst loading correlation {min(corr):.6f}"

    def test_fit_coordinates_match_flashpca_up_to_sign(self, model, flashpca_outputs):
        theirs = flashpca_outputs["fit_pc"][:, :N_PCS]
        ours = model.fit_coords * sign_alignment(model.fit_coords, theirs)
        corr = [abs(np.corrcoef(ours[:, i], theirs[:, i])[0, 1]) for i in range(N_PCS)]
        assert min(corr) > 0.9999, f"worst fit-PC correlation {min(corr):.6f}"


class TestProjectionParity:
    """Projection is where a standardisation mistake does its damage silently."""

    @pytest.fixture(scope="class")
    def projected(self, model):
        return SklearnPCABackend(n_components=N_PCS, random_state=42).project(PROJECT, model)

    def test_projection_matches_flashpca_up_to_sign(self, projected, flashpca_outputs):
        theirs = flashpca_outputs["project_pc"][:, :N_PCS]
        ours = projected * sign_alignment(projected, theirs)
        corr = [abs(np.corrcoef(ours[:, i], theirs[:, i])[0, 1]) for i in range(N_PCS)]
        assert min(corr) > 0.9999, f"worst projected-PC correlation {min(corr):.6f}"

    def test_leading_projected_components_agree_numerically_not_just_in_shape(
        self, projected, flashpca_outputs
    ):
        # Correlation tolerates a scale error; this catches one.
        theirs = flashpca_outputs["project_pc"][:, :5]
        ours = (projected * sign_alignment(projected, flashpca_outputs["project_pc"]))[:, :5]
        np.testing.assert_allclose(ours, theirs, atol=1e-4)

    def test_projection_is_chunk_invariant_on_real_data(self, model, projected):
        chunked = SklearnPCABackend(
            n_components=N_PCS, random_state=42, variant_chunk_size=20000
        ).project(PROJECT, model)
        np.testing.assert_allclose(chunked, projected, atol=1e-10)


class TestFacadeParity:
    """The two backends through the public PCA facade, on the real cohort.

    This is the gate for making the Python backend the default: it compares the
    `sample_id, dim_1, ...` frames the pipeline actually consumes, not internal
    arrays, so a discrepancy in sample ordering or ID formatting shows up too.
    """

    @pytest.fixture(scope="class")
    def frames(self, tmp_path_factory):
        from manifold_genetics.pca.flashpca import PCA

        out = tmp_path_factory.mktemp("facade")
        flash = PCA(n_components=N_PCS, flashpca_path=str(FLASHPCA), backend="flashpca")
        flash_fit = flash.fit_transform(FIT, output_path=out / "flash_fit.csv")
        flash_proj = flash.project(PROJECT, output_path=out / "flash_proj.csv")

        py = PCA(n_components=N_PCS, backend="python")
        py_fit = py.fit_transform(FIT, output_path=out / "py_fit.csv")
        py_proj = py.project(PROJECT, output_path=out / "py_proj.csv")
        return flash_fit, flash_proj, py_fit, py_proj

    def test_sample_ids_and_order_are_identical(self, frames):
        flash_fit, flash_proj, py_fit, py_proj = frames
        assert py_fit.sample_id.tolist() == flash_fit.sample_id.tolist()
        assert py_proj.sample_id.tolist() == flash_proj.sample_id.tolist()

    def test_column_layout_is_identical(self, frames):
        flash_fit, _, py_fit, _ = frames
        assert list(py_fit.columns) == list(flash_fit.columns)

    def test_fit_coordinates_agree_up_to_sign(self, frames):
        flash_fit, _, py_fit, _ = frames
        theirs = flash_fit.filter(like="dim_").to_numpy()
        ours = py_fit.filter(like="dim_").to_numpy() * sign_alignment(
            py_fit.filter(like="dim_").to_numpy(), theirs
        )
        corr = [abs(np.corrcoef(ours[:, i], theirs[:, i])[0, 1]) for i in range(N_PCS)]
        assert min(corr) > 0.9999, f"worst fit correlation {min(corr):.6f}"

    def test_projected_coordinates_agree_up_to_sign(self, frames):
        _, flash_proj, _, py_proj = frames
        theirs = flash_proj.filter(like="dim_").to_numpy()
        ours = py_proj.filter(like="dim_").to_numpy() * sign_alignment(
            py_proj.filter(like="dim_").to_numpy(), theirs
        )
        corr = [abs(np.corrcoef(ours[:, i], theirs[:, i])[0, 1]) for i in range(N_PCS)]
        assert min(corr) > 0.9999, f"worst projection correlation {min(corr):.6f}"
