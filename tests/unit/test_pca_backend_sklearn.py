"""Unit tests for the pure-Python PCA backend.

The numerical contract these pin was reverse-engineered from real flashpca
output on the HGDP data (see tests/integration/test_pca_flashpca_parity.py for
the check against flashpca itself):

    U, S, Vt     = SVD(standardised genotypes)
    eigenvalues  = S**2 / n_variants
    loadings     = Vt.T, unit-norm columns
    PC (fit)     = U * sqrt(eigenvalues)
    PC (project) = X_new @ loadings / sqrt(n_variants)

The last line is the one worth guarding: projection must standardise the new
cohort with the **reference** cohort's mean and SD. Recomputing them on the new
cohort produces coordinates that look reasonable and are not in the reference
space at all.
"""

import numpy as np
import pytest

from manifold_genetics.pca.backends.sklearn_backend import (
    PROJECT_BYTES_PER_DOSAGE,
    SklearnPCABackend,
)
from manifold_genetics.pca.flashpca import PCA

# 2 -> 00, 1 -> 10, 0 -> 11, NaN -> 01  (inverse of the reader's decoding)
_DOSAGE_TO_CODE = {2: 0b00, 1: 0b10, 0: 0b11}


def write_plink(path_prefix, dosages):
    """Write .bed/.bim/.fam for a (n_samples, n_variants) A1-dosage matrix."""
    dosages = np.asarray(dosages, dtype=float)
    n_samples, n_variants = dosages.shape
    bpv = (n_samples + 3) // 4

    out = bytearray(b"\x6c\x1b\x01")
    for v in range(n_variants):
        buf = bytearray(bpv)
        for s in range(n_samples):
            d = dosages[s, v]
            code = 0b01 if np.isnan(d) else _DOSAGE_TO_CODE[int(d)]
            buf[s // 4] |= code << (2 * (s % 4))
        out += buf
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    (path_prefix.parent / f"{path_prefix.name}.bed").write_bytes(bytes(out))

    with open(path_prefix.parent / f"{path_prefix.name}.bim", "w") as fh:
        for v in range(n_variants):
            fh.write(f"1\tsnp{v}\t0\t{v + 1}\tA\tG\n")
    with open(path_prefix.parent / f"{path_prefix.name}.fam", "w") as fh:
        for s in range(n_samples):
            fh.write(f"S{s}\tS{s}\t0\t0\t0\t-9\n")
    return path_prefix


@pytest.fixture
def structured_cohort():
    """Two clearly separated groups, so PC1 is a real signal and not noise."""
    rng = np.random.default_rng(0)
    n_variants = 60
    group_a = rng.binomial(2, 0.2, size=(15, n_variants)).astype(float)
    group_b = rng.binomial(2, 0.7, size=(15, n_variants)).astype(float)
    return np.vstack([group_a, group_b])


class TestFit:
    def test_model_shapes_follow_the_requested_component_count(self, tmp_path, structured_cohort):
        prefix = write_plink(tmp_path / "ref", structured_cohort)

        model = SklearnPCABackend(n_components=4, random_state=0).fit(prefix)

        n_variants = structured_cohort.shape[1]
        assert model.loadings.shape == (n_variants, 4)
        assert model.eigenvalues.shape == (4,)
        assert model.mean.shape == (n_variants,)
        assert model.sd.shape == (n_variants,)

    def test_eigenvalues_are_scaled_by_variant_count(self, tmp_path, structured_cohort):
        # eigenvalue = S**2 / n_variants, matching flashpca's .eigenval exactly.
        prefix = write_plink(tmp_path / "ref", structured_cohort)
        backend = SklearnPCABackend(n_components=3, random_state=0)

        model = backend.fit(prefix)

        from manifold_genetics.pca.plink import read_bed_dosages
        from manifold_genetics.pca.standardize import binom2_stats, standardize_dosages

        n_s, n_v = structured_cohort.shape
        dos = read_bed_dosages(prefix, n_samples=n_s, n_variants=n_v)
        X = standardize_dosages(dos, *binom2_stats(dos))
        s = np.linalg.svd(X, compute_uv=False)[:3]
        np.testing.assert_allclose(model.eigenvalues, s**2 / n_v, rtol=1e-6)

    def test_loadings_have_unit_norm_columns(self, tmp_path, structured_cohort):
        prefix = write_plink(tmp_path / "ref", structured_cohort)

        model = SklearnPCABackend(n_components=3, random_state=0).fit(prefix)

        np.testing.assert_allclose(np.linalg.norm(model.loadings, axis=0), 1.0, rtol=1e-6)

    def test_eigenvalues_are_descending(self, tmp_path, structured_cohort):
        prefix = write_plink(tmp_path / "ref", structured_cohort)

        model = SklearnPCABackend(n_components=5, random_state=0).fit(prefix)

        assert np.all(np.diff(model.eigenvalues) <= 0)


class TestProject:
    def test_projecting_the_reference_onto_itself_reproduces_the_fit_coordinates(
        self, tmp_path, structured_cohort
    ):
        prefix = write_plink(tmp_path / "ref", structured_cohort)
        backend = SklearnPCABackend(n_components=3, random_state=0)

        model = backend.fit(prefix)
        projected = backend.project(prefix, model)

        np.testing.assert_allclose(projected, model.fit_coords, atol=1e-8)

    def test_projection_uses_reference_statistics_not_the_new_cohort_s(
        self, tmp_path, structured_cohort
    ):
        """The defining property of projection onto a reference space.

        The second cohort here has deliberately different allele frequencies. If
        the backend recomputed mean/SD on it, its coordinates would be centred on
        its own mean -- near zero -- instead of displaced from the reference,
        which is the entire point of projecting.
        """
        ref_prefix = write_plink(tmp_path / "ref", structured_cohort)
        rng = np.random.default_rng(7)
        shifted = rng.binomial(2, 0.95, size=(12, structured_cohort.shape[1])).astype(float)
        new_prefix = write_plink(tmp_path / "new", shifted)

        backend = SklearnPCABackend(n_components=3, random_state=0)
        model = backend.fit(ref_prefix)
        projected = backend.project(new_prefix, model)

        from manifold_genetics.pca.plink import read_bed_dosages
        from manifold_genetics.pca.standardize import binom2_stats, standardize_dosages

        dos = read_bed_dosages(new_prefix, n_samples=12, n_variants=shifted.shape[1])
        expected = (
            standardize_dosages(dos, model.mean, model.sd)
            @ model.loadings
            / np.sqrt(shifted.shape[1])
        )
        np.testing.assert_allclose(projected, expected, atol=1e-10)

        self_standardised = (
            standardize_dosages(dos, *binom2_stats(dos))
            @ model.loadings
            / np.sqrt(shifted.shape[1])
        )
        assert not np.allclose(
            projected, self_standardised, atol=1e-3
        ), "projection matched cohort-recomputed statistics; it must use the reference's"

    def test_rejects_a_cohort_whose_variants_do_not_match_the_reference(
        self, tmp_path, structured_cohort
    ):
        # Silently projecting a mismatched variant set produces confident nonsense.
        ref_prefix = write_plink(tmp_path / "ref", structured_cohort)
        fewer = write_plink(tmp_path / "few", structured_cohort[:, :10])

        backend = SklearnPCABackend(n_components=3, random_state=0)
        model = backend.fit(ref_prefix)

        with pytest.raises(ValueError, match="variant"):
            backend.project(fewer, model)


class TestChunking:
    def test_chunked_projection_matches_unchunked(self, tmp_path, structured_cohort):
        """Chunking exists so a 486k-sample cohort is never materialised whole.

        It must be an implementation detail with no numerical consequence.
        """
        prefix = write_plink(tmp_path / "ref", structured_cohort)
        model = SklearnPCABackend(n_components=3, random_state=0).fit(prefix)

        whole = SklearnPCABackend(n_components=3, random_state=0).project(prefix, model)
        chunked = SklearnPCABackend(n_components=3, random_state=0, variant_chunk_size=7).project(
            prefix, model
        )

        np.testing.assert_allclose(chunked, whole, atol=1e-10)

    def test_a_cohort_too_large_to_hold_is_chunked_without_being_asked(self):
        """Projection must bound its own memory, as ``max_fit_memory_gb`` does for fit.

        Until this was added, ``variant_chunk_size`` defaulted to None and was
        set nowhere outside these tests, so every real projection read the whole
        cohort in one pass. On UK Biobank (486,748 samples x 120,849 variants)
        that is 14.7 GB of raw bytes plus a 58.8 GB uint8 unpack buffer; because
        ``np.empty`` commits lazily, the allocation succeeds and the process is
        OOM-killed while faulting pages in -- no MemoryError, no traceback.
        Observed on Narval in a 32 GB allocation, 2026-09-12.
        """
        backend = SklearnPCABackend(n_components=20)

        chunk = backend._resolve_project_chunk_size(n_samples=486_748, n_variants=120_849)

        assert chunk < 120_849, "a cohort far past the budget was not chunked"
        peak = 486_748 * chunk * PROJECT_BYTES_PER_DOSAGE
        assert peak <= backend.max_project_memory_gb * 1024**3

    def test_a_cohort_that_fits_is_read_in_one_pass(self, structured_cohort):
        """Chunking costs one read per chunk, so it must not switch on early."""
        backend = SklearnPCABackend(n_components=3)

        assert backend._resolve_project_chunk_size(n_samples=3_340, n_variants=120_849) == 120_849

    def test_an_explicit_chunk_size_overrides_the_budget(self):
        backend = SklearnPCABackend(n_components=3, variant_chunk_size=7)

        assert backend._resolve_project_chunk_size(n_samples=3_340, n_variants=120_849) == 7


class TestMemoryBudgetsAreReachable:
    """The budgets must be settable by whoever knows how much memory the job has.

    Both defaulted to 8 GB and were reachable only by constructing the backend
    directly, which the pipeline does not do. So a 59,264 x 169,829 fit -- 75 GB
    dense -- streamed at roughly nineteen times the wall clock on a node with
    128 GB free, and no config key or flag could say otherwise. Asking SLURM for
    more memory changed nothing, which is the opposite of what a user expects.
    """

    def test_pca_passes_the_fit_budget_to_the_backend(self):
        pca = PCA(n_components=3, backend="python", max_fit_memory_gb=64.0)

        assert pca._py_backend.max_fit_memory_gb == 64.0

    def test_pca_passes_the_project_budget_to_the_backend(self):
        pca = PCA(n_components=3, backend="python", max_project_memory_gb=24.0)

        assert pca._py_backend.max_project_memory_gb == 24.0

    def test_a_raised_fit_budget_holds_a_matrix_that_would_otherwise_stream(self):
        """The point of the knob: 54 GB dense fits in 64 GB and must not stream."""
        backend = SklearnPCABackend(n_components=20, max_fit_memory_gb=64.0)

        assert backend._resolve_fit_chunk_size(n_samples=60_000, n_variants=120_849) is None

    def test_the_default_still_streams_that_matrix(self):
        backend = SklearnPCABackend(n_components=20)

        assert backend._resolve_fit_chunk_size(n_samples=60_000, n_variants=120_849) is not None
