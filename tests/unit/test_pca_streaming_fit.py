"""Memory-bounded fit, so the Python backend can handle real cohort sizes.

The straightforward fit materialises the standardised matrix: for the 60,000
sample geosketch fit set at 172,152 variants that is 82 GB, which is why the
Python backend cannot be the default without this. A streaming randomized range
finder never holds more than one variant chunk plus two thin blocks --
``O(n_variants * l + n_samples * l)``, about 110 MB at these sizes, independent
of how many samples the cohort has.

Correctness is defined by agreement with the unchunked path, which is itself
pinned against flashpca in tests/integration/test_pca_flashpca_parity.py.
Subspace agreement is the right criterion: two randomized SVDs drawing different
random matrices recover the same invariant subspace, not the same floating-point
numbers.
"""

import numpy as np
import pytest

from manifold_genetics.pca.backends.sklearn_backend import SklearnPCABackend

from .test_pca_backend_sklearn import write_plink


@pytest.fixture
def cohort(tmp_path):
    """Structured cohort with a decaying spectrum, so components are ordered."""
    rng = np.random.default_rng(3)
    n_variants = 200
    blocks = [rng.binomial(2, p, size=(20, n_variants)).astype(float) for p in (0.15, 0.45, 0.75)]
    return write_plink(tmp_path / "cohort", np.vstack(blocks))


def subspace_agreement(a, b, k):
    """Largest principal angle cosine between the two k-dim column spaces."""
    qa = np.linalg.qr(a[:, :k])[0]
    qb = np.linalg.qr(b[:, :k])[0]
    return np.linalg.svd(qa.T @ qb, compute_uv=False).min()


class TestStreamingFit:
    def test_streaming_fit_recovers_the_same_eigenvalues(self, cohort):
        whole = SklearnPCABackend(n_components=5, random_state=0).fit(cohort)
        streamed = SklearnPCABackend(n_components=5, random_state=0, fit_chunk_size=37).fit(cohort)

        np.testing.assert_allclose(streamed.eigenvalues, whole.eigenvalues, rtol=1e-6)

    def test_streaming_fit_recovers_the_same_loading_subspace(self, cohort):
        whole = SklearnPCABackend(n_components=5, random_state=0).fit(cohort)
        streamed = SklearnPCABackend(n_components=5, random_state=0, fit_chunk_size=37).fit(cohort)

        assert subspace_agreement(streamed.loadings, whole.loadings, 5) > 0.9999

    def test_streaming_fit_agrees_per_component_up_to_sign(self, cohort):
        whole = SklearnPCABackend(n_components=5, random_state=0).fit(cohort)
        streamed = SklearnPCABackend(n_components=5, random_state=0, fit_chunk_size=37).fit(cohort)

        corr = [
            abs(np.corrcoef(streamed.loadings[:, i], whole.loadings[:, i])[0, 1]) for i in range(5)
        ]
        assert min(corr) > 0.999, f"worst loading correlation {min(corr):.6f}"

    def test_streaming_fit_coordinates_match_the_unchunked_ones(self, cohort):
        whole = SklearnPCABackend(n_components=4, random_state=0).fit(cohort)
        streamed = SklearnPCABackend(n_components=4, random_state=0, fit_chunk_size=50).fit(cohort)

        corr = [
            abs(np.corrcoef(streamed.fit_coords[:, i], whole.fit_coords[:, i])[0, 1])
            for i in range(4)
        ]
        assert min(corr) > 0.999

    def test_result_does_not_depend_on_the_chunk_size(self, cohort):
        a = SklearnPCABackend(n_components=4, random_state=0, fit_chunk_size=23).fit(cohort)
        b = SklearnPCABackend(n_components=4, random_state=0, fit_chunk_size=200).fit(cohort)

        np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, rtol=1e-8)
        assert subspace_agreement(a.loadings, b.loadings, 4) > 0.99999

    def test_streaming_fit_produces_a_usable_model(self, cohort):
        backend = SklearnPCABackend(n_components=3, random_state=0, fit_chunk_size=64)

        model = backend.fit(cohort)
        projected = backend.project(cohort, model)

        np.testing.assert_allclose(projected, model.fit_coords, atol=1e-8)

    def test_loadings_remain_unit_norm_under_streaming(self, cohort):
        model = SklearnPCABackend(n_components=4, random_state=0, fit_chunk_size=31).fit(cohort)

        np.testing.assert_allclose(np.linalg.norm(model.loadings, axis=0), 1.0, rtol=1e-8)

    def test_streaming_statistics_match_the_unchunked_ones_exactly(self, cohort):
        # Per-variant mean/SD are computed chunk by chunk. They are the flashpca
        # contract, so they must be exact, not merely close.
        whole = SklearnPCABackend(n_components=3, random_state=0).fit(cohort)
        streamed = SklearnPCABackend(n_components=3, random_state=0, fit_chunk_size=17).fit(cohort)

        np.testing.assert_allclose(streamed.mean, whole.mean, atol=1e-12)
        np.testing.assert_allclose(streamed.sd, whole.sd, atol=1e-12)
