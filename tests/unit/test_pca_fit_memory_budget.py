"""Streaming must switch itself on before the in-memory fit runs out of memory.

Once the Python backend is the default, nobody passes ``fit_chunk_size``. At
60,000 samples and 172,152 variants the dense standardised matrix is 82 GB, so
an unguarded default would not fail over -- it would die.

It must also stay *off* when the matrix does fit: streaming reads the ``.bed``
once per half-iteration, 42 times at ``n_iter=20``, which cost 11m39s on HGDP
against the in-memory path's 36s. Always streaming would make every small run
nineteen times slower.

The decision is a pure function of the two dimensions and the budget, so it is
tested directly rather than inferred from timings.
"""

import pytest

from manifold_genetics.pca.backends.sklearn_backend import SklearnPCABackend

GB = 1024**3


class TestChunkSizeDecision:
    def test_small_cohort_is_read_whole(self):
        # HGDP fit: 3,400 x 172,152 x 8 bytes = 4.7 GB, under an 8 GB budget.
        backend = SklearnPCABackend(n_components=20, max_fit_memory_gb=8.0)

        assert backend._resolve_fit_chunk_size(3_400, 172_152) is None

    def test_large_cohort_is_streamed(self):
        # geosketch fit: 60,000 x 172,152 x 8 bytes = 82 GB.
        backend = SklearnPCABackend(n_components=20, max_fit_memory_gb=8.0)

        chunk = backend._resolve_fit_chunk_size(60_000, 172_152)

        assert chunk is not None and chunk > 0

    def test_the_chosen_chunk_fits_the_budget(self):
        backend = SklearnPCABackend(n_components=20, max_fit_memory_gb=8.0)

        chunk = backend._resolve_fit_chunk_size(60_000, 172_152)

        assert chunk * 60_000 * 8 <= 8.0 * GB

    def test_an_explicit_chunk_size_always_wins(self):
        backend = SklearnPCABackend(n_components=20, max_fit_memory_gb=1000.0, fit_chunk_size=123)

        assert backend._resolve_fit_chunk_size(3_400, 172_152) == 123

    def test_a_cohort_so_wide_one_variant_exceeds_the_budget_still_yields_a_chunk(self):
        # Pathological but must not return 0 and loop forever.
        backend = SklearnPCABackend(n_components=2, max_fit_memory_gb=0.000001)

        chunk = backend._resolve_fit_chunk_size(1_000_000, 10)

        assert chunk >= 1

    @pytest.mark.parametrize("budget", [0.5, 4.0, 64.0])
    def test_a_bigger_budget_never_means_a_smaller_chunk(self, budget):
        smaller = SklearnPCABackend(max_fit_memory_gb=budget)._resolve_fit_chunk_size(
            50_000, 200_000
        )
        bigger = SklearnPCABackend(max_fit_memory_gb=budget * 2)._resolve_fit_chunk_size(
            50_000, 200_000
        )

        if smaller is not None and bigger is not None:
            assert bigger >= smaller


class TestAutoStreamingProducesTheSameAnswer:
    def test_forcing_a_tiny_budget_matches_the_unchunked_fit(self, tmp_path):
        """Auto-selected streaming is the same computation, not an approximation."""
        import numpy as np

        from .test_pca_backend_sklearn import write_plink

        rng = np.random.default_rng(5)
        dosages = np.vstack([rng.binomial(2, p, size=(20, 150)).astype(float) for p in (0.2, 0.6)])
        prefix = write_plink(tmp_path / "c", dosages)

        whole = SklearnPCABackend(n_components=4, random_state=0).fit(prefix)
        auto = SklearnPCABackend(n_components=4, random_state=0, max_fit_memory_gb=1e-7).fit(prefix)

        np.testing.assert_allclose(auto.eigenvalues, whole.eigenvalues, rtol=1e-6)
