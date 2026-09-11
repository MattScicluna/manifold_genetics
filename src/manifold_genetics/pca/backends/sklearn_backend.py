"""Pure-Python PCA backend -- no external binary.

Reproduces flashpca's parameterisation exactly, so outputs are interchangeable
with the binary's. The contract was reverse-engineered from real flashpca output
on the HGDP cohort rather than from documentation, and is pinned by
``tests/integration/test_pca_flashpca_parity.py``:

    U, S, Vt     = SVD(standardised genotypes)
    eigenvalues  = S**2 / n_variants
    loadings     = Vt.T                     (unit-norm columns)
    PC (fit)     = U * sqrt(eigenvalues)
    PC (project) = X_new @ loadings / sqrt(n_variants)

Projection standardises the new cohort with the **reference** cohort's mean and
SD. Recomputing them on the new cohort yields coordinates that are not in the
reference space.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.utils.extmath import randomized_svd

from ..plink import count_lines, read_bed_dosages, read_bim_variants, read_fam
from ..standardize import binom2_stats, standardize_dosages
from .base import PCABackend, PCAModel

logger = logging.getLogger(__name__)

__all__ = ["SklearnPCABackend"]

PathLike = Union[str, Path]


class SklearnPCABackend(PCABackend):
    """PCA via randomized SVD over genotypes read with the built-in .bed reader."""

    def __init__(
        self,
        n_components: int = 20,
        random_state: Optional[int] = 42,
        n_iter: int = 20,
        n_oversamples: Optional[int] = None,
        variant_chunk_size: Optional[int] = None,
        fit_chunk_size: Optional[int] = None,
        max_fit_memory_gb: float = 8.0,
    ):
        """
        Args:
            n_components: Number of PCs.
            random_state: Seed for the randomized SVD.
            n_iter: Power iterations. The default of 20 is not arbitrary --
                see the note on accuracy below.
            n_oversamples: Extra random vectors for the range finder. Defaults
                to ``max(40, 2 * n_components)``.
            variant_chunk_size: Variants per chunk when projecting. ``None``
                reads the cohort in one pass. Chunking bounds peak memory for
                large cohorts and must not change the result.
            fit_chunk_size: Variants per chunk when fitting. ``None`` lets
                ``max_fit_memory_gb`` decide; an explicit value always wins.
            max_fit_memory_gb: Budget for the dense standardised matrix. Below it
                the matrix is held whole; above it the fit streams, with peak
                memory ``O((n_variants + n_samples) * (n_components +
                n_oversamples))`` -- about 110 MB at real cohort sizes.
                60,000 samples at 172,152 variants would be 82 GB dense, so an
                unguarded default would not degrade, it would die. Streaming is
                not free either: it reads the ``.bed`` once per half-iteration
                (42 times at ``n_iter=20``; 11m39s on HGDP against 36s), so it
                must stay off whenever the matrix does fit.

        Accuracy: a genotype eigenvalue spectrum has a long flat tail -- on the
        HGDP cohort, PC13-PC20 span 2.81 to 2.28 with gaps as small as 0.010 --
        and near-equal eigenvalues are exactly where a randomized SVD loses the
        trailing components. sklearn's defaults (``n_iter=5``,
        ``n_oversamples=10``) give a 3e-3 eigenvalue error and 0.993 loading
        correlation against flashpca on PC20. ``n_iter=20`` with 40 oversamples
        reaches 1.5e-7, which is flashpca's own text-output precision and
        indistinguishable from an exact SVD -- while taking 36s against the
        exact SVD's 147s.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_oversamples = (
            n_oversamples if n_oversamples is not None else max(40, 2 * n_components)
        )
        self.variant_chunk_size = variant_chunk_size
        self.fit_chunk_size = fit_chunk_size
        self.max_fit_memory_gb = max_fit_memory_gb

    @staticmethod
    def _dims(prefix: PathLike):
        return count_lines(f"{prefix}.fam"), count_lines(f"{prefix}.bim")

    def fit(self, plink_prefix: PathLike) -> PCAModel:
        n_samples, n_variants = self._dims(plink_prefix)
        if self.n_components > min(n_samples, n_variants):
            raise ValueError(
                f"n_components={self.n_components} exceeds min(n_samples={n_samples}, "
                f"n_variants={n_variants})"
            )

        logger.info(f"Fitting PCA on {n_samples} samples x {n_variants} variants")

        chunk = self._resolve_fit_chunk_size(n_samples, n_variants)
        if chunk:
            logger.info(
                f"Streaming fit in chunks of {chunk} variants "
                f"(dense matrix would be {n_samples * n_variants * 8 / 1024**3:.1f} GB)"
            )
            mean, sd = self._streaming_stats(plink_prefix, n_samples, n_variants, chunk)
            U, S, Vt, sum_sq = self._streaming_svd(
                plink_prefix, n_samples, n_variants, mean, sd, chunk
            )
        else:
            dosages = read_bed_dosages(plink_prefix, n_samples=n_samples, n_variants=n_variants)
            mean, sd = binom2_stats(dosages)
            X = standardize_dosages(dosages, mean, sd)
            del dosages
            sum_sq = float(np.sum(X**2))
            U, S, Vt = randomized_svd(
                X,
                n_components=self.n_components,
                n_iter=self.n_iter,
                n_oversamples=self.n_oversamples,
                random_state=self.random_state,
            )
            del X

        eigenvalues = S**2 / n_variants
        variant_ids, ref_alleles = read_bim_variants(plink_prefix)
        fids, iids = read_fam(plink_prefix)

        return PCAModel(
            mean=mean,
            sd=sd,
            loadings=Vt.T,
            eigenvalues=eigenvalues,
            variant_ids=variant_ids,
            ref_alleles=ref_alleles,
            fit_coords=U * np.sqrt(eigenvalues),
            fit_sample_ids=iids,
            fit_family_ids=fids,
            total_variance=sum_sq / n_variants,
        )

    def _resolve_fit_chunk_size(self, n_samples: int, n_variants: int) -> Optional[int]:
        """Variants per chunk, or None to hold the whole matrix.

        A pure function of the dimensions and the budget, so the choice can be
        tested directly rather than inferred from timings.
        """
        if self.fit_chunk_size:
            return self.fit_chunk_size

        budget = self.max_fit_memory_gb * 1024**3
        if n_samples * n_variants * 8 <= budget:
            return None

        return max(1, int(budget // (n_samples * 8)))

    def _chunks(self, n_variants: int, chunk: Optional[int] = None):
        size = chunk or n_variants
        for start in range(0, n_variants, size):
            yield start, min(start + size, n_variants)

    def _streaming_stats(self, prefix, n_samples, n_variants, chunk=None):
        """Per-variant mean/SD accumulated chunk by chunk.

        Exact rather than approximate: a chunk holds every sample for the
        variants it covers, so each variant's statistics are complete within it.
        """
        mean = np.empty(n_variants)
        sd = np.empty(n_variants)
        for start, stop in self._chunks(n_variants, chunk):
            dosages = read_bed_dosages(
                prefix,
                n_samples=n_samples,
                n_variants=n_variants,
                variants=slice(start, stop),
            )
            mean[start:stop], sd[start:stop] = binom2_stats(dosages)
        return mean, sd

    def _standardised_chunks(self, prefix, n_samples, n_variants, mean, sd, chunk=None):
        for start, stop in self._chunks(n_variants, chunk):
            dosages = read_bed_dosages(
                prefix,
                n_samples=n_samples,
                n_variants=n_variants,
                variants=slice(start, stop),
            )
            yield start, stop, standardize_dosages(dosages, mean[start:stop], sd[start:stop])

    def _streaming_svd(self, prefix, n_samples, n_variants, mean, sd, chunk=None):
        """Randomized SVD that never materialises the standardised matrix.

        Halko-Martinsson-Tropp with re-orthonormalised power iterations, with
        every product against X accumulated over variant chunks. Peak memory is
        the two thin blocks (n_variants x width and n_samples x width) plus one chunk.
        """
        width = min(self.n_components + self.n_oversamples, n_samples, n_variants)
        rng = np.random.default_rng(self.random_state)

        def stream():
            return self._standardised_chunks(prefix, n_samples, n_variants, mean, sd, chunk)

        Z = rng.standard_normal((n_variants, width))
        Y = np.zeros((n_samples, width))
        sum_sq = 0.0
        for start, stop, Xc in stream():
            Y += Xc @ Z[start:stop]
            sum_sq += float(np.sum(Xc**2))  # free on a pass we already make

        for _ in range(self.n_iter):
            Y, _ = np.linalg.qr(Y)
            Z = np.empty((n_variants, width))
            for start, stop, Xc in stream():
                Z[start:stop] = Xc.T @ Y
            Z, _ = np.linalg.qr(Z)
            Y = np.zeros((n_samples, width))
            for start, stop, Xc in stream():
                Y += Xc @ Z[start:stop]

        Q, _ = np.linalg.qr(Y)

        B = np.empty((width, n_variants))
        for start, stop, Xc in stream():
            B[:, start:stop] = Q.T @ Xc

        Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
        k = self.n_components
        return (Q @ Ub)[:, :k], S[:k], Vt[:k], sum_sq

    def project(self, plink_prefix: PathLike, model: PCAModel) -> np.ndarray:
        n_samples, n_variants = self._dims(plink_prefix)
        if n_variants != model.n_variants:
            raise ValueError(
                f"cohort at {plink_prefix} has {n_variants} variants but the model was fitted "
                f"on {model.n_variants}. Both cohorts must use the same variant set, in the "
                "same order."
            )

        scale = np.sqrt(model.n_variants)
        chunk = self.variant_chunk_size or n_variants
        coords = np.zeros((n_samples, model.n_components), dtype=np.float64)

        # Projection is a sum over variants, so chunking over them is exact.
        for start in range(0, n_variants, chunk):
            stop = min(start + chunk, n_variants)
            dosages = read_bed_dosages(
                plink_prefix,
                n_samples=n_samples,
                n_variants=n_variants,
                variants=slice(start, stop),
            )
            Xc = standardize_dosages(dosages, model.mean[start:stop], model.sd[start:stop])
            coords += Xc @ model.loadings[start:stop]

        return coords / scale
