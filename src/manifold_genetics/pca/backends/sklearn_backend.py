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

from ..plink import count_lines, read_bed_dosages, read_bim_variants, read_fam_ids
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
        dosages = read_bed_dosages(plink_prefix, n_samples=n_samples, n_variants=n_variants)
        mean, sd = binom2_stats(dosages)
        X = standardize_dosages(dosages, mean, sd)
        del dosages

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

        return PCAModel(
            mean=mean,
            sd=sd,
            loadings=Vt.T,
            eigenvalues=eigenvalues,
            variant_ids=variant_ids,
            ref_alleles=ref_alleles,
            fit_coords=U * np.sqrt(eigenvalues),
            fit_sample_ids=read_fam_ids(plink_prefix),
        )

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
