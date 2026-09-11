"""PCA backend interface.

Mirrors the plugin shape already used for admixture
(``admixture/backends/base.py``): an ABC with ``fit`` and ``project``, so the
external-binary implementation and the pure-Python one are interchangeable and
the pipeline does not know which it has.

``PCAModel`` is the artefact that makes a projection reproducible. It holds the
**reference** cohort's per-variant statistics, and projecting a second cohort
means applying those -- never statistics recomputed on the new cohort.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

__all__ = ["PCABackend", "PCAModel"]

PathLike = Union[str, Path]


@dataclass
class PCAModel:
    """A fitted PCA, in flashpca's parameterisation.

    Attributes:
        mean: ``(n_variants,)`` reference per-variant A1 dosage means.
        sd: ``(n_variants,)`` reference per-variant binomial SDs.
        loadings: ``(n_variants, n_components)`` unit-norm right singular vectors.
        eigenvalues: ``(n_components,)`` equal to ``S**2 / n_variants``.
        variant_ids: ``.bim`` variant IDs, in file order.
        ref_alleles: ``.bim`` A1 alleles, the allele the dosages count.
        fit_coords: ``(n_samples, n_components)`` coordinates of the fit cohort,
            ``U * sqrt(eigenvalues)``. Optional so a model can be reconstructed
            from flashpca's text artefacts, which do not include it.
        fit_sample_ids: sample IIDs of the fit cohort, if known.
    """

    mean: np.ndarray
    sd: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray
    variant_ids: Sequence[str]
    ref_alleles: Sequence[str]
    fit_coords: Optional[np.ndarray] = None
    fit_sample_ids: Optional[List[str]] = field(default=None)

    @property
    def n_variants(self) -> int:
        return int(self.loadings.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.loadings.shape[1])


class PCABackend(ABC):
    """Fit a PCA on one cohort and project another into that space."""

    @abstractmethod
    def fit(self, plink_prefix: PathLike) -> PCAModel:
        """Fit on the cohort at ``plink_prefix`` and return the model."""

    @abstractmethod
    def project(self, plink_prefix: PathLike, model: PCAModel) -> np.ndarray:
        """Project the cohort at ``plink_prefix`` into ``model``'s space.

        Returns ``(n_samples, n_components)`` coordinates on the same scale as
        ``model.fit_coords``.
        """
