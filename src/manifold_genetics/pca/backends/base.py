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

    def save(self, path: PathLike) -> None:
        """Persist to a .npz so a re-run can skip refitting."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            sd=self.sd,
            loadings=self.loadings,
            eigenvalues=self.eigenvalues,
            variant_ids=np.asarray(self.variant_ids, dtype=object),
            ref_alleles=np.asarray(self.ref_alleles, dtype=object),
            fit_coords=(self.fit_coords if self.fit_coords is not None else np.empty(0)),
            fit_sample_ids=np.asarray(self.fit_sample_ids or [], dtype=object),
        )

    @classmethod
    def load(cls, path: PathLike) -> "PCAModel":
        """Load a model written by :meth:`save`.

        Raises whatever numpy raises on a file that is not a readable .npz; the
        caller decides whether a corrupt checkpoint is fatal or merely ignored.
        """
        with np.load(Path(path), allow_pickle=True) as z:
            coords = z["fit_coords"]
            ids = list(z["fit_sample_ids"])
            return cls(
                mean=z["mean"],
                sd=z["sd"],
                loadings=z["loadings"],
                eigenvalues=z["eigenvalues"],
                variant_ids=list(z["variant_ids"]),
                ref_alleles=list(z["ref_alleles"]),
                fit_coords=coords if coords.size else None,
                fit_sample_ids=ids or None,
            )

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
