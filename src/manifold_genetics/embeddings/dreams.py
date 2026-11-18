"""
DREAMS: t-SNE regularised towards a PCA layout.

Uses the authors' openTSNE fork (https://github.com/berenslab/DREAMS, branch
``tp``; Kury, Kobak & Damrich, TMLR 2026). The fork installs under the
``openTSNE`` import name and adds ``regularization``/``reg_lambda``/
``reg_embedding``. ``reg_lambda`` moves the result from t-SNE (0) towards the
2-D PCA reference (1). As in the authors' example, the reference is the first
two PCs of the input, scaled so PC1 has standard deviation 1e-4, and is also used
as the initialisation.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import EmbeddingBase, optional_import

logger = logging.getLogger(__name__)

_INSTALL = "pip install 'openTSNE @ git+https://github.com/berenslab/DREAMS.git@tp'"


def _import_dreams_tsne():
    import inspect

    TSNE = optional_import("openTSNE", _INSTALL).TSNE
    if "reg_lambda" not in inspect.signature(TSNE.__init__).parameters:
        raise ImportError(f"The installed openTSNE is not the DREAMS fork: {_INSTALL}")
    return TSNE


def _fit_fork(embedder, X: np.ndarray):
    """Run the fork's ``TSNE.fit`` steps, working around a bug in the fork.

    At the fork's ``tp`` head (eeea6a6, 2025-12-19), ``prepare_initial`` stores
    ``"X"`` among the gradient-descent parameters, but the fork's
    ``gradient_descent.__call__`` does not accept ``X``, so ``TSNE.fit`` raises
    ``TypeError``. This repeats ``fit`` (early exaggeration, then the main phase)
    after dropping that entry; nothing else changes.
    """
    embedding = embedder.prepare_initial(X)
    embedding.gradient_descent_params.pop("X", None)
    embedding.optimize(
        n_iter=embedder.early_exaggeration_iter,
        exaggeration=embedder.early_exaggeration,
        momentum=embedder.initial_momentum,
        inplace=True,
        propagate_exception=True,
    )
    embedding.optimize(
        n_iter=embedder.n_iter,
        exaggeration=embedder.exaggeration,
        momentum=embedder.final_momentum,
        inplace=True,
        propagate_exception=True,
    )
    return embedding


class DREAMS(EmbeddingBase):
    """DREAMS embedding.

    Args:
        n_components: Embedding dimensions (the PCA reference uses the same number of PCs).
        perplexity: t-SNE perplexity.
        reg_lambda: Regularisation strength in [0, 1]; 0 is plain t-SNE.
        random_state: Seed.
        n_jobs: Threads.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        reg_lambda: float = 0.15,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        if not 0.0 <= reg_lambda <= 1.0:
            raise ValueError(f"reg_lambda must be in [0, 1], got {reg_lambda}")
        self.perplexity = perplexity
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs

    def _reference(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        ref = PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        return ref / ref[:, 0].std() * 1e-4

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "DREAMS":
        TSNE = _import_dreams_tsne()
        X_array, self._sample_ids = self._load_input_data(X)
        X_array = np.asarray(X_array, dtype=float)
        ref = self._reference(X_array)
        logger.info(
            f"Running DREAMS (perplexity={self.perplexity}, lambda={self.reg_lambda}) "
            f"on {len(X_array)} samples..."
        )
        embedder = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            initialization=ref,
            regularization=True,
            reg_lambda=self.reg_lambda,
            reg_embedding=ref,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._embedding = _fit_fork(embedder, X_array)
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("DREAMS must be fitted before transform")
        X_array, sample_ids = self._load_input_data(X)
        if sample_ids == self._sample_ids:
            return self._format_output(np.asarray(self._embedding), sample_ids)
        # openTSNE places new points against the fitted embedding (no regularisation term).
        Y = np.asarray(self._embedding.transform(np.asarray(X_array, dtype=float)))
        return self._format_output(Y, sample_ids)

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        self.fit(X)
        return self._format_output(np.asarray(self._embedding), self._sample_ids, output_path)
