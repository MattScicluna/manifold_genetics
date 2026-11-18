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

from .base import EmbeddingBase

logger = logging.getLogger(__name__)


def _import_dreams_tsne():
    try:
        from openTSNE import TSNE as _TSNE
    except ImportError as exc:
        raise ImportError(
            "DREAMS needs the authors' openTSNE fork: "
            "`pip install 'openTSNE @ git+https://github.com/berenslab/DREAMS.git@tp'`."
        ) from exc
    import inspect

    if "reg_lambda" not in inspect.signature(_TSNE.__init__).parameters:
        raise ImportError(
            "The installed openTSNE is the upstream release, not the DREAMS fork "
            "(no `reg_lambda`). Install from https://github.com/berenslab/DREAMS (branch tp)."
        )
    return _TSNE


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
        self._embedding = None

    def _reference(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        ref = PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        return ref / ref[:, 0].std() * 1e-4

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "DREAMS":
        _TSNE = _import_dreams_tsne()
        X_array, self._sample_ids = self._load_input_data(X)
        X_array = np.asarray(X_array, dtype=float)
        ref = self._reference(X_array)
        logger.info(
            f"Running DREAMS (perplexity={self.perplexity}, lambda={self.reg_lambda}) "
            f"on {len(X_array)} samples..."
        )
        embedder = _TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            initialization=ref,
            regularization=True,
            reg_lambda=self.reg_lambda,
            reg_embedding=ref,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._embedding = embedder.fit(X_array)
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
