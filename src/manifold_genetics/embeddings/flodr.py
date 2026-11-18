"""
FloDR: dimensionality reduction with an invertible normalising flow.

Thin wrapper around the authors' implementation (``pip install flodr``;
https://github.com/daenuprobst/flodr; Baraka & Probst, arXiv:2607.26278).
FloDR has a real ``transform``, so fitting on one cohort and projecting another
works as it does for PHATE and UMAP.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import EmbeddingBase, optional_import

logger = logging.getLogger(__name__)


class FloDR(EmbeddingBase):
    """FloDR embedding.

    Args:
        n_components: Embedding dimensions.
        w: Weight of the distance-order term; 0 keeps neighbourhoods only, larger
            values keep more global structure (FloDR default 2.0).
        max_iter: Training steps (``None`` = FloDR default, 6,400).
        random_state: Seed.
        device: Torch device (``None`` = GPU when FloDR finds one).
    """

    def __init__(
        self,
        n_components: int = 2,
        w: float = 2.0,
        max_iter: Optional[int] = None,
        random_state: Optional[int] = 42,
        device: Optional[str] = None,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.w = w
        self.max_iter = max_iter
        self.device = device
        self.model = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "FloDR":
        flodr = optional_import("flodr", "pip install 'manifold-genetics[flodr]' (Python >= 3.11)")
        X_array, self._sample_ids = self._load_input_data(X)
        logger.info(f"Fitting FloDR (w={self.w}) on {len(X_array)} samples...")
        self.model = flodr.FloDR(
            n_components=self.n_components,
            w=self.w,
            density=False,
            max_iter=self.max_iter,
            random_state=0 if self.random_state is None else self.random_state,
            device=self.device,
        ).fit(np.asarray(X_array, dtype=np.float32))
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FloDR must be fitted before transform")
        X_array, sample_ids = self._load_input_data(X)
        Y = np.asarray(self.model.transform(np.asarray(X_array, dtype=np.float32)))
        return self._format_output(Y, sample_ids)

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        self.fit(X)
        return self._format_output(np.asarray(self.model.embedding_), self._sample_ids, output_path)
