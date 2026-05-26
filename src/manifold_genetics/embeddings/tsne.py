"""
t-SNE (t-Distributed Stochastic Neighbor Embedding).

Wrapper for the t-SNE algorithm with a consistent API.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE as SklearnTSNE

from .base import EmbeddingBase
from ..utils.io import write_embedding_csv

logger = logging.getLogger(__name__)


class TSNE(EmbeddingBase):
    """
    t-SNE embedding for manifold learning.

    Examples:
        >>> # Basic usage
        >>> tsne_model = TSNE(n_components=2, perplexity=30)
        >>> embedding = tsne_model.fit_transform("pca_50.csv")
        >>>
        >>> # With custom parameters
        >>> tsne_model = TSNE(n_components=2, perplexity=50, learning_rate=200)
        >>> embedding = tsne_model.fit_transform("pca_50.csv", output_path="tsne_2d.csv")
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = "auto",
        n_iter: int = 1000,
        metric: str = "euclidean",
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize t-SNE embedding.

        Args:
            n_components: Number of embedding dimensions
            perplexity: Perplexity parameter (balance local vs global structure)
            learning_rate: Learning rate for optimization ('auto' or float)
            n_iter: Number of optimization iterations
            metric: Distance metric to use
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__(n_components=n_components, random_state=random_state)

        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.n_jobs = n_jobs

        # Initialize t-SNE model
        self.model = SklearnTSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=n_iter,  # sklearn uses max_iter, not n_iter
            metric=metric,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "TSNE":
        """
        Fit t-SNE on the data.

        Note: t-SNE does not support separate fit/transform. This method
        will store the data for later use in transform().

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            Self (for method chaining)
        """
        X_array, self._sample_ids = self._load_input_data(X)
        self._fit_data = X_array  # Store for transform
        self._is_fitted = True

        logger.info(f"✓ t-SNE prepared for {len(X_array)} samples")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Transform data using t-SNE.

        Note: t-SNE does not support out-of-sample extension. This will
        refit the model on the new data.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            DataFrame with sample_id and t-SNE coordinates
        """
        logger.warning(
            "t-SNE does not support out-of-sample projection. " "Refitting on new data..."
        )
        return self.fit_transform(X)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fit t-SNE and transform data in one step.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)
            output_path: Optional path to save output CSV

        Returns:
            DataFrame with sample_id and t-SNE coordinates
        """
        X_array, sample_ids = self._load_input_data(X)

        logger.info(f"Running t-SNE (perplexity={self.perplexity}) on {len(X_array)} samples...")
        embedding = self.model.fit_transform(X_array)
        self._is_fitted = True
        self._sample_ids = sample_ids

        logger.info(f"✓ t-SNE embedding complete: {embedding.shape}")

        # Format output with sample_id column
        return self._format_output(embedding, sample_ids, output_path)
