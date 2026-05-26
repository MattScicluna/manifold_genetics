"""
UMAP (Uniform Manifold Approximation and Projection).

Wrapper for the UMAP algorithm with a consistent API.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import umap

from .base import EmbeddingBase
from ..utils.io import write_embedding_csv

logger = logging.getLogger(__name__)


class UMAP(EmbeddingBase):
    """
    UMAP embedding for manifold learning.

    Examples:
        >>> # Basic usage
        >>> umap_model = UMAP(n_components=2, n_neighbors=15)
        >>> embedding = umap_model.fit_transform("pca_50.csv")
        >>>
        >>> # With custom parameters
        >>> umap_model = UMAP(n_components=2, n_neighbors=50, min_dist=0.1)
        >>> embedding = umap_model.fit_transform("pca_50.csv", output_path="umap_2d.csv")
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: Optional[int] = 42,
        n_epochs: Optional[int] = None,
        learning_rate: float = 1.0,
        n_jobs: int = -1,
    ):
        """
        Initialize UMAP embedding.

        Args:
            n_components: Number of embedding dimensions
            n_neighbors: Number of nearest neighbors
            min_dist: Minimum distance between points in embedding
            metric: Distance metric to use
            random_state: Random seed
            n_epochs: Number of training epochs (None = auto)
            learning_rate: Learning rate for optimization
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__(n_components=n_components, random_state=random_state)

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs

        # Initialize UMAP model
        self.model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            verbose=False,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "UMAP":
        """
        Fit UMAP on the data.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            Self (for method chaining)
        """
        X_array, self._sample_ids = self._load_input_data(X)

        logger.info(f"Fitting UMAP with n_neighbors={self.n_neighbors}...")
        self.model.fit(X_array)
        self._is_fitted = True

        logger.info(f"✓ UMAP fitted on {len(X_array)} samples")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Transform data using fitted UMAP.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            DataFrame with sample_id and UMAP coordinates
        """
        if not self._is_fitted:
            raise RuntimeError("UMAP not fitted. Call fit() first.")

        X_array, sample_ids = self._load_input_data(X)

        logger.info(f"Transforming {len(X_array)} samples with UMAP...")
        embedding = self.model.transform(X_array)

        return self._format_output(embedding, sample_ids)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fit UMAP and transform data in one step.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)
            output_path: Optional path to save output CSV

        Returns:
            DataFrame with sample_id and UMAP coordinates
        """
        X_array, sample_ids = self._load_input_data(X)

        logger.info(f"Running UMAP (n_neighbors={self.n_neighbors}) on {len(X_array)} samples...")
        embedding = self.model.fit_transform(X_array)
        self._is_fitted = True
        self._sample_ids = sample_ids

        logger.info(f"✓ UMAP embedding complete: {embedding.shape}")

        # Format output with sample_id column
        return self._format_output(embedding, sample_ids, output_path)
