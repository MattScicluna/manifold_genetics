"""
Diffusion Maps for manifold learning.

Wrapper for diffusion maps with a consistent API.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

from .base import EmbeddingBase
from ..utils.io import write_embedding_csv

logger = logging.getLogger(__name__)


class DiffusionMap(EmbeddingBase):
    """
    Diffusion Maps embedding for manifold learning.

    Examples:
        >>> # Basic usage
        >>> dm_model = DiffusionMap(n_components=2, knn=25)
        >>> embedding = dm_model.fit_transform("pca_50.csv")
        >>>
        >>> # With custom parameters
        >>> dm_model = DiffusionMap(n_components=2, knn=50, alpha=1.0)
        >>> embedding = dm_model.fit_transform("pca_50.csv", output_path="dm_2d.csv")
    """

    def __init__(
        self,
        n_components: int = 2,
        knn: int = 5,
        alpha: float = 1.0,
        t: int = 1,
        random_state: Optional[int] = 42,
    ):
        """
        Initialize Diffusion Maps embedding.

        Args:
            n_components: Number of embedding dimensions
            knn: Number of nearest neighbors for kernel
            alpha: Normalization parameter (0=no normalization, 1=Laplacian normalization)
            t: Diffusion time (powers of diffusion operator)
            random_state: Random seed
        """
        super().__init__(n_components=n_components, random_state=random_state)

        self.knn = knn
        self.alpha = alpha
        self.t = t

        # Storage for fitted model
        self._eigenvectors = None
        self._eigenvalues = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "DiffusionMap":
        """
        Fit Diffusion Maps on the data.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            Self (for method chaining)
        """
        X_array, self._sample_ids = self._load_input_data(X)

        logger.info(f"Fitting Diffusion Maps with knn={self.knn}, alpha={self.alpha}...")

        # Compute diffusion map
        self._compute_diffusion_map(X_array)
        self._is_fitted = True

        logger.info(f"✓ Diffusion Maps fitted on {len(X_array)} samples")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Transform data using fitted Diffusion Maps.

        Note: Diffusion maps does not naturally support out-of-sample extension.
        This will use Nyström extension method.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            DataFrame with sample_id and diffusion coordinates
        """
        if not self._is_fitted:
            raise RuntimeError("Diffusion Maps not fitted. Call fit() first.")

        logger.warning(
            "Diffusion Maps out-of-sample extension is experimental. "
            "Consider refitting on combined data."
        )

        # For now, just refit (proper Nyström extension is complex)
        return self.fit_transform(X)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fit Diffusion Maps and transform data in one step.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)
            output_path: Optional path to save output CSV

        Returns:
            DataFrame with sample_id and diffusion coordinates
        """
        X_array, sample_ids = self._load_input_data(X)

        logger.info(
            f"Running Diffusion Maps (knn={self.knn}, alpha={self.alpha}) "
            f"on {len(X_array)} samples..."
        )

        # Compute diffusion map
        embedding = self._compute_diffusion_map(X_array)
        self._is_fitted = True
        self._sample_ids = sample_ids

        logger.info(f"✓ Diffusion Maps embedding complete: {embedding.shape}")

        # Format output with sample_id column
        return self._format_output(embedding, sample_ids, output_path)

    def _compute_diffusion_map(self, X: np.ndarray) -> np.ndarray:
        """
        Compute diffusion map embedding.

        Algorithm:
        1. Compute pairwise distances
        2. Build k-NN kernel
        3. Apply alpha-normalization
        4. Compute diffusion operator
        5. Eigen-decomposition
        6. Extract diffusion coordinates
        """
        n_samples = X.shape[0]

        # 1. Compute pairwise distances
        distances = squareform(pdist(X, metric="euclidean"))

        # 2. Build k-NN kernel with adaptive bandwidth
        # For each point, find k-th nearest neighbor distance
        sorted_dists = np.sort(distances, axis=1)
        epsilon = sorted_dists[:, self.knn] ** 2  # Squared distance to k-th neighbor

        # Gaussian kernel with adaptive bandwidth
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.exp(-distances[i, j] ** 2 / (epsilon[i] + 1e-10))

        # Zero out non-k-NN connections for sparsity
        for i in range(n_samples):
            neighbors = np.argsort(distances[i])[: self.knn + 1]
            mask = np.ones(n_samples, dtype=bool)
            mask[neighbors] = False
            K[i, mask] = 0

        # 3. Apply alpha-normalization (anisotropic diffusion)
        if self.alpha > 0:
            q = np.sum(K, axis=1) ** self.alpha
            K = K / (q[:, None] * q[None, :] + 1e-10)

        # 4. Row-normalize to get diffusion operator (row-stochastic matrix)
        D = np.sum(K, axis=1)
        P = K / (D[:, None] + 1e-10)

        # 5. Apply diffusion time
        if self.t > 1:
            P = np.linalg.matrix_power(P, self.t)

        # 6. Eigen-decomposition
        # Get top n_components + 1 eigenvectors (skip trivial eigenvector)
        try:
            # Use sparse eigenvalue solver for efficiency
            P_sparse = csr_matrix(P)
            eigenvalues, eigenvectors = eigs(P_sparse, k=self.n_components + 1, which="LM")

            # Sort by eigenvalue magnitude
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Skip first eigenvector (trivial, constant)
            eigenvalues = eigenvalues[1 : self.n_components + 1]
            eigenvectors = eigenvectors[:, 1 : self.n_components + 1]

        except Exception as e:
            logger.warning(f"Sparse eigenvalue solver failed: {e}. Using dense solver.")
            eigenvalues, eigenvectors = np.linalg.eig(P)

            # Sort by eigenvalue magnitude
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Skip first eigenvector (trivial)
            eigenvalues = eigenvalues[1 : self.n_components + 1]
            eigenvectors = eigenvectors[:, 1 : self.n_components + 1]

        # Store for later use
        self._eigenvalues = eigenvalues.real
        self._eigenvectors = eigenvectors.real

        # 7. Diffusion coordinates (scale eigenvectors by eigenvalues)
        embedding = eigenvectors.real * eigenvalues.real[None, :]

        return embedding
