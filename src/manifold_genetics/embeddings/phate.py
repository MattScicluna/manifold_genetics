"""
PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding).

Wrapper for the PHATE algorithm with a consistent API.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import phate

from .base import EmbeddingBase
from ..utils.io import write_embedding_csv

logger = logging.getLogger(__name__)


class PHATE(EmbeddingBase):
    """
    PHATE embedding for manifold learning.

    Examples:
        >>> # Basic usage
        >>> phate_model = PHATE(n_components=2, knn=25)
        >>> embedding = phate_model.fit_transform("pca_50.csv")
        >>>
        >>> # With custom parameters
        >>> phate_model = PHATE(n_components=2, knn=50, t=15, gamma=1)
        >>> embedding = phate_model.fit_transform("pca_50.csv", output_path="phate_2d.csv")
    """

    def __init__(
        self,
        n_components: int = 2,
        knn: int = 5,
        t: Union[int, str] = "auto",
        decay: Optional[int] = 40,
        gamma: float = 1.0,
        n_pca: Optional[int] = 100,
        n_landmark: Optional[int] = 2000,
        random_landmarking: bool = False,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        mds_solver: str = "sgd",
        embed_batch_size: Optional[int] = None,
    ):
        """
        Initialize PHATE embedding.

        Args:
            n_components: Number of embedding dimensions
            knn: Number of nearest neighbors
            t: Diffusion time (int or 'auto')
            decay: Alpha decay parameter for graph kernel
            gamma: Informational distance parameter
            n_pca: Number of PCs to compute before PHATE (None to skip)
            n_landmark: Number of landmarks for efficiency
            random_landmarking: Use random landmark selection instead of spectral clustering
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
            mds_solver: MDS solver ('sgd' or 'smacof')
            embed_batch_size: Number of samples to transform at once (None for all)
        """
        super().__init__(n_components=n_components, random_state=random_state)

        self.knn = knn
        self.t = t
        self.decay = decay
        self.gamma = gamma
        self.n_pca = n_pca
        self.n_landmark = n_landmark
        self.random_landmarking = random_landmarking
        self.n_jobs = n_jobs
        self.mds_solver = mds_solver
        self.embed_batch_size = embed_batch_size

        # Initialize PHATE model
        self.model = phate.PHATE(
            n_components=n_components,
            knn=knn,
            t=t,
            decay=decay,
            gamma=gamma,
            n_pca=n_pca,
            n_landmark=n_landmark,
            random_landmarking=random_landmarking,
            random_state=random_state,
            n_jobs=n_jobs,
            mds_solver=mds_solver,
            verbose=0,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "PHATE":
        """
        Fit PHATE on the data.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            Self (for method chaining)
        """
        X_array, self._sample_ids = self._load_input_data(X)

        logger.info(f"Fitting PHATE with knn={self.knn}, t={self.t}...")
        self.model.fit(X_array)
        self._is_fitted = True

        logger.info(f"✓ PHATE fitted on {len(X_array)} samples")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Transform data using fitted PHATE.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            DataFrame with sample_id and PHATE coordinates
        """
        if not self._is_fitted:
            raise RuntimeError("PHATE not fitted. Call fit() first.")

        X_array, sample_ids = self._load_input_data(X)
        n_samples = len(X_array)

        # Check if batch processing is needed
        if self.embed_batch_size is not None and n_samples > self.embed_batch_size:
            logger.info(
                f"🔄 BATCH MODE: Transforming {n_samples} samples with PHATE in batches of {self.embed_batch_size}..."
            )

            # Process in batches
            embeddings = []
            n_batches = int(np.ceil(n_samples / self.embed_batch_size))

            for i in range(n_batches):
                start_idx = i * self.embed_batch_size
                end_idx = min((i + 1) * self.embed_batch_size, n_samples)

                logger.info(
                    f"  Processing batch {i+1}/{n_batches} (samples {start_idx}-{end_idx})..."
                )
                batch_data = X_array[start_idx:end_idx]
                batch_embedding = self.model.transform(batch_data)
                embeddings.append(batch_embedding)

            # Concatenate all batches
            embedding = np.vstack(embeddings)
            logger.info(f"✓ Batch transformation complete: {embedding.shape}")
        else:
            # Process all at once
            if self.embed_batch_size is not None:
                logger.warning(
                    f"embed_batch_size={self.embed_batch_size} set but not using batch mode "
                    f"(data size {n_samples} <= batch size)"
                )
            logger.info(f"Transforming {n_samples} samples with PHATE (no batching)...")
            embedding = self.model.transform(X_array)

        return self._format_output(embedding, sample_ids)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fit PHATE and transform data in one step.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)
            output_path: Optional path to save output CSV

        Returns:
            DataFrame with sample_id and PHATE coordinates
        """
        X_array, sample_ids = self._load_input_data(X)

        logger.info(f"Running PHATE (knn={self.knn}, t={self.t}) on {len(X_array)} samples...")
        embedding = self.model.fit_transform(X_array)
        self._is_fitted = True
        self._sample_ids = sample_ids

        logger.info(f"✓ PHATE embedding complete: {embedding.shape}")

        # Format output with sample_id column
        return self._format_output(embedding, sample_ids, output_path)
