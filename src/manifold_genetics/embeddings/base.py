"""
Base class for dimensionality reduction methods.

Provides a common interface for all embedding algorithms.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..utils.io import read_embedding_csv, write_embedding_csv


class EmbeddingBase(ABC):
    """
    Abstract base class for dimensionality reduction methods.

    All embedding methods should inherit from this class and implement
    the fit(), transform(), and fit_transform() methods.
    """

    def __init__(self, n_components: int = 2, random_state: Optional[int] = 42):
        """
        Initialize embedding method.

        Args:
            n_components: Number of dimensions for the embedding
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "EmbeddingBase":
        """
        Fit the embedding method on the data.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            Self (for method chaining)
        """

    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Transform data using the fitted embedding.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            DataFrame with sample_id and embedding coordinates
        """

    @abstractmethod
    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fit and transform data in one step.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)
            output_path: Optional path to save output CSV

        Returns:
            DataFrame with sample_id and embedding coordinates
        """

    def _load_input_data(
        self, X: Union[np.ndarray, pd.DataFrame, str, Path]
    ) -> tuple[np.ndarray, Optional[list]]:
        """
        Load and validate input data.

        Args:
            X: Input data in various formats

        Returns:
            Tuple of (numpy array, sample_ids)
        """
        sample_ids = None

        if isinstance(X, (str, Path)):
            # Load from CSV file
            df = read_embedding_csv(X)
            if "sample_id" in df.columns:
                sample_ids = df["sample_id"].tolist()
                # Remove sample_id column for numerical processing
                X_array = df.drop(columns=["sample_id"]).values
            else:
                sample_ids = df.index.tolist()
                X_array = df.values
        elif isinstance(X, pd.DataFrame):
            # Extract from DataFrame
            if "sample_id" in X.columns:
                sample_ids = X["sample_id"].tolist()
                X_array = X.drop(columns=["sample_id"]).values
            elif X.index.name == "sample_id" or "sample_id" in str(X.index.name):
                sample_ids = X.index.tolist()
                X_array = X.values
            else:
                sample_ids = X.index.tolist()
                X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
            # Generate default sample IDs
            sample_ids = [f"sample_{i}" for i in range(len(X_array))]
        else:
            raise TypeError(
                f"Unsupported input type: {type(X)}. "
                "Expected numpy array, DataFrame, or path to CSV."
            )

        return X_array, sample_ids

    def _format_output(
        self,
        embedding: np.ndarray,
        sample_ids: list,
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Format embedding output and optionally save to file.

        Args:
            embedding: Embedding coordinates (n_samples × n_components)
            sample_ids: List of sample IDs
            output_path: Optional path to save CSV

        Returns:
            DataFrame with sample_id and embedding coordinates (sample_id, dim_1, dim_2, ...)
        """
        # Create DataFrame with dimension columns
        n_dims = embedding.shape[1]
        dim_cols = [f"dim_{i + 1}" for i in range(n_dims)]
        df = pd.DataFrame(embedding, columns=dim_cols)

        # Add sample_id column as the first column
        df.insert(0, "sample_id", sample_ids)

        # Save if output path provided
        if output_path:
            write_embedding_csv(df, output_path)

        return df
