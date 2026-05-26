"""
Abstract base class for admixture backends.

Provides a clean interface for swapping between real neural-admixture compute,
precomputed results, and fake/mock backends for testing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union


class AdmixtureBackend(ABC):
    """
    Abstract base class for admixture computation backends.

    All backends must implement fit(), transform(), and fit_transform() methods
    that return admixture proportions in standardized CSV format.

    CSV format: sample_id,component_1,component_2,...,component_K
    """

    def __init__(self, k_min: int = 2, k_max: int = 10, force: bool = False, **kwargs):
        """
        Initialize backend.

        Args:
            k_min: Minimum number of ancestral populations
            k_max: Maximum number of ancestral populations
            force: If True, recompute even if outputs exist
            **kwargs: Backend-specific parameters
        """
        self.k_min = k_min
        self.k_max = k_max
        self.force = force

    @abstractmethod
    def fit(
        self,
        plink_prefix: Union[str, Path],
        output_dir: Union[str, Path],
        model_name: str = "fit",
    ) -> None:
        """
        Train/prepare admixture models on reference data.

        Args:
            plink_prefix: Path to PLINK file prefix (without extension)
            output_dir: Directory to save model outputs
            model_name: Name for the model (used in output filenames)
        """

    @abstractmethod
    def transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """
        Infer ancestry proportions on new samples.

        Args:
            plink_prefix: Path to PLINK file prefix
            output_prefix: Prefix for output CSV files (will create <prefix>.K.csv)

        Returns:
            Dictionary mapping K values to CSV file paths
            Example: {2: Path("output.2.csv"), 3: Path("output.3.csv")}
        """

    @abstractmethod
    def fit_transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """
        Train models and infer on the same data.

        Args:
            plink_prefix: Path to PLINK file prefix
            output_prefix: Prefix for output CSV files

        Returns:
            Dictionary mapping K values to CSV file paths
        """
