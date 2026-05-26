"""
Fake admixture backend for fast unit tests.

This backend generates random Q matrices on the fly, allowing for fast
unit tests without any file I/O or external dependencies.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ...utils.io import get_sample_ids_from_plink, validate_plink_files
from .base import AdmixtureBackend

logger = logging.getLogger(__name__)


class FakeAdmixtureBackend(AdmixtureBackend):
    """
    Fake admixture backend for fast unit tests.

    Generates random Q matrices (ancestry proportions) on the fly.
    Useful for unit tests that need admixture data but don't care about
    the actual values.
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 10,
        force: bool = False,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize fake backend.

        Args:
            k_min: Minimum number of ancestral populations
            k_max: Maximum number of ancestral populations
            force: If True, regenerate even if outputs exist
            random_seed: Random seed for reproducibility (default: None = random)
        """
        super().__init__(k_min, k_max, force)
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)

        logger.debug(f"Using FakeAdmixtureBackend with seed={random_seed}")

    def fit(
        self,
        plink_prefix: Union[str, Path],
        output_dir: Union[str, Path],
        model_name: str = "fit",
    ) -> None:
        """
        'Train' models (no-op for fake backend).

        Args:
            plink_prefix: Path to PLINK file prefix
            output_dir: Directory to save model outputs (not used)
            model_name: Name for the model
        """
        plink_prefix = validate_plink_files(plink_prefix)

        logger.info("=" * 60)
        logger.info("FAKE ADMIXTURE (NO TRAINING NEEDED)")
        logger.info("=" * 60)
        logger.info("Using fake backend - will generate random Q matrices")

    def transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """
        'Infer' ancestry proportions (generate random Q matrices).

        Args:
            plink_prefix: Path to PLINK file prefix
            output_prefix: Prefix for output CSV files

        Returns:
            Dictionary mapping K values to CSV file paths
        """
        plink_prefix = validate_plink_files(plink_prefix)
        csv_prefix = Path(output_prefix)
        csv_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Check if output files already exist
        base_prefix = csv_prefix.with_suffix("") if csv_prefix.suffix else csv_prefix
        csv_files = {k: Path(f"{base_prefix}.{k}.csv") for k in range(self.k_min, self.k_max + 1)}

        if all(f.exists() for f in csv_files.values()) and not self.force:
            logger.info("=" * 60)
            logger.info("FAKE ADMIXTURE INFERENCE")
            logger.info("=" * 60)
            logger.info(f"Output CSV files found, skipping...")
            return csv_files

        logger.info("=" * 60)
        logger.info("FAKE ADMIXTURE INFERENCE")
        logger.info("=" * 60)
        logger.info("Generating random Q matrices...")

        # Get sample IDs
        sample_ids = get_sample_ids_from_plink(plink_prefix)
        n_samples = len(sample_ids)

        # Generate random Q matrices for each K
        for k in range(self.k_min, self.k_max + 1):
            q_matrix = self._generate_random_q_matrix(n_samples, k)

            # Create DataFrame in standardized format
            component_cols = [f"component_{i + 1}" for i in range(k)]
            df = pd.DataFrame(q_matrix, columns=component_cols)
            df.insert(0, "sample_id", sample_ids)

            # Save to CSV
            output_file = csv_files[k]
            df.to_csv(output_file, index=False)
            logger.info(f"✓ Generated K={k}: {output_file} ({n_samples} samples)")

        return csv_files

    def fit_transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """
        'Train' and 'infer' (generate random Q matrices).

        Args:
            plink_prefix: Path to PLINK file prefix
            output_prefix: Prefix for output CSV files

        Returns:
            Dictionary mapping K values to CSV file paths
        """
        # For fake backend, fit_transform is the same as transform
        return self.transform(plink_prefix, output_prefix)

    def _generate_random_q_matrix(self, n_samples: int, k: int) -> np.ndarray:
        """
        Generate random Q matrix (ancestry proportions).

        Each row represents a sample and sums to 1.0.

        Args:
            n_samples: Number of samples
            k: Number of ancestral populations

        Returns:
            Array of shape (n_samples, k) with rows summing to 1.0
        """
        # Generate random values from Dirichlet distribution
        # Alpha=1.0 gives uniform distribution over the simplex
        alpha = np.ones(k)
        q_matrix = self._rng.dirichlet(alpha, size=n_samples)

        # Ensure rows sum to 1.0 (should already be true, but good to verify)
        q_matrix = q_matrix / q_matrix.sum(axis=1, keepdims=True)

        return q_matrix
