"""
Precomputed admixture backend for testing.

This backend loads precomputed admixture results from fixtures instead of
running expensive neural-admixture training. Useful for integration tests.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

from ...utils.io import validate_plink_files
from .base import AdmixtureBackend

logger = logging.getLogger(__name__)


class PrecomputedAdmixtureBackend(AdmixtureBackend):
    """
    Precomputed admixture backend for testing.

    Loads precomputed admixture results from test fixtures instead of
    running real neural-admixture computation.
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 10,
        force: bool = False,
        fixtures_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize precomputed backend.

        Args:
            k_min: Minimum number of ancestral populations
            k_max: Maximum number of ancestral populations
            force: If True, re-copy fixtures even if outputs exist
            fixtures_dir: Directory containing precomputed fixtures
                         (defaults to tests/fixtures/admixture/)
        """
        super().__init__(k_min, k_max, force)

        if fixtures_dir is None:
            # Default to tests/fixtures/admixture relative to package root
            fixtures_dir = Path(__file__).parents[4] / "tests" / "fixtures" / "admixture"

        self.fixtures_dir = Path(fixtures_dir)

        if not self.fixtures_dir.exists():
            raise FileNotFoundError(
                f"Fixtures directory not found: {self.fixtures_dir}. "
                f"Make sure test fixtures exist before using PrecomputedAdmixtureBackend."
            )

        logger.debug(f"Using fixtures from: {self.fixtures_dir}")

        # State tracking (not really needed for precomputed, but maintains interface)
        self._dataset_name: Optional[str] = None

    def fit(
        self,
        plink_prefix: Union[str, Path],
        output_dir: Union[str, Path],
        model_name: str = "fit",
    ) -> None:
        """
        'Train' models (actually just a no-op for precomputed backend).

        Args:
            plink_prefix: Path to PLINK file prefix (used to determine dataset name)
            output_dir: Directory to save model outputs (not used, just for compatibility)
            model_name: Name for the model
        """
        plink_prefix = validate_plink_files(plink_prefix)

        # Extract dataset name from path (e.g., "fit", "transform")
        self._dataset_name = Path(plink_prefix).stem

        logger.info("=" * 60)
        logger.info("PRECOMPUTED ADMIXTURE (NO TRAINING NEEDED)")
        logger.info("=" * 60)
        logger.info(f"Using precomputed fixtures for dataset: {self._dataset_name}")
        logger.info(f"Fixtures directory: {self.fixtures_dir}")

    def transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """
        'Infer' ancestry proportions (actually copy from fixtures).

        Args:
            plink_prefix: Path to PLINK file prefix
            output_prefix: Prefix for output CSV files

        Returns:
            Dictionary mapping K values to CSV file paths
        """
        plink_prefix = validate_plink_files(plink_prefix)
        csv_prefix = Path(output_prefix)
        csv_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Determine dataset name from plink prefix
        dataset_name = Path(plink_prefix).stem

        # Check if output files already exist
        base_prefix = csv_prefix.with_suffix("") if csv_prefix.suffix else csv_prefix
        csv_files = {
            k: Path(f"{base_prefix}.{k}.csv") for k in range(self.k_min, self.k_max + 1)
        }

        if all(f.exists() for f in csv_files.values()) and not self.force:
            logger.info("=" * 60)
            logger.info("PRECOMPUTED ADMIXTURE INFERENCE")
            logger.info("=" * 60)
            logger.info(f"Output CSV files found, skipping...")
            return csv_files

        logger.info("=" * 60)
        logger.info("PRECOMPUTED ADMIXTURE INFERENCE")
        logger.info("=" * 60)
        logger.info(f"Copying precomputed fixtures for dataset: {dataset_name}")

        # Copy fixtures to output location
        for k in range(self.k_min, self.k_max + 1):
            fixture_file = self.fixtures_dir / f"{dataset_name}.{k}.csv"

            if not fixture_file.exists():
                raise FileNotFoundError(
                    f"Fixture not found: {fixture_file}. "
                    f"Available fixtures: {list(self.fixtures_dir.glob('*.csv'))}"
                )

            output_file = csv_files[k]
            shutil.copy2(fixture_file, output_file)
            logger.info(f"✓ Copied K={k}: {fixture_file} -> {output_file}")

        return csv_files

    def fit_transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """
        'Train' and 'infer' (actually just copy fixtures).

        Args:
            plink_prefix: Path to PLINK file prefix
            output_prefix: Prefix for output CSV files

        Returns:
            Dictionary mapping K values to CSV file paths
        """
        plink_prefix = validate_plink_files(plink_prefix)
        csv_prefix = Path(output_prefix)
        output_dir = csv_prefix.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract dataset name
        dataset_name = Path(plink_prefix).stem
        self._dataset_name = dataset_name

        # Check if output files already exist
        base_prefix = csv_prefix.with_suffix("") if csv_prefix.suffix else csv_prefix
        csv_files = {
            k: Path(f"{base_prefix}.{k}.csv") for k in range(self.k_min, self.k_max + 1)
        }

        if all(f.exists() for f in csv_files.values()) and not self.force:
            logger.info("=" * 60)
            logger.info("PRECOMPUTED ADMIXTURE FIT_TRANSFORM")
            logger.info("=" * 60)
            logger.info(f"Output CSV files found, skipping...")
            return csv_files

        logger.info("=" * 60)
        logger.info("PRECOMPUTED ADMIXTURE FIT_TRANSFORM")
        logger.info("=" * 60)
        logger.info(f"Copying precomputed fixtures for dataset: {dataset_name}")

        # Copy fixtures to output location
        for k in range(self.k_min, self.k_max + 1):
            fixture_file = self.fixtures_dir / f"{dataset_name}.{k}.csv"

            if not fixture_file.exists():
                raise FileNotFoundError(
                    f"Fixture not found: {fixture_file}. "
                    f"Available fixtures: {list(self.fixtures_dir.glob('*.csv'))}"
                )

            output_file = csv_files[k]
            shutil.copy2(fixture_file, output_file)
            logger.info(f"✓ Copied K={k}: {fixture_file} -> {output_file}")

        return csv_files
