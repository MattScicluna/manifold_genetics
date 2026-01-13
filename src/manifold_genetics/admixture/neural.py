"""
Neural Admixture wrapper.

Provides a user-friendly API for training neural admixture models and inferring
ancestry proportions on new samples.

This class is now a thin wrapper around admixture backends, allowing for
easy swapping between real computation, precomputed fixtures, and fake data.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from .backends import AdmixtureBackend, NeuralAdmixtureBackend

logger = logging.getLogger(__name__)


class NeuralAdmixture:
    """
    Neural admixture wrapper for ancestry inference.

    This class provides a user-friendly API and delegates the actual computation
    to a backend. By default, it uses NeuralAdmixtureBackend for real computation,
    but you can inject different backends for testing.

    Examples:
        >>> # Default: real neural-admixture computation
        >>> admix = NeuralAdmixture(k_min=2, k_max=5)
        >>> admix.fit("data/hgdp.plink", output_dir="admixture/")
        >>> q_files = admix.transform("data/ukbb.plink", output_prefix="admixture/ukbb")
        >>>
        >>> # For testing: use precomputed backend
        >>> from manifold_genetics.admixture.backends import PrecomputedAdmixtureBackend
        >>> backend = PrecomputedAdmixtureBackend(k_min=2, k_max=3)
        >>> admix = NeuralAdmixture(backend=backend)
        >>> q_files = admix.fit_transform("data/test.plink", output_prefix="test/output")
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 10,
        neural_admixture_path: Optional[str] = None,
        force: bool = False,
        threads: Optional[int] = None,
        num_gpus: Optional[int] = None,
        batch_size: Optional[int] = None,
        backend: Optional[AdmixtureBackend] = None,
    ):
        """
        Initialize Neural Admixture analyzer.

        Args:
            k_min: Minimum number of ancestral populations
            k_max: Maximum number of ancestral populations
            neural_admixture_path: Path to executable (None = auto-detect)
            force: If True, retrain even if models exist
            threads: Number of threads to use. If None, attempts to detect
                     available CPUs (respecting SLURM/HPC limits).
            num_gpus: Number of GPUs to use. If None, uses 1 when CUDA is
                      available and 0 otherwise.
            batch_size: Batch size for training and inference. If None, uses
                        neural-admixture defaults.
            backend: Optional backend to use for computation. If None, uses
                     NeuralAdmixtureBackend (real computation). For testing,
                     you can inject PrecomputedAdmixtureBackend or FakeAdmixtureBackend.
        """
        self.k_min = k_min
        self.k_max = k_max
        self.force = force

        # Use provided backend or create default NeuralAdmixtureBackend
        if backend is None:
            backend = NeuralAdmixtureBackend(
                k_min=k_min,
                k_max=k_max,
                force=force,
                neural_admixture_path=neural_admixture_path,
                threads=threads,
                num_gpus=num_gpus,
                batch_size=batch_size,
            )

        self.backend = backend
        logger.debug(f"Using backend: {type(backend).__name__}")

    def fit(
        self,
        plink_prefix: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        model_name: str = "fit",
    ) -> "NeuralAdmixture":
        """
        Train neural admixture models on reference data.

        Args:
            plink_prefix: Path to PLINK file prefix (without extension)
            output_dir: Directory to save model outputs (default: ./admixture_outputs)
            model_name: Name for the model (used in output filenames)

        Returns:
            Self (for method chaining)
        """
        if output_dir is None:
            output_dir = Path.cwd() / "admixture_outputs"

        # Delegate to backend
        self.backend.fit(plink_prefix, output_dir, model_name)

        logger.info(f"Neural admixture fitted for K={self.k_min} to {self.k_max}")
        return self

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
        # Delegate to backend
        return self.backend.transform(plink_prefix, output_prefix)

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
        # Delegate to backend
        return self.backend.fit_transform(plink_prefix, output_prefix)
