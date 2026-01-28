"""
Neural-admixture backend for real admixture computation.

This backend wraps the actual neural-admixture executable and performs
real training and inference.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from ...utils.io import get_sample_ids_from_plink, validate_plink_files, _append_extension
from ...utils.tools import ToolResolver
from .base import AdmixtureBackend

logger = logging.getLogger(__name__)


class NeuralAdmixtureBackend(AdmixtureBackend):
    """
    Neural-admixture backend for real admixture computation.

    Uses the neural-admixture executable to train models and infer
    ancestry proportions.
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 10,
        force: bool = False,
        neural_admixture_path: Optional[str] = None,
        threads: Optional[int] = None,
        num_gpus: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize neural-admixture backend.

        Args:
            k_min: Minimum number of ancestral populations
            k_max: Maximum number of ancestral populations
            force: If True, retrain even if models exist
            neural_admixture_path: Path to executable (None = auto-detect)
            threads: Number of threads to use (None = auto-detect)
            num_gpus: Number of GPUs to use (None = auto-detect)
            batch_size: Batch size for training and inference
        """
        super().__init__(k_min, k_max, force)

        self.threads = self._resolve_threads(threads)
        self.num_gpus = self._resolve_num_gpus(num_gpus)
        self.batch_size = batch_size

        # Resolve neural-admixture path
        if neural_admixture_path is None:
            resolver = ToolResolver()
            neural_admixture_path = resolver.resolve_neural_admixture()

        self.nadm_exec = neural_admixture_path
        logger.debug(f"Using neural-admixture: {self.nadm_exec}")
        logger.debug(f"Using threads: {self.threads}")
        logger.debug(f"Using GPUs: {self.num_gpus}")

        # State tracking
        self._model_dir: Optional[Path] = None
        self._model_name: Optional[str] = None

    def _resolve_threads(self, requested_threads: Optional[int]) -> int:
        """
        Determines the safe number of threads to use.
        Prioritizes user input -> SLURM env vars -> OS affinity -> Total CPUs.
        """
        if requested_threads is not None and requested_threads > 0:
            return requested_threads

        # 1. Check SLURM environment (Best for HPC)
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            try:
                logger.debug(f"Detected SLURM_CPUS_PER_TASK: {slurm_cpus}")
                return int(slurm_cpus)
            except ValueError:
                pass

        # 2. Check Process Affinity (Best for Linux generally)
        if hasattr(os, "sched_getaffinity"):
            try:
                return len(os.sched_getaffinity(0))
            except Exception:
                pass

        # 3. Fallback to total system CPUs
        count = os.cpu_count()
        return count if count else 1

    def _resolve_num_gpus(self, requested_gpus: Optional[int]) -> int:
        """Resolve number of GPUs to use (default: 1 if CUDA available)."""
        if requested_gpus is not None:
            return max(0, requested_gpus)
        return 1 if torch.cuda.is_available() else 0

    def fit(
        self,
        plink_prefix: Union[str, Path],
        output_dir: Union[str, Path],
        model_name: str = "fit",
    ) -> None:
        """Train neural admixture models on reference data."""
        plink_prefix = validate_plink_files(plink_prefix)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._train(Path(plink_prefix), output_dir, model_name)

        self._model_dir = output_dir
        self._model_name = model_name

        logger.info(f"Neural admixture fitted for K={self.k_min} to {self.k_max}")

    def transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """Infer ancestry proportions on new samples."""
        if self._model_dir is None:
            raise RuntimeError("Model directory not set; fit() must be called before transform().")

        plink_prefix = validate_plink_files(plink_prefix)
        csv_prefix = Path(output_prefix)
        csv_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Check if CSV files already exist
        base_prefix = csv_prefix.with_suffix("") if csv_prefix.suffix else csv_prefix
        csv_files = {
            k: Path(f"{base_prefix}.{k}.csv") for k in range(self.k_min, self.k_max + 1)
        }

        if all(f.exists() for f in csv_files.values()) and not self.force:
            logger.info("=" * 60)
            logger.info("NEURAL ADMIXTURE INFERENCE")
            logger.info("=" * 60)
            logger.info(f"Output CSV files found in {csv_prefix.parent}, skipping inference...")
            return csv_files

        # Infer and get raw Q files
        out_name = csv_prefix.stem
        q_files = self._infer(Path(plink_prefix), self._model_dir, out_name)

        # Convert Q files to standardized CSV format
        csv_files = self._convert_q_files_to_csv(
            q_files, Path(plink_prefix), csv_prefix
        )

        # Remove raw inference Q files to avoid duplicates
        for q_path in q_files.values():
            try:
                Path(q_path).unlink(missing_ok=True)
            except Exception as e:
                logger.debug(f"Could not remove raw Q file {q_path}: {e}")

        return csv_files

    def fit_transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """Train models and infer on the same data."""
        plink_prefix = validate_plink_files(plink_prefix)
        csv_prefix = Path(output_prefix)
        output_dir = csv_prefix.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if CSV files already exist
        base_prefix = csv_prefix.with_suffix("") if csv_prefix.suffix else csv_prefix
        csv_files = {
            k: Path(f"{base_prefix}.{k}.csv") for k in range(self.k_min, self.k_max + 1)
        }

        # Check if both models and CSV files exist
        model_files = [
            output_dir / f"fit_k{k}.pt" for k in range(self.k_min, self.k_max + 1)
        ]
        if (all(f.exists() for f in csv_files.values()) and
            all(f.exists() for f in model_files) and
            not self.force):
            logger.info("=" * 60)
            logger.info("NEURAL ADMIXTURE FIT_TRANSFORM")
            logger.info("=" * 60)
            logger.info(f"Model and output CSV files found, skipping fit_transform...")
            self._model_dir = output_dir
            self._model_name = "fit"
            return csv_files

        # Train
        self._train(Path(plink_prefix), output_dir, model_name="fit")

        # Get Q files from training
        q_files = self._infer_on_training_data(output_dir)

        # Convert Q files to standardized CSV format
        csv_files = self._convert_q_files_to_csv(
            q_files, Path(plink_prefix), csv_prefix
        )

        self._model_dir = output_dir
        self._model_name = "fit"

        return csv_files

    def _train(
        self, plink_prefix: Path, output_dir: Path, model_name: str
    ) -> None:
        """Train neural admixture models."""
        logger.info("=" * 60)
        logger.info("NEURAL ADMIXTURE TRAINING")
        logger.info("=" * 60)

        plink_bed = _append_extension(plink_prefix, ".bed")

        model_files = [
            output_dir / f"{model_name}_k{k}.pt" for k in range(self.k_min, self.k_max + 1)
        ]
        if all(f.exists() for f in model_files) and not self.force:
            logger.info(f"Model checkpoint found in {output_dir}")
            return

        logger.info(f"Training models K={self.k_min}-{self.k_max}")
        logger.info(f"  Threads: {self.threads}")
        logger.info(f"  GPUs: {self.num_gpus}")

        for k in range(self.k_min, self.k_max + 1):
            k_specific_name = f"{model_name}_k{k}"
            logger.info(f"Training K={k}...")

            cmd = [
                self.nadm_exec,
                "train",
                "--k", str(k),
                "--name", k_specific_name,
                "--data_path", str(plink_bed),
                "--save_dir", str(output_dir),
                "--threads", str(self.threads),
                "--num_gpus", str(self.num_gpus),
            ]

            if self.batch_size is not None:
                cmd.extend(["--batch_size", str(self.batch_size)])

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Training failed for K={k}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                raise

        logger.info(f"✓ Training complete")

    def _infer(
        self, plink_prefix: Path, q_dir: Path, out_name: str
    ) -> Dict[int, Path]:
        """
        Infer ancestry proportions.

        Note: Inference is performed on CPU only due to a bug in neural-admixture
        where the entire dataset is loaded into GPU memory during inference.
        """
        logger.info("=" * 60)
        logger.info("NEURAL ADMIXTURE INFERENCE")
        logger.info("=" * 60)

        if self._model_dir is None:
            raise RuntimeError("Model directory not set; fit() must be called before inference.")

        plink_bed = _append_extension(plink_prefix, ".bed")
        q_dir.mkdir(parents=True, exist_ok=True)

        q_files = {
            k: q_dir / f"{out_name}.{k}.Q" for k in range(self.k_min, self.k_max + 1)
        }

        # Check existing
        if all(f.exists() for f in q_files.values()) and not self.force:
             logger.info("Checkpoints found, skipping...")
             return q_files

        # Estimate memory requirements and warn user
        try:
            sample_ids = get_sample_ids_from_plink(plink_prefix)
            n_samples = len(sample_ids)
            if n_samples > 10000:
                logger.warning(f"Large dataset detected ({n_samples:,} samples)")
                logger.warning("Neural-admixture loads entire dataset into memory during inference")
                logger.warning("This may require >100GB RAM. Consider high-memory compute nodes.")
        except Exception as e:
            logger.debug(
                "Unable to estimate memory requirements from PLINK files; "
                "proceeding without memory warning. Error: %s",
                e,
            )

        logger.info("Using CPU-only inference to avoid OOM from neural-admixture bug...")

        for k in range(self.k_min, self.k_max + 1):
            k_model_name = f"{self._model_name}_k{k}"
            logger.info(f"Inferring K={k}...")

            cmd = [
                self.nadm_exec,
                "infer",
                "--name", k_model_name,
                "--save_dir", str(self._model_dir),
                "--data_path", str(plink_bed),
                "--out_name", out_name,
                "--threads", str(self.threads),
                "--num_gpus", "0",  # Force CPU-only to avoid OOM bug
            ]

            if self.batch_size is not None:
                cmd.extend(["--batch_size", str(self.batch_size)])

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                if e.returncode == -9:  # SIGKILL
                    logger.error(f"Inference killed for K={k} (likely OOM)")
                    logger.error("Try requesting more memory (>100GB for large datasets)")
                    logger.error("Example: #SBATCH --mem=200GB")
                else:
                    logger.error(f"Inference failed for K={k}")
                    logger.error(e.stderr)
                raise

        return q_files

    def _infer_on_training_data(self, output_dir: Path) -> Dict[int, Path]:
        """Get Q files from training."""
        q_files = {}
        for k in range(self.k_min, self.k_max + 1):
            q_file = output_dir / f"fit_k{k}.{k}.Q"
            if q_file.exists():
                q_files[k] = q_file

        if not q_files:
             raise FileNotFoundError(f"No Q files found in {output_dir}")
        return q_files

    def _convert_q_files_to_csv(
        self,
        q_files: Dict[int, Path],
        plink_prefix: Path,
        csv_prefix: Path,
    ) -> Dict[int, Path]:
        """Convert raw Q files to standardized CSV format with sample_id column."""
        import pandas as pd

        # Get sample IDs from PLINK files
        sample_ids = get_sample_ids_from_plink(plink_prefix)

        csv_files = {}
        base_prefix = csv_prefix.with_suffix("") if csv_prefix.suffix else csv_prefix

        for k, q_file in q_files.items():
            # Read raw Q matrix
            q_matrix = pd.read_csv(q_file, sep=r"\s+", header=None)

            # Ensure we have matching number of samples
            if len(q_matrix) != len(sample_ids):
                logger.warning(
                    f"K={k}: Q matrix has {len(q_matrix)} samples, "
                    f"PLINK has {len(sample_ids)} samples. Using minimum."
                )
                min_len = min(len(q_matrix), len(sample_ids))
                q_matrix = q_matrix.iloc[:min_len]
                sample_ids_subset = sample_ids[:min_len]
            else:
                sample_ids_subset = sample_ids

            # Create standardized DataFrame
            component_cols = [f"component_{i+1}" for i in range(k)]
            df = pd.DataFrame(q_matrix.values, columns=component_cols)
            df.insert(0, 'sample_id', sample_ids_subset)

            # Save to CSV
            csv_file = Path(f"{base_prefix}.{k}.csv")
            df.to_csv(csv_file, index=False)
            csv_files[k] = csv_file

            logger.info(f"✓ Converted K={k} Q file to CSV: {csv_file}")

        return csv_files
