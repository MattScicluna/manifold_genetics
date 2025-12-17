"""
Neural Admixture wrapper.

Provides a user-friendly API for training neural admixture models and inferring
ancestry proportions on new samples.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

import torch

# Assuming these imports exist in your project structure
from ..utils.io import validate_plink_files
from ..utils.tools import ToolResolver

logger = logging.getLogger(__name__)


class NeuralAdmixture:
    """
    Neural admixture wrapper for ancestry inference.
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
        """
        self.k_min = k_min
        self.k_max = k_max
        self.force = force
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

        # Fitted state
        self._is_fitted = False
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

        # 3. Fallback to total system CPUs (Risky on shared nodes, but standard fallback)
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
        output_dir: Optional[Union[str, Path]] = None,
        model_name: str = "fit",
    ) -> "NeuralAdmixture":
        """Train neural admixture models on reference data."""
        plink_prefix = validate_plink_files(plink_prefix)

        if output_dir is None:
            output_dir = Path.cwd() / "admixture_outputs"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        self._train(plink_prefix, output_dir, model_name)

        self._model_dir = output_dir
        self._model_name = model_name
        self._is_fitted = True

        logger.info(f"Neural admixture fitted for K={self.k_min} to {self.k_max}")
        return self

    def transform(
        self,
        plink_prefix: Union[str, Path],
        output_prefix: Union[str, Path],
    ) -> Dict[int, Path]:
        """Infer ancestry proportions on new samples."""
        if not self._is_fitted:
            raise RuntimeError("Neural admixture not fitted. Call fit() first.")

        plink_prefix = validate_plink_files(plink_prefix)

        if self._model_dir is None:
            raise RuntimeError("Model directory not set; fit() must be called before transform().")

        # Where to write CSVs (prefix without K/extension).
        csv_prefix = Path(output_prefix)
        out_name = csv_prefix.stem

        csv_prefix.parent.mkdir(parents=True, exist_ok=True)
        q_output_dir = self._model_dir

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

        q_files = self._infer(plink_prefix, q_output_dir, out_name)

        # Convert Q files to standardized CSV format
        csv_files = self._convert_q_files_to_csv(
            q_files, plink_prefix, csv_prefix
        )

        # Remove raw inference Q files to avoid duplicates once CSVs are created
        for q_path in q_files.values():
            try:
                Path(q_path).unlink(missing_ok=True)
            except Exception as e:
                logger.debug(f"Could not remove raw Q file {q_path}: {e}")

        return csv_files  # Return CSV files instead of raw Q files

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
        csv_prefix.parent.mkdir(parents=True, exist_ok=True)

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
            self._is_fitted = True
            return csv_files

        self._train(plink_prefix, output_dir, model_name="fit")
        q_files = self._infer_on_training_data(output_dir)

        # Convert Q files to standardized CSV format
        csv_files = self._convert_q_files_to_csv(
            q_files, plink_prefix, csv_prefix
        )

        self._model_dir = output_dir
        self._model_name = "fit"
        self._is_fitted = True

        return csv_files  # Return CSV files instead of raw Q files

    def _train(
        self, plink_prefix: Path, output_dir: Path, model_name: str
    ) -> None:
        """Train neural admixture models."""
        logger.info("=" * 60)
        logger.info("NEURAL ADMIXTURE TRAINING")
        logger.info("=" * 60)

        plink_bed = plink_prefix.with_suffix(".bed")

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
        """Infer ancestry proportions."""
        logger.info("=" * 60)
        logger.info("NEURAL ADMIXTURE INFERENCE")
        logger.info("=" * 60)

        if self._model_dir is None:
            raise RuntimeError("Model directory not set; fit() must be called before inference.")

        plink_bed = plink_prefix.with_suffix(".bed")
        q_dir.mkdir(parents=True, exist_ok=True)

        q_files = {
            k: q_dir / f"{out_name}.{k}.Q" for k in range(self.k_min, self.k_max + 1)
        }
        
        # Check existing
        if all(f.exists() for f in q_files.values()) and not self.force:
             logger.info("Checkpoints found, skipping...")
             return q_files

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
                "--num_gpus", str(self.num_gpus),
            ]

            if self.batch_size is not None:
                cmd.extend(["--batch_size", str(self.batch_size)])

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
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
        """
        Convert raw Q files to standardized CSV format with sample_id column.
        
        Args:
            q_files: Dictionary mapping K values to raw Q file paths
            plink_prefix: PLINK file prefix to get sample IDs
            csv_prefix: Output path prefix (files will be written as <prefix>.<K>.csv)
            
        Returns:
            Dictionary mapping K values to CSV file paths
        """
        from ..utils.io import get_sample_ids_from_plink
        
        # Get sample IDs from PLINK files
        sample_ids = get_sample_ids_from_plink(plink_prefix)
        
        csv_files = {}
        
        base_prefix = csv_prefix.with_suffix("") if csv_prefix.suffix else csv_prefix

        for k, q_file in q_files.items():
            # Read raw Q matrix
            import pandas as pd
            import numpy as np
            
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
