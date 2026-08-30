"""
FlashPCA wrapper for fast principal component analysis.

Provides a user-friendly API for fitting PCA on reference data and projecting
new samples, with automatic output formatting and checkpointing.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from ..utils.io import (
    validate_plink_files,
    write_embedding_csv,
)
from ..utils.tools import ToolResolver

logger = logging.getLogger(__name__)


class PCA:
    """
    FlashPCA wrapper for principal component analysis.

    Examples:
        >>> # Fit PCA on reference data
        >>> pca = PCA(n_components=50)
        >>> pca_coords = pca.fit_transform(plink_prefix="data/hgdp")
        >>> pca_coords.to_csv("pca_50.csv", index=False)
        >>>
        >>> # Project new samples onto reference PCA
        >>> pca = PCA(n_components=50)
        >>> pca.fit(plink_prefix="data/hgdp_ref")
        >>> new_coords = pca.project(plink_prefix="data/ukbb")
    """

    def __init__(
        self,
        n_components: int = 20,
        flashpca_path: Optional[str] = None,
        force: bool = False,
    ):
        """
        Initialize PCA analyzer.

        Args:
            n_components: Number of principal components to compute
            flashpca_path: Path to flashpca executable (None = auto-detect)
            force: If True, recompute even if outputs exist
        """
        self.n_components = n_components
        self.force = force

        # Resolve flashpca path
        if flashpca_path is None:
            resolver = ToolResolver()
            flashpca_path = resolver.resolve_flashpca()

        self.flashpca = flashpca_path
        logger.debug(f"Using flashpca: {self.flashpca}")

        # Fitted state
        self._is_fitted = False
        self._loadings_path: Optional[Path] = None
        self._meansd_path: Optional[Path] = None
        self._fit_output_dir: Optional[Path] = None

    def fit(
        self,
        plink_prefix: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "PCA":
        """
        Fit PCA on reference data.

        Args:
            plink_prefix: Path to PLINK file prefix (without extension)
            output_dir: Directory to save outputs (default: ./pca_outputs)

        Returns:
            Self (for method chaining)
        """
        plink_prefix = validate_plink_files(plink_prefix)
        plink_prefix = Path(plink_prefix).expanduser()
        if not plink_prefix.is_absolute():
            plink_prefix = Path.cwd() / plink_prefix
        plink_prefix = Path(plink_prefix).expanduser()
        if not plink_prefix.is_absolute():
            plink_prefix = Path.cwd() / plink_prefix

        # Set default output directory
        if output_dir is None:
            output_dir = Path.cwd() / "pca_outputs"
        else:
            output_dir = Path(output_dir).expanduser()
            if not output_dir.is_absolute():
                output_dir = Path.cwd() / output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        self._fit_output_dir = output_dir

        # Run FlashPCA fit
        output_prefix = output_dir / "fit"
        outputs = self._run_flashpca_fit(plink_prefix, output_prefix)

        # Store reference files for projection
        self._loadings_path = outputs["loadings"].resolve()
        self._meansd_path = outputs["meansd"].resolve()
        self._is_fitted = True

        logger.info(f"PCA fitted with {self.n_components} components")
        return self

    def project(
        self,
        plink_prefix: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Project samples onto fitted PCA space.

        Args:
            plink_prefix: Path to PLINK file prefix
            output_path: Optional path to save CSV output
            output_dir: Optional directory for raw FlashPCA projection outputs

        Returns:
            DataFrame with sample_id and PCA coordinates (dim_1, dim_2, ...)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        plink_prefix = validate_plink_files(plink_prefix)
        plink_prefix = Path(plink_prefix).expanduser()
        if not plink_prefix.is_absolute():
            plink_prefix = Path.cwd() / plink_prefix

        # Run FlashPCA projection
        if output_dir is None:
            output_dir = self._fit_output_dir or Path.cwd() / "pca_outputs"
        output_dir = Path(output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use dataset-specific prefix so we don't reuse a cached projection from
        # a different PLINK dataset (e.g., fit vs. project subset).
        dataset_name = Path(plink_prefix).name
        output_prefix = output_dir / f"project_{dataset_name}"
        pc_file = self._run_flashpca_project(plink_prefix, output_prefix)

        # Convert to manylatents format
        df = self._convert_pc_to_csv(pc_file, plink_prefix)

        # Save if output path provided
        if output_path:
            write_embedding_csv(df, output_path)

        return df

    def fit_transform(
        self, plink_prefix: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Fit PCA and transform the same data.

        Args:
            plink_prefix: Path to PLINK file prefix
            output_path: Optional path to save CSV output

        Returns:
            DataFrame with sample_id and PCA coordinates
        """
        plink_prefix = validate_plink_files(plink_prefix)

        # Determine output directory
        if output_path:
            output_dir = Path(output_path).parent / "pca_outputs"
        else:
            output_dir = Path.cwd() / "pca_outputs"
        output_dir = output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Run FlashPCA fit
        output_dir = output_dir if output_dir.is_absolute() else (Path.cwd() / output_dir)
        output_prefix = output_dir / "fit"
        outputs = self._run_flashpca_fit(plink_prefix, output_prefix)

        # Convert to manylatents format
        df = self._convert_pc_to_csv(outputs["pc"], plink_prefix)

        # Save if output path provided
        if output_path:
            write_embedding_csv(df, output_path)

        # Update fitted state
        self._loadings_path = outputs["loadings"]
        self._meansd_path = outputs["meansd"]
        self._fit_output_dir = output_dir
        self._is_fitted = True

        return df

    def _run_flashpca_fit(self, plink_prefix: Path, output_prefix: Path) -> Dict[str, Path]:
        """Run FlashPCA fit command."""
        logger.info("=" * 70)
        logger.info("FLASHPCA FIT")
        logger.info("=" * 70)

        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Define expected output files
        outputs = {
            "pc": Path(f"{output_prefix}.PC"),
            "loadings": Path(f"{output_prefix}.loadings"),
            "meansd": Path(f"{output_prefix}.meansd"),
            "eigenvec": Path(f"{output_prefix}.eigenvec"),
            "eigenval": Path(f"{output_prefix}.eigenval"),
        }

        # Check for existing checkpoint
        all_exist = all(path.exists() for path in outputs.values())
        if all_exist and not self.force:
            logger.info(f"PCA outputs checkpoint found: {output_prefix}.*")
            logger.info("Skipping fit (use force=True to recompute)")
            return outputs

        # Build command
        cmd = [
            self.flashpca,
            "--bfile",
            str(plink_prefix),
            "--outpc",
            str(outputs["pc"]),
            "--outload",
            str(outputs["loadings"]),
            "--outmeansd",
            str(outputs["meansd"]),
            "--outvec",
            str(outputs["eigenvec"]),
            "--outval",
            str(outputs["eigenval"]),
            "-d",
            str(self.n_components),
        ]

        logger.info(f"Fitting PCA on reference data:")
        logger.info(f"  PLINK prefix: {plink_prefix}")
        logger.info(f"  Number of PCs: {self.n_components}")
        logger.info(f"  Output prefix: {output_prefix}")

        # Run FlashPCA
        # Set cwd to output directory to catch any extra files (like pve.txt)
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=output_prefix.parent,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FlashPCA fit failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(
                "FlashPCA fit failed. See logs for stdout/stderr. "
                f"Command: {' '.join(cmd)}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
            )

        # Verify outputs were created
        missing = [name for name, path in outputs.items() if not path.exists()]
        if missing:
            logger.warning(f"Expected output files not found: {missing}")
        else:
            logger.info(f"✓ PCA fit complete: {len(outputs)} output files")

        return outputs

    def _run_flashpca_project(self, plink_prefix: Path, output_prefix: Path) -> Path:
        """Run FlashPCA project command."""
        logger.info("=" * 70)
        logger.info("FLASHPCA PROJECT")
        logger.info("=" * 70)

        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        output_pc = Path(f"{output_prefix}.PC")

        # Check for existing checkpoint
        if output_pc.exists() and not self.force:
            logger.info(f"Projection checkpoint found: {output_pc}")
            logger.info("Skipping project (use force=True to recompute)")
            return output_pc

        # Build command
        cmd = [
            self.flashpca,
            "--bfile",
            str(plink_prefix),
            "--project",
            "--inload",
            str(self._loadings_path),
            "--inmeansd",
            str(self._meansd_path),
            "--outproj",
            str(output_pc),
            "-d",
            str(self.n_components),
        ]

        logger.info(f"Projecting samples onto reference PCA:")
        logger.info(f"  PLINK prefix: {plink_prefix}")
        logger.info(f"  Number of PCs: {self.n_components}")
        logger.info(f"  Output file: {output_pc}")

        # Run FlashPCA projection
        # Set cwd to output directory to catch any extra files (like pve.txt)
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=output_prefix.parent,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FlashPCA project failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(
                "FlashPCA project failed. See logs for stdout/stderr. "
                f"Command: {' '.join(cmd)}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
            )

        # Verify output was created
        if not output_pc.exists():
            raise FileNotFoundError(f"Expected output file not created: {output_pc}")

        logger.info(f"✓ Projection complete: {output_pc}")
        return output_pc

    def _convert_pc_to_csv(self, pc_file: Path, plink_prefix: Path) -> pd.DataFrame:
        """
        Convert FlashPCA .PC file to manylatents CSV format.

        FlashPCA .PC format: FID IID PC1 PC2 ... PCN (with header row)
        Manylatents format: sample_id,dim_1,dim_2,...,dim_N
        """
        # Read FlashPCA output with header
        df = pd.read_csv(pc_file, sep=r"\s+")

        # Get sample IDs from PLINK files (IID column); coerce to str to
        # avoid int64/object dtype mismatch when merging with label DataFrames.
        sample_ids = df["IID"].astype(str).values

        # Extract PC columns (skip FID and IID)
        pc_cols = [col for col in df.columns if col.startswith("PC")]
        pc_values = df[pc_cols].values

        # Create manylatents format DataFrame with sample_id column
        n_pcs = pc_values.shape[1]
        dim_cols = [f"dim_{i + 1}" for i in range(n_pcs)]

        result_df = pd.DataFrame(pc_values, columns=dim_cols)
        result_df.insert(0, "sample_id", sample_ids)

        logger.info(f"Converted {len(result_df)} samples with {n_pcs} PCs")
        return result_df
