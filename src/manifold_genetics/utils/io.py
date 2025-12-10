"""
File I/O utilities for genomics data.

Handles reading/writing PLINK files, CSV files, and format conversions.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_plink_files(plink_prefix: Union[str, Path]) -> Path:
    """
    Validate that required PLINK files exist.

    Args:
        plink_prefix: Path to PLINK file prefix (without extension)

    Returns:
        Path object of the validated prefix

    Raises:
        FileNotFoundError: If any required file is missing
    """
    plink_prefix = Path(plink_prefix)

    required_extensions = [".bed", ".bim", ".fam"]
    missing = []

    for ext in required_extensions:
        file_path = plink_prefix.with_suffix(ext)
        if not file_path.exists():
            missing.append(str(file_path))

    if missing:
        raise FileNotFoundError(
            f"Missing PLINK files for prefix '{plink_prefix}':\n"
            + "\n".join(f"  - {f}" for f in missing)
        )

    return plink_prefix


def read_embedding_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read embedding CSV file in manylatents format.

    Expected format:
    - Columns: sample_id, dim_1, dim_2, ..., dim_N 
    - If no sample_id column, assumes rows are ordered by sample index

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with embedding dimensions as columns (may include sample_id)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Validate format - check for dim_ columns
    dim_cols = [col for col in df.columns if col.startswith("dim_")]
    if not dim_cols:
        raise ValueError(f"CSV must have 'dim_' columns. Found: {list(df.columns)}")

    return df


def write_embedding_csv(
    embeddings: Union[np.ndarray, pd.DataFrame],
    output_path: Union[str, Path],
    sample_ids: Optional[list] = None,
) -> Path:
    """
    Write embeddings to CSV in manylatents format.

    Output format:
    - Columns: sample_id, dim_1, dim_2, ..., dim_N (includes sample_id if available)

    Args:
        embeddings: Numpy array (n_samples × n_dims) or DataFrame
        output_path: Path to output CSV file
        sample_ids: List of sample IDs (used if embeddings is numpy array)

    Returns:
        Path to written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame if needed
    if isinstance(embeddings, np.ndarray):
        n_dims = embeddings.shape[1]
        dim_cols = [f"dim_{i+1}" for i in range(n_dims)]
        df = pd.DataFrame(embeddings, columns=dim_cols)
        # Add sample_ids if provided
        if sample_ids is not None:
            df.insert(0, 'sample_id', sample_ids)
    else:
        # If DataFrame, use as-is (should already include sample_id if needed)
        df = embeddings.copy()

    df.to_csv(output_path, index=False)

    return output_path


def read_labels_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read labels/metadata CSV file.

    Expected format:
    - First column (or a column named 'sample_id'): Sample IDs
    - Remaining columns: Various labels (e.g., Population, Genetic_region)

    Args:
        file_path: Path to labels CSV file

    Returns:
        DataFrame with sample_id as index
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Labels file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Set sample_id as index if present
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    else:
        logger.warning(
            f"No 'sample_id' column found in {file_path}. Using first column as index."
        )
        df = df.set_index(df.columns[0])

    return df


def read_fam_file(fam_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read PLINK .fam file.

    Format: FID IID Father Mother Sex Phenotype

    Args:
        fam_path: Path to .fam file

    Returns:
        DataFrame with sample information
    """
    fam_path = Path(fam_path)
    if not fam_path.exists():
        raise FileNotFoundError(f"FAM file not found: {fam_path}")

    df = pd.read_csv(
        fam_path,
        sep=r"\s+",
        names=["FID", "IID", "Father", "Mother", "Sex", "Phenotype"],
        header=None,
    )

    return df


def get_sample_ids_from_plink(plink_prefix: Union[str, Path]) -> list:
    """
    Extract sample IDs from PLINK .fam file.

    Args:
        plink_prefix: Path to PLINK file prefix

    Returns:
        List of sample IDs (IID column from .fam file)
    """
    plink_prefix = validate_plink_files(plink_prefix)
    fam_path = plink_prefix.with_suffix(".fam")

    fam_df = read_fam_file(fam_path)
    return fam_df["IID"].tolist()


def read_sample_indices(indices_path: Union[str, Path]) -> list:
    """
    Read sample IDs from a text file (one ID per line).

    Args:
        indices_path: Path to file with sample IDs

    Returns:
        List of sample IDs
    """
    indices_path = Path(indices_path)
    if not indices_path.exists():
        raise FileNotFoundError(f"Indices file not found: {indices_path}")

    with open(indices_path, "r") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    return sample_ids


def read_colormap(colormap_path: Union[str, Path]) -> dict:
    """
    Read colormap JSON file.

    Expected format:
    {
        "label_column_name": {
            "label_value_1": "#FF0000",
            "label_value_2": "#00FF00",
            ...
        },
        ...
    }

    Args:
        colormap_path: Path to colormap JSON file

    Returns:
        Dictionary mapping label columns to {value: color} dicts
    """
    import json

    colormap_path = Path(colormap_path)
    if not colormap_path.exists():
        raise FileNotFoundError(f"Colormap file not found: {colormap_path}")

    with open(colormap_path, "r") as f:
        colormap = json.load(f)

    return colormap
