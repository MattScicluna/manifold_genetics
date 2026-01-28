"""
File I/O utilities for genomics data.

Handles reading/writing PLINK files, CSV files, and format conversions.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

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

    prefix_str = str(plink_prefix)
    for ext in required_extensions:
        if prefix_str.endswith(ext):
            file_path = Path(prefix_str)
        else:
            # Avoid Path.with_suffix so prefixes like ".noHLA.unrelated" are preserved
            file_path = Path(prefix_str + ext)
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


def read_bim_file(bim_path: Union[str, Path]) -> Dict[str, Tuple[str, str]]:
    """
    Read PLINK .bim file and return SNP allele information.

    BIM format: chr, snp_id, genetic_dist, position, allele1, allele2

    Args:
        bim_path: Path to .bim file

    Returns:
        Dictionary mapping SNP ID to (allele1, allele2) tuple
    """
    bim_path = Path(bim_path)
    if not bim_path.exists():
        raise FileNotFoundError(f"BIM file not found: {bim_path}")

    snp_alleles = {}
    with open(bim_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 6:
                continue

            snp_id = fields[1]  # Column 2: SNP ID
            allele1 = fields[4]  # Column 5: Allele 1
            allele2 = fields[5]  # Column 6: Allele 2

            snp_alleles[snp_id] = (allele1, allele2)

    return snp_alleles


def check_allele_compatibility(
    bim1_path: Union[str, Path],
    bim2_path: Union[str, Path],
    common_snps: Set[str]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Check allele compatibility between two PLINK datasets.

    Compares alleles for common SNPs and determines which SNPs need flipping
    and which are incompatible.

    Compatible allele combinations:
    - Exact match: A/T matches A/T
    - Complementary: A/T matches T/A (needs flipping)
    - Complementary: C/G matches G/C (needs flipping)

    Incompatible combinations (e.g., A/T vs C/G) are excluded.

    Args:
        bim1_path: Path to first .bim file
        bim2_path: Path to second .bim file
        common_snps: Set of common SNP IDs

    Returns:
        Tuple of (exact_matches, need_flip, incompatible) SNP ID sets
    """
    logger.info(f"Reading BIM files...")
    logger.info(f"  {bim1_path}")
    logger.info(f"  {bim2_path}")

    # Read allele information from both BIM files
    snps1 = read_bim_file(bim1_path)
    snps2 = read_bim_file(bim2_path)

    exact_matches = set()
    need_flip = set()
    incompatible = set()

    logger.info(f"Checking {len(common_snps)} common SNPs...")

    # Use tqdm for progress bar
    for snp in tqdm(common_snps, desc="Checking alleles", unit="SNP"):
        if snp not in snps1:
            logger.warning(f"SNP {snp} not found in first BIM file")
            incompatible.add(snp)
            continue
        if snp not in snps2:
            logger.warning(f"SNP {snp} not found in second BIM file")
            incompatible.add(snp)
            continue

        a1_1, a2_1 = snps1[snp]
        a1_2, a2_2 = snps2[snp]

        # Check for exact match
        if (a1_1 == a1_2 and a2_1 == a2_2):
            exact_matches.add(snp)
        # Check for complementary (needs flip)
        elif (a1_1 == a2_2 and a2_1 == a1_2):
            need_flip.add(snp)
        # Incompatible alleles
        else:
            incompatible.add(snp)

    logger.info(f"\nResults:")
    logger.info(f"  Exact matches: {len(exact_matches)}")
    logger.info(f"  Compatible flips: {len(need_flip)}")
    logger.info(f"  Incompatible/missing: {len(incompatible)}")
    logger.info(f"  Total usable SNPs: {len(exact_matches) + len(need_flip)}")

    return exact_matches, need_flip, incompatible
