"""
Admixture preservation metrics.

Measures how well genetic embeddings preserve admixture proportions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from ..utils.io import read_embedding_csv

logger = logging.getLogger(__name__)


def compute_admixture_preservation(
    embedding: Union[pd.DataFrame, str, Path],
    q_files: Dict[int, Union[str, Path]],
    k_value: Optional[int] = None,
    num_samples: int = 50000,
) -> dict:
    """
    Compute preservation of admixture distances in genetic embedding.

    Uses Spearman correlation between:
    - Admixture distances (from Q matrices)
    - Embedding distances (from genetic embedding)

    Args:
        embedding: DataFrame or path to embedding CSV
        q_files: Dictionary mapping K values to Q file paths
        k_value: Specific K value to use (None = use all K values)
        num_samples: Maximum number of pairwise distances to sample

    Returns:
        Dictionary with results for each K:
        {
            K: {
                'correlation': Spearman correlation,
                'p_value': Statistical significance,
                'n_samples': Number of samples,
                'n_pairs': Number of pairs compared
            }
        }
    """
    # Load embedding
    if isinstance(embedding, (str, Path)):
        embedding_df = read_embedding_csv(embedding)
    else:
        embedding_df = embedding

    # Get embedding coordinates and sample IDs
    embedding_cols = [col for col in embedding_df.columns if col.startswith("dim_")]
    embedding_coords = embedding_df[embedding_cols].values
    
    # Get sample IDs (either from sample_id column or index)
    if "sample_id" in embedding_df.columns:
        sample_ids = embedding_df["sample_id"].tolist()
    else:
        sample_ids = embedding_df.index.tolist()

    # Process each K value
    results = {}

    k_values = [k_value] if k_value is not None else sorted(q_files.keys())

    for k in k_values:
        if k not in q_files:
            logger.warning(f"K={k} not found in q_files, skipping")
            continue

        logger.info(f"Computing admixture preservation for K={k}...")

        # Load Q matrix
        q_matrix = _load_q_matrix(q_files[k])

        # Align with embedding (match sample IDs)
        aligned_q, aligned_emb = _align_matrices(
            q_matrix, embedding_coords, sample_ids
        )

        if len(aligned_q) < 2:
            logger.warning(f"Not enough samples for K={k}, skipping")
            continue

        # Compute admixture distances (using Euclidean distance on Q)
        admix_dists = pdist(aligned_q, metric="euclidean")

        # Compute embedding distances
        emb_dists = pdist(aligned_emb, metric="euclidean")

        # Subsample if needed
        n_pairs = len(admix_dists)
        if n_pairs > num_samples:
            indices = np.random.choice(n_pairs, num_samples, replace=False)
            admix_dists = admix_dists[indices]
            emb_dists = emb_dists[indices]

        # Compute Spearman correlation
        correlation, p_value = spearmanr(admix_dists, emb_dists)

        results[k] = {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "n_samples": len(aligned_q),
            "n_pairs": len(admix_dists),
        }

        logger.info(
            f"K={k}: correlation={correlation:.4f}, p={p_value:.2e}, n={len(aligned_q)}"
        )

    return results


def _load_q_matrix(q_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load Q matrix from CSV file with sample_id and component columns.

    Expected format: sample_id,component_1,component_2,...,component_K

    Returns:
        DataFrame with numerical admixture proportions (sample_id as index)
    """
    q_path = Path(q_path)

    if not q_path.exists():
        raise FileNotFoundError(f"Q file not found: {q_path}")

    # Try to load as CSV first (new format)
    try:
        df = pd.read_csv(q_path)
        if "sample_id" in df.columns:
            # New format with sample_id
            df = df.set_index("sample_id")
            # Keep only component columns
            component_cols = [col for col in df.columns if col.startswith("component_")]
            return df[component_cols]
    except:
        pass
        
    # Fallback: load as space-separated file (old format)
    logger.warning(f"Loading {q_path} as legacy format (no headers)")
    q_matrix = pd.read_csv(q_path, sep=r"\s+", header=None)
    return q_matrix


def _align_matrices(
    q_matrix: pd.DataFrame, embedding_coords: np.ndarray, sample_ids: List[str]
) -> tuple:
    """
    Align Q matrix with embedding by matching sample IDs.

    Args:
        q_matrix: Q matrix with sample_id as index (from CSV format)
        embedding_coords: Embedding coordinates (n_samples × n_dims)
        sample_ids: List of sample IDs from embedding

    Returns:
        Tuple of (aligned_q, aligned_embedding)
    """
    # Create DataFrame for embedding with sample_id as index
    embedding_df = pd.DataFrame(embedding_coords, index=sample_ids)
    
    # Find intersection of sample IDs
    common_samples = q_matrix.index.intersection(embedding_df.index)
    
    if len(common_samples) == 0:
        raise ValueError("No common sample IDs found between Q matrix and embedding")
        
    logger.info(f"Found {len(common_samples)} common samples for alignment "
                f"(Q matrix: {len(q_matrix)}, embedding: {len(embedding_df)})")
    
    # Align both matrices by common sample IDs
    q_aligned = q_matrix.loc[common_samples]
    embedding_aligned = embedding_df.loc[common_samples]
    
    return q_aligned.values, embedding_aligned.values


def compute_admixture_preservation_summary(
    embedding: Union[pd.DataFrame, str, Path],
    q_files: Dict[int, Union[str, Path]],
) -> pd.DataFrame:
    """
    Compute admixture preservation for all K values and return summary.

    Args:
        embedding: DataFrame or path to embedding CSV
        q_files: Dictionary mapping K values to Q file paths

    Returns:
        DataFrame with columns: K, correlation, p_value, n_samples, n_pairs
    """
    results = compute_admixture_preservation(embedding, q_files)

    # Convert to DataFrame
    summary_data = []
    for k, metrics in results.items():
        summary_data.append({"K": k, **metrics})

    return pd.DataFrame(summary_data)
