"""
Admixture preservation metrics.

Measures how well genetic embeddings preserve admixture proportions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..utils.io import read_admixture_csv, read_embedding_csv
from .pairs import sampled_pair_distances

logger = logging.getLogger(__name__)


def compute_admixture_preservation(
    embedding: Union[pd.DataFrame, str, Path],
    q_files: Dict[int, Union[str, Path]],
    k_value: Optional[int] = None,
    num_samples: int = 50000,
    subsample: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Compute preservation of admixture distances in genetic embedding.

    Uses Spearman correlation between:
    - Admixture distances (from Q matrices)
    - Embedding distances (from genetic embedding)

    Only ``num_samples`` sample pairs are compared. When the cohort has more
    pairs than that, the pairs are drawn first and distances computed for
    those alone, so memory stays O(num_samples) whatever the cohort size.

    Args:
        embedding: DataFrame or path to embedding CSV
        q_files: Dictionary mapping K values to Q file paths
        k_value: Specific K value to use (None = use all K values)
        num_samples: Maximum number of pairwise distances to sample
        subsample: If set, randomly subsample individuals to this count before
            computing distances. Applied consistently across all K values.
        seed: Seed for the individual subsample and for the pair draw. The
            same pairs are drawn for every K, so the K values are comparable.

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

    # Subsample individuals (consistent across all K values).
    if subsample is not None and len(sample_ids) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sample_ids), subsample, replace=False)
        sample_ids = [sample_ids[i] for i in idx]
        embedding_coords = embedding_coords[idx]
        logger.info(f"Subsampled individuals to {subsample} (from {len(embedding_df)})")

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
        aligned_q, aligned_emb = _align_matrices(q_matrix, embedding_coords, sample_ids)

        if len(aligned_q) < 2:
            logger.warning(f"Not enough samples for K={k}, skipping")
            continue

        # Euclidean distances on Q and on the embedding, for the same pairs
        # (every pair when they fit within num_samples, else a random subset).
        admix_dists, emb_dists = sampled_pair_distances(
            aligned_q, aligned_emb, num_samples, np.random.default_rng(seed)
        )

        # Compute Spearman correlation
        correlation, p_value = spearmanr(admix_dists, emb_dists)

        results[k] = {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "n_samples": len(aligned_q),
            "n_pairs": len(admix_dists),
        }

        logger.info(f"K={k}: correlation={correlation:.4f}, p={p_value:.2e}, n={len(aligned_q)}")

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
        df = read_admixture_csv(q_path)
        # New format with sample_id (already str-coerced by read_admixture_csv)
        df = df.set_index("sample_id")
        component_cols = [col for col in df.columns if col.startswith("component_")]
        return df[component_cols]
    except (ValueError, FileNotFoundError):
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

    logger.info(
        f"Found {len(common_samples)} common samples for alignment "
        f"(Q matrix: {len(q_matrix)}, embedding: {len(embedding_df)})"
    )

    # Align both matrices by common sample IDs
    q_aligned = q_matrix.loc[common_samples]
    embedding_aligned = embedding_df.loc[common_samples]

    return q_aligned.values, embedding_aligned.values
