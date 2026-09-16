"""
Geographic preservation metrics.

Measures how well genetic embeddings preserve geographic distances.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from ..utils.io import read_embedding_csv, read_labels_csv
from .pairs import n_pairs, pair_distances, sample_pairs

logger = logging.getLogger(__name__)


def compute_geographic_preservation(
    embedding: Union[pd.DataFrame, str, Path],
    geographic_coords: Union[pd.DataFrame, str, Path],
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
    num_samples: int = 50000,
    ignore_missing: bool = True,
    seed: int = 42,
) -> dict:
    """
    Compute preservation of geographic distances in genetic embedding.

    Uses Spearman correlation between:
    - Geographic distances (from lat/lon coordinates)
    - Embedding distances (from genetic embedding)

    Only ``num_samples`` sample pairs are compared. When the cohort has more
    pairs than that, the pairs are drawn first and distances computed for
    those alone, so memory stays O(num_samples) whatever the cohort size.

    Args:
        embedding: DataFrame or path to embedding CSV (sample_id, dim_1, dim_2, ...)
        geographic_coords: DataFrame or path to CSV with geographic coordinates
        longitude_col: Name of longitude column
        latitude_col: Name of latitude column
        num_samples: Maximum number of pairwise distances to sample
        ignore_missing: If True, ignore samples without geographic coordinates
        seed: Seed for the pair draw, so a rerun compares the same pairs

    Returns:
        Dictionary with:
        - correlation: Spearman correlation coefficient
        - p_value: Statistical significance
        - n_samples: Number of samples used
        - n_pairs: Number of pairwise distances compared
    """
    # Load data
    if isinstance(embedding, (str, Path)):
        embedding_df = read_embedding_csv(embedding)
    else:
        embedding_df = embedding

    # Ensure embedding has sample_id as index for joining
    if "sample_id" in embedding_df.columns:
        embedding_df = embedding_df.set_index("sample_id")

    if isinstance(geographic_coords, (str, Path)):
        geo_df = read_labels_csv(geographic_coords)
    else:
        geo_df = geographic_coords

    # Merge embedding with geographic coordinates
    merged_df = embedding_df.join(geo_df[[longitude_col, latitude_col]], how="inner")

    logger.info(
        f"Found {len(merged_df)} samples with both embedding and geographic data "
        f"(embedding: {len(embedding_df)}, geographic: {len(geo_df)})"
    )

    # Remove samples with missing coordinates
    if ignore_missing:
        before_count = len(merged_df)
        merged_df = merged_df.dropna(subset=[longitude_col, latitude_col])
        after_count = len(merged_df)
        if before_count > after_count:
            logger.info(f"Removed {before_count - after_count} samples with missing coordinates")

    if len(merged_df) < 2:
        raise ValueError(
            f"Need at least 2 samples with geographic coordinates, found {len(merged_df)}"
        )

    # Extract coordinates
    geo_coords = merged_df[[longitude_col, latitude_col]].values
    embedding_cols = [col for col in merged_df.columns if col.startswith("dim_")]
    embedding_coords = merged_df[embedding_cols].values

    # Geographic and embedding distances for the same pairs: every pair when
    # they fit within num_samples, else a random subset drawn before any
    # distance is computed.
    n = len(merged_df)
    if n_pairs(n) <= num_samples:
        logger.info(f"Computing geographic distances for {n} samples...")
        geo_dists = _haversine_distances(geo_coords)
        logger.info("Computing embedding distances...")
        embedding_dists = pdist(embedding_coords, metric="euclidean")
    else:
        i, j = sample_pairs(n, num_samples, np.random.default_rng(seed))
        logger.info(f"Computing geographic distances for {len(i)} pairs...")
        geo_dists = _haversine_pairs(geo_coords, i, j)
        logger.info("Computing embedding distances...")
        embedding_dists = pair_distances(embedding_coords, i, j)

    # Compute Spearman correlation
    correlation, p_value = spearmanr(geo_dists, embedding_dists)

    result = {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "n_samples": len(merged_df),
        "n_pairs": len(geo_dists),
    }

    logger.info(f"Geographic preservation: {correlation:.4f} (p={p_value:.2e})")
    return result


def _haversine_pairs(coords: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    Great-circle distance (km) between ``coords[i]`` and ``coords[j]``, row-wise.

    Args:
        coords: Array of shape (n_samples, 2) with [longitude, latitude] in degrees
        i: Indices of the first sample of each pair
        j: Indices of the second sample of each pair

    Returns:
        Array of ``len(i)`` distances
    """
    coords = np.asarray(coords)
    a_rad = np.deg2rad(coords[i].astype(np.float64, copy=False))
    b_rad = np.deg2rad(coords[j].astype(np.float64, copy=False))
    lon_a, lat_a = a_rad[:, 0], a_rad[:, 1]
    lon_b, lat_b = b_rad[:, 0], b_rad[:, 1]

    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(h))

    # Earth radius in km
    return 6371.0 * c


def _haversine_distances(coords: np.ndarray) -> np.ndarray:
    """
    Compute great-circle distances between all pairs of geographic coordinates.

    Only used when every pair is compared, so ``n`` is small; the sampled
    branch uses :func:`_haversine_pairs`.

    Args:
        coords: Array of shape (n_samples, 2) with [longitude, latitude] in degrees

    Returns:
        Array of pairwise distances (condensed form from pdist)
    """
    # Convert to radians
    coords_rad = np.deg2rad(coords)

    # Extract lon/lat
    lon = coords_rad[:, 0]
    lat = coords_rad[:, 1]

    # Compute pairwise haversine distances
    n = len(coords)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            # Haversine formula
            dlat = lat[j] - lat[i]
            dlon = lon[j] - lon[i]

            a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat[j]) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))

            # Earth radius in km
            r = 6371.0

            distance = r * c
            distances.append(distance)

    return np.array(distances)
