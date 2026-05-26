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

logger = logging.getLogger(__name__)


def compute_geographic_preservation(
    embedding: Union[pd.DataFrame, str, Path],
    geographic_coords: Union[pd.DataFrame, str, Path],
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
    num_samples: int = 50000,
    ignore_missing: bool = True,
) -> dict:
    """
    Compute preservation of geographic distances in genetic embedding.

    Uses Spearman correlation between:
    - Geographic distances (from lat/lon coordinates)
    - Embedding distances (from genetic embedding)

    Args:
        embedding: DataFrame or path to embedding CSV (sample_id, dim_1, dim_2, ...)
        geographic_coords: DataFrame or path to CSV with geographic coordinates
        longitude_col: Name of longitude column
        latitude_col: Name of latitude column
        num_samples: Maximum number of pairwise distances to sample
        ignore_missing: If True, ignore samples without geographic coordinates

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

    # Compute pairwise distances
    logger.info(f"Computing geographic distances for {len(merged_df)} samples...")
    geo_dists = _haversine_distances(geo_coords)

    logger.info(f"Computing embedding distances...")
    embedding_dists = pdist(embedding_coords, metric="euclidean")

    # Subsample if too many distances
    n_pairs = len(geo_dists)
    if n_pairs > num_samples:
        logger.info(f"Subsampling {num_samples} of {n_pairs} pairwise distances...")
        indices = np.random.choice(n_pairs, num_samples, replace=False)
        geo_dists = geo_dists[indices]
        embedding_dists = embedding_dists[indices]

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


def _haversine_distances(coords: np.ndarray) -> np.ndarray:
    """
    Compute great-circle distances between geographic coordinates.

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


def compute_geographic_preservation_by_distance_bin(
    embedding: Union[pd.DataFrame, str, Path],
    geographic_coords: Union[pd.DataFrame, str, Path],
    longitude_col: str = "Longitude",
    latitude_col: str = "Latitude",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute geographic preservation separately for distance bins.

    Useful for understanding if preservation varies with geographic distance
    (e.g., better for nearby vs distant populations).

    Args:
        embedding: DataFrame or path to embedding CSV
        geographic_coords: DataFrame or path to CSV with geographic coordinates
        longitude_col: Name of longitude column
        latitude_col: Name of latitude column
        n_bins: Number of distance bins

    Returns:
        DataFrame with columns:
        - bin_start: Start of distance bin (km)
        - bin_end: End of distance bin (km)
        - correlation: Spearman correlation for this bin
        - n_pairs: Number of pairs in this bin
    """
    # Load data
    if isinstance(embedding, (str, Path)):
        embedding_df = read_embedding_csv(embedding)
    else:
        embedding_df = embedding

    if isinstance(geographic_coords, (str, Path)):
        geo_df = read_labels_csv(geographic_coords)
    else:
        geo_df = geographic_coords

    # Merge and remove missing
    merged_df = embedding_df.join(geo_df[[longitude_col, latitude_col]], how="inner")
    merged_df = merged_df.dropna(subset=[longitude_col, latitude_col])

    # Extract coordinates
    geo_coords = merged_df[[longitude_col, latitude_col]].values
    embedding_cols = [col for col in merged_df.columns if col.startswith("dim_")]
    embedding_coords = merged_df[embedding_cols].values

    # Compute distances
    geo_dists = _haversine_distances(geo_coords)
    embedding_dists = pdist(embedding_coords, metric="euclidean")

    # Create distance bins
    bin_edges = np.percentile(geo_dists, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = 0  # Ensure first bin starts at 0
    bin_edges[-1] = np.inf  # Ensure last bin includes all distances

    # Compute correlation for each bin
    results = []
    for i in range(n_bins):
        mask = (geo_dists >= bin_edges[i]) & (geo_dists < bin_edges[i + 1])

        if np.sum(mask) < 2:
            continue

        geo_bin = geo_dists[mask]
        emb_bin = embedding_dists[mask]

        corr, _ = spearmanr(geo_bin, emb_bin)

        results.append(
            {
                "bin_start": bin_edges[i],
                "bin_end": bin_edges[i + 1] if i < n_bins - 1 else geo_dists.max(),
                "correlation": corr,
                "n_pairs": len(geo_bin),
            }
        )

    return pd.DataFrame(results)
