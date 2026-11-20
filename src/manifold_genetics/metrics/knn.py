"""
k-NN preservation.

The share of each sample's k nearest neighbours in a reference space (usually the
PCA coordinates the embedding was built from) that are also among its k nearest
neighbours in the embedding, averaged over samples. It measures local structure:
1 means every neighbourhood is kept, and a random embedding scores about k / n.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..utils.io import read_embedding_csv

logger = logging.getLogger(__name__)

PathOrFrame = Union[pd.DataFrame, str, Path]


def _coordinates(data: PathOrFrame) -> pd.DataFrame:
    """Numeric coordinates indexed by sample_id."""
    df = read_embedding_csv(data) if isinstance(data, (str, Path)) else data
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    df.index = df.index.astype(str)
    return df.select_dtypes("number")


def _neighbours(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k nearest other rows (the row itself excluded)."""
    return NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)


def compute_knn_preservation(
    embedding: PathOrFrame,
    reference: PathOrFrame,
    k: int = 10,
) -> dict:
    """
    Compute how many of each sample's nearest neighbours the embedding keeps.

    Args:
        embedding: DataFrame or path to embedding CSV (sample_id, dim_1, dim_2, ...)
        reference: DataFrame or path to CSV of reference coordinates with a sample_id
            column, usually the PCA coordinates the embedding was built from
        k: Number of neighbours compared per sample

    Samples missing from either input are left out; neighbours are found among the
    samples present in both.

    Returns:
        Dictionary with:
        - knn_preservation: mean share of shared neighbours, from 0 to 1
        - k: the number of neighbours compared
        - n_samples: number of samples scored
    """
    emb = _coordinates(embedding)
    ref = _coordinates(reference)
    ids = emb.index.intersection(ref.index)
    if len(ids) <= k:
        raise ValueError(f"Need more than k={k} samples present in both inputs, got {len(ids)}")
    logger.info(f"k-NN preservation (k={k}) on {len(ids)} samples")

    nn_ref = _neighbours(ref.loc[ids].values, k)
    nn_emb = _neighbours(emb.loc[ids].values, k)
    # Each row holds k distinct indices, so the overlap is the number of shared entries.
    shared = (nn_ref[:, :, None] == nn_emb[:, None, :]).any(axis=2).sum(axis=1)
    return {
        "knn_preservation": float(shared.mean() / k),
        "k": int(k),
        "n_samples": int(len(ids)),
    }
