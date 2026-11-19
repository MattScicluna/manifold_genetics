"""
Separation inflation.

Rank-based preservation metrics (admixture, geographic, k-NN overlap) do not change
when an embedding squeezes each group into a tight cluster: the order of pairwise
distances, and each sample's nearest neighbours, can survive the squeeze. This metric
is scale-aware. For each sample it compares how far the nearest member of another
group is with how far its own group's nearest neighbours are, in the embedding and
in a reference space (for example the PCA input or admixture proportions):

    ratio(x)     = distance to nearest sample of another group / mean distance to own k nearest neighbours
    inflation(x) = log ratio_embedding(x) - log ratio_reference(x)

The summary is the median over samples. Zero means groups are as separated as in the
reference; positive values mean the embedding shows groups as more separated than they
are (detached clusters); negative values mean it merges them.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..utils.io import read_embedding_csv, read_labels_csv

logger = logging.getLogger(__name__)

PathOrFrame = Union[pd.DataFrame, str, Path]


def _coordinates(data: PathOrFrame) -> pd.DataFrame:
    """Numeric coordinates indexed by sample_id."""
    df = read_embedding_csv(data) if isinstance(data, (str, Path)) else data
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    df.index = df.index.astype(str)
    return df.select_dtypes("number")


def separation_ratios(X: np.ndarray, groups: np.ndarray, k_own: int = 5) -> np.ndarray:
    """Per sample: distance to the nearest other-group sample / mean distance to own k NN.

    Distances of zero (identical coordinates, common for admixture proportions) are raised
    to the smallest positive distance seen in this space, so every ratio has a finite log.
    """
    d_own, d_other = np.empty(len(X)), np.empty(len(X))
    for g in np.unique(groups):
        own = groups == g
        own_nn = NearestNeighbors(n_neighbors=k_own + 1).fit(X[own])
        d_own[own] = own_nn.kneighbors(X[own])[0][:, 1:].mean(axis=1)
        d_other[own] = NearestNeighbors(n_neighbors=1).fit(X[~own]).kneighbors(X[own])[0][:, 0]
    both = np.concatenate([d_own, d_other])
    floor = both[both > 0].min() if (both > 0).any() else 1.0
    return np.maximum(d_other, floor) / np.maximum(d_own, floor)


def compute_separation_inflation(
    embedding: PathOrFrame,
    reference: PathOrFrame,
    labels: PathOrFrame,
    label_col: str,
    k_own: int = 5,
) -> dict:
    """
    Compare how separated groups look in the embedding with how separated they are in a reference.

    Args:
        embedding: DataFrame or path to embedding CSV (sample_id, dim_1, dim_2, ...)
        reference: DataFrame or path to CSV of reference coordinates with a sample_id
            column, e.g. the PCA coordinates the embedding was built from, or admixture
            proportions
        labels: DataFrame or path to labels CSV with a sample_id column
        label_col: Column of ``labels`` that defines the groups (e.g. Population)
        k_own: Number of own-group neighbours whose mean distance sets each sample's scale

    Groups with ``k_own`` or fewer samples are left out, and so are samples missing from
    any of the three inputs.

    Returns:
        Dictionary with:
        - inflation: median over samples of the log ratio difference (0 = as separated as
          the reference; > 0 more separated; < 0 merged)
        - per_group: median inflation of each group
        - n_samples, n_groups: what was scored
    """
    emb = _coordinates(embedding)
    ref = _coordinates(reference)
    lab = read_labels_csv(labels) if isinstance(labels, (str, Path)) else labels
    if "sample_id" in lab.columns:
        lab = lab.set_index("sample_id")
    lab.index = lab.index.astype(str)
    if label_col not in lab.columns:
        raise ValueError(f"Label column {label_col!r} not in labels (columns: {list(lab.columns)})")

    ids = emb.index.intersection(ref.index).intersection(lab.index[lab[label_col].notna()])
    groups = lab.loc[ids, label_col].astype(str).values
    sizes = pd.Series(groups).value_counts()
    keep = np.isin(groups, sizes.index[sizes > k_own])
    if pd.Series(groups[keep]).nunique() < 2:
        raise ValueError(f"Need at least two groups with more than {k_own} samples")
    ids, groups = ids[keep], groups[keep]
    logger.info(f"Separation inflation on {len(ids)} samples in {len(set(groups))} groups")

    diff = np.log(separation_ratios(emb.loc[ids].values, groups, k_own)) - np.log(
        separation_ratios(ref.loc[ids].values, groups, k_own)
    )
    return {
        "inflation": float(np.median(diff)),
        "per_group": pd.Series(diff).groupby(groups).median().to_dict(),
        "n_samples": int(len(ids)),
        "n_groups": int(len(set(groups))),
    }
