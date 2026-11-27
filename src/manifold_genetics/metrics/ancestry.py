"""
Ancestry resolution.

How well the embedding keeps ancestries apart: each sample is labelled with its
largest admixture component, and a k-nearest-neighbour classifier predicts that
label from the embedding coordinates alone, scored by balanced accuracy under
cross-validation. 1 means every sample's neighbours in the embedding share its
main ancestry; chance is 1 / (number of groups).

It complements admixture preservation, a rank correlation over random pairs. That
score is dominated by the largest distances (for example between continents), so an
embedding can score well on it while overlapping smaller ancestries; ancestry
resolution weights every ancestry equally.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from ..utils.io import read_admixture_csv
from .knn import _coordinates

logger = logging.getLogger(__name__)

PathOrFrame = Union[pd.DataFrame, str, Path]


def _main_component(q: PathOrFrame) -> pd.Series:
    """Index (0-based) of each sample's largest admixture component, indexed by sample_id."""
    df = read_admixture_csv(q) if isinstance(q, (str, Path)) else q.copy()
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    df.index = df.index.astype(str)
    values = df.select_dtypes("number").to_numpy(dtype=float)
    if values.shape[1] < 2:
        raise ValueError(f"Need at least 2 admixture components, got {values.shape[1]}")
    return pd.Series(values.argmax(axis=1), index=df.index)


def compute_ancestry_resolution(
    embedding: PathOrFrame,
    q_file: PathOrFrame,
    n_neighbors: int = 15,
    n_folds: int = 5,
    min_group_size: int = 30,
    subsample: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Compute how well each sample's main ancestry can be predicted from the embedding.

    Args:
        embedding: DataFrame or path to embedding CSV (sample_id, dim_1, dim_2, ...)
        q_file: DataFrame or path to an admixture CSV (sample_id, component_1, ...,
            component_K), for example the Neural Admixture output for one K
        n_neighbors: Neighbours used by the k-nearest-neighbour classifier
        n_folds: Cross-validation folds (stratified by group)
        min_group_size: Groups (main components) with fewer samples are left out
        subsample: Score a random subset of this many samples (None: all)
        seed: Seed for the subsample and the fold split

    Samples missing from either input are left out.

    Returns:
        Dictionary with:
        - ancestry_resolution: balanced accuracy, from 0 to 1
        - chance: 1 / n_groups, the balanced accuracy of guessing
        - n_groups: number of groups scored
        - n_samples: number of samples scored
        - n_neighbors, n_folds: the settings used
    """
    emb = _coordinates(embedding)
    groups = _main_component(q_file)
    ids = emb.index.intersection(groups.index)
    rng = np.random.default_rng(seed)
    if subsample is not None and len(ids) > subsample:
        ids = ids[np.sort(rng.choice(len(ids), subsample, replace=False))]
    y = groups.loc[ids]
    sizes = y.value_counts()
    keep = y.isin(sizes.index[sizes >= max(min_group_size, n_folds)])
    ids, y = ids[keep.to_numpy()], y[keep].to_numpy()
    n_groups = len(np.unique(y))
    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 groups with {min_group_size} or more samples, got {n_groups}"
        )
    X = emb.loc[ids].to_numpy(dtype=float)
    k = min(n_neighbors, len(ids) - len(ids) // n_folds - 1)
    logger.info(
        f"Ancestry resolution ({k}-NN, {n_folds}-fold) on {len(ids)} samples, {n_groups} groups"
    )

    pred = np.empty_like(y)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(X, y):
        pred[test] = KNeighborsClassifier(n_neighbors=k).fit(X[train], y[train]).predict(X[test])
    return {
        "ancestry_resolution": float(balanced_accuracy_score(y, pred)),
        "chance": 1.0 / n_groups,
        "n_groups": int(n_groups),
        "n_samples": int(len(ids)),
        "n_neighbors": int(k),
        "n_folds": int(n_folds),
    }
