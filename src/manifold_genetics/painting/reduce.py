"""Coancestry features to ``sample_id, dim_1 ... dim_n`` coordinates.

Painting is all against all within the painted set, so there is no separate
model to project a second cohort through the way genotype PCA has. What does
carry over is the shape of the data: every row is one recipient's copying
profile over the *same* donors, the painted set. So the fit cohort's rows fit
the PCA and the project cohort's rows are projected through it, exactly as a
project cohort's genotypes are projected through loadings fitted elsewhere.

The PCA is sklearn's, on the dense matrix. The package's in-process genotype
backend is not reused, because it reads PLINK ``.bed`` files rather than an
array.
"""

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

__all__ = ["match_cohort", "painting_pca"]

logger = logging.getLogger(__name__)

# The full SVD of an n x n matrix is O(n^3); past this many samples the
# randomized solver is far faster. 20 power iterations matches the genotype
# backend's choice, for the same reason: near-equal trailing eigenvalues.
_FULL_SVD_MAX = 5000


def match_cohort(
    painted_ids: Sequence[str], fids: Sequence[str], iids: Sequence[str], *, cohort: str
) -> List[int]:
    """Row indices into the painted set for each sample of a ``.fam``, in ``.fam`` order.

    A VCF exported from PLINK names a sample either by its IID or by
    ``FID_IID``, depending on the export settings, so both are accepted.

    Raises:
        ValueError: a sample of the cohort was not painted. Dropping it would
            silently change the cohort, so the run stops and names a few.
    """
    index: Dict[str, int] = {}
    for i, sid in enumerate(painted_ids):
        index.setdefault(sid, i)
    rows, missing = [], []
    for fid, iid in zip(fids, iids):
        row = index.get(iid, index.get(f"{fid}_{iid}"))
        if row is None:
            missing.append(iid)
        else:
            rows.append(row)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(iids)} {cohort} samples are not in the phased VCF, for "
            f"example {', '.join(missing[:5])}. Painting needs every sample of the cohort; "
            "sample names in the VCF must equal the .fam IID or FID_IID."
        )
    return rows


def painting_pca(
    features: np.ndarray,
    sample_ids: Sequence[str],
    *,
    fit_rows: Sequence[int],
    project_rows: Sequence[int],
    n_components: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit PCA on ``fit_rows`` of ``features``; project ``fit_rows`` and ``project_rows``.

    ``n_components`` is capped at one less than the number of fit samples,
    which is the most an n-sample PCA can resolve, with a warning.

    Returns:
        ``(fit_coords, project_coords)``, each ``sample_id, dim_1 ... dim_k``.
    """
    fit_rows, project_rows = list(fit_rows), list(project_rows)
    if len(fit_rows) < 2:
        raise ValueError("The fit cohort needs at least two painted samples")
    k = min(n_components, len(fit_rows) - 1, features.shape[1])
    if k < n_components:
        logger.warning(
            f"Painting PCA: {n_components} components requested but the fit cohort "
            f"supports {k}; writing {k}."
        )
    if len(fit_rows) <= _FULL_SVD_MAX:
        pca = PCA(n_components=k, svd_solver="full")
    else:
        pca = PCA(
            n_components=k, svd_solver="randomized", iterated_power=20, random_state=random_state
        )
    pca.fit(features[fit_rows])

    def frame(rows: List[int]) -> pd.DataFrame:
        coords = pca.transform(features[rows])
        df = pd.DataFrame(coords, columns=[f"dim_{i}" for i in range(1, k + 1)])
        df.insert(0, "sample_id", [str(sample_ids[r]) for r in rows])
        return df

    return frame(fit_rows), frame(project_rows)
