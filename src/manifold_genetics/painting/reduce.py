"""Coancestry features to ``sample_id, dim_1 ... dim_n`` coordinates.

Painting is all against all within the painted set, so there is no separate
model to project a second cohort through the way genotype PCA has. What does
carry over is the shape of the data: every row is one recipient's copying
profile over the *same* donors, the painted set. So the fit cohort's rows fit
the PCA and the project cohort's rows are projected through it, exactly as a
project cohort's genotypes are projected through loadings fitted elsewhere.

The decomposition is an uncentered SVD of the dense feature matrix, as both
published recipes take it (see ``painting_pca``). The package's in-process
genotype backend is not reused, because it reads PLINK ``.bed`` files rather
than an array.
"""

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd, svd_flip

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
    method: str = "hc",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Uncentered SVD of ``fit_rows`` of ``features``, then coordinates for both cohorts.

    Neither published recipe centers the columns: fineSTRUCTURE's ``mypca``
    takes the eigenvectors of ``X X^T`` after centering each *row*, and the
    haplotype components are the SVD of ``log10(A + 1)`` as it stands. So
    sklearn's ``PCA``, which centers columns, is not used. With
    ``X_fit = U D V^T``, a row ``x`` gets ``x V`` for ``finestructure`` (the
    fit rows get ``U D``, the scale of genotype PC scores, while ``mypca`` itself
    reports ``U``, which differs only by one scale per component) and
    ``x V D^(-1/2)`` for ``hc`` (the fit rows get ``U sqrt(D)``, as published).

    ``n_components`` is capped at one less than the number of fit samples,
    with a warning.

    Returns:
        ``(fit_coords, project_coords)``, each ``sample_id, dim_1 ... dim_k``.
    """
    if method not in ("finestructure", "hc"):
        raise ValueError(f"Unknown painting method: {method!r}")
    fit_rows, project_rows = list(fit_rows), list(project_rows)
    if len(fit_rows) < 2:
        raise ValueError("The fit cohort needs at least two painted samples")
    k = min(n_components, len(fit_rows) - 1, features.shape[1])
    if k < n_components:
        logger.warning(
            f"Painting PCA: {n_components} components requested but the fit cohort "
            f"supports {k}; writing {k}."
        )
    x_fit = features[fit_rows]
    if len(fit_rows) <= _FULL_SVD_MAX:
        u, d, vt = np.linalg.svd(x_fit, full_matrices=False)
        u, d, vt = u[:, :k], d[:k], vt[:k]
    else:
        u, d, vt = randomized_svd(x_fit, k, n_iter=20, random_state=random_state, flip_sign=False)
    # Deterministic signs: the largest loading of each component is positive.
    u, vt = svd_flip(u, vt, u_based_decision=False)
    scale = np.ones_like(d) if method == "finestructure" else 1.0 / np.sqrt(np.maximum(d, 1e-300))
    loadings = vt.T * scale

    def frame(rows: List[int]) -> pd.DataFrame:
        coords = features[rows] @ loadings
        df = pd.DataFrame(coords, columns=[f"dim_{i}" for i in range(1, k + 1)])
        df.insert(0, "sample_id", [str(sample_ids[r]) for r in rows])
        return df

    return frame(fit_rows), frame(project_rows)
