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

__all__ = ["MISSING_POLICIES", "match_cohort", "painting_pca"]

logger = logging.getLogger(__name__)

# The full SVD of an n x n matrix is O(n^3); past this many samples the
# randomized solver is far faster. 20 power iterations matches the genotype
# backend's choice, for the same reason: near-equal trailing eigenvalues.
_FULL_SVD_MAX = 5000


#: What to do with a cohort sample that the phased VCF does not have.
MISSING_POLICIES = ("error", "drop")

_EXAMPLES = 5


def _split_keys(name: str) -> List[str]:
    """Each way of cutting ``name`` at an underscore, both halves, lowercased."""
    tokens = name.lower().split("_")
    keys = []
    for i in range(1, len(tokens)):
        keys.append("_".join(tokens[:i]))
        keys.append("_".join(tokens[i:]))
    return keys


def _near_matches(painted_ids: Sequence[str], absent: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    """``{iid: vcf_name}`` for absent samples whose name differs only in format.

    Caught: a difference of case, a VCF name that is ``<prefix>_IID`` or
    ``IID_<suffix>`` with a prefix other than the FID, and a ``.fam`` IID that
    is itself ``FID_IID`` when the VCF has the bare IID.
    """
    whole: Dict[str, str] = {}
    parts: Dict[str, str] = {}
    for vid in painted_ids:
        whole.setdefault(vid.lower(), vid)
        for key in _split_keys(vid):
            parts.setdefault(key, vid)
    near = {}
    for fid, iid in absent:
        for name in (iid, f"{fid}_{iid}"):
            hit = whole.get(name.lower()) or parts.get(name.lower())
            if hit is None:
                hit = next((whole[k] for k in _split_keys(name) if k in whole), None)
            if hit is not None:
                near[iid] = hit
                break
    return near


def _missing_message(
    painted_ids: Sequence[str],
    absent: Sequence[Tuple[str, str]],
    n_cohort: int,
    cohort: str,
    source: str,
) -> str:
    near = _near_matches(painted_ids, absent)
    lines = [f"{len(absent)} of {n_cohort} {cohort} samples are not in {source}."]
    if near:
        pairs = ", ".join(f"{iid} vs {vid}" for iid, vid in list(near.items())[:_EXAMPLES])
        verb = "looks" if len(near) == 1 else "look"
        lines.append(
            f"{len(near)} of them {verb} like an ID format mismatch rather than absent "
            f"samples (.fam IID vs VCF sample: {pairs}). A VCF sample matches only when "
            "its name equals the .fam IID or FID_IID exactly, so rename one side to "
            "agree, for example with bcftools reheader --samples."
        )
    really_absent = [iid for _, iid in absent if iid not in near]
    if really_absent:
        lines.append(
            f"{len(really_absent)} {'is' if len(really_absent) == 1 else 'are'} absent "
            "from the VCF under any naming, "
            f"for example {', '.join(really_absent[:_EXAMPLES])}."
        )
    return " ".join(lines)


def match_cohort(
    painted_ids: Sequence[str],
    fids: Sequence[str],
    iids: Sequence[str],
    *,
    cohort: str,
    missing: str = "error",
    source: str = "the phased VCF",
) -> Tuple[List[int], List[str]]:
    """Row indices into the painted set for each sample of a ``.fam``, in ``.fam`` order.

    A VCF exported from PLINK names a sample either by its IID or by
    ``FID_IID``, depending on the export settings, so both are accepted.

    Args:
        painted_ids: Sample names in the VCF, in VCF order.
        fids: The cohort's ``.fam`` FIDs.
        iids: The cohort's ``.fam`` IIDs.
        cohort: ``fit`` or ``project``, for messages.
        missing: ``error`` stops on a cohort sample that the VCF lacks, and
            ``drop`` leaves it out of the cohort with a warning.
        source: How messages name the VCF.

    Returns:
        ``(rows, ids)``: the matched rows and the IIDs they belong to.

    Raises:
        ValueError: a cohort sample is not in the VCF and ``missing`` is
            ``error``, or no cohort sample is. The message separates names that
            differ only in format from samples that are absent, and says how to
            proceed.
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(
            f"Unknown painting missing policy: {missing!r}. Choose from: "
            f"{', '.join(MISSING_POLICIES)}"
        )
    index: Dict[str, int] = {}
    for i, sid in enumerate(painted_ids):
        index.setdefault(sid, i)
    rows, kept, absent = [], [], []
    for fid, iid in zip(fids, iids):
        row = index.get(iid, index.get(f"{fid}_{iid}"))
        if row is None:
            absent.append((fid, iid))
        else:
            rows.append(row)
            kept.append(iid)
    if not absent:
        return rows, kept

    message = _missing_message(painted_ids, absent, len(iids), cohort, source)
    if not rows:
        raise ValueError(f"{message} No {cohort} sample is left to paint.")
    if missing == "error":
        raise ValueError(
            f"{message} Painting needs every sample of the cohort by default. To go on "
            "without the missing ones, set painting: missing: drop in the config "
            "(--missing drop for manifold-genetics paint), which leaves them out of the "
            f"{cohort} cohort with a warning. Or remove them from the .fam."
        )
    logger.warning(f"{message} They are left out of the {cohort} cohort (missing: drop).")
    return rows, kept


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
