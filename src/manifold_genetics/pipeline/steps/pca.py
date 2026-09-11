"""In-process PCA step.

``run_pca()`` is the seam shared by the ``manifold-genetics pca`` subcommand and
``run_pca_step()``, so the CLI and the orchestrator cannot drift in how they fit
and project (spec goal 2). ``run_pca_step()`` adds the pipeline's config, path
and checkpoint layer on top.

The real work still happens in a separate process: ``PCA`` shells out to the
flashpca binary. That boundary is L1 and is unchanged here.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ...pca import PCA
from ..config import IOConfig, PCAConfig
from .paths import pca_output_paths

logger = logging.getLogger(__name__)

__all__ = ["PCAStepResult", "pca_output_paths", "run_pca", "run_pca_step"]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PCAStepResult:
    """Typed outputs of the PCA step.

    ``coords_df`` is excluded from comparison: DataFrame ``==`` is elementwise
    and would make a generated ``__eq__`` raise.
    """

    fit_pca: Path
    project_pca: Path
    coords_df: Optional[pd.DataFrame] = field(default=None, compare=False)
    skipped: bool = False


def run_pca(
    fit_plink: PathLike,
    project_plink: Optional[PathLike] = None,
    *,
    project_output: PathLike,
    fit_output: Optional[PathLike] = None,
    flashpca_dir: Optional[PathLike] = None,
    n_pcs: int = 50,
    force: bool = False,
    backend: str = "python",
) -> pd.DataFrame:
    """Run FlashPCA and write coordinate CSVs.

    Args:
        fit_plink: PLINK prefix the PCA model is fitted on.
        project_plink: PLINK prefix to project into that space. ``None`` fits and
            projects the same dataset (the ``pca --input X`` shape).
        project_output: CSV path for the projected coordinates.
        fit_output: Optional CSV path for the fit set's own coordinates.
        flashpca_dir: Directory for flashpca's raw intermediate outputs.
        n_pcs: Number of principal components.
        force: Recompute even when cached flashpca outputs exist.
        backend: ``"flashpca"`` or ``"python"``; see PCAConfig.

    Returns:
        DataFrame written to ``project_output`` (sample_id, dim_1, ..., dim_N).
    """
    pca = PCA(n_components=n_pcs, force=force, backend=backend)

    if project_plink is None:
        return pca.fit_transform(fit_plink, output_path=project_output)

    pca.fit(fit_plink, output_dir=flashpca_dir)
    if fit_output:
        pca.project(fit_plink, output_path=fit_output)
    return pca.project(project_plink, output_path=project_output)


def _existing_pcs_mismatch(project_pca: Path, n_pcs: int) -> bool:
    """True when a cached project PCA CSV cannot be trusted for ``n_pcs``.

    Reusing a cached file with the wrong number of ``dim_`` columns would feed
    wrong-dimensionality coordinates to every downstream step.
    """
    if not project_pca.exists():
        return False
    try:
        header = pd.read_csv(project_pca, nrows=1)
    except Exception as e:  # unreadable/corrupt cache — recompute rather than crash
        logger.warning(f"Could not check existing PCA file: {e}")
        return True

    existing_n_pcs = len([col for col in header.columns if col.startswith("dim_")])
    if existing_n_pcs != n_pcs:
        logger.info(f"PCA component mismatch: existing={existing_n_pcs}, requested={n_pcs}")
        logger.info("Forcing PCA recomputation...")
        return True
    return False


def run_pca_step(io: IOConfig, pca: PCAConfig) -> PCAStepResult:
    """Fit PCA on the fit cohort and project the project cohort.

    Idempotent: ``PCA(force=False)`` reuses cached flashpca outputs, and a cached
    CSV whose component count disagrees with ``pca.n_pcs`` forces a recompute.
    """
    paths = pca_output_paths(io, pca)
    fit_pca, project_pca = paths["fit_pca"], paths["project_pca"]
    flashpca_dir = paths["flashpca_dir"]

    project_pca.parent.mkdir(parents=True, exist_ok=True)
    flashpca_dir.mkdir(parents=True, exist_ok=True)

    force = pca.force or _existing_pcs_mismatch(project_pca, pca.n_pcs)

    logger.info(f"Running PCA (fit: {io.fit_plink}, project: {io.project_plink})")
    coords = run_pca(
        io.fit_plink,
        io.project_plink,
        fit_output=fit_pca,
        project_output=project_pca,
        flashpca_dir=flashpca_dir,
        n_pcs=pca.n_pcs,
        force=force,
        backend=pca.backend,
    )

    return PCAStepResult(fit_pca=fit_pca, project_pca=project_pca, coords_df=coords)
