"""In-process haplotype painting step: pbwt coancestry in place of genotype PCA.

``run_painting()`` is the seam shared by the ``manifold-genetics paint``
subcommand and ``run_painting_step()``, as ``run_pca()`` is for PCA. Its output
is the same ``sample_id, dim_1 ... dim_n`` pair of CSVs the PCA step writes, so
the embedding, visualization and metrics steps take it unchanged.

How the fit and project cohorts are handled: every sample in the phased VCF is
painted against every other, once. The summed matrix has one row per recipient
and one column per donor, and the donors are the whole painted set for every
row. The fit cohort's rows fit the PCA and the project cohort's rows are
projected through it. A sample in the VCF that is in neither cohort still
serves as a donor, which is what makes a reference panel in the VCF useful.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

from ...painting.matrix import (
    build_features,
    method_matrix,
    read_paint_matrix,
    read_pbwt_samples,
)
from ...painting.pbwt import expand_phased_vcf, paint_chromosome
from ...painting.reduce import match_cohort, painting_pca
from ...pca.plink import read_fam
from ..config import IOConfig, PaintingConfig, PCAConfig
from .paths import painting_output_paths

logger = logging.getLogger(__name__)

__all__ = [
    "PaintingStepResult",
    "painting_output_paths",
    "run_painting",
    "run_painting_step",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PaintingStepResult:
    """Typed outputs of the painting step.

    ``fit_pca`` and ``project_pca`` are named as in ``PCAStepResult`` because
    they play the same part: the coordinates the embedding reads.
    """

    fit_pca: Optional[Path]
    project_pca: Optional[Path]
    coancestry: Optional[Path] = None
    coords_df: Optional[pd.DataFrame] = field(default=None, compare=False)
    skipped: bool = False


def _paint_all(
    pbwt: str, vcfs: Mapping[str, Path], roots: Mapping[str, Path], matrix: str, force: bool
):
    """Paint every chromosome. Returns ``(sample_ids, {chromosome: matrix})``."""
    sample_ids = None
    matrices = {}
    for chrom, vcf in vcfs.items():
        outputs = paint_chromosome(pbwt, vcf, roots[chrom], force=force)
        ids = read_pbwt_samples(outputs["samples"])
        if sample_ids is None:
            sample_ids, first = ids, chrom
        elif ids != sample_ids:
            raise ValueError(
                f"The phased VCF for chromosome {chrom} does not list the same samples in the "
                f"same order as chromosome {first}. Painting matrices can only be summed "
                "when every chromosome has identical samples."
            )
        labels, values = read_paint_matrix(outputs[matrix])
        if len(labels) != len(ids):
            raise ValueError(
                f"{outputs[matrix]} has {len(labels)} rows but pbwt listed {len(ids)} samples. "
                "Painting assumes diploid samples."
            )
        matrices[chrom] = values
    return sample_ids, matrices


def _cohort_rows(sample_ids: Sequence[str], plink: Optional[PathLike], cohort: str):
    """``(rows, ids)`` for a cohort: every painted sample when ``plink`` is None."""
    if plink is None:
        return list(range(len(sample_ids))), list(sample_ids)
    fids, iids = read_fam(plink)
    return match_cohort(sample_ids, fids, iids, cohort=cohort), list(iids)


def run_painting(
    phased_vcf: PathLike,
    *,
    chromosomes: Sequence[str],
    pbwt_dir: PathLike,
    project_output: PathLike,
    fit_output: Optional[PathLike] = None,
    coancestry_output: Optional[PathLike] = None,
    fit_plink: Optional[PathLike] = None,
    project_plink: Optional[PathLike] = None,
    method: str = "finestructure",
    n_pcs: int = 50,
    force: bool = False,
    pbwt: Optional[str] = None,
) -> pd.DataFrame:
    """Paint each chromosome, build coancestry features and reduce them by PCA.

    Args:
        phased_vcf: VCF path template containing ``{chr}``.
        chromosomes: Names substituted for ``{chr}``.
        pbwt_dir: Where pbwt's per-chromosome outputs go (reused on a rerun).
        project_output: CSV for the project cohort's coordinates.
        fit_output: Optional CSV for the fit cohort's coordinates.
        coancestry_output: Optional CSV for the summed matrix, recipients by donors:
            chunk counts for ``finestructure``, chunk lengths in centimorgans for ``hc``.
        fit_plink: PLINK prefix whose ``.fam`` names the fit cohort. ``None`` fits
            on every painted sample.
        project_plink: PLINK prefix whose ``.fam`` names the project cohort.
            ``None`` projects the fit cohort itself.
        method: ``finestructure`` (Lawson et al. 2012, chunk counts) or ``hc``
            (Yang et al. 2025, chunk lengths). See ``painting.matrix``.
        n_pcs: Principal components to keep (capped by the fit cohort's size).
        force: Repaint chromosomes even when their outputs are cached.
        pbwt: The pbwt executable. Resolved with ``ToolResolver`` when ``None``.

    Returns:
        DataFrame written to ``project_output`` (sample_id, dim_1, ..., dim_k).
    """
    if pbwt is None:
        from ...utils.tools import ToolResolver

        pbwt = ToolResolver().resolve_pbwt()

    matrix = method_matrix(method)
    vcfs = expand_phased_vcf(phased_vcf, chromosomes)
    pbwt_dir = Path(pbwt_dir)
    roots = {c: pbwt_dir / f"chr{c}" for c in chromosomes}
    sample_ids, per_chromosome = _paint_all(pbwt, vcfs, roots, matrix, force)
    summed, features = build_features(per_chromosome, method, labels=sample_ids)
    del per_chromosome
    logger.info(f"Painted {len(sample_ids)} samples on {len(vcfs)} chromosome(s)")

    if coancestry_output:
        coancestry_output = Path(coancestry_output)
        coancestry_output.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(summed, index=sample_ids, columns=sample_ids)
        frame.index.name = "sample_id"
        frame.to_csv(coancestry_output)

    # Coordinates are written under the .fam's IIDs, which is what the labels
    # file is keyed on, even when the VCF names samples FID_IID.
    fit_rows, fit_ids = _cohort_rows(sample_ids, fit_plink, "fit")
    if project_plink is None:
        project_rows, project_ids = fit_rows, fit_ids
    else:
        project_rows, project_ids = _cohort_rows(sample_ids, project_plink, "project")

    fit_df, project_df = painting_pca(
        features,
        sample_ids,
        fit_rows=fit_rows,
        project_rows=project_rows,
        n_components=n_pcs,
        method=method,
    )
    fit_df["sample_id"] = fit_ids
    project_df["sample_id"] = project_ids

    if fit_output:
        Path(fit_output).parent.mkdir(parents=True, exist_ok=True)
        fit_df.to_csv(fit_output, index=False)
    project_output = Path(project_output)
    project_output.parent.mkdir(parents=True, exist_ok=True)
    project_df.to_csv(project_output, index=False)
    return project_df


def run_painting_step(
    io: IOConfig, pca: PCAConfig, painting: PaintingConfig, *, pbwt: Optional[str] = None
) -> PaintingStepResult:
    """Paint the phased VCFs and write coordinates for the fit and project cohorts."""
    paths = painting_output_paths(io, pca, painting)
    logger.info(
        f"Painting {io.phased_vcf} ({len(painting.chromosomes)} chromosomes, "
        f"method: {painting.method})"
    )
    coords = run_painting(
        io.phased_vcf,
        chromosomes=painting.chromosomes,
        pbwt_dir=paths["pbwt_dir"],
        project_output=paths["project_pca"],
        fit_output=paths["fit_pca"],
        coancestry_output=paths["coancestry"],
        fit_plink=io.fit_plink,
        project_plink=io.project_plink,
        method=painting.method,
        n_pcs=pca.n_pcs,
        force=painting.force,
        pbwt=pbwt,
    )
    return PaintingStepResult(
        fit_pca=paths["fit_pca"],
        project_pca=paths["project_pca"],
        coancestry=paths["coancestry"],
        coords_df=coords,
    )
