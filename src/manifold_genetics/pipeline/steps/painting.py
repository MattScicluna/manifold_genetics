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
    chromosome_weight,
    features_from_sum,
    method_matrix,
    read_paint_matrix,
    read_pbwt_samples,
)
from ...painting.pbwt import expand_phased_vcf, paint_chromosome
from ...painting.reduce import match_cohort, painting_pca
from ...painting.vcfheader import read_vcf_samples
from ...pca.plink import read_fam
from ..config import IOConfig, PaintingConfig, PCAConfig
from .paths import painting_output_paths

logger = logging.getLogger(__name__)

__all__ = [
    "PaintingStepResult",
    "check_samples",
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
    pbwt: str, vcfs: Mapping[str, Path], roots: Mapping[str, Path], method: str, force: bool
):
    """Paint every chromosome and sum its weighted matrix. Returns ``(sample_ids, summed)``.

    Summed one chromosome at a time, so at most two N x N matrices are held.
    """
    matrix = method_matrix(method)
    sample_ids = None
    summed = None
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
        weighted = values * chromosome_weight(method, values, chrom)
        summed = weighted if summed is None else summed + weighted
        del values, weighted
    return sample_ids, summed


def _describe_difference(ids: Sequence[str], first: Sequence[str]) -> str:
    in_ids, in_first = set(ids), set(first)
    extra = [s for s in ids if s not in in_first]
    lacking = [s for s in first if s not in in_ids]
    if not extra and not lacking:
        return "It has the same samples in a different order."
    parts = [f"It has {len(ids)} samples and the first has {len(first)}."]
    if extra:
        parts.append(f"{len(extra)} only in it, for example {', '.join(extra[:5])}.")
    if lacking:
        parts.append(f"{len(lacking)} missing from it, for example {', '.join(lacking[:5])}.")
    return " ".join(parts)


def _cohort_rows(
    sample_ids: Sequence[str],
    plink: Optional[PathLike],
    cohort: str,
    missing: str,
    source: str,
):
    """``(rows, ids)`` for a cohort: every painted sample when ``plink`` is None."""
    if plink is None:
        return list(range(len(sample_ids))), list(sample_ids)
    fids, iids = read_fam(plink)
    return match_cohort(sample_ids, fids, iids, cohort=cohort, missing=missing, source=source)


def check_samples(
    vcfs: Mapping[str, Path],
    *,
    fit_plink: Optional[PathLike] = None,
    project_plink: Optional[PathLike] = None,
    missing: str = "error",
):
    """Check the phased VCFs' samples against each other and the cohorts, before painting.

    Reads only the VCF headers, so a mismatch is reported in seconds rather than
    after every chromosome has been painted.

    Returns:
        ``(sample_ids, (fit_rows, fit_ids), (project_rows, project_ids))``, where
        the rows index ``sample_ids``, the VCF's samples in file order.

    Raises:
        FileNotFoundError: a chromosome's VCF does not exist.
        ValueError: the chromosomes list different samples, or a cohort sample
            is not in the VCF and ``missing`` is ``error``.
    """
    sample_ids, first = None, None
    for chrom, vcf in vcfs.items():
        ids = read_vcf_samples(vcf)
        if sample_ids is None:
            sample_ids, first = ids, chrom
        elif ids != sample_ids:
            raise ValueError(
                f"The phased VCF for chromosome {chrom} ({vcf}) does not list the same "
                f"samples in the same order as chromosome {first}. "
                f"{_describe_difference(ids, sample_ids)} Painting matrices can only be "
                "summed when every chromosome has identical samples."
            )
    source = f"the phased VCFs ({vcfs[first]} and the other chromosomes)"
    fit = _cohort_rows(sample_ids, fit_plink, "fit", missing, source)
    if project_plink is None:
        project = fit
    else:
        project = _cohort_rows(sample_ids, project_plink, "project", missing, source)
    logger.info(
        f"Phased VCFs list {len(sample_ids)} samples on {len(vcfs)} chromosome(s): "
        f"{len(fit[0])} fit, {len(project[0])} project"
    )
    return sample_ids, fit, project


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
    method: str = "hc",
    n_pcs: int = 50,
    force: bool = False,
    pbwt: Optional[str] = None,
    missing: str = "error",
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
        missing: ``error`` (the default) stops before painting when a cohort
            sample is not in the VCFs. ``drop`` leaves such samples out of
            their cohort with a warning.

    Returns:
        DataFrame written to ``project_output`` (sample_id, dim_1, ..., dim_k).
    """
    if pbwt is None:
        from ...utils.tools import ToolResolver

        pbwt = ToolResolver().resolve_pbwt()

    method_matrix(method)  # an unknown method fails before any painting
    vcfs = expand_phased_vcf(phased_vcf, chromosomes)
    # Every check that needs only the VCF headers and the .fam files runs here,
    # before pbwt, which takes most of an hour on a genome.
    vcf_ids, (fit_rows, fit_ids), (project_rows, project_ids) = check_samples(
        vcfs, fit_plink=fit_plink, project_plink=project_plink, missing=missing
    )
    pbwt_dir = Path(pbwt_dir)
    roots = {c: pbwt_dir / f"chr{c}" for c in chromosomes}
    sample_ids, summed = _paint_all(pbwt, vcfs, roots, method, force)
    if sample_ids != vcf_ids:
        raise ValueError(
            f"pbwt listed {len(sample_ids)} samples but the VCF headers list {len(vcf_ids)}, "
            f"or in another order. Delete {pbwt_dir} or set force to repaint."
        )
    features = features_from_sum(summed, method, labels=sample_ids)
    logger.info(f"Painted {len(sample_ids)} samples on {len(vcfs)} chromosome(s)")

    if coancestry_output:
        coancestry_output = Path(coancestry_output)
        coancestry_output.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(summed, index=sample_ids, columns=sample_ids)
        frame.index.name = "sample_id"
        frame.to_csv(coancestry_output)

    # Coordinates are written under the .fam's IIDs, which is what the labels
    # file is keyed on, even when the VCF names samples FID_IID.
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
        missing=painting.missing,
    )
    return PaintingStepResult(
        fit_pca=paths["fit_pca"],
        project_pca=paths["project_pca"],
        coancestry=paths["coancestry"],
        coords_df=coords,
    )
