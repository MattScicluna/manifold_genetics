"""`subsample`: choose the fit samples of a cohort. Samples only, never SNPs."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.tools import ToolResolver
from .cohort import (
    PathLike,
    check_label_coverage,
    filter_geographic_to_fam,
    filter_labels_to_fam,
    read_cohort,
    read_fam_ids,
    write_cohort_config,
)

logger = logging.getLogger(__name__)


def _plink2_keep(bfile: Path, keep: Path, out: Path, plink2: str) -> None:
    """``scaffold._run_plink2_keep``, looked up at call time: scaffold imports
    this package for the label-coverage rule, so a module-level import here
    would be circular."""
    from .. import scaffold

    scaffold._run_plink2_keep(bfile, keep, out, plink2)


def _run_pca(fit_plink, project_plink, **kwargs) -> pd.DataFrame:
    """``pipeline.steps.pca.run_pca``, looked up at call time for the same reason
    as ``_plink2_keep``: the pipeline package imports this one."""
    from ..pipeline.steps.pca import run_pca

    return run_pca(fit_plink, project_plink, **kwargs)


# How many PCs to compute for a sketch when no --pca is given, and how many
# samples to fit them on: the settings of the UK Biobank pilot
# (examples/ukbb/fullpca_geosketch_pilot), whose fit on 100k took 1.5 h.
SKETCH_N_PCS = 20
SKETCH_POOL = 100_000


_GROUP_FORMAT = "COLUMN=PATTERN:COUNT, e.g. race_ethnicity=White|European:10000"


@dataclass(frozen=True)
class Group:
    column: str
    pattern: str
    count: int


def parse_group(spec: str) -> Group:
    column, sep, rest = spec.partition("=")
    pattern, sep2, count_text = rest.rpartition(":")
    if not sep or not sep2 or not column or not pattern:
        raise ValueError(f"--group {spec!r} is not of the form {_GROUP_FORMAT}")
    try:
        count = int(count_text)
    except ValueError:
        raise ValueError(f"--group {spec!r}: COUNT must be an integer ({_GROUP_FORMAT})")
    if count <= 0:
        raise ValueError(f"--group {spec!r}: COUNT must be positive ({_GROUP_FORMAT})")
    return Group(column, pattern, count)


def select_by_groups(
    labels: pd.DataFrame, groups: Sequence[Group], *, include_rest: bool, seed: int
) -> List[str]:
    """Sample IDs chosen group by group; a sample is taken at most once.

    This is examples/aou/shared/select_samples.py with the column named rather
    than guessed, which is what makes it serve UK Biobank and All of Us alike.
    """
    taken: List[str] = []
    taken_set: set = set()
    matched: set = set()
    for group in groups:
        if group.column not in labels.columns:
            raise ValueError(
                f"column {group.column!r} is not in the labels (have: {', '.join(labels.columns)})"
            )
        mask = (
            labels[group.column]
            .astype(str)
            .str.contains(group.pattern, case=False, regex=True, na=False)
        )
        matched.update(labels.loc[mask, "sample_id"])
        candidates = labels.loc[mask & ~labels["sample_id"].isin(taken_set), "sample_id"]
        if len(candidates) > group.count:
            chosen = candidates.sample(n=group.count, random_state=seed)
            logger.info(
                "%s=%s: %d available, taking %d",
                group.column,
                group.pattern,
                len(candidates),
                group.count,
            )
        else:
            chosen = candidates
            logger.info(
                "%s=%s: %d available, taking all", group.column, group.pattern, len(candidates)
            )
        taken.extend(chosen)
        taken_set.update(chosen)
    if include_rest:
        rest = labels.loc[~labels["sample_id"].isin(matched), "sample_id"]
        logger.info("rest: %d samples matched no group", len(rest))
        taken.extend(rest)
    return taken


def select_by_geosketch(
    pca: pd.DataFrame,
    n: int,
    seed: int,
    sketch: Optional[Callable] = None,
    n_pcs: Optional[int] = None,
) -> List[str]:
    """Geometric sketching (Hie et al. 2019) on a run's PCA coordinates.

    Mirrors examples/_shared/select_samples_geosketch.py: every column but
    ``sample_id`` is treated as a dimension, optionally truncated to the first
    ``n_pcs`` of them, and cast to ``float32`` before sketching. As in that
    script, ``n`` is clamped to the number of rows available (with a warning)
    rather than passed straight to ``gs``, which requires ``n <= len(X)``.

    Raises:
        ValueError: ``pca`` has no rows (nothing survived the ``.fam`` filter).
    """
    if len(pca) == 0:
        raise ValueError("no PCA rows match the cohort's .fam")
    if sketch is None:
        from geosketch import gs as sketch
    if n > len(pca):
        logger.warning(
            "geosketch: requested %d samples but only %d available; using %d", n, len(pca), len(pca)
        )
        n = len(pca)
    dim_cols = [c for c in pca.columns if c != "sample_id"]
    if n_pcs is not None:
        dim_cols = dim_cols[:n_pcs]
    X = pca[dim_cols].to_numpy().astype(np.float32)
    index = np.sort(np.asarray(sketch(X, n, seed=seed, replace=False)))
    return list(pca["sample_id"].iloc[index])


def _label_free_pca(
    project_plink: Path,
    out_dir: Path,
    *,
    pool: int,
    n_pcs: int,
    seed: int,
    max_memory_gb: Optional[float],
    keep_runner: Callable,
    pca_runner: Callable,
) -> Path:
    """PCs of every cohort sample, chosen without looking at a label.

    The PCA is fitted on ``pool`` samples drawn uniformly from the ``.fam`` and
    every sample is projected in. Geosketch needs such coordinates -- it cannot
    run on hundreds of thousands of genotype columns -- but a random pool makes
    them as imbalanced as the cohort is. That is acceptable because they are
    used only to choose the fit set; ``run`` then fits a fresh PCA on it.

    Skipped when ``out_dir/sketch_pca.csv`` exists, since the fit can take hours.
    """
    from ..scaffold import _write_keep_file

    csv = out_dir / "sketch_pca.csv"
    if csv.exists():
        logger.info("Reusing %s; delete it to recompute the sketch PCs", csv)
        return csv

    work = out_dir / "sketch_pca"
    work.mkdir(parents=True, exist_ok=True)
    fam = Path(f"{project_plink}.fam")
    ids = pd.Series(read_fam_ids(project_plink))
    if pool >= len(ids):
        logger.info("Sketch PCs: fitting on all %d samples", len(ids))
        fit_plink = project_plink
    else:
        logger.info("Sketch PCs: fitting on a random %d of %d samples", pool, len(ids))
        keep = work / "pool_samples.txt"
        _write_keep_file(fam, ids.sample(n=pool, random_state=seed), keep)
        fit_plink = work / "sketch_pool"
        keep_runner(project_plink, keep, fit_plink, ToolResolver().resolve_plink2())

    budget = (
        {}
        if max_memory_gb is None
        else dict(max_fit_memory_gb=max_memory_gb, max_project_memory_gb=max_memory_gb)
    )
    partial = work / "sketch_pca.partial.csv"
    pca_runner(
        fit_plink,
        project_plink,
        project_output=partial,
        flashpca_dir=work / "pca",
        n_pcs=n_pcs,
        **budget,
    )
    # Written under another name and renamed, so a killed run never leaves a
    # truncated CSV that the next run would take as finished.
    partial.replace(csv)
    if fit_plink != project_plink:
        for ext in ("bed", "bim", "fam"):
            Path(f"{fit_plink}.{ext}").unlink(missing_ok=True)
    return csv


def subsample(
    config: PathLike,
    out_dir: PathLike,
    *,
    groups: Sequence[Group] = (),
    include_rest: bool = False,
    seed: int = 42,
    fit_samples: Optional[PathLike] = None,
    geosketch: Optional[int] = None,
    pca: Optional[PathLike] = None,
    n_pcs: Optional[int] = None,
    sketch_pool: Optional[int] = None,
    max_memory_gb: Optional[float] = None,
    force: bool = False,
    keep_runner: Callable = _plink2_keep,
    pca_runner: Callable = _run_pca,
) -> Path:
    """Write a cohort directory whose fit set is a chosen subset of the project set.

    The input's fit side is dropped: given a projection (HGDP fit, biobank
    project), the output fits on a subset of the biobank.

    ``geosketch`` without ``pca`` sketches in PCs computed here without labels
    (see ``_label_free_pca``): ``n_pcs`` of them (default 20), fitted on
    ``sketch_pool`` random samples (default 100,000), saved as
    ``out_dir/sketch_pca.csv``. With ``pca``, ``n_pcs`` instead limits how many
    of its columns are used.

    Raises:
        ValueError: not exactly one of ``groups``, ``fit_samples`` and
            ``geosketch`` given, or ``sketch_pool`` given without the PCs being
            computed here (``geosketch`` and no ``pca``); or the
            project labels cover less than half the project ``.fam`` (checked
            before plink runs -- the fit set is drawn from that ``.fam``, so
            its coverage can only be what the input's is).
        FileExistsError: ``out_dir/config.yaml`` exists and ``force`` is False.
    """
    if sum([bool(groups), fit_samples is not None, geosketch is not None]) != 1:
        raise ValueError(
            "choose fit samples by exactly one of --group, --fit-samples or --geosketch"
        )
    if sketch_pool is not None and (geosketch is None or pca is not None):
        raise ValueError("--sketch-pool applies only to --geosketch without --pca")
    out_dir = Path(out_dir).expanduser().resolve()
    config_path = out_dir / "config.yaml"
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} exists. Pass --force to overwrite it.")

    cohort = read_cohort(config)
    project = cohort.project
    check_label_coverage(project.labels, project.plink)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    from ..scaffold import _write_keep_file

    keep = data_dir / "fit_samples.txt"
    if fit_samples is not None:
        lines = [line for line in Path(fit_samples).read_text().splitlines() if line.strip()]
        if not lines:
            raise ValueError("no samples were selected; the --fit-samples file is empty")
        shutil.copy(fit_samples, keep)
    elif geosketch is not None:
        available = set(read_fam_ids(project.plink))
        if pca is None:
            pca = _label_free_pca(
                project.plink,
                out_dir,
                pool=SKETCH_POOL if sketch_pool is None else sketch_pool,
                n_pcs=SKETCH_N_PCS if n_pcs is None else n_pcs,
                seed=seed,
                max_memory_gb=max_memory_gb,
                keep_runner=keep_runner,
                pca_runner=pca_runner,
            )
        pca_df = pd.read_csv(pca, dtype={"sample_id": str})
        pca_df = pca_df[pca_df["sample_id"].isin(available)].reset_index(drop=True)
        chosen = select_by_geosketch(pca_df, geosketch, seed=seed, n_pcs=n_pcs)
        if len(chosen) == 0:
            raise ValueError(
                "no samples were selected; check --group patterns against the label values"
            )
        _write_keep_file(Path(f"{project.plink}.fam"), pd.Series(chosen), keep)
        logger.info(
            "Selected %d of %d samples for the fit set via geosketch", len(chosen), len(available)
        )
    else:
        labels = pd.read_csv(project.labels, dtype={"sample_id": str}, low_memory=False)
        available = set(read_fam_ids(project.plink))
        labels = labels[labels["sample_id"].isin(available)]
        chosen = select_by_groups(labels, groups, include_rest=include_rest, seed=seed)
        if len(chosen) == 0:
            raise ValueError(
                "no samples were selected; check --group patterns against the label values"
            )
        _write_keep_file(Path(f"{project.plink}.fam"), pd.Series(chosen), keep)
        logger.info("Selected %d of %d samples for the fit set", len(chosen), len(available))

    keep_runner(project.plink, keep, data_dir / "fit_subset", ToolResolver().resolve_plink2())
    for ext in ("bed", "bim", "fam"):
        target = data_dir / f"project_subset.{ext}"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(Path(f"{project.plink}.{ext}").resolve())

    filter_labels_to_fam(project.labels, data_dir / "fit_subset", data_dir / "fit_labels.csv")
    filter_labels_to_fam(
        project.labels, data_dir / "project_subset", data_dir / "project_labels.csv"
    )
    shutil.copy(project.colormap, out_dir / "colormap_fit.json")
    shutil.copy(project.colormap, out_dir / "colormap_project.json")

    data = {
        "fit_plink": "data/fit_subset",
        "project_plink": "data/project_subset",
        "fit_labels": "data/fit_labels.csv",
        "project_labels": "data/project_labels.csv",
        "fit_colormap": "colormap_fit.json",
        "project_colormap": "colormap_project.json",
        "output_dir": "outputs",
    }
    # subsample keeps the project side intact -- it only chooses the fit set --
    # so a geographic file the input covers is still valid, filtered to the
    # (unchanged) project .fam.
    if cohort.geographic_coords is not None:
        filter_geographic_to_fam(
            cohort.geographic_coords, data_dir / "project_subset", data_dir / "geographic.csv"
        )
        data["geographic_coords"] = "data/geographic.csv"

    return write_cohort_config(
        out_dir,
        based_on=cohort,
        preset="subsample",
        data=data,
        written_by="manifold-genetics subsample",
    )


__all__ = ["Group", "parse_group", "select_by_groups", "select_by_geosketch", "subsample"]
