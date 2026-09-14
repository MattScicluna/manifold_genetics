"""`subsample`: choose the fit samples of a cohort. Samples only, never SNPs."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..scaffold import _run_plink2_keep, _write_keep_file
from ..utils.tools import ToolResolver
from .cohort import PathLike, filter_labels_to_fam, read_cohort, read_fam_ids, write_cohort_config

logger = logging.getLogger(__name__)

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
        ImportError: ``sketch`` was not given and ``geosketch`` is not installed.
    """
    if len(pca) == 0:
        raise ValueError("no PCA rows match the cohort's .fam")
    if sketch is None:
        try:
            from geosketch import gs as sketch
        except ImportError:
            raise ImportError(
                "--geosketch needs the geosketch extra: "
                "pip install 'manifold-genetics[geosketch]'"
            ) from None
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
    force: bool = False,
    keep_runner: Callable = _run_plink2_keep,
) -> Path:
    """Write a cohort directory whose fit set is a chosen subset of the project set.

    The input's fit side is dropped: given a projection (HGDP fit, biobank
    project), the output fits on a subset of the biobank.

    Raises:
        ValueError: not exactly one of ``groups``, ``fit_samples`` and
            ``geosketch`` given, or ``geosketch`` given without ``pca``.
        FileExistsError: ``out_dir/config.yaml`` exists and ``force`` is False.
    """
    if sum([bool(groups), fit_samples is not None, geosketch is not None]) != 1:
        raise ValueError(
            "choose fit samples by exactly one of --group, --fit-samples or --geosketch"
        )
    if geosketch is not None and pca is None:
        raise ValueError("--geosketch requires --pca")
    out_dir = Path(out_dir).expanduser().resolve()
    config_path = out_dir / "config.yaml"
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} exists. Pass --force to overwrite it.")

    cohort = read_cohort(config)
    project = cohort.project
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    keep = data_dir / "fit_samples.txt"
    if fit_samples is not None:
        shutil.copy(fit_samples, keep)
    elif geosketch is not None:
        available = set(read_fam_ids(project.plink))
        pca_df = pd.read_csv(pca, dtype={"sample_id": str})
        pca_df = pca_df[pca_df["sample_id"].isin(available)].reset_index(drop=True)
        chosen = select_by_geosketch(pca_df, geosketch, seed=seed, n_pcs=n_pcs)
        _write_keep_file(Path(f"{project.plink}.fam"), pd.Series(chosen), keep)
        logger.info(
            "Selected %d of %d samples for the fit set via geosketch", len(chosen), len(available)
        )
    else:
        labels = pd.read_csv(project.labels, dtype={"sample_id": str}, low_memory=False)
        available = set(read_fam_ids(project.plink))
        labels = labels[labels["sample_id"].isin(available)]
        chosen = select_by_groups(labels, groups, include_rest=include_rest, seed=seed)
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

    return write_cohort_config(
        out_dir,
        based_on=cohort,
        preset="subsample",
        data={
            "fit_plink": "data/fit_subset",
            "project_plink": "data/project_subset",
            "fit_labels": "data/fit_labels.csv",
            "project_labels": "data/project_labels.csv",
            "fit_colormap": "colormap_fit.json",
            "project_colormap": "colormap_project.json",
            "output_dir": "outputs",
        },
        written_by="manifold-genetics subsample",
    )


__all__ = ["Group", "parse_group", "select_by_groups", "select_by_geosketch", "subsample"]
