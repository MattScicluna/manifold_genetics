"""`preprocess`: one cohort directory in, one out, SNPs filtered in between."""

import dataclasses
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

from ..utils.tools import ToolResolver
from .cohort import (
    PathLike,
    bed_expected_size,
    bed_is_complete,
    check_label_coverage,
    filter_geographic_to_fam,
    filter_labels_to_fam,
    filter_labels_to_ids,
    read_cohort,
    read_fam_ids,
    write_cohort_config,
)
from .flags import PreprocessOptions, shell_argv
from .references import default_tools_dir

logger = logging.getLogger(__name__)


def _run(argv: List[str]) -> None:
    logger.info("Running: %s", " ".join(argv))
    subprocess.run(argv, check=True)


def _first_colormap_column(colormap: Path) -> Optional[str]:
    import json

    keys = list(json.loads(colormap.read_text()))
    return keys[0] if keys else None


def preprocess(
    fit_config: PathLike,
    out_dir: PathLike,
    *,
    project_config: Optional[PathLike] = None,
    options: PreprocessOptions = PreprocessOptions(),
    force: bool = False,
    runner: Callable[[List[str]], None] = _run,
) -> Path:
    """Filter one cohort, or intersect two, into a new cohort directory.

    One config: its own fit and project sets are the two sides; the output keeps
    the input's preset and label layout. Two configs: the first's fit side and
    the second's project side; the output is a ``projection`` with separate
    labels and colormaps per side.

    Args:
        runner: Executes the shell argv. Replaced in tests.

    Returns:
        The config written.

    Raises:
        FileExistsError: ``out_dir/config.yaml`` exists and ``force`` is False.
        ValueError: a label file covers less than half its ``.fam``. Checked
            before the shell runs: it removes no samples, so the input's
            coverage is the output's.
        subprocess.CalledProcessError: the shell failed; its own output says why.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    config_path = out_dir / "config.yaml"
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} exists. Pass --force to overwrite it.")

    fit_cohort = read_cohort(fit_config)
    project_cohort = read_cohort(project_config) if project_config else fit_cohort
    fit, project = fit_cohort.fit, project_cohort.project
    two_sided = project_config is not None or not fit_cohort.shared_labels

    # Pre-flight, before the hours-long shell: the labels that will be filtered
    # afterwards must already cover their genotypes.
    for labels, plink in dict.fromkeys([(fit.labels, fit.plink), (project.labels, project.plink)]):
        check_label_coverage(labels, plink)

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if options.tools_dir is None:
        options = dataclasses.replace(options, tools_dir=default_tools_dir())

    resolver = ToolResolver()
    runner(
        shell_argv(
            fit.plink,
            project.plink,
            data_dir,
            options,
            plink2=resolver.resolve_plink2(),
            plink=resolver.resolve_plink1(),
            python=sys.executable,
        )
    )
    for name in ("fit_subset", "project_subset"):
        prefix = data_dir / name
        bed = data_dir / f"{name}.bed"
        if not bed.exists():
            raise RuntimeError(f"the shell finished without writing {bed}")
        if not bed_is_complete(prefix):
            expected = bed_expected_size(prefix)
            actual = bed.stat().st_size
            raise RuntimeError(
                f"{prefix} is incomplete: expected {expected} bytes but found {actual} "
                "(the shell may have been killed mid-write)"
            )

    data = {"fit_plink": "data/fit_subset", "project_plink": "data/project_subset"}
    visualization = None
    if two_sided:
        filter_labels_to_fam(fit.labels, data_dir / "fit_subset", data_dir / "fit_labels.csv")
        filter_labels_to_fam(
            project.labels, data_dir / "project_subset", data_dir / "project_labels.csv"
        )
        shutil.copy(fit.colormap, out_dir / "colormap_fit.json")
        shutil.copy(project.colormap, out_dir / "colormap_project.json")
        data.update(
            fit_labels="data/fit_labels.csv",
            project_labels="data/project_labels.csv",
            fit_colormap="colormap_fit.json",
            project_colormap="colormap_project.json",
        )
        preset = (
            "projection" if project_config is not None else (fit_cohort.preset or "whole_cohort")
        )
        if preset == "projection":
            visualization = {
                "projection_plot_fit_column": _first_colormap_column(fit.colormap),
                "projection_plot_project_column": _first_colormap_column(project.colormap),
            }
    else:
        # One label file covering both sides: filter it to their union, fit first.
        _filter_shared_labels(fit.labels, data_dir)
        shutil.copy(fit.colormap, out_dir / "colormap.json")
        data.update(labels="data/labels.csv", colormap="colormap.json")
        preset = fit_cohort.preset or "whole_cohort"

    # Neither preprocess nor subsample removes a project-side sample, so a
    # geographic file the input still covers is still valid for the output --
    # filtered to the (possibly reordered) project .fam it now sits beside. The
    # two-config case takes the project config's file, matching its .fam.
    if project_cohort.geographic_coords is not None:
        filter_geographic_to_fam(
            project_cohort.geographic_coords,
            data_dir / "project_subset",
            data_dir / "geographic.csv",
        )
        data["geographic_coords"] = "data/geographic.csv"

    data["output_dir"] = "outputs"

    return write_cohort_config(
        out_dir,
        based_on=fit_cohort,
        preset=preset,
        data=data,
        visualization=visualization,
        written_by="manifold-genetics preprocess",
    )


def _filter_shared_labels(labels: Path, data_dir: Path) -> None:
    # Union of both sides' ids, fit first and order-preserving, so a sample that
    # appears in both subsets is not requested (and so not duplicated) twice.
    ids = list(
        dict.fromkeys(
            read_fam_ids(data_dir / "fit_subset") + read_fam_ids(data_dir / "project_subset")
        )
    )
    filter_labels_to_ids(labels, ids, data_dir / "labels.csv")


__all__ = ["preprocess"]
