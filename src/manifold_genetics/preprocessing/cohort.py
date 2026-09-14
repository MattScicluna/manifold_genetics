"""The cohort directory: config.yaml, colormap(s), data/ with genotypes and labels.

`acquire` writes one; `preprocess` and `subsample` read one and write another;
`run` consumes one. Everything here is about reading that layout through the
config and writing it back with the paths moved."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml

from ..pipeline.configfile import load_config

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Side:
    """One half of a cohort: its genotypes, the labels that cover them, the colours."""

    plink: Path
    labels: Path
    colormap: Path


@dataclass(frozen=True)
class CohortConfig:
    path: Path
    raw: dict
    preset: Optional[str]
    fit: Side
    project: Side
    shared_labels: bool


def read_cohort(config_path: PathLike) -> CohortConfig:
    """Read a config and resolve both sides.

    A side's labels are ``<side>_labels`` when given, else the shared ``labels``;
    colormaps likewise. That is the precedence ``run`` applies.
    """
    path = Path(config_path).expanduser().resolve()
    resolved = load_config(path)
    raw = yaml.safe_load(path.read_text()) or {}

    def side(name: str) -> Side:
        labels = resolved.get(f"{name}_labels") or resolved.get("labels")
        colormap = resolved.get(f"{name}_colormap") or resolved.get("colormap")
        if labels is None or colormap is None:
            raise ValueError(f"{path} names no labels or colormap for the {name} side")
        return Side(
            plink=Path(resolved[f"{name}_plink"]), labels=Path(labels), colormap=Path(colormap)
        )

    return CohortConfig(
        path=path,
        raw=raw,
        preset=raw.get("preset"),
        fit=side("fit"),
        project=side("project"),
        shared_labels="labels" in resolved and "fit_labels" not in resolved,
    )


def read_fam_ids(prefix: PathLike) -> List[str]:
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, dtype=str, usecols=[0, 1])
    return list(fam[1])


def filter_labels_to_fam(labels: PathLike, prefix: PathLike, out: PathLike) -> int:
    """Write the rows of ``labels`` for the samples in ``prefix``.fam, in .fam order.

    Raises:
        ValueError: a sample in the .fam has no label row. A label file must
            cover its genotypes; a filtered copy that silently dropped samples
            would produce a figure with grey points and no error.
    """
    ids = read_fam_ids(prefix)
    frame = pd.read_csv(labels, dtype={"sample_id": str}, low_memory=False)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{labels} has no sample_id column")
    missing = set(ids) - set(frame["sample_id"])
    if missing:
        example = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"{len(missing)} of {len(ids)} samples in {prefix}.fam are not in {labels} "
            f"(e.g. {example}). Labels must cover the genotypes they describe."
        )
    # Drop duplicate sample_ids *before* indexing by the .fam order, so a label
    # file with repeated ids does not expand `.loc[ids]` into extra rows.
    frame = frame.drop_duplicates("sample_id")
    kept = frame.set_index("sample_id").loc[ids].reset_index()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False)
    return len(kept)


_CARRIED_SECTIONS = ("pca", "admixture", "embedding", "visualization", "skip")


def write_cohort_config(
    out_dir: Path,
    *,
    based_on: CohortConfig,
    preset: str,
    data: Dict[str, str],
    visualization: Optional[Dict[str, str]] = None,
    written_by: str,
) -> Path:
    """Write ``out_dir/config.yaml``: the input's settings with a new data section.

    The data section is replaced wholesale so that no stale key survives -- a
    ``labels`` left beside ``fit_labels`` is exactly the shape ``run`` rejects.
    """
    document: Dict[str, object] = {"preset": preset, "data": dict(data)}
    for section in _CARRIED_SECTIONS:
        if section in based_on.raw and based_on.raw[section]:
            document[section] = dict(based_on.raw[section])
    if visualization:
        document.setdefault("visualization", {})
        document["visualization"].update(visualization)  # type: ignore[union-attr]

    header = (
        f"# Written by `{written_by}` from {based_on.path}.\n"
        "#\n"
        "#   manifold-genetics run config.yaml --dry-run   # print the settings, do nothing\n"
        "#   manifold-genetics run config.yaml             # do the work\n"
        "#\n"
        "# Paths are relative to this file.\n\n"
    )
    path = out_dir / "config.yaml"
    path.write_text(header + yaml.safe_dump(document, sort_keys=False))
    logger.info("Wrote %s", path)
    return path


__all__ = [
    "CohortConfig",
    "Side",
    "filter_labels_to_fam",
    "read_cohort",
    "read_fam_ids",
    "write_cohort_config",
]
