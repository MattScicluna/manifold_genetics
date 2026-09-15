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

from ..pca.plink import read_fam_ids  # re-exported: one .fam reader for the package
from ..pipeline.configfile import PRESET_OWNED_EMBEDDING_KEYS, load_config

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
    geographic_coords: Optional[Path]


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
        plink_key = f"{name}_plink"
        if plink_key not in resolved:
            raise ValueError(
                f"{path} has no data.{plink_key}; a cohort config names both "
                "fit_plink and project_plink"
            )
        return Side(plink=Path(resolved[plink_key]), labels=Path(labels), colormap=Path(colormap))

    geographic_coords = resolved.get("geographic_coords")

    return CohortConfig(
        path=path,
        raw=raw,
        preset=raw.get("preset"),
        fit=side("fit"),
        project=side("project"),
        shared_labels="labels" in resolved
        and "fit_labels" not in resolved
        and "project_labels" not in resolved,
        geographic_coords=Path(geographic_coords) if geographic_coords is not None else None,
    )


def _count_lines(path: PathLike) -> int:
    # Not pandas: this only needs a line count, and pandas' startup and dtype
    # inference cost is wasted on a check that runs on every preprocess call.
    count = 0
    with open(path, "rb") as fh:
        for _ in fh:
            count += 1
    return count


def bed_expected_size(prefix: PathLike) -> int:
    """Bytes a complete PLINK 1 SNP-major ``.bed`` must have for ``prefix``.

    A 3-byte magic header plus one packed byte-column per variant, each column
    ``ceil(n_samples / 4)`` bytes (four 2-bit genotypes per byte).
    """
    n_variants = _count_lines(f"{prefix}.bim")
    n_samples = _count_lines(f"{prefix}.fam")
    return 3 + n_variants * ((n_samples + 3) // 4)


def bed_is_complete(prefix: PathLike) -> bool:
    """Whether ``prefix.bed`` is the full file rather than a truncated write.

    Python counterpart of ``preprocessing/common.sh``'s ``bed_is_complete``: an
    OOM-killed preprocessing run once left a partial ``.bed`` on disk, and an
    existence-only check treated it as a finished output on the next run.
    """
    bed, bim, fam = Path(f"{prefix}.bed"), Path(f"{prefix}.bim"), Path(f"{prefix}.fam")
    if not (bed.exists() and bim.exists() and fam.exists()):
        return False
    return bed.stat().st_size == bed_expected_size(prefix)


# The one label-coverage rule. `acquire custom` and `acquire aou` accept a label
# file that describes at least this fraction of the .fam and `run` draws the
# rest grey; `preprocess` and `subsample` apply the same rule, and check it
# before any long computation, so a cohort that `acquire` accepted is never
# rejected hours later.
MIN_LABEL_COVERAGE = 0.5


def _read_labels(labels: PathLike) -> pd.DataFrame:
    frame = pd.read_csv(labels, dtype={"sample_id": str}, low_memory=False)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{labels} has no sample_id column")
    return frame


def _coverage(labels: PathLike, frame: pd.DataFrame, ids: List[str]) -> float:
    """The fraction of ``ids`` with a row in ``frame``; raises below the rule."""
    known = set(frame["sample_id"])
    missing = [i for i in dict.fromkeys(ids) if i not in known]
    n_ids = len(set(ids))
    coverage = (n_ids - len(missing)) / max(n_ids, 1)
    if coverage < MIN_LABEL_COVERAGE:
        example = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} of {n_ids} samples ({1 - coverage:.1%}) are not in {labels} "
            f"(e.g. {example}). A label file must cover at least {MIN_LABEL_COVERAGE:.0%} of "
            f"the genotypes it describes; this one covers {coverage:.1%}."
        )
    if missing:
        logger.warning(
            "%d of %d samples (%.1f%%) have no row in %s (e.g. %s); they will be drawn grey.",
            len(missing),
            n_ids,
            100 * (1 - coverage),
            labels,
            ", ".join(missing[:5]),
        )
    return coverage


def check_label_coverage(labels: PathLike, prefix: PathLike) -> float:
    """The fraction of ``prefix``.fam that ``labels`` describes.

    Raises:
        ValueError: ``labels`` has no ``sample_id`` column, or covers less than
            ``MIN_LABEL_COVERAGE`` of the ``.fam``. Between that and full
            coverage a warning names the uncovered samples.
    """
    return _coverage(labels, _read_labels(labels), read_fam_ids(prefix))


def filter_labels_to_ids(labels: PathLike, ids: List[str], out: PathLike) -> int:
    """Write the rows of ``labels`` for ``ids``, in that order.

    Ids without a row are skipped with a warning; ``run`` draws them grey, as it
    would had the labels been handed to it directly.

    Raises:
        ValueError: ``labels`` has no ``sample_id`` column, or fewer than
            ``MIN_LABEL_COVERAGE`` of ``ids`` have a row -- the rule
            ``acquire custom`` applies to a label file it is handed.
    """
    frame = _read_labels(labels)
    _coverage(labels, frame, ids)
    # Drop duplicate sample_ids *before* indexing by the requested order, so a
    # label file with repeated ids does not expand `.loc[ids]` into extra rows.
    frame = frame.drop_duplicates("sample_id").set_index("sample_id")
    covered = [i for i in ids if i in frame.index]
    kept = frame.loc[covered].reset_index()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False)
    return len(kept)


def filter_labels_to_fam(labels: PathLike, prefix: PathLike, out: PathLike) -> int:
    """Write the rows of ``labels`` for the samples in ``prefix``.fam, in .fam order.

    Raises:
        ValueError: fewer than ``MIN_LABEL_COVERAGE`` of the .fam has a label row.
    """
    return filter_labels_to_ids(labels, read_fam_ids(prefix), out)


def filter_geographic_to_fam(geographic: PathLike, prefix: PathLike, out: PathLike) -> int:
    """Write the rows of ``geographic`` for the samples in ``prefix``.fam, in .fam order.

    Unlike labels, geographic coordinates are legitimately partial -- there is
    no ``MIN_LABEL_COVERAGE``-style floor and a sample missing a row is simply
    omitted, silently, rather than warned about.
    """
    ids = read_fam_ids(prefix)
    frame = pd.read_csv(geographic, dtype={"sample_id": str}, low_memory=False)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{geographic} has no sample_id column")
    frame = frame.drop_duplicates("sample_id").set_index("sample_id")
    covered = [i for i in ids if i in frame.index]
    kept = frame.loc[covered].reset_index()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False)
    return len(kept)


_CARRIED_SECTIONS = ("pca", "admixture", "embedding", "visualization", "skip")

# Dropped from a carried `embedding` section when the output's preset differs
# from the input's: each is either the preset's own dataset-role choice
# (`input_mode`) or a landmarking/knn default the preset sets (see PRESETS in
# configfile.py). Carrying them past a preset change would silently override
# what the new preset would otherwise set.
_DROPPED_ON_PRESET_CHANGE = ("input_mode", *sorted(PRESET_OWNED_EMBEDDING_KEYS))


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

    When ``preset`` differs from ``based_on.preset``, a carried ``embedding``
    section has its dataset-role and landmarking keys dropped (see
    ``_DROPPED_ON_PRESET_CHANGE``), with one warning per key naming the value
    that was dropped and the preset that now sets it -- otherwise a hand-edited
    ``input_mode`` or ``n_landmark`` would silently override the new preset.
    """
    document: Dict[str, object] = {"preset": preset, "data": dict(data)}
    for section in _CARRIED_SECTIONS:
        if section in based_on.raw and based_on.raw[section]:
            carried = dict(based_on.raw[section])
            if section == "embedding" and preset != based_on.preset:
                for key in _DROPPED_ON_PRESET_CHANGE:
                    if key in carried:
                        old_value = carried.pop(key)
                        logger.warning(
                            "`embedding.%s: %s` dropped: the `%s` preset sets it.",
                            key,
                            old_value,
                            preset,
                        )
            if carried:
                document[section] = carried
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
    "MIN_LABEL_COVERAGE",
    "CohortConfig",
    "Side",
    "bed_expected_size",
    "bed_is_complete",
    "check_label_coverage",
    "filter_geographic_to_fam",
    "filter_labels_to_fam",
    "filter_labels_to_ids",
    "read_cohort",
    "read_fam_ids",
    "write_cohort_config",
]
