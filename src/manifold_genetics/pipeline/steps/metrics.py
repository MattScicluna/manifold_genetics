"""In-process metrics steps.

Both steps take plain paths rather than upstream step-result objects, so they
stay decoupled from whichever layer produced the embedding CSV and the Q files
(spec §"L2 — step layer"). ``cmd_metrics_*`` and the orchestrator both call
these, so validation lives here rather than in the CLI handlers — otherwise the
pipeline would lose the validation it used to get by shelling out.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from ...metrics import compute_admixture_preservation, compute_geographic_preservation
from ...utils.validation import (
    validate_admixture_csv,
    validate_embedding_csv,
    validate_geographic_csv,
    validate_sample_id_overlap,
)
from .paths import metrics_output_paths

logger = logging.getLogger(__name__)

__all__ = [
    "MetricsStepResult",
    "metrics_output_paths",
    "run_admixture_metrics_step",
    "run_geographic_metrics_step",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class MetricsStepResult:
    """Typed outputs of a metrics step. ``values`` is the parsed JSON artifact."""

    path: Path
    values: dict = field(default_factory=dict)
    skipped: bool = False


def _write_and_reload(values: dict, out: PathLike) -> MetricsStepResult:
    """Write ``values`` as JSON and return what the file actually contains.

    Reading back is deliberate: ``compute_admixture_preservation`` returns int K
    keys, while the JSON artifact — and therefore every existing consumer of
    ``run_pipeline()``'s metrics dict — has string keys.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(values, f, indent=2)
    with open(out) as f:
        return MetricsStepResult(path=out, values=json.load(f))


def run_geographic_metrics_step(
    embedding_csv: PathLike,
    geo_csv: PathLike,
    out: PathLike,
    *,
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
    num_samples: int = 50000,
    ignore_missing: bool = True,
) -> MetricsStepResult:
    """Correlate pairwise geographic distance with pairwise embedding distance."""
    validate_embedding_csv(embedding_csv)
    validate_geographic_csv(geo_csv, longitude_col, latitude_col)
    validate_sample_id_overlap(embedding_csv, geo_csv, "embedding", "geographic coordinates")

    logger.info("Computing geographic preservation...")
    values = compute_geographic_preservation(
        embedding=embedding_csv,
        geographic_coords=geo_csv,
        longitude_col=longitude_col,
        latitude_col=latitude_col,
        num_samples=num_samples,
        ignore_missing=ignore_missing,
    )
    return _write_and_reload(values, out)


def run_admixture_metrics_step(
    embedding_csv: PathLike,
    q_prefix: PathLike,
    k_range: Iterable[int],
    out: PathLike,
    *,
    k_value: Optional[int] = None,
    num_samples: int = 50000,
    subsample: Optional[int] = None,
) -> MetricsStepResult:
    """Correlate pairwise admixture distance with pairwise embedding distance.

    ``q_prefix`` names the Q files as ``<prefix>.{K}.csv`` (spec constraint B).
    """
    q_prefix = Path(q_prefix)
    k_values = list(k_range)
    if not k_values:
        raise ValueError(f"No admixture files found for K range: {k_range!r}")

    validate_embedding_csv(embedding_csv)
    validate_admixture_csv(str(q_prefix), k_values)
    first_q = f"{q_prefix}.{k_values[0]}.csv"
    validate_sample_id_overlap(embedding_csv, first_q, "embedding", "admixture")

    q_files = {k: Path(f"{q_prefix}.{k}.csv") for k in k_values}

    logger.info("Computing admixture preservation...")
    values = compute_admixture_preservation(
        embedding=embedding_csv,
        q_files=q_files,
        k_value=k_value,
        num_samples=num_samples,
        subsample=subsample,
    )
    return _write_and_reload(values, out)
