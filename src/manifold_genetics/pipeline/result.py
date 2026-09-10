"""Typed outputs of a pipeline run.

``PipelineResult`` replaces the loose eighteen-key ``results`` dict that
``Pipeline.run()`` has threaded through the pipeline. Most of those keys were
already reachable through the step-layer results this object holds (e.g. the
old ``pca_file`` is ``result.pca.project_pca``) — collapsing that duplication
is the point of this type, so it deliberately does not re-expose them.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

from .steps.admixture import AdmixtureStepResult
from .steps.embedding import EmbeddingStepResult
from .steps.metrics import MetricsStepResult
from .steps.pca import PCAStepResult

__all__ = ["PipelineResult"]


@dataclass(frozen=True)
class PipelineResult:
    """Typed outputs of a pipeline run. Replaces the loose results dict."""

    # Compute step results; None when that step was skipped
    pca: Optional[PCAStepResult] = None
    admixture: Optional[AdmixtureStepResult] = None
    embedding: Optional[EmbeddingStepResult] = None
    geographic_metrics: Optional[MetricsStepResult] = None
    admixture_metrics: Optional[MetricsStepResult] = None

    # Figures, by family
    pca_figures: Tuple[Path, ...] = ()
    fit_embedding_figures: Tuple[Path, ...] = ()
    embedding_figures: Tuple[Path, ...] = ()
    projection_plot: Optional[Path] = None
    admixture_figures: Mapping[str, Path] = field(default_factory=dict)

    # Steps whose failure was tolerated (spec constraint D)
    failed_steps: Tuple[str, ...] = ()

    @property
    def figures(self) -> Tuple[Path, ...]:
        """Every figure produced, in stage order."""
        figures = (
            tuple(self.pca_figures)
            + tuple(self.fit_embedding_figures)
            + tuple(self.embedding_figures)
        )
        if self.projection_plot is not None:
            figures += (self.projection_plot,)
        figures += tuple(self.admixture_figures.values())
        return figures

    @property
    def metrics(self) -> dict:
        """The metrics mapping in the shape the CLI summary prints:
        {"geographic": {...}, "admixture": {"2": {...}}}; empty when neither ran.
        """
        values = {}
        if self.geographic_metrics is not None:
            values["geographic"] = self.geographic_metrics.values
        if self.admixture_metrics is not None:
            values["admixture"] = self.admixture_metrics.values
        return values
