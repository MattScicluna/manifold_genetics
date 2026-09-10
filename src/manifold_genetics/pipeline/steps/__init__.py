"""In-process pipeline steps.

Each compute step lives in its own module and exposes a pure
``<step>_output_paths()`` helper (defined once in ``steps.paths``) plus a
``run_<step>_step()`` function. Both the ``manifold-genetics`` CLI subcommands
and the pipeline orchestrator call these directly, in-process.
"""

from .admixture import (
    AdmixtureStepResult,
    run_admixture,
    run_admixture_step,
)
from .embedding import (
    EmbeddingStepResult,
    build_embedding_model,
    run_embedding,
    run_embedding_step,
)
from .metrics import (
    MetricsStepResult,
    metrics_output_paths,
    run_admixture_metrics_step,
    run_geographic_metrics_step,
)
from .paths import admixture_output_paths, embedding_output_paths
from .pca import PCAStepResult, pca_output_paths, run_pca, run_pca_step

__all__ = [
    "AdmixtureStepResult",
    "EmbeddingStepResult",
    "MetricsStepResult",
    "PCAStepResult",
    "admixture_output_paths",
    "build_embedding_model",
    "embedding_output_paths",
    "metrics_output_paths",
    "pca_output_paths",
    "run_admixture",
    "run_admixture_metrics_step",
    "run_admixture_step",
    "run_embedding",
    "run_embedding_step",
    "run_geographic_metrics_step",
    "run_pca",
    "run_pca_step",
]
