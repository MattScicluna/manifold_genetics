"""In-process pipeline steps.

Each compute step lives in its own module and exposes a pure
``<step>_output_paths()`` helper (defined once in ``steps.paths``) plus a
``run_<step>_step()`` function. Both the ``manifold-genetics`` CLI subcommands
and the pipeline orchestrator call these directly, in-process.
"""

from .paths import admixture_output_paths, embedding_output_paths
from .pca import PCAStepResult, pca_output_paths, run_pca, run_pca_step

__all__ = [
    "PCAStepResult",
    "admixture_output_paths",
    "embedding_output_paths",
    "pca_output_paths",
    "run_pca",
    "run_pca_step",
]
