"""
manifold-genetics: Genetic analysis with PCA, Admixture, and manifold learning.

A lightweight, batteries-included Python package for genetic analysis with
dimensionality reduction and visualization.
"""

__version__ = "0.2.1"

from .admixture.neural import NeuralAdmixture
from .embeddings.diffusion_map import DiffusionMap
from .embeddings.phate import PHATE
from .embeddings.tsne import TSNE
from .embeddings.umap import UMAP

# Import main user-facing classes and functions
from .pca.flashpca import PCA
from .pipeline.configfile import load_config
from .pipeline.orchestrator import Pipeline
from .pipeline.runner import run_pipeline
from .visualization.plotting import plot_embedding, visualize

# The public API. Anything not named here is an implementation detail and may
# change in a patch release -- including the backends, the PLINK reader and
# the standardisation helpers, which exist to serve this surface, not users.
__all__ = [
    "run_pipeline",
    "load_config",
    "PCA",
    "NeuralAdmixture",
    "PHATE",
    "UMAP",
    "TSNE",
    "DiffusionMap",
    "visualize",
    "plot_embedding",
    "Pipeline",
]
