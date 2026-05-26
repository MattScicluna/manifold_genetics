"""
manifold-genetics: Genetic analysis with PCA, Admixture, and manifold learning.

A lightweight, batteries-included Python package for genetic analysis with
dimensionality reduction and visualization.
"""

__version__ = "0.1.0"

from .admixture.neural import NeuralAdmixture
from .embeddings.diffusion_map import DiffusionMap
from .embeddings.phate import PHATE
from .embeddings.tsne import TSNE
from .embeddings.umap import UMAP

# Import main user-facing classes and functions
from .pca.flashpca import PCA
from .pipeline.orchestrator import Pipeline
from .visualization.plotting import plot_embedding, visualize

__all__ = [
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
