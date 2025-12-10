"""
manifold-genetics: Genetic analysis with PCA, Admixture, and manifold learning.

A lightweight, batteries-included Python package for genetic analysis with
dimensionality reduction and visualization.
"""

__version__ = "0.1.0"

# Import main user-facing classes and functions
from .pca.flashpca import PCA
from .admixture.neural import NeuralAdmixture
from .embeddings.phate import PHATE
from .embeddings.umap import UMAP
from .embeddings.tsne import TSNE
from .embeddings.diffusion_map import DiffusionMap
from .visualization.plotting import visualize, plot_embedding
from .pipeline.orchestrator import Pipeline

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
