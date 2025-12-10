"""Embedding modules for manifold learning."""

from .base import EmbeddingBase
from .phate import PHATE
from .umap import UMAP
from .tsne import TSNE
from .diffusion_map import DiffusionMap

__all__ = ["EmbeddingBase", "PHATE", "UMAP", "TSNE", "DiffusionMap"]
