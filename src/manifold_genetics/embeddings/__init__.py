"""Embedding modules for manifold learning."""

from .base import EmbeddingBase
from .diffusion_map import DiffusionMap
from .phate import PHATE
from .tsne import TSNE
from .umap import UMAP

__all__ = ["EmbeddingBase", "PHATE", "UMAP", "TSNE", "DiffusionMap"]
