"""Embedding modules for manifold learning."""

from .autoencoder import Autoencoder
from .base import EmbeddingBase
from .diffusion_map import DiffusionMap
from .dreams import DREAMS
from .flodr import FloDR
from .phate import PHATE
from .popvae import PopVAE
from .tsne import TSNE
from .umap import UMAP

__all__ = [
    "EmbeddingBase",
    "PHATE",
    "UMAP",
    "TSNE",
    "DiffusionMap",
    "DREAMS",
    "FloDR",
    "PopVAE",
    "Autoencoder",
]
