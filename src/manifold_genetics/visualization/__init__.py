"""Visualization module for plotting embeddings and admixture results."""

from .plotting import (
    plot_admixture_bar_grid,
    plot_admixture_embedding_grid,
    plot_embedding,
    plot_knn_composition,
    plot_pca_pairs,
    plot_projection,
    visualize,
)

__all__ = [
    "plot_embedding",
    "plot_pca_pairs",
    "plot_projection",
    "plot_knn_composition",
    "visualize",
    "plot_admixture_bar_grid",
    "plot_admixture_embedding_grid",
]
