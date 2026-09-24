"""Visualization module for plotting embeddings and admixture results."""

from .plotting import (
    DEFAULT_MAX_POINTS_3D,
    plot_admixture_bar_grid,
    plot_admixture_embedding_3d,
    plot_admixture_embedding_grid,
    plot_embedding,
    plot_embedding_3d,
    plot_knn_composition,
    plot_pca_pairs,
    plot_projection,
    visualize,
    visualize_3d,
)

__all__ = [
    "DEFAULT_MAX_POINTS_3D",
    "plot_embedding",
    "plot_embedding_3d",
    "plot_pca_pairs",
    "plot_projection",
    "plot_knn_composition",
    "visualize",
    "visualize_3d",
    "plot_admixture_bar_grid",
    "plot_admixture_embedding_grid",
    "plot_admixture_embedding_3d",
]
