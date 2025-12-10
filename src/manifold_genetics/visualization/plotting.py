"""
Visualization functions for genetic embeddings and admixture.

Provides publication-ready plots with customizable colormaps.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.io import read_colormap, read_embedding_csv, read_labels_csv

matplotlib.use("Agg")  # Non-interactive backend
logger = logging.getLogger(__name__)


def plot_embedding(
    embedding: Union[pd.DataFrame, str, Path],
    labels: Union[pd.DataFrame, str, Path],
    colormap: Union[Dict, str, Path],
    output_path: Union[str, Path],
    title: Optional[str] = None,
    figsize: tuple = (6, 4),
    point_size: float = 1.0,
    alpha: float = 0.6,
    show_legend: bool = True,
) -> Path:
    """
    Plot 2D embedding colored by labels.

    Args:
        embedding: DataFrame or path to embedding CSV (sample_id, dim_1, dim_2)
        labels: DataFrame or path to labels CSV (sample_id, label_columns)
        colormap: Dict or path to colormap JSON {label_col: {value: color}}
        output_path: Path to save figure
        title: Optional plot title
        figsize: Figure size (width, height)
        point_size: Size of scatter plot points
        alpha: Transparency of points
        show_legend: Whether to show legend

    Returns:
        Path to saved figure
    """
    # Load data
    if isinstance(embedding, (str, Path)):
        embedding_df = read_embedding_csv(embedding)
    else:
        embedding_df = embedding

    if isinstance(labels, (str, Path)):
        labels_df = read_labels_csv(labels)
    else:
        labels_df = labels

    if isinstance(colormap, (str, Path)):
        colormap_dict = read_colormap(colormap)
    else:
        colormap_dict = colormap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge embedding with labels on sample_id to ensure correct alignment
    merged_df = embedding_df.merge(labels_df, on='sample_id', how='inner')

    # Create figure
    fig, axes = plt.subplots(
        1, len(colormap_dict), figsize=(figsize[0] * len(colormap_dict), figsize[1])
    )
    if len(colormap_dict) == 1:
        axes = [axes]

    # Plot for each label column in colormap
    for ax, (label_col, color_dict) in zip(axes, colormap_dict.items()):
        if label_col not in merged_df.columns:
            logger.warning(f"Column '{label_col}' not found in labels, skipping")
            continue

        # Get unique labels
        unique_labels = sorted(merged_df[label_col].unique())

        # Plot each label
        for label in unique_labels:
            mask = merged_df[label_col] == label
            label_data = merged_df[mask]

            color = color_dict.get(label, "#D3D3D3")  # Default to gray

            ax.scatter(
                label_data["dim_1"],
                label_data["dim_2"],
                s=20,
                alpha=0.6,
                color=color,
                edgecolors='none',
                label=label,
            )

        # Remove ticks, tick labels, axis labels, and titles
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        # Add legend if requested and reasonable number of labels
        if show_legend and len(unique_labels) <= 50:
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                frameon=False,
                fontsize=8,
                markerscale=2,
            )

    # No titles needed

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved embedding plot: {output_path}")
    return output_path


def plot_pca_pairs(
    pca_coords: Union[pd.DataFrame, str, Path],
    labels: Union[pd.DataFrame, str, Path],
    colormap: Union[Dict, str, Path],
    output_path: Union[str, Path],
    label_column: str,
    n_pcs: int = 20,
    title: Optional[str] = None,
) -> Path:
    """
    Plot PC pairs (PC1 vs PC2, PC3 vs PC4, etc.) colored by a label.

    Args:
        pca_coords: DataFrame or path to PCA coordinates CSV
        labels: DataFrame or path to labels CSV
        colormap: Dict or path to colormap JSON
        output_path: Path to save figure
        label_column: Name of column to color by
        n_pcs: Number of PCs to plot
        title: Optional plot title

    Returns:
        Path to saved figure
    """
    # Load data
    if isinstance(pca_coords, (str, Path)):
        pca_df = read_embedding_csv(pca_coords)
    else:
        pca_df = pca_coords

    if isinstance(labels, (str, Path)):
        labels_df = read_labels_csv(labels)
    else:
        labels_df = labels

    if isinstance(colormap, (str, Path)):
        colormap_dict = read_colormap(colormap)
    else:
        colormap_dict = colormap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge PCA with labels on sample_id to ensure correct alignment
    merged_df = pca_df.merge(labels_df, on='sample_id', how='inner')

    # Get PC columns
    pc_cols = [f"dim_{i+1}" for i in range(n_pcs)]
    available_pcs = [col for col in pc_cols if col in merged_df.columns]
    n_pairs = len(available_pcs) // 2

    if n_pairs == 0:
        raise ValueError(f"Need at least 2 PCs, found {len(available_pcs)}")

    # Create grid - use 5x5 to accommodate up to 25 pairs (50 PCs)
    n_cols = 5
    n_rows = 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get color mapping for the label column
    if label_column in colormap_dict:
        color_dict = colormap_dict[label_column]
    else:
        logger.warning(f"Label column '{label_column}' not in colormap, using default colors")
        unique_labels = merged_df[label_column].unique()
        color_dict = {label: f"C{i}" for i, label in enumerate(unique_labels)}

    # Plot each PC pair
    for pair_idx in range(n_pairs):
        ax = axes[pair_idx]

        pc_x_idx = pair_idx * 2
        pc_y_idx = pair_idx * 2 + 1
        pc_x_col = available_pcs[pc_x_idx]
        pc_y_col = available_pcs[pc_y_idx]

        # Get unique labels
        unique_labels = sorted(merged_df[label_column].unique())

        # Plot each label
        for label in unique_labels:
            mask = merged_df[label_column] == label
            label_data = merged_df[mask]

            color = color_dict.get(label, "#D3D3D3")

            ax.scatter(
                label_data[pc_x_col],
                label_data[pc_y_col],
                s=5,
                alpha=0.6,
                color=color,
                edgecolors='none',
                label=label,
            )

        # Remove ticks, tick labels, axis labels, and titles
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    # Hide extra subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis("off")

    # Add legend
    if len(unique_labels) <= 50:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_dict.get(label, "#D3D3D3"),
                markersize=8,
                label=label,
            )
            for label in unique_labels
        ]
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title=label_column,
            frameon=False,
            fontsize=8,
        )

    # No titles needed

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved PCA pairs plot: {output_path}")
    return output_path


def visualize(
    embedding: Union[pd.DataFrame, str, Path],
    labels: Union[pd.DataFrame, str, Path],
    colormap: Union[Dict, str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: str = "embedding",
) -> List[Path]:
    """
    Create all standard visualization plots.

    Generates a plot for each label column in the colormap.

    Args:
        embedding: DataFrame or path to embedding CSV
        labels: DataFrame or path to labels CSV
        colormap: Dict or path to colormap JSON
        output_dir: Directory to save plots (default: current directory)
        output_prefix: Prefix for output filenames

    Returns:
        List of paths to saved figures
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load colormap to get label columns
    if isinstance(colormap, (str, Path)):
        colormap_dict = read_colormap(colormap)
    else:
        colormap_dict = colormap

    # Generate plots for each label column
    output_paths = []
    for label_col in colormap_dict.keys():
        output_path = output_dir / f"{output_prefix}_by_{label_col}.png"

        plot_embedding(
            embedding=embedding,
            labels=labels,
            colormap={label_col: colormap_dict[label_col]},
            output_path=output_path,
            title=f"Embedding colored by {label_col}",
        )

        output_paths.append(output_path)

    logger.info(f"Generated {len(output_paths)} visualization plots")
    return output_paths
