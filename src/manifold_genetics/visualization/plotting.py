"""
Visualization functions for genetic embeddings and admixture.

Provides publication-ready plots with customizable colormaps.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import linear_sum_assignment

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
    point_size: float = 4.0,
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

    # Reset index if sample_id is the index (from read_labels_csv)
    if labels_df.index.name == 'sample_id':
        labels_df = labels_df.reset_index()

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

        # FIRST: Plot samples with missing data in gray (background layer)
        missing_mask = merged_df[label_col].isna()
        if missing_mask.sum() > 0:
            ax.scatter(
                merged_df.loc[missing_mask, "dim_1"],
                merged_df.loc[missing_mask, "dim_2"],
                s=point_size,
                alpha=alpha * 0.5,  # More transparent for background
                color='lightgray',
                edgecolors='none',
                label='Unknown',
                rasterized=True,
                zorder=1,  # Low z-order for background
            )

        # SECOND: Plot each color group separately (foreground layer)
        # Use the ordering from the color_dict (Python 3.7+ preserves insertion order)
        # Plot in REVERSE order so that the first items in colormap appear on top
        color_groups = [k for k in color_dict.keys() if k in merged_df[label_col].values]
        for label in reversed(color_groups):
            mask = merged_df[label_col] == label
            label_data = merged_df[mask]
            color = color_dict.get(label, "#D3D3D3")  # Default to gray

            ax.scatter(
                label_data["dim_1"],
                label_data["dim_2"],
                s=point_size,
                alpha=alpha,
                color=color,
                edgecolors='none',
                label=label,
                rasterized=True,
                zorder=2,  # Higher z-order for foreground
            )

        # THIRD: Create legend using Patches instead of scatter handles
        # This ensures legend follows the color_dict order
        legend_elements = [
            Patch(facecolor=color_dict[g], label=g)
            for g in color_dict.keys()
            if g in merged_df[label_col].values
        ]
        if merged_df[label_col].isna().any():
            legend_elements.append(Patch(facecolor='lightgray', label='Unknown'))

        # Remove ticks, tick labels, axis labels, and titles
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        # Add legend if requested and reasonable number of labels
        if show_legend and len(legend_elements) <= 50:
            ax.legend(
                handles=legend_elements,
                fontsize=8,
                framealpha=0.9,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
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
    point_size: float = 4.0,
    alpha: float = 0.6,
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

    # Reset index if sample_id is the index (from read_labels_csv)
    if labels_df.index.name == 'sample_id':
        labels_df = labels_df.reset_index()

    if isinstance(colormap, (str, Path)):
        colormap_dict = read_colormap(colormap)
    else:
        colormap_dict = colormap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge PCA with labels on sample_id to ensure correct alignment
    merged_df = pca_df.merge(labels_df, on='sample_id', how='inner')

    # Validate label column exists in the data
    if label_column not in merged_df.columns:
        available_cols = [col for col in merged_df.columns if col not in ['sample_id'] + [f"dim_{i+1}" for i in range(50)]]
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: Label column '{label_column}' not found in labels data\n"
            f"{'='*70}\n"
            f"The colormap contains a key '{label_column}', but this column does not\n"
            f"exist in the labels CSV file.\n\n"
            f"Available label columns in the data:\n"
            f"  {', '.join(available_cols)}\n\n"
            f"Columns in colormap:\n"
            f"  {', '.join(colormap_dict.keys())}\n\n"
            f"Possible solutions:\n"
            f"  1. Update the labels CSV to include a '{label_column}' column\n"
            f"  2. Remove '{label_column}' from the colormap JSON\n"
            f"  3. Use --admixture-group-column with a column that exists in the labels\n"
            f"{'='*70}\n"
        )
        raise ValueError(error_msg)

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

    # Store all handles and labels for unified legend
    all_handles = []
    all_labels = []

    # Plot each PC pair
    for pair_idx in range(n_pairs):
        ax = axes[pair_idx]

        pc_x_idx = pair_idx * 2
        pc_y_idx = pair_idx * 2 + 1
        pc_x_col = available_pcs[pc_x_idx]
        pc_y_col = available_pcs[pc_y_idx]

        # FIRST: Plot samples with missing data in gray (background layer)
        missing_mask = merged_df[label_column].isna()
        if missing_mask.sum() > 0:
            ax.scatter(
                merged_df.loc[missing_mask, pc_x_col],
                merged_df.loc[missing_mask, pc_y_col],
                s=point_size,
                alpha=alpha * 0.5,  # More transparent for background
                color='lightgray',
                edgecolors='none',
                label='Unknown' if pair_idx == 0 else '',
                rasterized=True,
                zorder=1,  # Low z-order for background
            )

        # SECOND: Plot each color group separately (foreground layer)
        # Plot in REVERSE order so that the first items in colormap appear on top
        color_groups = [k for k in color_dict.keys() if k in merged_df[label_column].values]
        for label in reversed(color_groups):
            mask = merged_df[label_column] == label
            label_data = merged_df[mask]
            color = color_dict.get(label, "#D3D3D3")

            ax.scatter(
                label_data[pc_x_col],
                label_data[pc_y_col],
                s=point_size,
                alpha=alpha,
                color=color,
                edgecolors='none',
                label=label if pair_idx == 0 else '',  # Only label on first plot
                rasterized=True,
                zorder=2,  # Higher z-order for foreground
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

    # Create legend using Patches to match the color_dict order
    legend_elements = [
        Patch(facecolor=color_dict[g], label=g)
        for g in color_dict.keys()
        if g in merged_df[label_column].values
    ]
    if merged_df[label_column].isna().any():
        legend_elements.append(Patch(facecolor='lightgray', label='Unknown'))

    # Add legend with patches
    if len(legend_elements) <= 50:
        fig.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title=label_column,
            fontsize=8,
            framealpha=0.9,
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
    dataset_prefix: str = "",
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
        dataset_prefix: Prefix for dataset type (e.g., "fit_" or "transform_")

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
        output_path = output_dir / f"{dataset_prefix}{output_prefix}_by_{label_col}.png"

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


# --------------------------------------------------------------------------- #
# Admixture visualizations
# --------------------------------------------------------------------------- #

def _load_admixture_csv(
    q_prefix: Union[str, Path],
    k_values: Sequence[int],
) -> Dict[int, pd.DataFrame]:
    """
    Load admixture CSV files of the form <prefix>.<K>.csv into DataFrames.

    Returns a dict mapping K -> DataFrame.
    """
    q_prefix = Path(q_prefix)
    admixture = {}
    for k in k_values:
        path = Path(f"{q_prefix}.{k}.csv")
        if not path.exists():
            logger.warning(f"Admixture file not found for K={k}: {path}")
            continue
        df = pd.read_csv(path)
        if "sample_id" not in df.columns:
            logger.warning(f"K={k}: sample_id column missing in {path}, skipping")
            continue
        admixture[k] = df
    return admixture


def _get_component_columns(df: pd.DataFrame, k: Optional[int] = None) -> List[str]:
    """Return ordered component columns (component_1 ... component_k)."""
    comp_cols = [c for c in df.columns if str(c).startswith("component_")]
    if k:
        comp_cols = [f"component_{i}" for i in range(1, k + 1) if f"component_{i}" in comp_cols]
    comp_cols = sorted(comp_cols, key=lambda c: int(c.split("_")[1]))
    return comp_cols


def plot_admixture_bar_grid(
    q_prefix: Union[str, Path],
    labels: Union[pd.DataFrame, str, Path],
    group_column: str,
    k_values: Sequence[int],
    output_path: Union[str, Path],
    colors: Optional[List[str]] = None,
    subsample_per_group: Optional[int] = None,
    sort_groups: bool = True,
    group_order: Optional[Sequence[str]] = None,
    colormap: Optional[Union[Dict, str, Path]] = None,
    within_group_order: Optional[str] = 'chron',
    component_colors_output: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Plot stacked admixture barplots for multiple K values (one row per K).

    Args:
        q_prefix: Prefix for admixture CSVs (<prefix>.<K>.csv)
        labels: Labels table (path or DataFrame) with sample_id and grouping column
        group_column: Column in labels used for grouping/sorting bars
        k_values: Iterable of K values to plot
        output_path: Where to save the figure
        colors: Optional list of colors to reuse across Ks
        subsample_per_group: If set, subsample each group to this many rows
        sort_groups: If True, sort groups alphabetically; if False, preserve input order
        group_order: Optional explicit ordering of groups (takes precedence over sort_groups)
        colormap: Optional colormap to derive group ordering from (if group_column matches a key)
        within_group_order: Method for ordering samples within each group:
            - 'chron': Sort by dominant component (highest mean) for clear gradients
            - 'tree': Hierarchical clustering using Ward's method
            - None: Keep original order within groups

    Notes:
        Component colors are always aligned across K values using sequential
        Hungarian matching on component similarity, so stable ancestry
        components keep the same color between K levels.
    """
    if isinstance(labels, (str, Path)):
        labels_df = read_labels_csv(labels)
    else:
        labels_df = labels
    if labels_df.index.name == "sample_id":
        labels_df = labels_df.reset_index()

    admixture = _load_admixture_csv(q_prefix, k_values)
    if not admixture:
        raise ValueError("No admixture CSVs found for requested K values.")

    k_values = [k for k in k_values if k in admixture]
    if not k_values:
        raise ValueError("No valid K values remained after loading admixture CSVs.")

    # Derive ordering once: explicit > colormap (matching group_column) > optional sort.
    derived_order = list(group_order) if group_order is not None else None
    if derived_order is None and colormap is not None:
        cmap_dict = read_colormap(colormap) if isinstance(colormap, (str, Path)) else colormap
        for key, entry in cmap_dict.items():
            if key.lower() == group_column.lower():
                derived_order = list(entry.keys())
                break

    prepared_by_k: Dict[int, pd.DataFrame] = {}
    comp_cols_by_k: Dict[int, List[str]] = {}
    component_values_by_k: Dict[int, pd.DataFrame] = {}

    # Preprocess each K once (merge, grouping order, within-group order).
    for k in k_values:
        df = admixture[k].merge(labels_df, on="sample_id", how="inner")
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in merged data for K={k}.")

        # Optional subsample per group (kept deterministic).
        if subsample_per_group:
            parts = []
            for _, sub in df.groupby(group_column, sort=False, dropna=False):
                if len(sub) > subsample_per_group:
                    sub = sub.sample(n=subsample_per_group, random_state=42)
                parts.append(sub)
            df = pd.concat(parts, axis=0, ignore_index=True)

        if derived_order:
            present_groups = set(df[group_column])
            order = [g for g in derived_order if g in present_groups]
            missing_groups = [g for g in df[group_column].dropna().unique().tolist() if g not in order]
            order.extend(missing_groups)
            cat = pd.Categorical(df[group_column], categories=order, ordered=True)
            df[group_column] = cat
            df = df.sort_values(group_column)
        elif sort_groups:
            df = df.sort_values(group_column)

        comp_cols = _get_component_columns(df, k)
        if not comp_cols:
            raise ValueError(f"No component columns found for K={k}.")

        # Apply within-group ordering for smooth gradient effect.
        if within_group_order == "chron":
            sorted_groups = []
            for _, group_df in df.groupby(group_column, sort=False, observed=False, dropna=False):
                component_means = group_df[comp_cols].mean()
                dominant_comp = component_means.idxmax()
                sorted_group = group_df.sort_values(dominant_comp, ascending=False)
                sorted_groups.append(sorted_group)
            df = pd.concat(sorted_groups, axis=0, ignore_index=True)
        elif within_group_order == "tree":
            sorted_groups = []
            for _, group_df in df.groupby(group_column, sort=False, observed=False, dropna=False):
                if len(group_df) > 1:
                    data = group_df[comp_cols].to_numpy()
                    linkage_matrix = linkage(data, method="ward")
                    order = leaves_list(linkage_matrix)
                    sorted_group = group_df.iloc[order]
                else:
                    sorted_group = group_df
                sorted_groups.append(sorted_group)
            df = pd.concat(sorted_groups, axis=0, ignore_index=True)

        prepared_by_k[k] = df
        comp_cols_by_k[k] = comp_cols
        component_values_by_k[k] = df[["sample_id", *comp_cols]].drop_duplicates("sample_id").set_index("sample_id")

    # Build an alignment sample set that is shared across all K values.
    common_sample_ids = set(component_values_by_k[k_values[0]].index)
    for k in k_values[1:]:
        common_sample_ids &= set(component_values_by_k[k].index)
    if not common_sample_ids:
        raise ValueError("No shared sample_id values found across K values for component alignment.")

    alignment_sample_ids = list(common_sample_ids)
    alignment_max_samples = 12000
    if len(alignment_sample_ids) > alignment_max_samples:
        # Stratified sample by group to keep large cohorts tractable while preserving structure.
        ref_df = prepared_by_k[k_values[0]][["sample_id", group_column]].drop_duplicates("sample_id")
        ref_df = ref_df[ref_df["sample_id"].isin(common_sample_ids)]

        sampled_parts: List[pd.Series] = []
        total = len(ref_df)
        for _, group_df in ref_df.groupby(group_column, dropna=False):
            target_n = max(1, int(round((len(group_df) / total) * alignment_max_samples)))
            target_n = min(target_n, len(group_df))
            sampled_parts.append(group_df.sample(n=target_n, random_state=42)["sample_id"])

        sampled_ids = pd.concat(sampled_parts, axis=0).drop_duplicates().tolist()
        if len(sampled_ids) > alignment_max_samples:
            sampled_ids = (
                pd.Series(sampled_ids).sample(n=alignment_max_samples, random_state=42).tolist()
            )
        elif len(sampled_ids) < alignment_max_samples:
            sampled_set = set(sampled_ids)
            remaining_ids = [sid for sid in alignment_sample_ids if sid not in sampled_set]
            need = min(alignment_max_samples - len(sampled_ids), len(remaining_ids))
            if need > 0:
                sampled_ids.extend(pd.Series(remaining_ids).sample(n=need, random_state=42).tolist())

        alignment_sample_ids = sampled_ids
        logger.info(
            "Admixture alignment subsampling: using %d / %d shared samples.",
            len(alignment_sample_ids),
            len(common_sample_ids),
        )
    else:
        logger.info(
            "Admixture alignment: using all %d shared samples.",
            len(alignment_sample_ids),
        )

    # Build aligned matrices in the same sample order for similarity matching.
    aligned_matrix_by_k: Dict[int, np.ndarray] = {}
    for k in k_values:
        aligned_matrix_by_k[k] = component_values_by_k[k].loc[alignment_sample_ids, comp_cols_by_k[k]].to_numpy()

    # Create a reusable palette. Start from user colors if provided, then extend deterministically.
    if colors:
        palette = list(colors)
    else:
        palette = [to_hex(c) for c in plt.get_cmap("tab20").colors]

    max_needed_colors = max(len(cols) for cols in comp_cols_by_k.values())
    if len(palette) < max_needed_colors:
        extra = plt.get_cmap("hsv")(np.linspace(0, 1, max_needed_colors - len(palette), endpoint=False))
        palette.extend(to_hex(c) for c in extra)

    # Track component lineage across Ks via Hungarian matching.
    lineages_by_k: Dict[int, Dict[str, int]] = {}
    first_k = k_values[0]
    lineages_by_k[first_k] = {col: idx for idx, col in enumerate(comp_cols_by_k[first_k])}
    next_lineage = len(lineages_by_k[first_k])

    for prev_k, curr_k in zip(k_values[:-1], k_values[1:]):
        prev_cols = comp_cols_by_k[prev_k]
        curr_cols = comp_cols_by_k[curr_k]
        prev_mat = aligned_matrix_by_k[prev_k]
        curr_mat = aligned_matrix_by_k[curr_k]

        # Pearson correlation between components (clipped to non-negative for matching).
        prev_center = prev_mat - prev_mat.mean(axis=0, keepdims=True)
        curr_center = curr_mat - curr_mat.mean(axis=0, keepdims=True)
        prev_norm = np.linalg.norm(prev_center, axis=0)
        curr_norm = np.linalg.norm(curr_center, axis=0)
        denom = np.outer(prev_norm, curr_norm)
        similarity = np.divide(
            prev_center.T @ curr_center,
            denom,
            out=np.zeros((len(prev_cols), len(curr_cols))),
            where=denom > 0,
        )
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
        similarity = np.clip(similarity, a_min=0.0, a_max=None)

        row_ind, col_ind = linear_sum_assignment(-similarity)
        curr_lineages: Dict[str, int] = {}
        for prev_idx, curr_idx in zip(row_ind, col_ind):
            curr_lineages[curr_cols[curr_idx]] = lineages_by_k[prev_k][prev_cols[prev_idx]]

        # Any unmatched current components are genuinely new lineages.
        for col in curr_cols:
            if col not in curr_lineages:
                curr_lineages[col] = next_lineage
                next_lineage += 1

        lineages_by_k[curr_k] = curr_lineages

    if len(palette) < next_lineage:
        extra = plt.get_cmap("hsv")(np.linspace(0, 1, next_lineage - len(palette), endpoint=False))
        palette.extend(to_hex(c) for c in extra)

    lineage_colors = {idx: palette[idx] for idx in range(next_lineage)}

    component_colors_dict = {
        k: {col: lineage_colors[lineages_by_k[k][col]] for col in comp_cols_by_k[k]}
        for k in k_values
    }
    if component_colors_output is not None:
        out = Path(component_colors_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({str(k): v for k, v in component_colors_dict.items()}, f, indent=2)
        logger.info(f"Saved component colors: {out}")

    n_rows = len(k_values)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        df = prepared_by_k[k]
        comp_cols = comp_cols_by_k[k]
        component_colors = [lineage_colors[lineages_by_k[k][col]] for col in comp_cols]
        df[comp_cols].plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=1.0,
            edgecolor="none",
            color=component_colors,
        )

        # Remove ticks/legend; add group separators
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.get_legend().remove()
        ax.set_ylabel(f"K={k}")
        ax.set_ylim([0, 1])  # Fix y-axis to [0, 1] for proportions

        # Draw separators between groups
        boundaries = []
        offset = 0
        for _, sub in df.groupby(group_column, sort=False, observed=False):
            offset += len(sub)
            boundaries.append(offset)
        for pos in boundaries[:-1]:
            ax.axvline(x=pos - 0.5, linestyle="--", color="black", alpha=0.5)

        # Add group labels only on the last subplot
        if ax is axes[-1]:
            group_sizes = df[group_column].value_counts(sort=False)
            labels_order = list(group_sizes.index)
            cum_sizes = np.cumsum(group_sizes.values)
            starts = [0] + list(cum_sizes[:-1])
            mids = [(s + e) / 2 for s, e in zip(starts, cum_sizes)]
            ax.set_xticks(mids)
            ax.set_xticklabels(labels_order, rotation=45, ha="right", fontsize=10)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved admixture bar plot: {output_path}")
    return output_path


def plot_admixture_embedding_grid(
    embedding: Union[pd.DataFrame, str, Path],
    q_prefix: Union[str, Path],
    k_values: Sequence[int],
    output_path: Union[str, Path],
    pc_x: int = 1,
    pc_y: int = 2,
    subsample: Optional[int] = None,
    component_colormap: Optional[Union[Dict, str, Path]] = None,
) -> Path:
    """
    Plot embedding colored by admixture components in a grid.

    One row per K, columns = max(Ks). Each cell is a component heatmap (seismic 0-1).
    When component_colormap is provided (path to JSON exported by plot_admixture_bar_grid),
    each subplot uses a white-to-component-color gradient matching the bar chart.
    """
    if isinstance(component_colormap, (str, Path)):
        with open(component_colormap) as f:
            component_colormap = json.load(f)

    if isinstance(embedding, (str, Path)):
        emb_df = read_embedding_csv(embedding)
    else:
        emb_df = embedding

    admixture = _load_admixture_csv(q_prefix, k_values)
    if not admixture:
        raise ValueError("No admixture CSVs found for requested K values.")

    k_values = [k for k in k_values if k in admixture]
    max_k = max(k_values)
    n_rows = len(k_values)
    n_cols = max_k

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    pc_x_col = f"dim_{pc_x}"
    pc_y_col = f"dim_{pc_y}"
    if pc_x_col not in emb_df.columns or pc_y_col not in emb_df.columns:
        raise ValueError(f"PC columns {pc_x_col} and/or {pc_y_col} not found in embedding.")

    for row_idx, k in enumerate(k_values):
        q_df = admixture[k]
        merged = emb_df.merge(q_df, on="sample_id", how="inner")
        if subsample and len(merged) > subsample:
            merged = merged.sample(n=subsample, random_state=42)

        comp_cols = _get_component_columns(merged, k)
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            comp_idx = col_idx + 1
            if comp_idx > k:
                ax.axis("off")
                continue
            comp_col = f"component_{comp_idx}"
            if component_colormap is not None:
                color = component_colormap[str(k)][comp_col]
                cmap = LinearSegmentedColormap.from_list(
                    f"comp_{k}_{comp_idx}", ["white", color], N=256
                )
            else:
                cmap = "seismic"
            scatter = ax.scatter(
                merged[pc_x_col],
                merged[pc_y_col],
                c=merged[comp_col],
                cmap=cmap,
                s=3,
                alpha=0.6,
                vmin=0,
                vmax=1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"K={k} • Comp {comp_idx}", fontsize=10)

            # Add a colorbar on the last column for each row
            if col_idx == n_cols - 1:
                plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Admixture proportion")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved admixture embedding grid: {output_path}")
    return output_path


def plot_projection(
    fit_embedding: Union[pd.DataFrame, str, Path],
    project_embedding: Union[pd.DataFrame, str, Path],
    fit_labels: Union[pd.DataFrame, str, Path],
    project_labels: Union[pd.DataFrame, str, Path],
    fit_colormap: Union[Dict, str, Path],
    project_colormap: Union[Dict, str, Path],
    output_path: Union[str, Path],
    fit_label_column: str,
    project_label_column: str,
    title: Optional[str] = None,
    figsize: tuple = (6, 4),
    point_size: float = 4.0,
    alpha: float = 0.6,
    linewidth: float = 0.8,
    fit_marker: str = '^',
    project_marker: str = 'o',
    show_legend: bool = True,
) -> Path:
    """
    Plot fit and projection embeddings together with different marker shapes.

    Allows different label columns for fit and project datasets - they don't
    need to match. Each dataset is colored by its own colormap.

    Args:
        fit_embedding: DataFrame or path to fit embedding CSV (sample_id, dim_1, dim_2)
        project_embedding: DataFrame or path to project embedding CSV
        fit_labels: DataFrame or path to fit labels CSV
        project_labels: DataFrame or path to project labels CSV
        fit_colormap: Dict or path to fit colormap JSON
        project_colormap: Dict or path to project colormap JSON
        output_path: Path to save figure
        fit_label_column: Label column from fit colormap to use for coloring fit data
        project_label_column: Label column from project colormap to use for coloring project data
        title: Optional plot title
        figsize: Figure size (width, height)
        point_size: Size of scatter plot points
        alpha: Transparency of points
        linewidth: Edge width for hollow markers
        fit_marker: Marker shape for fit data (default: triangle '^')
        project_marker: Marker shape for project data (default: circle 'o')
        show_legend: Whether to show legend

    Returns:
        Path to saved figure
    """
    # Load fit embedding
    if isinstance(fit_embedding, (str, Path)):
        fit_emb_df = read_embedding_csv(fit_embedding)
    else:
        fit_emb_df = fit_embedding

    # Load project embedding
    if isinstance(project_embedding, (str, Path)):
        project_emb_df = read_embedding_csv(project_embedding)
    else:
        project_emb_df = project_embedding

    # Load fit labels
    if isinstance(fit_labels, (str, Path)):
        fit_labels_df = read_labels_csv(fit_labels)
    else:
        fit_labels_df = fit_labels

    # Reset index if sample_id is the index
    if fit_labels_df.index.name == 'sample_id':
        fit_labels_df = fit_labels_df.reset_index()

    # Load project labels
    if isinstance(project_labels, (str, Path)):
        project_labels_df = read_labels_csv(project_labels)
    else:
        project_labels_df = project_labels

    # Reset index if sample_id is the index
    if project_labels_df.index.name == 'sample_id':
        project_labels_df = project_labels_df.reset_index()

    # Load colormaps
    if isinstance(fit_colormap, (str, Path)):
        fit_cmap_dict = read_colormap(fit_colormap)
    else:
        fit_cmap_dict = fit_colormap

    if isinstance(project_colormap, (str, Path)):
        project_cmap_dict = read_colormap(project_colormap)
    else:
        project_cmap_dict = project_colormap

    # Validate columns exist in colormaps
    if fit_label_column not in fit_cmap_dict:
        raise ValueError(
            f"Fit column '{fit_label_column}' not in fit colormap. "
            f"Available: {list(fit_cmap_dict.keys())}"
        )
    if project_label_column not in project_cmap_dict:
        raise ValueError(
            f"Project column '{project_label_column}' not in project colormap. "
            f"Available: {list(project_cmap_dict.keys())}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge embeddings with labels
    fit_merged = fit_emb_df.merge(fit_labels_df, on='sample_id', how='inner')
    project_merged = project_emb_df.merge(project_labels_df, on='sample_id', how='inner')

    # Get color dictionaries for the specified columns
    fit_color_dict = fit_cmap_dict[fit_label_column]
    project_color_dict = project_cmap_dict[project_label_column]

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get all unique labels from both datasets
    fit_labels_set = set()
    project_labels_set = set()
    if fit_label_column in fit_merged.columns:
        fit_labels_set.update(fit_merged[fit_label_column].dropna().unique())
    if project_label_column in project_merged.columns:
        project_labels_set.update(project_merged[project_label_column].dropna().unique())

    # LAYER 1: Plot missing data in lightgray for both datasets
    # Fit missing data
    if fit_label_column in fit_merged.columns:
        fit_missing_mask = fit_merged[fit_label_column].isna()
        if fit_missing_mask.sum() > 0:
            ax.scatter(
                fit_merged.loc[fit_missing_mask, "dim_1"],
                fit_merged.loc[fit_missing_mask, "dim_2"],
                marker=fit_marker,
                s=point_size,
                facecolors='none',
                edgecolors='lightgray',
                linewidths=linewidth,
                alpha=alpha * 0.5,
                rasterized=True,
                zorder=1,
            )

    # Project missing data
    if project_label_column in project_merged.columns:
        project_missing_mask = project_merged[project_label_column].isna()
        if project_missing_mask.sum() > 0:
            ax.scatter(
                project_merged.loc[project_missing_mask, "dim_1"],
                project_merged.loc[project_missing_mask, "dim_2"],
                marker=project_marker,
                s=point_size,
                facecolors='none',
                edgecolors='lightgray',
                linewidths=linewidth,
                alpha=alpha * 0.5,
                rasterized=True,
                zorder=1,
            )

    # LAYER 2: Plot colored groups in REVERSE order
    # Plot fit data groups
    fit_color_groups = [g for g in fit_color_dict.keys() if g in fit_labels_set]
    for label in reversed(fit_color_groups):
        if fit_label_column in fit_merged.columns:
            fit_mask = fit_merged[fit_label_column] == label
            if fit_mask.sum() > 0:
                fit_color = fit_color_dict[label]
                ax.scatter(
                    fit_merged.loc[fit_mask, "dim_1"],
                    fit_merged.loc[fit_mask, "dim_2"],
                    marker=fit_marker,
                    s=point_size,
                    facecolors='none',
                    edgecolors=fit_color,
                    linewidths=linewidth,
                    alpha=alpha,
                    rasterized=True,
                    zorder=2,
                )

    # Plot project data groups
    project_color_groups = [g for g in project_color_dict.keys() if g in project_labels_set]
    for label in reversed(project_color_groups):
        if project_label_column in project_merged.columns:
            project_mask = project_merged[project_label_column] == label
            if project_mask.sum() > 0:
                project_color = project_color_dict[label]
                ax.scatter(
                    project_merged.loc[project_mask, "dim_1"],
                    project_merged.loc[project_mask, "dim_2"],
                    marker=project_marker,
                    s=point_size,
                    facecolors='none',
                    edgecolors=project_color,
                    linewidths=linewidth,
                    alpha=alpha,
                    rasterized=True,
                    zorder=2,
                )

    # Create legend with two sections: fit and project
    if show_legend:
        legend_elements = []

        # Add fit legend entries (if not too many)
        if len(fit_color_groups) <= 25:
            for label in fit_color_groups:
                color = fit_color_dict[label]
                legend_elements.append(Patch(facecolor=color, label=f"▲ {label}"))

        # Add project legend entries (if not too many)
        if len(project_color_groups) <= 25:
            for label in project_color_groups:
                color = project_color_dict[label]
                legend_elements.append(Patch(facecolor=color, label=f"● {label}"))

        # Add "Unknown" if there's missing data
        has_missing = (
            (fit_label_column in fit_merged.columns and fit_merged[fit_label_column].isna().any()) or
            (project_label_column in project_merged.columns and project_merged[project_label_column].isna().any())
        )
        if has_missing:
            legend_elements.append(Patch(facecolor='lightgray', label='Unknown'))

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                fontsize=7,
                framealpha=0.9,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                title=f"▲ fit: {fit_label_column}\n● transform: {project_label_column}"
            )

    # Remove ticks, tick labels, axis labels, and titles
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved projection plot: {output_path}")
    return output_path
