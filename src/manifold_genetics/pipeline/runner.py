"""
Canonical pipeline function API for manifold-genetics.

This module provides a single `run_pipeline()` function that serves as the
canonical entry point for running the full genetic analysis pipeline. Both
the CLI and example scripts use this function to ensure consistency.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from .orchestrator import Pipeline

logger = logging.getLogger(__name__)


def run_pipeline(
    fit_plink: Union[str, Path],
    project_plink: Union[str, Path],
    output_dir: Union[str, Path],
    labels: Optional[Union[str, Path]] = None,
    colormap: Optional[Union[str, Path]] = None,
    # Optional overrides for cross-cohort analysis
    fit_labels: Optional[Union[str, Path]] = None,
    project_labels: Optional[Union[str, Path]] = None,
    fit_colormap: Optional[Union[str, Path]] = None,
    project_colormap: Optional[Union[str, Path]] = None,
    # Geographic coordinates for metrics
    geographic_coords: Optional[Union[str, Path]] = None,
    # PCA parameters
    n_pcs: int = 50,
    # Admixture parameters
    k_min: int = 2,
    k_max: int = 10,
    admix_threads: Optional[int] = None,
    admix_gpus: Optional[int] = None,
    admix_batch_size: Optional[int] = None,
    admixture_backend: Optional[object] = None,
    # Embedding parameters
    embedding: str = "phate",
    embedding_params: Optional[Dict] = None,
    embedding_input: str = "both",
    # Visualization parameters
    admix_group_column: Optional[str] = None,
    admix_within_group_order: Optional[str] = "chron",
    projection_plot_fit_column: Optional[str] = None,
    projection_plot_project_column: Optional[str] = None,
    # Skip flags
    skip_pca: bool = False,
    skip_admixture: bool = False,
    skip_embedding: bool = False,
    skip_visualization: bool = False,
    skip_pca_visualization: bool = False,
    skip_admixture_visualization: bool = False,
    skip_metrics: bool = False,
) -> Dict:
    """
    Run the complete manifold-genetics pipeline.

    This is the canonical entry point for running the full genetic analysis pipeline.
    It coordinates PCA, Admixture, Embeddings, Visualization, and Metrics computation.

    Args:
        fit_plink: Path to fit subset PLINK files (prefix for .bed/.bim/.fam)
        project_plink: Path to project subset PLINK files (prefix for .bed/.bim/.fam)
        output_dir: Directory for all outputs
        labels: Path to labels CSV (used for both fit and project if not overridden)
        colormap: Path to colormap JSON (used for both fit and project if not overridden)
        fit_labels: Optional override labels CSV for fit dataset
        project_labels: Optional override labels CSV for project dataset
        fit_colormap: Optional override colormap JSON for fit dataset
        project_colormap: Optional override colormap JSON for project dataset
        geographic_coords: Optional path to geographic coordinates CSV for metrics
        n_pcs: Number of principal components (default: 50)
        k_min: Minimum K for admixture (default: 2)
        k_max: Maximum K for admixture (default: 10)
        admix_threads: Number of threads for neural admixture (None = auto-detect)
        admix_gpus: Number of GPUs for neural admixture (None = auto-detect)
        admix_batch_size: Batch size for neural admixture training
        admixture_backend: Optional AdmixtureBackend instance for testing
                          (if None, uses neural-admixture via CLI)
        embedding: Embedding method - 'phate', 'umap', 'tsne', or 'diffusion_map' (default: 'phate')
        embedding_params: Optional dictionary of embedding-specific parameters
        embedding_input: Which dataset to embed - 'fit', 'project', or 'both' (default: 'both')
        admix_group_column: Column for grouping in admixture barplots (None = use first colormap key)
        admix_within_group_order: Method for ordering samples within groups ('chron', 'tree', or None)
        projection_plot_fit_column: Column from fit colormap to use for projection plot
        projection_plot_project_column: Column from project colormap to use for projection plot
        skip_pca: Skip PCA step
        skip_admixture: Skip admixture step
        skip_embedding: Skip embedding step
        skip_visualization: Skip embedding visualization step
        skip_pca_visualization: Skip PCA visualization step
        skip_admixture_visualization: Skip admixture visualization step
        skip_metrics: Skip metrics computation

    Returns:
        Dictionary with paths to outputs and computed metrics. Keys include:
        - fit_pca_file: Path to fit PCA coordinates CSV
        - project_pca_file: Path to project PCA coordinates CSV
        - pca_coords: pandas DataFrame with PCA coordinates
        - admixture_dir: Path to admixture output directory
        - fit_q_files: Dict mapping K -> fit admixture CSV path
        - project_q_files: Dict mapping K -> project admixture CSV path
        - embedding_file: Path to embedding coordinates CSV
        - embedding_coords: pandas DataFrame with embedding coordinates
        - pca_figures: List of PCA plot paths
        - embedding_figures: List of embedding plot paths
        - admixture_figures: Dict with admixture plot paths
        - metrics: Dict with geographic and admixture preservation metrics

    Examples:
        >>> # Basic usage with shared labels/colormap
        >>> results = run_pipeline(
        ...     fit_plink="data/fit_subset",
        ...     project_plink="data/project_subset",
        ...     labels="labels.csv",
        ...     colormap="colormap.json",
        ...     output_dir="results/",
        ...     n_pcs=50,
        ...     k_min=2,
        ...     k_max=10,
        ...     embedding="phate",
        ...     embedding_params={"knn": 100, "t": 3}
        ... )

        >>> # Cross-cohort analysis with separate labels/colormaps
        >>> results = run_pipeline(
        ...     fit_plink="data/hgdp_fit",
        ...     project_plink="data/ukbb_project",
        ...     fit_labels="hgdp_labels.csv",
        ...     project_labels="ukbb_labels.csv",
        ...     fit_colormap="hgdp_colors.json",
        ...     project_colormap="ukbb_colors.json",
        ...     output_dir="cross_cohort_results/",
        ... )

        >>> # Skip steps selectively
        >>> results = run_pipeline(
        ...     fit_plink="data/fit",
        ...     project_plink="data/project",
        ...     labels="labels.csv",
        ...     colormap="colormap.json",
        ...     output_dir="results/",
        ...     skip_pca=True,  # Use existing PCA
        ...     skip_metrics=True  # Don't compute metrics
        ... )

    Note:
        You must provide either:

        * A shared set of `labels` and `colormap`, which will be used for both the
          fit and project cohorts, **or**
        * Separate values for all of `fit_labels`, `project_labels`, `fit_colormap`,
          and `project_colormap` for cross-cohort analysis.
    """
    # Validate labels/colormap arguments
    if not labels and not (fit_labels and project_labels):
        raise ValueError(
            "Must provide either 'labels' (used for both fit and project) OR both "
            "'fit_labels' and 'project_labels'. Providing only one of 'fit_labels' or "
            "'project_labels' without 'labels' is not allowed."
        )
    if not colormap and not (fit_colormap and project_colormap):
        raise ValueError(
            "Must provide either 'colormap' (used for both fit and project) OR both "
            "'fit_colormap' and 'project_colormap'. Providing only one of "
            "'fit_colormap' or 'project_colormap' without 'colormap' is not allowed."
        )

    # Create Pipeline instance
    pipeline = Pipeline(
        fit_plink_prefix=fit_plink,
        project_plink_prefix=project_plink,
        labels=labels,
        colormap=colormap,
        output_dir=output_dir,
        geographic_coords=geographic_coords,
        fit_labels=fit_labels,
        project_labels=project_labels,
        fit_colormap=fit_colormap,
        project_colormap=project_colormap,
        admixture_backend=admixture_backend,
        projection_plot_fit_column=projection_plot_fit_column,
        projection_plot_project_column=projection_plot_project_column,
    )

    # Run pipeline with all parameters
    results = pipeline.run(
        n_pcs=n_pcs,
        k_min=k_min,
        k_max=k_max,
        embedding=embedding,
        embedding_params=embedding_params,
        embedding_input=embedding_input,
        skip_pca=skip_pca,
        skip_admixture=skip_admixture,
        skip_embedding=skip_embedding,
        skip_visualization=skip_visualization,
        skip_pca_visualization=skip_pca_visualization,
        skip_admixture_visualization=skip_admixture_visualization,
        skip_metrics=skip_metrics,
        admix_group_column=admix_group_column,
        admix_within_group_order=admix_within_group_order,
        admix_threads=admix_threads,
        admix_gpus=admix_gpus,
        admix_batch_size=admix_batch_size,
    )

    logger.info("Pipeline execution complete")
    return results
