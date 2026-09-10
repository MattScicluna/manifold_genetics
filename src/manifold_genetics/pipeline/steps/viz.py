"""In-process visualization steps.

Each function here is one of the four inline visualization blocks that used to
live in ``Pipeline.run()``, extracted verbatim: same plotting function, same
keyword arguments, same figure filenames, same iteration order. Imports are
module-level, which is what fixes issue #72 — the old code imported
``read_colormap`` inside the PCA-visualization branch, making the name
function-local for the rest of ``run()`` and raising ``UnboundLocalError`` in
the admixture bar-plot block whenever ``--skip-pca-visualization`` was passed.

Visualization is NON-FATAL by spec constraint D, but that is enforced by the
orchestrator, not here: these step functions may raise, and it is the
orchestrator's ``_run_viz`` wrapper that catches. The one exception is the
projection-plot sub-block inside ``run_embedding_viz_step``, which keeps its
own pre-existing try/except so that a projection-plot failure does not discard
the fit/project figures already produced in the same step.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from ...utils.io import read_colormap
from ...visualization import (
    plot_admixture_bar_grid,
    plot_admixture_embedding_grid,
    plot_pca_pairs,
    plot_projection,
    visualize,
)
from ..config import IOConfig, VizConfig
from .admixture import AdmixtureStepResult
from .embedding import EmbeddingStepResult
from .paths import figure_output_paths

logger = logging.getLogger(__name__)

__all__ = [
    "VizStepResult",
    "plot_pca_pair_grids",
    "run_admixture_embedding_viz_step",
    "run_admixture_viz_step",
    "run_embedding_viz_step",
    "run_pca_viz_step",
]


def plot_pca_pair_grids(
    pca_coords,
    labels,
    colormap: Mapping,
    output_dir: Path,
    n_pcs: int,
) -> list:
    """Write one PCA-pairs grid per label column in ``colormap``.

    The shared seam behind both ``manifold-genetics plot-pca`` and the pipeline's
    PCA visualization step, so the ``pca_pairs_by_{column}.png`` naming lives in
    one place.

    Returns the figure paths, in colormap-key order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    for label_col in colormap.keys():
        output_path = output_dir / f"pca_pairs_by_{label_col}.png"
        plot_path = plot_pca_pairs(
            pca_coords=pca_coords,
            labels=labels,
            colormap=colormap,
            output_path=output_path,
            label_column=label_col,
            n_pcs=n_pcs,
            title=f"PCA Pairs by {label_col}",
        )
        figure_paths.append(plot_path)

    return figure_paths


@dataclass(frozen=True)
class VizStepResult:
    """Typed outputs of a visualization step: the figures it produced."""

    figures: Tuple[Path, ...] = ()
    failed: bool = False
    # Populated only by run_embedding_viz_step. That step feeds three distinct
    # results keys, which a flat `figures` tuple cannot reconstruct.
    fit_figures: Tuple[Path, ...] = ()
    project_figures: Tuple[Path, ...] = ()
    projection_plot: Optional[Path] = None


def run_pca_viz_step(io: IOConfig, viz: VizConfig, *, pca_file: Path, n_pcs: int) -> VizStepResult:
    """Plot one PCA-pairs grid per column in the project colormap."""
    paths = figure_output_paths(io)
    pca_figures_dir = paths["pca"]

    logger.info("Plotting PCA pairs grid via plot_pca_pairs")
    colormap_dict = read_colormap(io.project_colormap)
    pca_figure_paths = plot_pca_pair_grids(
        pca_coords=pca_file,
        labels=io.project_labels,
        colormap=colormap_dict,
        output_dir=pca_figures_dir,
        n_pcs=n_pcs,
    )
    for plot_path in pca_figure_paths:
        logger.info(f"Saved PCA pairs plot: {plot_path}")

    logger.info(f"Created PCA plots: {len(pca_figure_paths)} figures")
    return VizStepResult(figures=tuple(pca_figure_paths))


def run_embedding_viz_step(
    io: IOConfig, viz: VizConfig, *, embedding: EmbeddingStepResult, method: str
) -> VizStepResult:
    """Plot fit figures (if a fit embedding exists), project figures, and the
    fit/project projection plot (if both projection columns are configured).
    """
    paths = figure_output_paths(io)
    embedding_figures_dir = paths["embeddings"]
    embedding_figures_dir.mkdir(parents=True, exist_ok=True)

    fit_figure_paths = []
    project_figure_paths = []
    projection_plot_path = None

    # Fit visualizations (only when a fit embedding was produced).
    if embedding.fit_embedding_file is not None:
        logger.info("Creating fit embedding visualizations...")
        fit_figure_paths = visualize(
            embedding=embedding.fit_embedding_file,
            labels=io.fit_labels,
            colormap=io.fit_colormap,
            output_dir=embedding_figures_dir,
            output_prefix=method,
            dataset_prefix="fit_",
        )
        logger.info(f"Created {len(fit_figure_paths)} fit embedding figures")

    # Project visualizations — always.
    logger.info("Creating project embedding visualizations...")
    project_figure_paths = visualize(
        embedding=embedding.embedding_file,
        labels=io.project_labels,
        colormap=io.project_colormap,
        output_dir=embedding_figures_dir,
        output_prefix=method,
        dataset_prefix="project_",
    )
    logger.info(f"Created {len(project_figure_paths)} project embedding figures")

    # Projection plot (fit + project together), only in cross-projection mode.
    if (
        embedding.fit_embedding_file is not None
        and viz.projection_plot_fit_column
        and viz.projection_plot_project_column
    ):
        logger.info("Creating projection plot (fit + project together)...")

        candidate_projection_plot_path = (
            embedding_figures_dir / f"{method}_projection_fit_{viz.projection_plot_fit_column}"
            f"_project_{viz.projection_plot_project_column}.png"
        )

        try:
            plot_projection(
                fit_embedding=embedding.fit_embedding_file,
                project_embedding=embedding.embedding_file,
                fit_labels=io.fit_labels,
                project_labels=io.project_labels,
                fit_colormap=io.fit_colormap,
                project_colormap=io.project_colormap,
                output_path=candidate_projection_plot_path,
                fit_label_column=viz.projection_plot_fit_column,
                project_label_column=viz.projection_plot_project_column,
            )

            projection_plot_path = candidate_projection_plot_path
            logger.info(f"Projection plot saved: {projection_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to create projection plot: {e}")

    figures = list(fit_figure_paths) + list(project_figure_paths)
    if projection_plot_path is not None:
        figures.append(projection_plot_path)

    return VizStepResult(
        figures=tuple(figures),
        fit_figures=tuple(fit_figure_paths),
        project_figures=tuple(project_figure_paths),
        projection_plot=projection_plot_path,
    )


def run_admixture_viz_step(
    io: IOConfig, viz: VizConfig, *, admixture: AdmixtureStepResult
) -> VizStepResult:
    """Plot the admixture bar grid for the project cohort."""
    paths = figure_output_paths(io)
    admix_figures_dir = paths["admixture"]
    admix_figures_dir.mkdir(parents=True, exist_ok=True)

    cmap_dict = read_colormap(io.project_colormap)
    # Grouping column: user-specified if provided, else first colormap key.
    group_col = viz.admix_group_column
    if not group_col:
        group_col = next(iter(cmap_dict.keys()))

    bar_plot_path = paths["admixture_bars"]
    plot_admixture_bar_grid(
        q_prefix=admixture.q_prefix,
        labels=io.project_labels,
        group_column=group_col,
        k_values=admixture.k_values,
        output_path=bar_plot_path,
        colormap=cmap_dict,
        subsample_per_group=300,
        within_group_order=viz.admix_within_group_order,
    )

    return VizStepResult(figures=(bar_plot_path,))


def run_admixture_embedding_viz_step(
    io: IOConfig, *, embedding: EmbeddingStepResult, admixture: AdmixtureStepResult
) -> VizStepResult:
    """Plot the admixture-coloured embedding grid for the project cohort."""
    paths = figure_output_paths(io)
    admix_figures_dir = paths["admixture"]
    admix_figures_dir.mkdir(parents=True, exist_ok=True)

    emb_plot_path = paths["admixture_colored_embedding"]
    plot_admixture_embedding_grid(
        embedding=embedding.embedding_file,
        q_prefix=admixture.q_prefix,
        k_values=admixture.k_values,
        output_path=emb_plot_path,
    )
    logger.info(f"Saved admixture-colored embedding plot: {emb_plot_path}")

    return VizStepResult(figures=(emb_plot_path,))
