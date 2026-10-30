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
from typing import List, Mapping, Optional, Tuple

from ...utils.io import read_colormap
from ...visualization import (
    plot_admixture_bar_grid,
    plot_admixture_embedding_3d,
    plot_admixture_embedding_grid,
    plot_embedding_grid,
    plot_pca_pairs,
    plot_projection,
    visualize,
    visualize_3d,
)
from ..config import IOConfig, VizConfig
from .admixture import AdmixtureStepResult
from .embedding import EmbeddingStepResult, run_embedding, select_embedding_inputs
from .paths import embedding_grid_output_paths, figure_output_paths
from .pca import PCAStepResult

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingVizResult",
    "VizStepResult",
    "plot_pca_pair_grids",
    "run_admixture_embedding_viz_step",
    "run_admixture_viz_step",
    "run_embedding_grid_viz_step",
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
    # Parts of the step that failed without failing it, e.g. one grid panel.
    failed_substeps: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EmbeddingVizResult:
    """Outputs of the embedding visualization step, which feeds three distinct
    figure families and guards its projection plot separately.
    """

    fit_figures: Tuple[Path, ...] = ()
    project_figures: Tuple[Path, ...] = ()
    projection_plot: Optional[Path] = None
    # Sub-steps that failed inside an otherwise-successful step. The embedding
    # viz step guards its projection plot separately, so that a projection
    # failure does not lose the fit/project figures — this is how that failure
    # still reaches the pipeline's failed-step list.
    failed_substeps: Tuple[str, ...] = ()

    @property
    def figures(self) -> Tuple[Path, ...]:
        """All figures, in production order: fit, then project, then projection."""
        figures = tuple(self.fit_figures) + tuple(self.project_figures)
        if self.projection_plot is not None:
            figures += (self.projection_plot,)
        return figures


def run_pca_viz_step(io: IOConfig, *, pca_file: Path, n_pcs: int) -> VizStepResult:
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


def _embedding_dims(path) -> int:
    """Number of ``dim_*`` columns in an embedding CSV, from its header alone."""
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    return sum(1 for c in header if c.startswith("dim_"))


def _visualize_3d_if_possible(embedding_file, labels, colormap, output_dir, method, dataset_prefix):
    """Write the rotatable HTML figures for a >=3-D embedding.

    A 3-D embedding drawn only by ``visualize`` would be its face-on projection
    as a PNG, with nothing to say a third axis exists. plotly is optional, so
    its absence is a warning naming the extra rather than a failed step.
    """
    if _embedding_dims(embedding_file) < 3:
        return []
    try:
        return visualize_3d(
            embedding=embedding_file,
            labels=labels,
            colormap=colormap,
            output_dir=output_dir,
            output_prefix=f"{method}_3d",
            dataset_prefix=dataset_prefix,
        )
    except ImportError as exc:
        logger.warning(f"Embedding has 3 dimensions but the interactive figure was skipped: {exc}")
        return []


def run_embedding_viz_step(
    io: IOConfig, viz: VizConfig, *, embedding: EmbeddingStepResult, method: str
) -> EmbeddingVizResult:
    """Plot fit figures (if a fit embedding exists), project figures, and the
    fit/project projection plot (if both projection columns are configured).
    """
    paths = figure_output_paths(io)
    embedding_figures_dir = paths["embeddings"]
    embedding_figures_dir.mkdir(parents=True, exist_ok=True)

    fit_figure_paths = []
    project_figure_paths = []
    projection_plot_path = None
    failed_substeps = []

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
        fit_figure_paths = list(fit_figure_paths) + list(
            _visualize_3d_if_possible(
                embedding.fit_embedding_file,
                io.fit_labels,
                io.fit_colormap,
                embedding_figures_dir,
                method,
                "fit_",
            )
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
    project_figure_paths = list(project_figure_paths) + list(
        _visualize_3d_if_possible(
            embedding.embedding_file,
            io.project_labels,
            io.project_colormap,
            embedding_figures_dir,
            method,
            "project_",
        )
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
            failed_substeps.append("projection_plot")

    return EmbeddingVizResult(
        fit_figures=tuple(fit_figure_paths),
        project_figures=tuple(project_figure_paths),
        projection_plot=projection_plot_path,
        failed_substeps=tuple(failed_substeps),
    )


_AXIS_NAMES = {"phate": "PHATE", "umap": "UMAP", "tsne": "t-SNE", "diffusion_map": "DM"}


def _embed_grid_cell(name, inputs, output_dir, method, params, failed) -> Optional[Path]:
    """Embed one grid panel, or record it as failed and return None.

    One panel failing -- a perplexity too large for the cohort, say -- leaves
    that panel empty rather than losing the whole comparison.
    """
    output = output_dir / f"{name}.csv"
    fit_input, project_input = inputs
    try:
        run_embedding(
            fit_input,
            project_input,
            project_output=output,
            method=method,
            params={**params, "n_components": 2},
        )
    except Exception as e:
        logger.warning(f"Embedding grid panel {name!r} failed: {e}", exc_info=True)
        failed.append(name)
        return None
    return output


def _plot_grid_per_column(grid, labels, colormap_path, output_dir, stem, **kwargs) -> List[Path]:
    """One grid figure per colormap column, as ``visualize`` does for single plots."""
    figures = []
    for label_col in read_colormap(colormap_path):
        figures.append(
            plot_embedding_grid(
                embeddings=grid,
                labels=labels,
                colormap=colormap_path,
                label_column=label_col,
                output_path=output_dir / f"{stem}_by_{label_col}.png",
                **kwargs,
            )
        )
    return figures


def run_embedding_grid_viz_step(
    io: IOConfig,
    viz: VizConfig,
    *,
    pca: PCAStepResult,
    input_mode: str,
    base_params: Optional[Mapping] = None,
) -> VizStepResult:
    """Embed and plot the method-comparison and PHATE-parameter grids.

    Each panel embeds the same PCA coordinates as the run's own embedding
    (``input_mode``), and is written to ``embeddings/grids/`` before it is
    drawn. ``base_params`` are the run's PHATE parameters, which the PHATE grid
    varies around, so a preset's landmarking carries into every panel.
    """
    if not viz.method_grid and not viz.phate_grid:
        return VizStepResult()

    inputs = select_embedding_inputs(input_mode, pca)
    labels, colormap = (
        (io.fit_labels, io.fit_colormap)
        if input_mode == "fit"
        else (io.project_labels, io.project_colormap)
    )
    paths = embedding_grid_output_paths(io)
    paths["embeddings"].mkdir(parents=True, exist_ok=True)
    figures, failed = [], []

    if viz.method_grid:
        methods = list(viz.method_grid)
        row = [
            _embed_grid_cell(
                f"method_grid_{method}",
                inputs,
                paths["embeddings"],
                method,
                dict(viz.method_grid[method] or {}),
                failed,
            )
            for method in methods
        ]
        figures += _plot_grid_per_column(
            [row],
            labels,
            colormap,
            paths["figures"],
            "method_grid",
            col_titles=[_AXIS_NAMES[m] for m in methods],
            axis_names=[[_AXIS_NAMES[m] for m in methods]],
        )

    if viz.phate_grid:
        fixed = {k: v for k, v in (base_params or {}).items() if k != "n_components"}
        fixed.update({k: v for k, v in viz.phate_grid.items() if not isinstance(v, list)})
        swept = [(k, v) for k, v in viz.phate_grid.items() if isinstance(v, list)]
        # One swept parameter reads best as a single row of columns.
        (row_key, row_values), (col_key, col_values) = (
            swept if len(swept) == 2 else [(None, [None]), swept[0]]
        )
        grid = []
        for rv in row_values:
            cells = []
            for cv in col_values:
                params = {**fixed, col_key: cv}
                name = f"phate_grid_{col_key}-{cv}"
                if row_key is not None:
                    params[row_key] = rv
                    name = f"phate_grid_{row_key}-{rv}_{col_key}-{cv}"
                cells.append(
                    _embed_grid_cell(name, inputs, paths["embeddings"], "phate", params, failed)
                )
            grid.append(cells)
        figures += _plot_grid_per_column(
            grid,
            labels,
            colormap,
            paths["figures"],
            "phate_grid",
            row_titles=None if row_key is None else [f"{row_key} = {v}" for v in row_values],
            col_titles=[f"{col_key} = {v}" for v in col_values],
            axis_names=[["PHATE"] * len(col_values)] * len(row_values),
            title="PHATE parameter grid",
        )

    return VizStepResult(figures=tuple(figures), failed_substeps=tuple(failed))


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

    interactive = admixture_embedding_3d_if_possible(
        embedding.embedding_file,
        q_prefix=admixture.q_prefix,
        k_values=admixture.k_values,
        output_path=paths["admixture_colored_embedding_3d"],
    )
    return VizStepResult(figures=(emb_plot_path, *interactive))


def admixture_embedding_3d_if_possible(embedding_file, **kwargs) -> List[Path]:
    """Write the single-file interactive admixture figure for a >=3-D embedding.

    The seam shared by ``plot-admixture-embedding`` and the pipeline step: the
    2-D grid is always drawn, and this adds the HTML alongside it when there is
    a third axis to show. As with the label figures, missing plotly is a
    warning naming the extra rather than a failed step.
    """
    if _embedding_dims(embedding_file) < 3:
        return []
    try:
        return [plot_admixture_embedding_3d(embedding=embedding_file, **kwargs)]
    except ImportError as exc:
        logger.warning(f"Embedding has 3 dimensions but the interactive figure was skipped: {exc}")
        return []
