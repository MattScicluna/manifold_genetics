"""
Pipeline orchestrator for end-to-end genetic analysis.

Coordinates PCA, Admixture, Embeddings, Visualization, and Metrics.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from .config import AdmixtureConfig, EmbeddingConfig, PCAConfig, build_configs
from .result import PipelineResult
from .steps.admixture import run_admixture_step
from .steps.embedding import run_embedding_step
from .steps.metrics import run_admixture_metrics_step, run_geographic_metrics_step
from .steps.paths import metrics_output_paths, pca_output_paths
from .steps.pca import PCAStepResult, run_pca_step
from .steps.viz import (
    run_admixture_embedding_viz_step,
    run_admixture_viz_step,
    run_embedding_viz_step,
    run_pca_viz_step,
)

logger = logging.getLogger(__name__)


def _run_viz(name: str, failed: list, fn):
    """Run a visualization step, converting any failure into a warning.

    Spec constraint D: visualization is non-fatal. A broken plot must never fail
    `manifold-genetics pipeline`, but it must not vanish silently either — the
    step name lands in the returned results so the CLI can shout about it.
    """
    try:
        return fn()
    except Exception as e:
        logger.warning(f"Visualization step {name!r} failed: {e}", exc_info=True)
        failed.append(name)
        return None


def _resolve_skipped_pca(io, pca: PCAConfig) -> PCAStepResult:
    """Build a ``PCAStepResult`` from cached PCA output when ``--skip-pca`` is set.

    Populates whichever of the fit/project CSVs exist on disk and leaves the
    other ``None``. Never raises: callers that actually need one of these
    paths for a step that is about to run must check the result themselves
    and fail with their own clear message (spec constraint C).
    """
    paths = pca_output_paths(io, pca)
    fit_pca = paths["fit_pca"] if paths["fit_pca"].exists() else None
    project_pca = paths["project_pca"] if paths["project_pca"].exists() else None
    return PCAStepResult(fit_pca=fit_pca, project_pca=project_pca, skipped=True)


class Pipeline:
    """
    End-to-end pipeline for genetic analysis.

    Examples:
        >>> # Full pipeline
        >>> pipeline = Pipeline(
        ...     fit_plink_prefix="data/fit_subset",
        ...     project_plink_prefix="data/project_subset",
        ...     labels="labels.csv",
        ...     colormap="colormap.json",
        ...     output_dir="results/"
        ... )
        >>> results = pipeline.run(
        ...     n_pcs=50,
        ...     k_min=2, k_max=10,
        ...     embedding="phate", knn=25
        ... )
    """

    def __init__(
        self,
        fit_plink_prefix: Union[str, Path],
        project_plink_prefix: Union[str, Path],
        labels: Optional[Union[str, Path]] = None,
        colormap: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = None,
        geographic_coords: Optional[Union[str, Path]] = None,
        fit_labels: Optional[Union[str, Path]] = None,
        project_labels: Optional[Union[str, Path]] = None,
        fit_colormap: Optional[Union[str, Path]] = None,
        project_colormap: Optional[Union[str, Path]] = None,
        admixture_backend: Optional[object] = None,
        projection_plot_fit_column: Optional[str] = None,
        projection_plot_project_column: Optional[str] = None,
    ):
        """
        Initialize pipeline.

        Args:
            fit_plink_prefix: Path to fit subset PLINK files
            project_plink_prefix: Path to project subset PLINK files
            labels: Path to labels CSV (used for both fit and project if not overridden)
            colormap: Path to colormap JSON (used for both fit and project if not overridden)
            output_dir: Directory for outputs
            geographic_coords: Optional path to geographic coordinates
            fit_labels: Optional override labels CSV for fit dataset
            project_labels: Optional override labels CSV for project dataset
            fit_colormap: Optional override colormap JSON for fit dataset
            project_colormap: Optional override colormap JSON for project dataset
            admixture_backend: Optional AdmixtureBackend instance for testing
                              (if None, constructs a real NeuralAdmixtureBackend, which
                              runs neural-admixture in its own child process)
            projection_plot_fit_column: Column from fit colormap to use for projection plot
            projection_plot_project_column: Column from project colormap to use for projection plot

        Note:
            Must provide either (labels + colormap) OR (fit_labels + project_labels + fit_colormap + project_colormap)
        """
        # build_configs() performs all labels/colormap argument-shape validation
        # (no filesystem access) and must run before output_dir is created below —
        # tests/unit/test_runner.py::test_validation_fires_before_output_dir_created
        # guards exactly this ordering.
        #
        # __init__ only receives the IO and viz-relevant arguments; run()-time
        # parameters (n_pcs, k_min/k_max, embedding, the skip flags, and the two
        # admixture-viz ordering knobs) are not passed here, so the pca/admixture/
        # embedding/skips configs this call produces are built from defaults and
        # discarded — run() builds its own from its own arguments, exactly as
        # before. This keeps `Pipeline(...)` and `run_pipeline(...)` signatures
        # unchanged and avoids storing per-run state on the instance.
        configs = build_configs(
            fit_plink=fit_plink_prefix,
            project_plink=project_plink_prefix,
            output_dir=output_dir,
            labels=labels,
            colormap=colormap,
            fit_labels=fit_labels,
            project_labels=project_labels,
            fit_colormap=fit_colormap,
            project_colormap=project_colormap,
            geographic_coords=geographic_coords,
            projection_plot_fit_column=projection_plot_fit_column,
            projection_plot_project_column=projection_plot_project_column,
        )

        self._io = configs.io
        self._viz_config = configs.viz

        # Loose attributes kept for backward compatibility (existing callers and
        # tests read these directly off the Pipeline instance).
        self.fit_plink_prefix = configs.io.fit_plink
        self.project_plink_prefix = configs.io.project_plink
        self.output_dir = configs.io.output_dir
        self.geographic_coords = configs.io.geographic_coords
        self.fit_labels = configs.io.fit_labels
        self.project_labels = configs.io.project_labels
        self.fit_colormap = configs.io.fit_colormap
        self.project_colormap = configs.io.project_colormap
        self.labels = Path(labels) if labels else None
        self.colormap = Path(colormap) if colormap else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store admixture backend (for testing)
        self.admixture_backend = admixture_backend

        # Store projection column settings
        self.projection_plot_fit_column = projection_plot_fit_column
        self.projection_plot_project_column = projection_plot_project_column

    def run(
        self,
        n_pcs: int = 50,
        pca_backend: str = "python",
        k_min: int = 2,
        k_max: int = 10,
        embedding: str = "phate",
        embedding_params: Optional[Dict] = None,
        embedding_input: str = "both",
        skip_pca: bool = False,
        skip_admixture: bool = False,
        skip_embedding: bool = False,
        skip_visualization: bool = False,
        skip_pca_visualization: bool = False,
        skip_admixture_visualization: bool = False,
        admix_group_column: Optional[str] = None,
        admix_within_group_order: Optional[str] = "chron",
        skip_metrics: bool = False,
        admix_threads: Optional[int] = None,
        admix_gpus: Optional[int] = None,
        admix_batch_size: Optional[int] = 400,
    ) -> PipelineResult:
        """
        Run full pipeline.

        Args:
            pca_backend: 'flashpca' (external binary) or 'python' (in process)
            n_pcs: Number of principal components
            k_min: Minimum K for admixture
            k_max: Maximum K for admixture
            embedding: Embedding method ('phate', 'umap', 'tsne', 'diffusion_map')
            embedding_params: Optional parameters for embedding
            embedding_input: Which dataset to embed - 'fit', 'project', or 'both' (default)
            skip_pca: Skip PCA step
            skip_admixture: Skip admixture step
            skip_embedding: Skip embedding step
            skip_visualization: Skip embedding visualization step
            skip_pca_visualization: Skip PCA visualization step
            admix_group_column: Column for grouping in admixture barplots (None = use first colormap key)
            admix_within_group_order: Method for ordering samples within groups ('chron', 'tree', or None)
            skip_metrics: Skip metrics computation
            admix_threads: Threads to use for neural admixture (None = auto-detect)
            admix_gpus: Number of GPUs for neural admixture (None = auto-detect)

        Returns:
            PipelineResult with the typed outputs of every stage that ran.
            ``admixture`` and ``embedding`` are None when their stage did not
            run (``skip_admixture`` / ``skip_embedding``), as are
            ``geographic_metrics`` and ``admixture_metrics`` when their
            metric did not run. ``pca`` is different: it is never None —
            under ``skip_pca`` it is still a ``PCAStepResult``, but with
            ``skipped=True`` and ``fit_pca``/``project_pca`` populated only
            from whatever cached output already exists on disk (possibly
            both None). Check ``.skipped``, not truthiness, to tell whether
            PCA ran. Figure families are empty tuples/dicts when their stage
            did not run or produced nothing.
        """
        failed = []

        io = self._io
        pca_cfg = PCAConfig(n_pcs=n_pcs, backend=pca_backend)
        pca_paths = pca_output_paths(io, pca_cfg)
        # admix_group_column / admix_within_group_order are run()-time parameters,
        # not init-time ones — self._viz_config only carries the init-time
        # projection-plot columns, so those two fields are overridden per call.
        # Two run() calls with different values must not interfere with each
        # other, which a stored, mutated VizConfig on self would risk.
        viz_cfg = dataclasses.replace(
            self._viz_config,
            admix_group_column=admix_group_column,
            admix_within_group_order=admix_within_group_order,
        )

        # ---- Step 1: PCA ----
        if not skip_pca:
            logger.info("=" * 70)
            logger.info("STEP 1: PCA")
            logger.info("=" * 70)

            r_pca = run_pca_step(io, pca_cfg)
        else:
            # PCA skipped — resolve expected paths so embedding can still run
            r_pca = _resolve_skipped_pca(io, pca_cfg)

        # ---- Step 1.5: PCA Visualization (independent of PCA computation) ----
        pca_figures = ()
        if not skip_pca_visualization:
            logger.info("=" * 70)
            logger.info("STEP 1.5: PCA VISUALIZATION")
            logger.info("=" * 70)

            pca_file = pca_paths["project_pca"]

            if pca_file.exists():
                pca_viz_result = _run_viz(
                    "pca_viz",
                    failed,
                    lambda: run_pca_viz_step(io, pca_file=pca_file, n_pcs=n_pcs),
                )
                if pca_viz_result is not None:
                    pca_figures = tuple(pca_viz_result.figures)
            else:
                logger.warning(f"PCA file not found: {pca_file}")
                logger.warning("Run with --skip-pca=False to compute PCA first")

        # ---- Step 2: Admixture ----
        r_admix = None
        admixture_figures = {}
        if not skip_admixture:
            logger.info("=" * 70)
            logger.info("STEP 2: ADMIXTURE")
            logger.info("=" * 70)

            admix_cfg = AdmixtureConfig(
                k_min=k_min,
                k_max=k_max,
                threads=admix_threads,
                num_gpus=admix_gpus,
                batch_size=admix_batch_size,
            )

            r_admix = run_admixture_step(io, admix_cfg, backend=self.admixture_backend)

            # Admixture bar plot (placed in figures/admixture/)
            if not skip_admixture_visualization:
                admix_viz_result = _run_viz(
                    "admixture_viz",
                    failed,
                    lambda: run_admixture_viz_step(io, viz_cfg, admixture=r_admix),
                )
                if admix_viz_result is not None:
                    bars = admix_viz_result.figures[0] if admix_viz_result.figures else None
                    if bars is not None:
                        admixture_figures["bars"] = bars

        # ---- Step 3: Embedding ----
        r_emb = None
        if not skip_embedding:
            logger.info("=" * 70)
            logger.info(f"STEP 3: EMBEDDING ({embedding.upper()})")
            logger.info("=" * 70)

            if r_pca.fit_pca is None and r_pca.project_pca is None:
                raise RuntimeError(
                    "No PCA files found. PCA was skipped (--skip-pca) and no cached "
                    f"output exists at {pca_paths['fit_pca']} or "
                    f"{pca_paths['project_pca']}. Re-run without --skip-pca to "
                    "compute PCA, or place existing PCA CSVs at those paths."
                )

            emb_cfg = EmbeddingConfig(
                method=embedding,
                input_mode=embedding_input,
                params=dict(embedding_params or {}),
            )

            # PCA outputs may have come from the step or been resolved from disk
            # under --skip-pca; either way the embedding step takes them as a result.
            r_emb = run_embedding_step(io, emb_cfg, pca=r_pca)

        # ---- Step 4: Embedding Visualization ----
        fit_embedding_figures = ()
        embedding_figures = ()
        projection_plot = None
        if not skip_visualization and not skip_embedding:
            logger.info("=" * 70)
            logger.info("STEP 4: EMBEDDING VISUALIZATION")
            logger.info("=" * 70)

            emb_viz_result = _run_viz(
                "embedding_viz",
                failed,
                lambda: run_embedding_viz_step(io, viz_cfg, embedding=r_emb, method=embedding),
            )
            if emb_viz_result is not None:
                if r_emb.fit_embedding_file is not None:
                    fit_embedding_figures = tuple(emb_viz_result.fit_figures)
                embedding_figures = tuple(emb_viz_result.project_figures)
                if emb_viz_result.projection_plot is not None:
                    projection_plot = emb_viz_result.projection_plot
                failed.extend(emb_viz_result.failed_substeps)

        # ---- Step 4.5: Admixture-Colored Embedding Visualization (requires embedding) ----
        if not skip_admixture_visualization and not skip_embedding and not skip_admixture:
            logger.info("=" * 70)
            logger.info("STEP 4.5: ADMIXTURE-COLORED EMBEDDING VISUALIZATION")
            logger.info("=" * 70)

            if r_emb is not None and r_admix is not None:
                admix_emb_viz_result = _run_viz(
                    "admixture_embedding_viz",
                    failed,
                    lambda: run_admixture_embedding_viz_step(
                        io, embedding=r_emb, admixture=r_admix
                    ),
                )
                if admix_emb_viz_result is not None:
                    admix_emb = (
                        admix_emb_viz_result.figures[0] if admix_emb_viz_result.figures else None
                    )
                    if admix_emb is not None:
                        admixture_figures["admixture_colored_embedding"] = admix_emb
            else:
                logger.warning(
                    "Skipping admixture-colored embedding visualization - missing embedding or admixture data"
                )

        # ---- Step 5: Metrics ----
        r_geo = None
        r_admix_metrics = None
        if not skip_metrics and not skip_embedding:
            logger.info("=" * 70)
            logger.info("STEP 5: METRICS")
            logger.info("=" * 70)

            metrics_paths = metrics_output_paths(io)
            # Created unconditionally: the output tree has always contained metrics/
            # even when neither metric runs.
            metrics_paths["geographic"].parent.mkdir(parents=True, exist_ok=True)

            # Geographic preservation
            if self.geographic_coords:
                r_geo = run_geographic_metrics_step(
                    r_emb.embedding_file,
                    self.geographic_coords,
                    metrics_paths["geographic"],
                )

            # Admixture preservation
            if not skip_admixture and r_admix is not None:
                r_admix_metrics = run_admixture_metrics_step(
                    r_emb.embedding_file,
                    r_admix.q_prefix,
                    range(k_min, k_max + 1),
                    metrics_paths["admixture"],
                )

        # Summary
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.output_dir}")

        return PipelineResult(
            pca=r_pca,
            admixture=r_admix,
            embedding=r_emb,
            geographic_metrics=r_geo,
            admixture_metrics=r_admix_metrics,
            pca_figures=pca_figures,
            fit_embedding_figures=fit_embedding_figures,
            embedding_figures=embedding_figures,
            projection_plot=projection_plot,
            admixture_figures=admixture_figures,
            failed_steps=tuple(failed),
        )
