"""
Pipeline orchestrator for end-to-end genetic analysis.

Coordinates PCA, Admixture, Embeddings, Visualization, and Metrics.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

from ..embeddings import PHATE, TSNE, UMAP, DiffusionMap
from ..visualization import (
    plot_admixture_bar_grid,
    plot_admixture_embedding_grid,
    plot_projection,
    visualize,
)
from .config import EmbeddingConfig, IOConfig, PCAConfig
from .steps.embedding import run_embedding_step
from .steps.metrics import run_admixture_metrics_step, run_geographic_metrics_step
from .steps.paths import metrics_output_paths, pca_output_paths
from .steps.pca import PCAStepResult, run_pca_step

logger = logging.getLogger(__name__)


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
                              (if None, uses neural-admixture via CLI)
            projection_plot_fit_column: Column from fit colormap to use for projection plot
            projection_plot_project_column: Column from project colormap to use for projection plot

        Note:
            Must provide either (labels + colormap) OR (fit_labels + project_labels + fit_colormap + project_colormap)
        """
        self.fit_plink_prefix = Path(fit_plink_prefix)
        self.project_plink_prefix = Path(project_plink_prefix)
        self.output_dir = Path(output_dir)
        self.geographic_coords = Path(geographic_coords) if geographic_coords else None

        # Handle labels: either use shared labels or separate fit/project labels
        if labels is not None:
            self.labels = Path(labels)
            self.fit_labels = Path(fit_labels) if fit_labels else self.labels
            self.project_labels = Path(project_labels) if project_labels else self.labels
        else:
            # No shared labels provided, must have separate fit/project labels
            if not fit_labels or not project_labels:
                raise ValueError(
                    "Must provide either 'labels' OR both 'fit_labels' and 'project_labels'"
                )
            self.labels = None
            self.fit_labels = Path(fit_labels)
            self.project_labels = Path(project_labels)

        # Handle colormap: either use shared colormap or separate fit/project colormaps
        if colormap is not None:
            self.colormap = Path(colormap)
            self.fit_colormap = Path(fit_colormap) if fit_colormap else self.colormap
            self.project_colormap = Path(project_colormap) if project_colormap else self.colormap
        else:
            # No shared colormap provided, must have separate fit/project colormaps
            if not fit_colormap or not project_colormap:
                raise ValueError(
                    "Must provide either 'colormap' OR both 'fit_colormap' and 'project_colormap'"
                )
            self.colormap = None
            self.fit_colormap = Path(fit_colormap)
            self.project_colormap = Path(project_colormap)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store admixture backend (for testing)
        self.admixture_backend = admixture_backend

        # Store projection column settings
        self.projection_plot_fit_column = projection_plot_fit_column
        self.projection_plot_project_column = projection_plot_project_column

    def _io_config(self) -> IOConfig:
        """Adapt the pipeline's loose attributes to the step layer's IOConfig.

        Transitional: PR 5 wires ``build_configs()`` into ``__init__`` and drops
        this. ``__init__`` has already guaranteed every field is set.
        """
        return IOConfig(
            fit_plink=self.fit_plink_prefix,
            project_plink=self.project_plink_prefix,
            output_dir=self.output_dir,
            fit_labels=self.fit_labels,
            project_labels=self.project_labels,
            fit_colormap=self.fit_colormap,
            project_colormap=self.project_colormap,
            geographic_coords=self.geographic_coords,
        )

    def run(
        self,
        n_pcs: int = 50,
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
        admix_batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Run full pipeline.

        Args:
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
            Dictionary with paths to outputs and computed metrics
        """
        results = {}

        io = self._io_config()
        pca_cfg = PCAConfig(n_pcs=n_pcs)
        pca_paths = pca_output_paths(io, pca_cfg)

        # Step 1: PCA
        if not skip_pca:
            logger.info("=" * 70)
            logger.info("STEP 1: PCA")
            logger.info("=" * 70)

            pca_result = run_pca_step(io, pca_cfg)

            results["fit_pca_file"] = pca_result.fit_pca
            results["project_pca_file"] = pca_result.project_pca
            results["pca_file"] = pca_result.project_pca
            results["pca_coords"] = pca_result.coords_df

        else:
            # PCA skipped — resolve expected paths so embedding can still run
            if pca_paths["fit_pca"].exists():
                results["fit_pca_file"] = pca_paths["fit_pca"]
            if pca_paths["project_pca"].exists():
                results["project_pca_file"] = pca_paths["project_pca"]
                results["pca_file"] = pca_paths["project_pca"]

        # Step 1.5: PCA Visualization (independent of PCA computation)
        if not skip_pca_visualization:
            logger.info("=" * 70)
            logger.info("STEP 1.5: PCA VISUALIZATION")
            logger.info("=" * 70)

            pca_file = pca_paths["project_pca"]

            if pca_file.exists():
                pca_figures_dir = self.output_dir / "figures" / "pca"
                pca_figures_dir.mkdir(parents=True, exist_ok=True)

                logger.info("Plotting PCA pairs grid via plot_pca_pairs")
                # Use plot_pca_pairs to generate a single grid covering PC pairs
                from ..utils.io import read_colormap
                from ..visualization import plot_pca_pairs

                colormap_dict = read_colormap(self.project_colormap)
                pca_figure_paths = []
                for label_col in colormap_dict.keys():
                    output_path = pca_figures_dir / f"pca_pairs_by_{label_col}.png"
                    plot_path = plot_pca_pairs(
                        pca_coords=pca_file,
                        labels=self.project_labels,
                        colormap=colormap_dict,
                        output_path=output_path,
                        label_column=label_col,
                        n_pcs=n_pcs,
                        title=f"PCA Pairs by {label_col}",
                    )
                    pca_figure_paths.append(plot_path)
                    logger.info(f"Saved PCA pairs plot: {plot_path}")

                results["pca_figures"] = pca_figure_paths
                logger.info(f"Created PCA plots: {len(pca_figure_paths)} figures")
            else:
                logger.warning(f"PCA file not found: {pca_file}")
                logger.warning("Run with --skip-pca=False to compute PCA first")

        # Step 2: Admixture
        if not skip_admixture:
            logger.info("=" * 70)
            logger.info("STEP 2: ADMIXTURE")
            logger.info("=" * 70)

            admix_dir = self.output_dir / "admixture"
            admix_dir.mkdir(exist_ok=True)
            admix_checkpoints_dir = admix_dir / "checkpoints"
            admix_checkpoints_dir.mkdir(parents=True, exist_ok=True)

            # Use provided backend or fall back to CLI
            if self.admixture_backend is not None:
                logger.info("Running admixture with provided backend (testing mode)")

                # Fit on fit dataset
                self.admixture_backend.fit(
                    str(self.fit_plink_prefix), str(admix_checkpoints_dir), model_name="fit"
                )

                # Apply to both fit and project datasets
                fit_q_files_from_backend = self.admixture_backend.fit_transform(
                    str(self.fit_plink_prefix), str(admix_dir / "fit")
                )

                project_q_files_from_backend = self.admixture_backend.transform(
                    str(self.project_plink_prefix), str(admix_dir / "project")
                )

                logger.info(f"✓ Admixture complete using {type(self.admixture_backend).__name__}")
            else:
                logger.info("Running admixture via CLI")
                admix_cmd = [
                    "manifold-genetics",
                    "admixture",
                    "--fit-plink",
                    str(self.fit_plink_prefix),
                    "--project-plink",
                    str(self.project_plink_prefix),
                    "--neuraladmixture-output-dir",
                    str(admix_checkpoints_dir),
                    "--fit-output",
                    str(admix_dir / "fit"),
                    "--project-output",
                    str(admix_dir / "project"),
                    "--k-min",
                    str(k_min),
                    "--k-max",
                    str(k_max),
                ]
                if admix_threads:
                    admix_cmd.extend(["--threads", str(admix_threads)])
                if admix_gpus is not None:
                    admix_cmd.extend(["--num-gpus", str(admix_gpus)])
                if admix_batch_size is not None:
                    admix_cmd.extend(["--neuraladmixture-batch-size", str(admix_batch_size)])

                subprocess.run(admix_cmd, check=True)

            fit_q_files = {k: admix_dir / f"fit.{k}.csv" for k in range(k_min, k_max + 1)}
            project_q_files = {k: admix_dir / f"project.{k}.csv" for k in range(k_min, k_max + 1)}

            results["admixture_dir"] = admix_dir
            results["admixture_checkpoints_dir"] = admix_checkpoints_dir
            results["fit_q_files"] = fit_q_files
            results["project_q_files"] = project_q_files
            results["q_files"] = project_q_files

            # Admixture bar plot (placed in figures/admixture/)
            if not skip_admixture_visualization:
                admix_figures_dir = self.output_dir / "figures" / "admixture"
                admix_figures_dir.mkdir(parents=True, exist_ok=True)

                cmap_dict = read_colormap(self.project_colormap)
                # Grouping column: user-specified if provided, else first colormap key
                group_col = admix_group_column
                if not group_col:
                    group_col = next(iter(cmap_dict.keys()))

                # Bar plot (using project set by default)
                bar_plot_path = admix_figures_dir / "project_bars.png"
                plot_admixture_bar_grid(
                    q_prefix=admix_dir / "project",
                    labels=self.project_labels,
                    group_column=group_col,
                    k_values=range(k_min, k_max + 1),
                    output_path=bar_plot_path,
                    colormap=cmap_dict,
                    subsample_per_group=300,
                    within_group_order=admix_within_group_order,
                )

                results.setdefault("admixture_figures", {})["bars"] = bar_plot_path

        # Step 3: Embedding
        if not skip_embedding:
            logger.info("=" * 70)
            logger.info(f"STEP 3: EMBEDDING ({embedding.upper()})")
            logger.info("=" * 70)

            if "fit_pca_file" not in results and "project_pca_file" not in results:
                raise RuntimeError(
                    "No PCA files found. Run without --skip-pca, or ensure "
                    f"{self.output_dir / 'pca'} contains fit_pca_{{n}}.csv / project_pca_{{n}}.csv."
                )

            emb_cfg = EmbeddingConfig(
                method=embedding,
                input_mode=embedding_input,
                params=dict(embedding_params or {}),
            )

            # PCA outputs may have come from the step or been resolved from disk
            # under --skip-pca; either way the embedding step takes them as a result.
            pca_result = PCAStepResult(
                fit_pca=results.get("fit_pca_file"),
                project_pca=results.get("project_pca_file"),
            )

            emb_result = run_embedding_step(io, emb_cfg, pca=pca_result)

            embedding_file = emb_result.embedding_file
            results["embedding_file"] = embedding_file
            results["embedding_coords"] = emb_result.coords_df

            if emb_result.fit_embedding_file is not None:
                results["fit_embedding_file"] = emb_result.fit_embedding_file

        # Step 4: Embedding Visualization
        if not skip_visualization and not skip_embedding:
            logger.info("=" * 70)
            logger.info("STEP 4: EMBEDDING VISUALIZATION")
            logger.info("=" * 70)

            embedding_figures_dir = self.output_dir / "figures" / "embeddings"
            embedding_figures_dir.mkdir(parents=True, exist_ok=True)

            # Step 4.1: Generate FIT visualizations (for all columns in fit_colormap)
            if "fit_embedding_file" in results and self.fit_labels and self.fit_colormap:
                logger.info("Creating fit embedding visualizations...")
                fit_figure_paths = visualize(
                    embedding=results["fit_embedding_file"],
                    labels=self.fit_labels,
                    colormap=self.fit_colormap,
                    output_dir=embedding_figures_dir,
                    output_prefix=embedding,
                    dataset_prefix="fit_",
                )
                results["fit_embedding_figures"] = fit_figure_paths
                logger.info(f"Created {len(fit_figure_paths)} fit embedding figures")

            # Step 4.2: Generate PROJECT visualizations (for all columns in project_colormap)
            logger.info("Creating project embedding visualizations...")
            project_figure_paths = visualize(
                embedding=embedding_file,
                labels=self.project_labels,
                colormap=self.project_colormap,
                output_dir=embedding_figures_dir,
                output_prefix=embedding,
                dataset_prefix="project_",
            )
            results["embedding_figures"] = project_figure_paths
            logger.info(f"Created {len(project_figure_paths)} project embedding figures")

            # Step 4.3: Projection Plot (fit + project together) if cross-projection mode
            if (
                "fit_embedding_file" in results
                and self.projection_plot_fit_column
                and self.projection_plot_project_column
            ):
                logger.info("Creating projection plot (fit + project together)...")

                projection_plot_path = (
                    embedding_figures_dir
                    / f"{embedding}_projection_fit_{self.projection_plot_fit_column}_project_{self.projection_plot_project_column}.png"
                )

                try:
                    plot_projection(
                        fit_embedding=results["fit_embedding_file"],
                        project_embedding=results["embedding_file"],
                        fit_labels=self.fit_labels,
                        project_labels=self.project_labels,
                        fit_colormap=self.fit_colormap,
                        project_colormap=self.project_colormap,
                        output_path=projection_plot_path,
                        fit_label_column=self.projection_plot_fit_column,
                        project_label_column=self.projection_plot_project_column,
                    )

                    results["projection_plot"] = projection_plot_path
                    logger.info(f"Projection plot saved: {projection_plot_path}")
                except Exception as e:
                    logger.warning(f"Failed to create projection plot: {e}")

        # Step 4.5: Admixture-Colored Embedding Visualization (requires embedding to exist)
        if not skip_admixture_visualization and not skip_embedding and not skip_admixture:
            logger.info("=" * 70)
            logger.info("STEP 4.5: ADMIXTURE-COLORED EMBEDDING VISUALIZATION")
            logger.info("=" * 70)

            if "embedding_file" in results and "admixture_dir" in results:
                admix_figures_dir = self.output_dir / "figures" / "admixture"
                admix_figures_dir.mkdir(parents=True, exist_ok=True)

                # Admixture-colored embedding grid plot
                emb_plot_path = admix_figures_dir / "project_admixture_colored_embedding.png"
                plot_admixture_embedding_grid(
                    embedding=results["embedding_file"],
                    q_prefix=results["admixture_dir"] / "project",
                    k_values=range(k_min, k_max + 1),
                    output_path=emb_plot_path,
                )
                results.setdefault("admixture_figures", {})[
                    "admixture_colored_embedding"
                ] = emb_plot_path
                logger.info(f"Saved admixture-colored embedding plot: {emb_plot_path}")
            else:
                logger.warning(
                    "Skipping admixture-colored embedding visualization - missing embedding or admixture data"
                )

        # Step 5: Metrics
        if not skip_metrics and not skip_embedding:
            logger.info("=" * 70)
            logger.info("STEP 5: METRICS")
            logger.info("=" * 70)

            metrics = {}

            metrics_paths = metrics_output_paths(io)
            # Created unconditionally: the output tree has always contained metrics/
            # even when neither metric runs.
            metrics_paths["geographic"].parent.mkdir(parents=True, exist_ok=True)

            # Geographic preservation
            if self.geographic_coords:
                geo_result = run_geographic_metrics_step(
                    embedding_file,
                    self.geographic_coords,
                    metrics_paths["geographic"],
                )
                metrics["geographic"] = geo_result.values

            # Admixture preservation
            if not skip_admixture and "admixture_dir" in results:
                admix_result = run_admixture_metrics_step(
                    embedding_file,
                    results["admixture_dir"] / "project",
                    range(k_min, k_max + 1),
                    metrics_paths["admixture"],
                )
                metrics["admixture"] = admix_result.values

            results["metrics"] = metrics

        # Summary
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.output_dir}")

        return results

    def _get_embedding_model(self, method: str, params: Optional[Dict] = None):
        """Get embedding model instance."""
        if params is None:
            params = {}

        # Set defaults for each method
        if method == "phate":
            defaults = {"n_components": 2, "knn": 25}
            defaults.update(params)
            return PHATE(**defaults)
        elif method == "umap":
            defaults = {"n_components": 2, "n_neighbors": 15}
            defaults.update(params)
            return UMAP(**defaults)
        elif method == "tsne":
            defaults = {"n_components": 2, "perplexity": 30}
            defaults.update(params)
            return TSNE(**defaults)
        elif method == "diffusion_map":
            defaults = {"n_components": 2, "knn": 25}
            defaults.update(params)
            return DiffusionMap(**defaults)
        else:
            raise ValueError(
                f"Unknown embedding method: {method}. "
                "Choose from: phate, umap, tsne, diffusion_map"
            )
