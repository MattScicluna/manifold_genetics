"""
Pipeline orchestrator for end-to-end genetic analysis.

Coordinates PCA, Admixture, Embeddings, Visualization, and Metrics.
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from ..pca import PCA
from ..admixture import NeuralAdmixture
from ..embeddings import PHATE, UMAP, TSNE, DiffusionMap
from ..visualization import (
    visualize,
    plot_pca_pairs,
    plot_admixture_bar_grid,
    plot_admixture_embedding_grid,
)
from ..utils.io import read_colormap
from ..metrics import compute_geographic_preservation, compute_admixture_preservation

logger = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end pipeline for genetic analysis.

    Examples:
        >>> # Full pipeline
        >>> pipeline = Pipeline(
        ...     fit_plink_prefix="data/fit_subset",
        ...     transform_plink_prefix="data/transform_subset",
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
        transform_plink_prefix: Union[str, Path],
        labels: Union[str, Path],
        colormap: Union[str, Path],
        output_dir: Union[str, Path],
        geographic_coords: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize pipeline.

        Args:
            fit_plink_prefix: Path to fit subset PLINK files
            transform_plink_prefix: Path to transform subset PLINK files
            labels: Path to labels CSV
            colormap: Path to colormap JSON
            output_dir: Directory for outputs
            geographic_coords: Optional path to geographic coordinates
        """
        self.fit_plink_prefix = Path(fit_plink_prefix)
        self.transform_plink_prefix = Path(transform_plink_prefix)
        self.labels = Path(labels)
        self.colormap = Path(colormap)
        self.output_dir = Path(output_dir)
        self.geographic_coords = (
            Path(geographic_coords) if geographic_coords else None
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        n_pcs: int = 50,
        k_min: int = 2,
        k_max: int = 10,
        embedding: str = "phate",
        embedding_params: Optional[Dict] = None,
        skip_pca: bool = False,
        skip_admixture: bool = False,
        skip_embedding: bool = False,
        skip_visualization: bool = False,
        skip_pca_visualization: bool = False,
        skip_admixture_visualization: bool = False,
        admix_group_column: Optional[str] = None,
        skip_metrics: bool = False,
        admix_threads: Optional[int] = None,
        admix_gpus: Optional[int] = None,
        flashpca_output_dir: Optional[Union[str, Path]] = None,
        neuraladmixture_output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Run full pipeline.

        Args:
            n_pcs: Number of principal components
            k_min: Minimum K for admixture
            k_max: Maximum K for admixture
            embedding: Embedding method ('phate', 'umap', 'tsne', 'diffusion_map')
            embedding_params: Optional parameters for embedding
            skip_pca: Skip PCA step
            skip_admixture: Skip admixture step
            skip_embedding: Skip embedding step
            skip_visualization: Skip embedding visualization step
            skip_pca_visualization: Skip PCA visualization step
            skip_metrics: Skip metrics computation
            admix_threads: Threads to use for neural admixture (None = auto-detect)
            admix_gpus: Number of GPUs for neural admixture (None = auto-detect)

        Returns:
            Dictionary with paths to outputs and computed metrics
        """
        results = {}

        # Step 1: PCA
        if not skip_pca:
            logger.info("=" * 70)
            logger.info("STEP 1: PCA")
            logger.info("=" * 70)

            pca_dir = self.output_dir / "pca"
            pca_dir.mkdir(exist_ok=True)

            fit_pca_file = pca_dir / f"fit_pca_{n_pcs}.csv"
            transform_pca_file = pca_dir / f"transform_pca_{n_pcs}.csv"
            flashpca_dir = pca_dir / "flashpca_outputs"
            flashpca_dir.mkdir(parents=True, exist_ok=True)

            # Check if existing PCA outputs have correct number of components
            force_pca = False
            if transform_pca_file.exists():
                try:
                    existing_pca = pd.read_csv(transform_pca_file, nrows=1)
                    # Count dimension columns (dim_1, dim_2, ...)
                    existing_n_pcs = len([col for col in existing_pca.columns if col.startswith('dim_')])
                    if existing_n_pcs != n_pcs:
                        logger.info(f"PCA component mismatch: existing={existing_n_pcs}, requested={n_pcs}")
                        logger.info("Forcing PCA recomputation...")
                        force_pca = True
                except Exception as e:
                    logger.warning(f"Could not check existing PCA file: {e}")
                    force_pca = True

            logger.info(f"Running PCA via CLI (fit: {self.fit_plink_prefix}, project: {self.transform_plink_prefix})")
            pca_cmd = [
                "manifold-genetics",
                "pca",
                "--fit-plink",
                str(self.fit_plink_prefix),
                "--project-plink",
                str(self.transform_plink_prefix),
                "--fit-output",
                str(fit_pca_file),
                "--project-output",
                str(transform_pca_file),
                "--flashpca-output-dir",
                str(flashpca_dir),
                "--n-pcs",
                str(n_pcs),
            ]
            if force_pca:
                pca_cmd.append("--force")

            subprocess.run(pca_cmd, check=True)

            pca_coords = pd.read_csv(transform_pca_file)

            results["fit_pca_file"] = fit_pca_file
            results["transform_pca_file"] = transform_pca_file
            results["pca_file"] = transform_pca_file
            results["pca_coords"] = pca_coords

        # Step 1.5: PCA Visualization (independent of PCA computation)
        if not skip_pca_visualization:
            logger.info("=" * 70)
            logger.info("STEP 1.5: PCA VISUALIZATION")
            logger.info("=" * 70)

            pca_dir = self.output_dir / "pca"
            pca_file = pca_dir / f"transform_pca_{n_pcs}.csv"

            if pca_file.exists():
                pca_figures_dir = self.output_dir / "figures" / "pca"
                pca_figures_dir.mkdir(parents=True, exist_ok=True)

                logger.info("Plotting PCA pairs grid via plot_pca_pairs")
                # Use plot_pca_pairs to generate a single grid covering PC pairs
                from ..visualization import plot_pca_pairs
                from ..utils.io import read_colormap

                colormap_dict = read_colormap(self.colormap)
                pca_figure_paths = []
                for label_col in colormap_dict.keys():
                    output_path = pca_figures_dir / f"pca_pairs_by_{label_col}.png"
                    plot_path = plot_pca_pairs(
                        pca_coords=pca_file,
                        labels=self.labels,
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

            logger.info("Running admixture via CLI")
            admix_cmd = [
                "manifold-genetics",
                "admixture",
                "--fit-plink",
                str(self.fit_plink_prefix),
                "--project-plink",
                str(self.transform_plink_prefix),
                "--neuraladmixture-output-dir",
                str(admix_checkpoints_dir),
                "--fit-output",
                str(admix_dir / "fit"),
                "--project-output",
                str(admix_dir / "transform"),
                "--k-min",
                str(k_min),
                "--k-max",
                str(k_max),
            ]
            if admix_threads:
                admix_cmd.extend(["--threads", str(admix_threads)])
            if admix_gpus is not None:
                admix_cmd.extend(["--num-gpus", str(admix_gpus)])

            subprocess.run(admix_cmd, check=True)

            fit_q_files = {k: admix_dir / f"fit.{k}.csv" for k in range(k_min, k_max + 1)}
            transform_q_files = {k: admix_dir / f"transform.{k}.csv" for k in range(k_min, k_max + 1)}

            results["admixture_dir"] = admix_dir
            results["admixture_checkpoints_dir"] = admix_checkpoints_dir
            results["fit_q_files"] = fit_q_files
            results["transform_q_files"] = transform_q_files
            results["q_files"] = transform_q_files

            # Admixture-specific visualizations (bar + embedding grid)
            if not skip_admixture_visualization:
                admix_figures_dir = admix_dir / "figures"
                admix_figures_dir.mkdir(parents=True, exist_ok=True)

                cmap_dict = read_colormap(self.colormap)
                # Grouping column: user-specified if provided, else first colormap key
                group_col = admix_group_column
                if not group_col:
                    group_col = next(iter(cmap_dict.keys()))

                # Bar plot (using transform set by default)
                bar_plot_path = admix_figures_dir / "admixture_bars_transform.png"
                plot_admixture_bar_grid(
                    q_prefix=admix_dir / "transform",
                    labels=self.labels,
                    group_column=group_col,
                    k_values=range(k_min, k_max + 1),
                    output_path=bar_plot_path,
                    colormap=cmap_dict,
                )

                # Embedding-based plot (requires embedding step)
                if not skip_embedding:
                    embedding_file = results.get("embedding_file")
                    if embedding_file:
                        emb_plot_path = admix_figures_dir / "admixture_embedding_transform.png"
                        plot_admixture_embedding_grid(
                            embedding=embedding_file,
                            q_prefix=admix_dir / "transform",
                            k_values=range(k_min, k_max + 1),
                            output_path=emb_plot_path,
                        )
                        results.setdefault("admixture_figures", {})["embedding"] = emb_plot_path

                results.setdefault("admixture_figures", {})["bars"] = bar_plot_path

        # Step 3: Embedding
        if not skip_embedding and not skip_pca:
            logger.info("=" * 70)
            logger.info(f"STEP 3: EMBEDDING ({embedding.upper()})")
            logger.info("=" * 70)

            embedding_dir = self.output_dir / "embeddings"
            embedding_dir.mkdir(exist_ok=True)
            embedding_file = embedding_dir / f"{embedding}_2d.csv"

            logger.info("Running embedding via CLI on transform PCA coordinates")
            embed_cmd = [
                "manifold-genetics",
                "embed",
                "--method",
                embedding,
                "--input",
                str(results["transform_pca_file"]),
                "--project-output",
                str(embedding_file),
            ]

            if embedding == "phate":
                embed_cmd.extend(["--knn", str(embedding_params.get("knn", 25))])
                t_val = embedding_params.get("t", "auto")
                embed_cmd.extend(["--t", str(t_val)])
            elif embedding == "umap":
                embed_cmd.extend(["--n-neighbors", str(embedding_params.get("n_neighbors", 15))])
                embed_cmd.extend(["--min-dist", str(embedding_params.get("min_dist", 0.1))])
            elif embedding == "tsne":
                embed_cmd.extend(["--perplexity", str(embedding_params.get("perplexity", 30))])
            elif embedding == "diffusion_map":
                embed_cmd.extend(["--knn", str(embedding_params.get("knn", 25))])

            subprocess.run(embed_cmd, check=True)
            embedding_coords = pd.read_csv(embedding_file)

            results["embedding_file"] = embedding_file
            results["embedding_coords"] = embedding_coords

        # Step 4: Embedding Visualization
        if not skip_visualization and not skip_embedding:
            logger.info("=" * 70)
            logger.info("STEP 4: EMBEDDING VISUALIZATION")
            logger.info("=" * 70)

            embedding_figures_dir = self.output_dir / "figures" / "embeddings"
            embedding_figures_dir.mkdir(parents=True, exist_ok=True)

            figure_paths = visualize(
                embedding=embedding_file,
                labels=self.labels,
                colormap=self.colormap,
                output_dir=embedding_figures_dir,
                output_prefix=embedding,
            )

            results["embedding_figures"] = figure_paths

        # Step 5: Metrics
        if not skip_metrics and not skip_embedding:
            logger.info("=" * 70)
            logger.info("STEP 5: METRICS")
            logger.info("=" * 70)

            metrics = {}

            metrics_dir = self.output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            # Geographic preservation via CLI
            if self.geographic_coords:
                logger.info("Computing geographic preservation via CLI...")
                geo_out = metrics_dir / "geographic.json"
                geo_cmd = [
                    "manifold-genetics",
                    "metrics-geographic",
                    "--embedding",
                    str(embedding_file),
                    "--geographic",
                    str(self.geographic_coords),
                    "--output",
                    str(geo_out),
                ]
                subprocess.run(geo_cmd, check=True)
                with open(geo_out) as f:
                    metrics["geographic"] = json.load(f)

            # Admixture preservation via CLI
            if not skip_admixture and "admixture_dir" in results:
                logger.info("Computing admixture preservation via CLI...")
                admix_out = metrics_dir / "admixture.json"
                admix_cmd = [
                    "manifold-genetics",
                    "metrics-admixture",
                    "--embedding",
                    str(embedding_file),
                    "--q-dir",
                    str(results["admixture_dir"]),
                    "--output",
                    str(admix_out),
                    "--k-min",
                    str(k_min),
                    "--k-max",
                    str(k_max),
                ]
                subprocess.run(admix_cmd, check=True)
                with open(admix_out) as f:
                    metrics["admixture"] = json.load(f)

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
