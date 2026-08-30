"""
Resolved, validated configuration for the manifold-genetics pipeline.

Six frozen sub-config dataclasses plus ``build_configs()``, which turns the
loose keyword arguments of ``run_pipeline`` / the ``pipeline`` CLI subcommand
into these typed objects and performs argument-shape validation (no filesystem
access).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_EMBEDDING_METHODS = ("phate", "umap", "tsne", "diffusion_map")
_EMBEDDING_INPUT_MODES = ("fit", "project", "both")


@dataclass(frozen=True)
class IOConfig:
    """Input data locations, the output directory, and per-cohort labels/colormaps.

    After ``build_configs`` runs, ``fit_*`` and ``project_*`` are always both set
    (to the shared value when the caller passed only ``labels`` / ``colormap``).
    """

    fit_plink: Path
    project_plink: Path
    output_dir: Path
    fit_labels: Path
    project_labels: Path
    fit_colormap: Path
    project_colormap: Path
    geographic_coords: Optional[Path] = None


@dataclass(frozen=True)
class PCAConfig:
    n_pcs: int = 50
    force: bool = False


@dataclass(frozen=True)
class AdmixtureConfig:
    k_min: int = 2
    k_max: int = 10
    threads: Optional[int] = None
    num_gpus: Optional[int] = None
    batch_size: Optional[int] = None


@dataclass(frozen=True)
class EmbeddingConfig:
    method: str = "phate"
    input_mode: str = "both"
    params: dict = field(default_factory=dict)  # method-specific; treated read-only


@dataclass(frozen=True)
class VizConfig:
    admix_group_column: Optional[str] = None
    admix_within_group_order: Optional[str] = "chron"
    projection_plot_fit_column: Optional[str] = None
    projection_plot_project_column: Optional[str] = None


@dataclass(frozen=True)
class SkipConfig:
    skip_pca: bool = False
    skip_admixture: bool = False
    skip_embedding: bool = False
    skip_pca_visualization: bool = False
    skip_embedding_visualization: bool = False
    skip_admixture_visualization: bool = False
    skip_metrics: bool = False


@dataclass(frozen=True)
class PipelineConfigs:
    io: IOConfig
    pca: PCAConfig
    admixture: AdmixtureConfig
    embedding: EmbeddingConfig
    viz: VizConfig
    skips: SkipConfig
