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
_WITHIN_GROUP_ORDERS = ("chron", "tree", "none", None)


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


def _as_path(value) -> Optional[Path]:
    return None if value is None else Path(value)


def build_configs(
    *,
    fit_plink,
    project_plink,
    output_dir,
    labels=None,
    colormap=None,
    fit_labels=None,
    project_labels=None,
    fit_colormap=None,
    project_colormap=None,
    geographic_coords=None,
    n_pcs: int = 50,
    force_pca: bool = False,
    k_min: int = 2,
    k_max: int = 10,
    admix_threads: Optional[int] = None,
    admix_gpus: Optional[int] = None,
    admix_batch_size: Optional[int] = None,
    embedding: str = "phate",
    embedding_input: str = "both",
    embedding_params: Optional[dict] = None,
    admix_group_column: Optional[str] = None,
    admix_within_group_order: Optional[str] = "chron",
    projection_plot_fit_column: Optional[str] = None,
    projection_plot_project_column: Optional[str] = None,
    skip_pca: bool = False,
    skip_admixture: bool = False,
    skip_embedding: bool = False,
    skip_pca_visualization: bool = False,
    skip_embedding_visualization: bool = False,
    skip_admixture_visualization: bool = False,
    skip_metrics: bool = False,
) -> PipelineConfigs:
    """Resolve loose pipeline kwargs into validated ``PipelineConfigs``.

    Argument-shape validation only; no filesystem access.
    """
    # --- labels: shared OR both separate ---
    if labels:
        eff_fit_labels = fit_labels if fit_labels is not None else labels
        eff_project_labels = project_labels if project_labels is not None else labels
    elif fit_labels and project_labels:
        eff_fit_labels, eff_project_labels = fit_labels, project_labels
    else:
        raise ValueError(
            "Must provide either 'labels' (used for both fit and project) OR both "
            "'fit_labels' and 'project_labels'. Providing only one of 'fit_labels' or "
            "'project_labels' without 'labels' is not allowed."
        )

    # --- colormap: shared OR both separate ---
    if colormap:
        eff_fit_colormap = fit_colormap if fit_colormap is not None else colormap
        eff_project_colormap = project_colormap if project_colormap is not None else colormap
    elif fit_colormap and project_colormap:
        eff_fit_colormap, eff_project_colormap = fit_colormap, project_colormap
    else:
        raise ValueError(
            "Must provide either 'colormap' (used for both fit and project) OR both "
            "'fit_colormap' and 'project_colormap'. Providing only one of "
            "'fit_colormap' or 'project_colormap' without 'colormap' is not allowed."
        )

    # --- enum / range checks ---
    if embedding not in _EMBEDDING_METHODS:
        raise ValueError(
            f"Unknown embedding method: {embedding!r}. "
            f"Choose from: {', '.join(_EMBEDDING_METHODS)}"
        )
    if embedding_input not in _EMBEDDING_INPUT_MODES:
        raise ValueError(
            f"Unknown embedding_input: {embedding_input!r}. "
            f"Choose from: {', '.join(_EMBEDDING_INPUT_MODES)}"
        )
    if admix_within_group_order not in _WITHIN_GROUP_ORDERS:
        raise ValueError(
            f"Unknown admix_within_group_order: {admix_within_group_order!r}. "
            f"Choose from: chron, tree, none"
        )
    if k_min > k_max:
        raise ValueError(f"k_min ({k_min}) must be <= k_max ({k_max})")
    if n_pcs <= 0:
        raise ValueError(f"n_pcs must be a positive integer, got {n_pcs}")

    io = IOConfig(
        fit_plink=Path(fit_plink),
        project_plink=Path(project_plink),
        output_dir=Path(output_dir),
        fit_labels=Path(eff_fit_labels),
        project_labels=Path(eff_project_labels),
        fit_colormap=Path(eff_fit_colormap),
        project_colormap=Path(eff_project_colormap),
        geographic_coords=_as_path(geographic_coords),
    )
    return PipelineConfigs(
        io=io,
        pca=PCAConfig(n_pcs=n_pcs, force=force_pca),
        admixture=AdmixtureConfig(
            k_min=k_min,
            k_max=k_max,
            threads=admix_threads,
            num_gpus=admix_gpus,
            batch_size=admix_batch_size,
        ),
        embedding=EmbeddingConfig(
            method=embedding,
            input_mode=embedding_input,
            params=dict(embedding_params or {}),
        ),
        viz=VizConfig(
            admix_group_column=admix_group_column,
            admix_within_group_order=admix_within_group_order,
            projection_plot_fit_column=projection_plot_fit_column,
            projection_plot_project_column=projection_plot_project_column,
        ),
        skips=SkipConfig(
            skip_pca=skip_pca,
            skip_admixture=skip_admixture,
            skip_embedding=skip_embedding,
            skip_pca_visualization=skip_pca_visualization,
            skip_embedding_visualization=skip_embedding_visualization,
            skip_admixture_visualization=skip_admixture_visualization,
            skip_metrics=skip_metrics,
        ),
    )
