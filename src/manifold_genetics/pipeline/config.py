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

# What each method's model takes, as build_embedding_model reads it. The grids
# are always 2-D figures, so n_components is not settable in them.
GRID_METHOD_PARAMS = {
    "phate": ("knn", "t", "n_landmark", "random_landmarking", "embed_batch_size", "gamma"),
    "umap": ("n_neighbors", "min_dist"),
    "tsne": ("perplexity",),
    "diffusion_map": ("knn",),
}


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
    # "python" runs in process and needs no binary; "flashpca" shells out to the
    # external one. They agree to 1.5e-7 and write the same artefact set
    # (tests/integration/test_pca_flashpca_parity.py). Python is the default so a
    # pip-installed package works on any platform.
    backend: str = "python"
    # Budgets in GB for the in-process backend; see SklearnPCABackend. Ignored by
    # flashpca, which manages its own memory.
    max_fit_memory_gb: float = 8.0
    max_project_memory_gb: float = 8.0


@dataclass(frozen=True)
class AdmixtureConfig:
    k_min: int = 2
    k_max: int = 10
    threads: Optional[int] = None
    num_gpus: Optional[int] = None
    # neural-admixture batches the whole dataset at once when no batch size is
    # given, which is a bug in the tool; 400 is the workaround this project has
    # used. It is a correctness guard, not a tuning knob, so it defaults on rather
    # than being something each caller must remember. Pass None to opt out.
    batch_size: Optional[int] = 400


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
    # {method: params}: one column per method, each embedded with its own
    # parameters. Treated read-only.
    method_grid: Optional[dict] = None
    # {param: [values]} for up to two PHATE parameters, rows then columns;
    # scalar entries are fixed overrides. Treated read-only.
    phate_grid: Optional[dict] = None


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


def _check_grid_params(method: str, params, where: str) -> None:
    unknown = sorted(set(params) - set(GRID_METHOD_PARAMS[method]))
    if unknown:
        raise ValueError(
            f"Unknown {method} parameter(s) in {where}: {', '.join(unknown)}. "
            f"Choose from: {', '.join(GRID_METHOD_PARAMS[method])}"
        )


def validate_method_grid(method_grid) -> None:
    """``{method: params or None}`` naming known methods and their parameters."""
    if not isinstance(method_grid, dict) or not method_grid:
        raise ValueError("method_grid must map at least one method to its parameters")
    for method, params in method_grid.items():
        if method not in GRID_METHOD_PARAMS:
            raise ValueError(
                f"Unknown method in method_grid: {method!r}. "
                f"Choose from: {', '.join(_EMBEDDING_METHODS)}"
            )
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"method_grid[{method!r}] must be a mapping of parameters")
        _check_grid_params(method, params or {}, "method_grid")


def validate_phate_grid(phate_grid) -> None:
    """``{param: [values] | value}`` with one or two swept (list-valued) parameters."""
    if not isinstance(phate_grid, dict) or not phate_grid:
        raise ValueError("phate_grid must map PHATE parameters to lists of values")
    _check_grid_params("phate", phate_grid, "phate_grid")
    swept = [k for k, v in phate_grid.items() if isinstance(v, list)]
    if not 1 <= len(swept) <= 2:
        raise ValueError(
            "phate_grid must give a list of values for one or two parameters "
            f"(the rows, then the columns); got {len(swept)}: {', '.join(swept) or 'none'}"
        )
    for key in swept:
        if not phate_grid[key]:
            raise ValueError(f"phate_grid[{key!r}] is an empty list")


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
    admix_batch_size: Optional[int] = 400,
    embedding: str = "phate",
    embedding_input: str = "both",
    embedding_params: Optional[dict] = None,
    admix_group_column: Optional[str] = None,
    admix_within_group_order: Optional[str] = "chron",
    projection_plot_fit_column: Optional[str] = None,
    projection_plot_project_column: Optional[str] = None,
    method_grid: Optional[dict] = None,
    phate_grid: Optional[dict] = None,
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
    if method_grid is not None:
        validate_method_grid(method_grid)
    if phate_grid is not None:
        validate_phate_grid(phate_grid)

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
            method_grid=method_grid,
            phate_grid=phate_grid,
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
