"""
Resolved, validated configuration for the manifold-genetics pipeline.

Seven frozen sub-config dataclasses plus ``build_configs()``, which turns the
loose keyword arguments of ``run_pipeline`` / the ``pipeline`` CLI subcommand
into these typed objects and performs argument-shape validation (no filesystem
access).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

_EMBEDDING_METHODS = ("phate", "umap", "tsne", "diffusion_map")
_EMBEDDING_INPUT_MODES = ("fit", "project", "both")
_WITHIN_GROUP_ORDERS = ("chron", "tree", "none", None)
_PAINTING_METHODS = ("finestructure", "hc")
_PAINTING_MISSING = ("error", "drop")
# Components taken from each of the genotype PCA and the painting when
# painting.genotype_weight mixes them.
GENOTYPE_WEIGHT_COMPONENTS = 10
# PHATE's t when painting provides the embedding input and neither the config
# nor its preset sets t. On painted HGDP+1KGP it beat automatic t and t 3.
PAINTING_DEFAULT_T = 10


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
    # Phased VCFs for haplotype painting, one per chromosome: a path containing
    # "{chr}". Only read when painting is enabled.
    phased_vcf: Optional[Path] = None


@dataclass(frozen=True)
class PaintingConfig:
    """Haplotype painting with pbwt, replacing genotype PCA as the embedding input.

    ``method`` picks the published recipe that turns pbwt's output into PCA
    features: ``finestructure`` (Lawson et al. 2012, chunk counts) or ``hc``
    (Yang et al. 2025, chunk lengths). The number of components is
    the PCA section's ``n_pcs``, so switching painting on changes nothing else
    about the run's shape.

    ``missing`` says what happens to a fit or project sample that the phased
    VCFs lack: ``error`` stops before painting, ``drop`` leaves it out of its
    cohort with a warning. ``pbwt_dir`` is where pbwt's per-chromosome output
    goes, ``<output_dir>/painting/pbwt`` when ``None``. pbwt's output does not
    depend on the method, so runs with different methods or output
    directories can share one.

    ``genotype_weight`` (in [0, 1]) mixes the genotype PCA back in. The top
    ``GENOTYPE_WEIGHT_COMPONENTS`` components of each are concatenated, each
    block scaled so the genotype's share of squared distance is the weight.
    0, the default, embeds the painting alone and runs no genotype PCA. 1
    embeds only the genotype PCA's top components and skips painting.
    """

    enabled: bool = False
    method: str = "hc"
    chromosomes: Tuple[str, ...] = tuple(str(c) for c in range(1, 23))
    force: bool = False
    missing: str = "error"
    pbwt_dir: Optional[Path] = None
    genotype_weight: float = 0.0

    @property
    def paints(self) -> bool:
        """Whether the painting runs: enabled, and not replaced outright by the genotype."""
        return self.enabled and self.genotype_weight < 1


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
    painting: PaintingConfig = field(default_factory=PaintingConfig)


def _as_path(value) -> Optional[Path]:
    return None if value is None else Path(value)


# The run's settings that every PHATE grid panel inherits unless the grid sets
# them. They decide what a panel costs, not what it shows, so a grid must not
# silently fall back to exact, O(n^2) PHATE when the run uses landmarks.
GRID_INHERITED_PHATE_PARAMS = ("n_landmark", "random_landmarking", "embed_batch_size")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_int(value) -> bool:
    return _is_int(value) and value > 0


# Per grid parameter: (accepts the value, what it must be).
_GRID_PARAM_TYPES = {
    "knn": (_is_positive_int, "a positive integer"),
    "n_neighbors": (_is_positive_int, "a positive integer"),
    "t": (lambda v: v == "auto" or _is_positive_int(v), "a positive integer or 'auto'"),
    "n_landmark": (lambda v: v is None or _is_positive_int(v), "a positive integer or none"),
    "random_landmarking": (lambda v: isinstance(v, bool), "true or false"),
    "embed_batch_size": (lambda v: v is None or _is_positive_int(v), "a positive integer or none"),
    "gamma": (_is_number, "a number"),
    "min_dist": (_is_number, "a number"),
    "perplexity": (lambda v: _is_number(v) and v > 0, "a positive number"),
}


def _check_grid_params(method: str, params, where: str) -> None:
    unknown = sorted(set(params) - set(GRID_METHOD_PARAMS[method]))
    if unknown:
        raise ValueError(
            f"Unknown {method} parameter(s) in {where}: {', '.join(unknown)}. "
            f"Choose from: {', '.join(GRID_METHOD_PARAMS[method])}"
        )
    for key, value in params.items():
        accepts, expected = _GRID_PARAM_TYPES[key]
        for v in value if isinstance(value, list) else [value]:
            if not accepts(v):
                raise ValueError(f"{where}: {key} must be {expected}, got {v!r}")


def check_landmarking(params, inherited=None) -> None:
    """Refuse ``random_landmarking`` without ``n_landmark``.

    Landmarking is disabled when ``n_landmark`` is None, so random landmarking
    would silently do nothing. ``params`` may hold lists (a ``phate_grid``
    sweep), and then any combination pairing the two is refused. ``inherited``
    is the run's own embedding parameters: a grid that turns random landmarking
    on and leaves ``n_landmark`` to the run takes the run's. An inherited
    ``random_landmarking`` is not checked here, because a panel without
    landmarks drops it (``grid_phate_params``).
    """

    def as_list(value):
        return value if isinstance(value, list) else [value]

    if not any(as_list(params.get("random_landmarking"))):
        return
    source = params if "n_landmark" in params else (inherited or {})
    if any(n is None for n in as_list(source.get("n_landmark"))):
        raise ValueError(
            "random_landmarking is set but n_landmark is not. Landmarking is disabled "
            "when n_landmark is none, so random_landmarking would have no effect."
        )


def grid_phate_params(inherited, overrides) -> dict:
    """One PHATE grid panel's parameters: ``inherited`` from the run, then the
    panel's own ``overrides``.

    A panel whose ``n_landmark`` is None runs without landmarks, so an inherited
    ``random_landmarking`` is dropped there rather than contradicting it. An
    explicit one is refused earlier, by ``check_landmarking``.
    """
    overrides = dict(overrides or {})
    params = {**(inherited or {}), **overrides}
    if params.get("n_landmark") is None and "random_landmarking" not in overrides:
        params.pop("random_landmarking", None)
    return params


def validate_method_grid(method_grid, run_params=None) -> None:
    """``{method: params or None}`` naming known methods and their parameters.

    ``run_params`` are the run's embedding parameters, which a PHATE panel
    inherits its landmarking from.
    """
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
        if any(isinstance(v, list) for v in (params or {}).values()):
            raise ValueError(
                f"method_grid[{method!r}] takes one value per parameter. "
                "Sweep a PHATE parameter with phate_grid instead."
            )
        _check_grid_params(method, params or {}, "method_grid")
        if method == "phate":
            check_landmarking(params or {}, run_params)


def validate_phate_grid(phate_grid, run_params=None) -> None:
    """``{param: [values] | value}`` with one or two swept (list-valued) parameters.

    ``run_params`` are the run's embedding parameters, which every panel
    inherits its landmarking from.
    """
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
    check_landmarking(phate_grid, run_params)


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
    phased_vcf=None,
    painting: bool = False,
    painting_method: str = "hc",
    painting_chromosomes=None,
    force_painting: bool = False,
    painting_missing: str = "error",
    painting_pbwt_dir=None,
    painting_genotype_weight: float = 0.0,
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
    painting_cfg = build_painting_config(
        phased_vcf=phased_vcf,
        enabled=painting,
        method=painting_method,
        chromosomes=painting_chromosomes,
        force=force_painting,
        missing=painting_missing,
        pbwt_dir=painting_pbwt_dir,
        genotype_weight=painting_genotype_weight,
    )
    if method_grid is not None:
        validate_method_grid(method_grid, embedding_params)
    if phate_grid is not None:
        validate_phate_grid(phate_grid, embedding_params)

    io = IOConfig(
        fit_plink=Path(fit_plink),
        project_plink=Path(project_plink),
        output_dir=Path(output_dir),
        fit_labels=Path(eff_fit_labels),
        project_labels=Path(eff_project_labels),
        fit_colormap=Path(eff_fit_colormap),
        project_colormap=Path(eff_project_colormap),
        geographic_coords=_as_path(geographic_coords),
        phased_vcf=_as_path(phased_vcf),
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
        painting=painting_cfg,
    )


def build_painting_config(
    *,
    phased_vcf=None,
    enabled: bool = False,
    method: str = "hc",
    chromosomes=None,
    force: bool = False,
    missing: str = "error",
    pbwt_dir=None,
    genotype_weight: float = 0.0,
) -> PaintingConfig:
    """Validate the painting settings into a ``PaintingConfig``. No filesystem access.

    Raises:
        ValueError: an unknown method or ``missing`` policy, an ``hc`` run on a
            chromosome with no known genetic length, an unparseable chromosome list, or painting
            enabled without a ``phased_vcf`` containing ``{chr}``, or a
            ``genotype_weight`` that is not a number in [0, 1].
    """
    # Imported here: the painting package pulls in sklearn, which the rest of
    # this module has no need for.
    from ..painting.matrix import CHROMOSOME_CM
    from ..painting.pbwt import expand_phased_vcf, parse_chromosomes

    if method not in _PAINTING_METHODS:
        raise ValueError(
            f"Unknown painting method: {method!r}. Choose from: {', '.join(_PAINTING_METHODS)}"
        )
    if missing not in _PAINTING_MISSING:
        raise ValueError(
            f"Unknown painting missing policy: {missing!r}. Choose from: "
            f"{', '.join(_PAINTING_MISSING)}"
        )
    if (
        isinstance(genotype_weight, bool)
        or not isinstance(genotype_weight, (int, float))
        or not 0 <= genotype_weight <= 1
    ):
        raise ValueError(
            f"painting genotype_weight must be a number from 0 (painting only) to 1 "
            f"(genotype PCA only), got {genotype_weight!r}"
        )
    names = tuple(parse_chromosomes(chromosomes))
    if method == "hc":
        unknown = [c for c in names if c.removeprefix("chr") not in CHROMOSOME_CM]
        if unknown:
            raise ValueError(
                f"Painting method hc needs each chromosome's genetic length, which is known "
                f"only for autosomes 1-22, not {', '.join(unknown)}. Use method: "
                "finestructure for these."
            )
    if enabled:
        if phased_vcf is None:
            raise ValueError(
                "Painting is enabled but no phased_vcf is set. Name one per chromosome in "
                "the data section, for example phased_vcf: data/chr{chr}.vcf.gz"
            )
        expand_phased_vcf(phased_vcf, names)  # raises when {chr} is missing
    return PaintingConfig(
        enabled=bool(enabled),
        method=method,
        chromosomes=names,
        force=bool(force),
        missing=missing,
        pbwt_dir=_as_path(pbwt_dir),
        genotype_weight=float(genotype_weight),
    )
