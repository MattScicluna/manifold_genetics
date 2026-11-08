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
    """

    enabled: bool = False
    method: str = "finestructure"
    chromosomes: Tuple[str, ...] = tuple(str(c) for c in range(1, 23))
    force: bool = False


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
    painting_method: str = "finestructure",
    painting_chromosomes=None,
    force_painting: bool = False,
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
    )

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
    method: str = "finestructure",
    chromosomes=None,
    force: bool = False,
) -> PaintingConfig:
    """Validate the painting settings into a ``PaintingConfig``. No filesystem access.

    Raises:
        ValueError: an unknown method, an ``hc`` run on a chromosome with no known
            genetic length, an unparseable chromosome list, or painting
            enabled without a ``phased_vcf`` containing ``{chr}``.
    """
    # Imported here: the painting package pulls in sklearn, which the rest of
    # this module has no need for.
    from ..painting.matrix import CHROMOSOME_CM
    from ..painting.pbwt import expand_phased_vcf, parse_chromosomes

    if method not in _PAINTING_METHODS:
        raise ValueError(
            f"Unknown painting method: {method!r}. Choose from: {', '.join(_PAINTING_METHODS)}"
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
        enabled=bool(enabled), method=method, chromosomes=names, force=bool(force)
    )
