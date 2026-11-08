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
_EMBEDDING_INIT = ("none", "genotype")
_PAINTING_COMBINE = ("concat", "kernel_sum", "kernel_product")
_PAINTING_COMBINE_NORMS = ("degree", "markov")


def check_embedding_init(method: str, params: Optional[dict], painting_enabled: bool) -> None:
    """Validate the EXPERIMENTAL ``init`` and ``init_blend`` embedding params.

    ``init: genotype`` starts painting PHATE's final MDS from the genotype-PCA
    PHATE layout, so it needs painting on and PHATE as the method.
    ``init_blend`` (alpha in [0, 1]) needs ``init: genotype``.

    Raises:
        ValueError: an unknown ``init``, an ``init_blend`` out of range or without
            ``init: genotype``, or ``init`` with painting off or another method.
    """
    params = params or {}
    init = params.get("init")
    init = "none" if init is None else init
    blend = params.get("init_blend")
    if init not in _EMBEDDING_INIT:
        raise ValueError(
            f"Unknown embedding init: {init!r}. Choose from: {', '.join(_EMBEDDING_INIT)}"
        )
    if blend is not None:
        if isinstance(blend, bool) or not isinstance(blend, (int, float)) or not 0 <= blend <= 1:
            raise ValueError(f"embedding init_blend must be a number in [0, 1], got {blend!r}")
        if init != "genotype":
            raise ValueError(
                "embedding init_blend blends in the genotype layout, so it needs init: genotype"
            )
    if init == "genotype":
        if method != "phate":
            raise ValueError(
                f"embedding init: genotype works only with method phate, not {method!r}"
            )
        if not painting_enabled:
            raise ValueError(
                "embedding init: genotype starts the painting PHATE from the genotype "
                "PHATE layout, so it needs painting enabled (painting: enabled: true)"
            )


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
    """

    enabled: bool = False
    method: str = "hc"
    chromosomes: Tuple[str, ...] = tuple(str(c) for c in range(1, 23))
    force: bool = False
    missing: str = "error"
    pbwt_dir: Optional[Path] = None
    # EXPERIMENTAL: combine the top ``combine_pcs`` genotype PCs with the top
    # painting PCs (see docs/painting.md). ``concat`` embeds the two blocks side
    # by side, each scaled to equal weight. ``kernel_sum`` adds PHATE's two
    # affinity matrices, ``combine_weight`` being the genotype's share and
    # ``combine_norm`` how each is scaled first. ``kernel_product`` is
    # alternating diffusion. None is off.
    combine_genotype: Optional[str] = None
    combine_pcs: int = 10
    combine_weight: float = 0.5
    combine_norm: str = "degree"


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
    painting_method: str = "hc",
    painting_chromosomes=None,
    force_painting: bool = False,
    painting_missing: str = "error",
    painting_pbwt_dir=None,
    painting_combine_genotype: Optional[str] = None,
    painting_combine_pcs: int = 10,
    painting_combine_weight: float = 0.5,
    painting_combine_norm: str = "degree",
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
        missing=painting_missing,
        pbwt_dir=painting_pbwt_dir,
        combine_genotype=painting_combine_genotype,
        combine_pcs=painting_combine_pcs,
        combine_weight=painting_combine_weight,
        combine_norm=painting_combine_norm,
        n_pcs=n_pcs,
    )
    check_embedding_init(embedding, embedding_params, painting_cfg.enabled)
    check_painting_combine(painting_cfg, embedding, embedding_input, embedding_params)

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
    method: str = "hc",
    chromosomes=None,
    force: bool = False,
    missing: str = "error",
    pbwt_dir=None,
    combine_genotype: Optional[str] = None,
    combine_pcs: int = 10,
    combine_weight: float = 0.5,
    combine_norm: str = "degree",
    n_pcs: Optional[int] = None,
) -> PaintingConfig:
    """Validate the painting settings into a ``PaintingConfig``. No filesystem access.

    Raises:
        ValueError: an unknown method or ``missing`` policy, an ``hc`` run on a
            chromosome with no known genetic length, an unparseable chromosome list, or painting
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
    if missing not in _PAINTING_MISSING:
        raise ValueError(
            f"Unknown painting missing policy: {missing!r}. Choose from: "
            f"{', '.join(_PAINTING_MISSING)}"
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
    if combine_genotype in (None, False, "none"):
        combine_genotype = None
    elif combine_genotype not in _PAINTING_COMBINE:
        raise ValueError(
            f"Unknown painting combine_genotype: {combine_genotype!r}. Choose from: "
            f"none, {', '.join(_PAINTING_COMBINE)}"
        )
    if combine_genotype is not None:
        if not enabled:
            raise ValueError(
                "painting combine_genotype combines the genotype PCA with the painting, "
                "so it needs painting enabled (painting: enabled: true)"
            )
        if isinstance(combine_pcs, bool) or not isinstance(combine_pcs, int) or combine_pcs < 1:
            raise ValueError(
                f"painting combine_pcs must be a positive integer, got {combine_pcs!r}"
            )
        if n_pcs is not None and combine_pcs > n_pcs:
            raise ValueError(
                f"painting combine_pcs ({combine_pcs}) is more than pca n_pcs ({n_pcs}), "
                "the number of components each view has"
            )
        if (
            isinstance(combine_weight, bool)
            or not isinstance(combine_weight, (int, float))
            or not 0 <= combine_weight <= 1
        ):
            raise ValueError(
                f"painting combine_weight must be a number in [0, 1], got {combine_weight!r}"
            )
        if combine_norm not in _PAINTING_COMBINE_NORMS:
            raise ValueError(
                f"Unknown painting combine_norm: {combine_norm!r}. Choose from: "
                f"{', '.join(_PAINTING_COMBINE_NORMS)}"
            )
    return PaintingConfig(
        enabled=bool(enabled),
        method=method,
        chromosomes=names,
        force=bool(force),
        missing=missing,
        pbwt_dir=_as_path(pbwt_dir),
        combine_genotype=combine_genotype,
        combine_pcs=int(combine_pcs),
        combine_weight=float(combine_weight),
        combine_norm=combine_norm,
    )


def check_painting_combine(
    painting: PaintingConfig, method: str, input_mode: str, params: Optional[dict]
) -> None:
    """The kernel modes of ``combine_genotype`` build PHATE's own kernels, embed
    one cohort and have no landmarks, so they need PHATE, an ``input_mode`` other
    than ``both``, and ``n_landmark`` unset.

    Raises:
        ValueError: a kernel mode with another method, ``both``, or landmarks.
    """
    mode = painting.combine_genotype
    if mode is None or not mode.startswith("kernel"):
        return
    if method != "phate":
        raise ValueError(
            f"painting combine_genotype: {mode} combines PHATE's affinity matrices, so it "
            f"works only with embedding method phate, not {method!r}. Use concat for "
            "other methods."
        )
    if input_mode == "both":
        raise ValueError(
            f"painting combine_genotype: {mode} cannot place a second cohort in the "
            "embedding, so it needs embedding input_mode fit or project, not both "
            "(the projection preset)"
        )
    if (params or {}).get("n_landmark") is not None:
        raise ValueError(
            f"painting combine_genotype: {mode} runs exact PHATE, so it needs " "n_landmark: none"
        )
    if ((params or {}).get("init") or "none") != "none":
        raise ValueError(
            f"painting combine_genotype: {mode} and embedding init cannot be used together"
        )
