"""In-process embedding step.

``run_embedding()`` is the seam shared by the ``manifold-genetics embed``
subcommand and ``run_embedding_step()``, and ``build_embedding_model()`` is the
one place a method name plus a params dict becomes a model — so the CLI and the
orchestrator cannot drift in how an embedding is parameterised (spec goal 2).

The defaults here reproduce what the CLI path has always produced. In
particular PHATE receives ``n_landmark`` explicitly (``None`` when unset),
because PHATE's own default is 2000 and ``cmd_embed`` has always overridden it.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

import pandas as pd

from ...embeddings import PHATE, TSNE, UMAP, DiffusionMap, InitializedPHATE
from ...embeddings.phate import CombinedKernelPHATE
from ...utils.validation import validate_embedding_csv
from ..config import EmbeddingConfig, IOConfig
from .paths import embedding_init_output_paths, embedding_output_paths
from .pca import PCAStepResult

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingStepResult",
    "build_embedding_model",
    "embedding_output_paths",
    "run_embedding",
    "run_embedding_step",
    "run_genotype_init_embedding",
    "run_kernel_combined_embedding",
    "KernelCombine",
    "select_embedding_inputs",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class EmbeddingStepResult:
    """Typed outputs of the embedding step.

    ``fit_embedding_file`` is set only in ``both`` mode. ``coords_df`` is
    excluded from comparison: DataFrame ``==`` is elementwise and would make a
    generated ``__eq__`` raise.
    """

    embedding_file: Path
    fit_embedding_file: Optional[Path] = None
    coords_df: Optional[pd.DataFrame] = field(default=None, compare=False)
    skipped: bool = False


def _phate_kwargs(p: Mapping, n_components: int) -> dict:
    return dict(
        n_components=n_components,
        knn=p.get("knn", 25),
        t=p.get("t", "auto"),
        n_landmark=p.get("n_landmark"),
        random_landmarking=p.get("random_landmarking", False),
        embed_batch_size=p.get("embed_batch_size"),
        gamma=p.get("gamma", 1.0),
    )


@dataclass(frozen=True)
class KernelCombine:
    """EXPERIMENTAL ``painting: combine_genotype: kernel_sum|kernel_product``.

    ``genotype`` is the genotype PCA of the cohorts the painting covers. The
    top ``n_pcs`` of each view build its PHATE kernel, and the kernels are
    combined as :func:`~manifold_genetics.embeddings.phate.combine_kernels`
    describes.
    """

    genotype: PCAStepResult
    mode: str = "kernel_sum"
    weight: float = 0.5
    norm: str = "degree"
    n_pcs: int = 10


def build_embedding_model(
    method: str, params: Optional[Mapping] = None, init_layout: Optional[pd.DataFrame] = None
):
    """Construct the embedding model for ``method`` from ``params``.

    Every parameter the CLI path passes is passed here too, including ones whose
    value is ``None`` — omitting them would silently fall back to the embedding
    class's own defaults, which differ (notably ``PHATE(n_landmark=2000)``).

    ``params["n_components"]`` selects the embedding dimensionality and defaults
    to 2. Setting it to 3 produces ``dim_1, dim_2, dim_3`` output, which
    ``plot_embedding_3d`` renders.

    EXPERIMENTAL: ``init_layout`` (sample_id plus one column per component)
    makes PHATE start its final MDS from that layout, with
    ``params["init_blend"]`` as the optional blend weight. See
    :class:`~manifold_genetics.embeddings.phate.InitializedPHATE`.
    """
    p = dict(params or {})

    # Every method accepts n_components; 2 keeps the historical default, so an
    # existing config that never mentions it produces exactly what it did before.
    n_components = int(p.get("n_components", 2))
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    if method == "phate":
        kwargs = _phate_kwargs(p, n_components)
        if init_layout is not None:
            return InitializedPHATE(
                init_layout=init_layout, init_blend=p.get("init_blend"), **kwargs
            )
        return PHATE(**kwargs)
    if init_layout is not None:
        raise ValueError(f"An init layout works only with method phate, not {method!r}")
    if method == "umap":
        return UMAP(
            n_components=n_components,
            n_neighbors=p.get("n_neighbors", 15),
            min_dist=p.get("min_dist", 0.1),
        )
    if method == "tsne":
        return TSNE(n_components=n_components, perplexity=p.get("perplexity", 30))
    if method == "diffusion_map":
        return DiffusionMap(n_components=n_components, knn=p.get("knn", 25))

    raise ValueError(
        f"Unknown embedding method: {method}. Choose from: phate, umap, tsne, diffusion_map"
    )


def run_embedding(
    fit_input: PathLike,
    project_input: Optional[PathLike] = None,
    *,
    project_output: PathLike,
    fit_output: Optional[PathLike] = None,
    method: str = "phate",
    params: Optional[Mapping] = None,
) -> pd.DataFrame:
    """Fit an embedding on ``fit_input`` and project ``project_input`` into it.

    ``project_input=None`` means "project the fitted dataset itself" — fit then
    transform the same CSV. This is deliberately not ``fit_transform``.

    Returns the DataFrame written to ``project_output``.
    """
    if fit_input is not None:
        validate_embedding_csv(fit_input)
    if project_input is not None and project_input != fit_input:
        validate_embedding_csv(project_input)

    if project_input is None:
        project_input = fit_input

    model = build_embedding_model(method, params)
    model.fit(fit_input)

    if fit_output:
        fit_output = Path(fit_output)
        fit_output.parent.mkdir(parents=True, exist_ok=True)
        model.transform(fit_input).to_csv(fit_output, index=False)

    embedding = model.transform(project_input)
    project_output = Path(project_output)
    project_output.parent.mkdir(parents=True, exist_ok=True)
    embedding.to_csv(project_output, index=False)
    return embedding


def select_embedding_inputs(input_mode: str, pca: PCAStepResult):
    """The ``(fit_input, project_input)`` pair ``input_mode`` embeds.

    Shared by the embedding step and the embedding-grid figures, so a grid
    always shows the same data as the run's own embedding.

    Raises:
        RuntimeError: the PCA coordinates the mode needs were not produced or
            found on disk.
    """
    if input_mode == "fit":
        logger.info("Embedding fit PCA coordinates only (mode: fit-only)")
        fit_input, project_input = pca.fit_pca, None
    elif input_mode == "project":
        logger.info("Embedding project PCA coordinates only (mode: project-only)")
        fit_input, project_input = pca.project_pca, None
    else:  # "both"
        logger.info("Embedding: fit on fit PCA, apply to project PCA (mode: both)")
        fit_input, project_input = pca.fit_pca, pca.project_pca

    if fit_input is None:
        raise RuntimeError(
            f"Embedding mode {input_mode!r} requires the "
            f"{'project' if input_mode == 'project' else 'fit'} PCA coordinates, "
            "but they were not produced or found on disk."
        )
    if input_mode == "both" and project_input is None:
        raise RuntimeError(
            f"Embedding mode {input_mode!r} requires the project PCA coordinates, "
            "but they were not produced or found on disk."
        )
    return fit_input, project_input


def run_embedding_step(
    io: IOConfig,
    emb: EmbeddingConfig,
    *,
    pca: PCAStepResult,
    init_pca: Optional[PCAStepResult] = None,
    kernel: Optional[KernelCombine] = None,
) -> EmbeddingStepResult:
    """Embed the PCA coordinates according to ``emb.input_mode``.

    * ``fit``     — fit and project the fit cohort's PCA coordinates.
    * ``project`` — fit and project the project cohort's PCA coordinates.
    * ``both``    — fit on the fit cohort, apply that embedding to the project
      cohort, and additionally write the fit cohort's own embedding.

    EXPERIMENTAL: with ``emb.params["init"] == "genotype"``, ``pca`` holds the
    painting coordinates and ``init_pca`` the genotype PCA of the same cohorts.
    See :func:`run_genotype_init_embedding`. With ``kernel``, PHATE runs on
    the combination of the painting's and the genotype PCA's kernels. See
    :func:`run_kernel_combined_embedding`.

    Raises:
        RuntimeError: the PCA coordinates ``emb.input_mode`` needs were not
            produced or found on disk (e.g. ``--skip-pca`` with only one of the
            fit/project PCA files present, while the selected mode needs the
            other one).
    """
    if kernel is not None:
        if (emb.params.get("init") or "none") != "none":
            raise ValueError("embedding init and a kernel combine_genotype cannot be combined")
        return run_kernel_combined_embedding(io, emb, pca=pca, kernel=kernel)
    if (emb.params.get("init") or "none") == "genotype":
        if init_pca is None:
            raise RuntimeError("embedding init: genotype needs the genotype PCA coordinates")
        return run_genotype_init_embedding(io, emb, pca=pca, genotype_pca=init_pca)

    paths = embedding_output_paths(io, emb)
    embedding_file = paths["embedding"]

    fit_input, project_input = select_embedding_inputs(emb.input_mode, pca)

    fit_embedding_file = paths.get("fit_embedding") if project_input is not None else None

    coords = run_embedding(
        fit_input,
        project_input,
        fit_output=fit_embedding_file,
        project_output=embedding_file,
        method=emb.method,
        params=emb.params,
    )

    return EmbeddingStepResult(
        embedding_file=embedding_file,
        fit_embedding_file=fit_embedding_file,
        coords_df=coords,
    )


def _select_inputs(emb: EmbeddingConfig, coords: PCAStepResult, what: str):
    """The (fit, project) CSVs ``emb.input_mode`` embeds. Project is None outside ``both``."""
    if emb.input_mode == "fit":
        fit_input, project_input = coords.fit_pca, None
    elif emb.input_mode == "project":
        fit_input, project_input = coords.project_pca, None
    else:  # "both"
        fit_input, project_input = coords.fit_pca, coords.project_pca
    if fit_input is None or (emb.input_mode == "both" and project_input is None):
        raise RuntimeError(
            f"Embedding mode {emb.input_mode!r} requires {what} coordinates that were "
            "not produced or found on disk."
        )
    return fit_input, project_input


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_genotype_init_embedding(
    io: IOConfig, emb: EmbeddingConfig, *, pca: PCAStepResult, genotype_pca: PCAStepResult
) -> EmbeddingStepResult:
    """EXPERIMENTAL: painting PHATE started from the genotype PHATE layout.

    1. Fit PHATE, with the same settings, on the genotype PCA of the cohort
       ``emb.input_mode`` selects, and write that layout
       (``<method>_init_genotype_<n>d.csv``).
    2. Fit PHATE on the painting coordinates (``pca``), but run its final MDS
       from the genotype layout of the fit samples (or on the blend, with
       ``init_blend``), and write it where the embedding always goes.
    3. Write the Procrustes disparity between the two layouts to
       ``<method>_init_report_<n>d.json``.
    """
    paths = embedding_output_paths(io, emb)
    init_paths = embedding_init_output_paths(io, emb)
    geno_fit, geno_project = _select_inputs(emb, genotype_pca, "genotype PCA")
    paint_fit, paint_project = _select_inputs(emb, pca, "painting")
    for f in {geno_fit, geno_project, paint_fit, paint_project} - {None}:
        validate_embedding_csv(f)

    logger.info("EXPERIMENTAL init: genotype. Embedding the genotype PCA first")
    geno_model = build_embedding_model(emb.method, emb.params)
    geno_model.fit(geno_fit)
    geno_fit_layout = geno_model.transform(geno_fit)
    if geno_project is None:
        _write(geno_fit_layout, init_paths["genotype_embedding"])
    else:
        _write(geno_fit_layout, init_paths["genotype_fit_embedding"])
        _write(geno_model.transform(geno_project), init_paths["genotype_embedding"])

    logger.info("EXPERIMENTAL init: genotype. Painting PHATE from the genotype layout")
    model = build_embedding_model(emb.method, emb.params, init_layout=geno_fit_layout)
    model.fit(paint_fit)

    fit_embedding_file = paths.get("fit_embedding") if paint_project is not None else None
    if fit_embedding_file is not None:
        _write(model.transform(paint_fit), fit_embedding_file)
    coords = model.transform(paint_project if paint_project is not None else paint_fit)
    _write(coords, paths["embedding"])

    report = {
        "init": "genotype",
        "init_blend": model.init_blend,
        "procrustes_disparity_fit_samples": model.init_disparity,
        "n_fit_samples": int(len(geno_fit_layout)),
        "n_landmark": model.n_landmark,
        "genotype_embedding": str(init_paths["genotype_embedding"]),
    }
    init_paths["report"].write_text(json.dumps(report, indent=2) + "\n")
    logger.info(f"EXPERIMENTAL init: genotype. Report in {init_paths['report']}")

    return EmbeddingStepResult(
        embedding_file=paths["embedding"],
        fit_embedding_file=fit_embedding_file,
        coords_df=coords,
    )


def _top_pcs(path: PathLike, n: int, what: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str})
    if df.shape[1] - 1 < n:
        raise ValueError(
            f"painting combine_pcs is {n}, but the {what} coordinates in {path} have only "
            f"{df.shape[1] - 1} components"
        )
    return df.iloc[:, : n + 1]


def run_kernel_combined_embedding(
    io: IOConfig, emb: EmbeddingConfig, *, pca: PCAStepResult, kernel: KernelCombine
) -> EmbeddingStepResult:
    """EXPERIMENTAL: PHATE on the combined painting and genotype kernels.

    Embeds the one cohort ``emb.input_mode`` selects (``fit`` or ``project``):
    there is no out-of-sample extension, so ``both`` is refused. The rows are
    the painting's samples, and each needs a genotype PCA row.
    """
    if emb.method != "phate":
        raise ValueError(f"A kernel combine_genotype needs method phate, not {emb.method!r}")
    if emb.input_mode == "both":
        raise ValueError("A kernel combine_genotype embeds one cohort: input_mode fit or project")
    paint_file, _ = _select_inputs(emb, pca, "painting")
    geno_file, _ = _select_inputs(emb, kernel.genotype, "genotype PCA")
    paint = _top_pcs(paint_file, kernel.n_pcs, "painting")
    geno = _top_pcs(geno_file, kernel.n_pcs, "genotype")

    p = dict(emb.params)
    model = CombinedKernelPHATE(
        second_view=geno,
        mode=kernel.mode,
        weight=kernel.weight,
        norm=kernel.norm,
        **_phate_kwargs(p, int(p.get("n_components", 2))),
    )
    coords = model.fit_transform(paint)
    path = embedding_output_paths(io, emb)["embedding"]
    _write(coords, path)
    logger.info(
        f"EXPERIMENTAL combine_genotype: {kernel.mode} (genotype weight {kernel.weight}, "
        f"norm {kernel.norm}, {kernel.n_pcs} PCs per view) in {path}"
    )
    return EmbeddingStepResult(embedding_file=path, coords_df=coords)
