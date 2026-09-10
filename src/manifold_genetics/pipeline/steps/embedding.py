"""In-process embedding step.

``run_embedding()`` is the seam shared by the ``manifold-genetics embed``
subcommand and ``run_embedding_step()``, and ``build_embedding_model()`` is the
one place a method name plus a params dict becomes a model — so the CLI and the
orchestrator cannot drift in how an embedding is parameterised (spec goal 2).

The defaults here reproduce what the CLI path has always produced. In
particular PHATE receives ``n_landmark`` explicitly (``None`` when unset),
because PHATE's own default is 2000 and ``cmd_embed`` has always overridden it.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

import pandas as pd

from ...embeddings import PHATE, TSNE, UMAP, DiffusionMap
from ..config import EmbeddingConfig, IOConfig
from .paths import embedding_output_paths
from .pca import PCAStepResult

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingStepResult",
    "build_embedding_model",
    "embedding_output_paths",
    "run_embedding",
    "run_embedding_step",
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


def build_embedding_model(method: str, params: Optional[Mapping] = None):
    """Construct the embedding model for ``method`` from ``params``.

    Every parameter the CLI path passes is passed here too, including ones whose
    value is ``None`` — omitting them would silently fall back to the embedding
    class's own defaults, which differ (notably ``PHATE(n_landmark=2000)``).
    """
    p = dict(params or {})

    if method == "phate":
        return PHATE(
            n_components=2,
            knn=p.get("knn", 25),
            t=p.get("t", "auto"),
            n_landmark=p.get("n_landmark"),
            random_landmarking=p.get("random_landmarking", False),
            embed_batch_size=p.get("embed_batch_size"),
        )
    if method == "umap":
        return UMAP(
            n_components=2,
            n_neighbors=p.get("n_neighbors", 15),
            min_dist=p.get("min_dist", 0.1),
        )
    if method == "tsne":
        return TSNE(n_components=2, perplexity=p.get("perplexity", 30))
    if method == "diffusion_map":
        return DiffusionMap(n_components=2, knn=p.get("knn", 25))

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


def run_embedding_step(
    io: IOConfig, emb: EmbeddingConfig, *, pca: PCAStepResult
) -> EmbeddingStepResult:
    """Embed the PCA coordinates according to ``emb.input_mode``.

    * ``fit``     — fit and project the fit cohort's PCA coordinates.
    * ``project`` — fit and project the project cohort's PCA coordinates.
    * ``both``    — fit on the fit cohort, apply that embedding to the project
      cohort, and additionally write the fit cohort's own embedding.

    Raises:
        RuntimeError: the PCA coordinates ``emb.input_mode`` needs were not
            produced or found on disk (e.g. ``--skip-pca`` with only one of the
            fit/project PCA files present, while the selected mode needs the
            other one).
    """
    paths = embedding_output_paths(io, emb)
    embedding_file = paths["embedding"]

    if emb.input_mode == "fit":
        logger.info("Embedding fit PCA coordinates only (mode: fit-only)")
        fit_input, project_input = pca.fit_pca, None
    elif emb.input_mode == "project":
        logger.info("Embedding project PCA coordinates only (mode: project-only)")
        fit_input, project_input = pca.project_pca, None
    else:  # "both"
        logger.info("Embedding: fit on fit PCA, apply to project PCA (mode: both)")
        fit_input, project_input = pca.fit_pca, pca.project_pca

    if fit_input is None:
        raise RuntimeError(
            f"Embedding mode {emb.input_mode!r} requires the "
            f"{'project' if emb.input_mode == 'project' else 'fit'} PCA coordinates, "
            "but they were not produced or found on disk."
        )
    if emb.input_mode == "both" and project_input is None:
        raise RuntimeError(
            f"Embedding mode {emb.input_mode!r} requires the project PCA coordinates, "
            "but they were not produced or found on disk."
        )

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
