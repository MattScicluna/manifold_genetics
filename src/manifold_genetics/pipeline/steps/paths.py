"""Pure output-path helpers — the single source of truth for where each pipeline
step writes. No I/O; deterministic from the config arguments.

These paths are a contract: downstream example scripts and the pipeline's
checkpoint logic depend on them (spec constraint B).
"""

from pathlib import Path
from typing import Dict

from ..config import (
    GENOTYPE_WEIGHT_COMPONENTS,
    AdmixtureConfig,
    EmbeddingConfig,
    IOConfig,
    PaintingConfig,
    PCAConfig,
)


def pca_output_paths(io: IOConfig, pca: PCAConfig) -> Dict[str, Path]:
    d = io.output_dir / "pca"
    return {
        "fit_pca": d / f"fit_pca_{pca.n_pcs}.csv",
        "project_pca": d / f"project_pca_{pca.n_pcs}.csv",
        "flashpca_dir": d / "flashpca_outputs",
    }


def painting_output_paths(io: IOConfig, pca: PCAConfig, painting: PaintingConfig) -> Dict:
    """Where the painting step writes. The coordinate CSVs mirror the PCA step's.

    The method is in every name that depends on it, so a run with ``hc`` and
    one with ``finestructure`` in the same output directory do not overwrite
    each other, and ``skip: pca`` finds only the method it was asked for.
    pbwt's own output does not depend on the method, so it is shared, and
    ``painting.pbwt_dir`` can move it outside the output directory.

    ``pbwt_roots`` maps each chromosome to the root pbwt writes under, so its
    files are ``<root>.chunkcounts.out`` and so on.
    """
    d = io.output_dir / "painting"
    pbwt_dir = painting.pbwt_dir if painting.pbwt_dir is not None else d / "pbwt"
    method, n = painting.method, pca.n_pcs
    return {
        "dir": d,
        "pbwt_dir": pbwt_dir,
        "pbwt_roots": {c: pbwt_dir / f"chr{c}" for c in painting.chromosomes},
        "coancestry": d / f"coancestry_{method}.csv",
        "fit_pca": d / f"fit_painting_{method}_pca_{n}.csv",
        "project_pca": d / f"project_painting_{method}_pca_{n}.csv",
        # painting.genotype_weight 1: the top genotype PCs the run embeds.
        "fit_genotype_top": d / f"fit_genotype_pca_top{GENOTYPE_WEIGHT_COMPONENTS}.csv",
        "project_genotype_top": d / f"project_genotype_pca_top{GENOTYPE_WEIGHT_COMPONENTS}.csv",
    }


def admixture_output_paths(io: IOConfig, admix: AdmixtureConfig) -> Dict[str, object]:
    d = io.output_dir / "admixture"
    return {
        "dir": d,
        "checkpoints_dir": d / "checkpoints",
        "fit_prefix": d / "fit",
        "project_prefix": d / "project",
        "fit_q_files": {k: d / f"fit.{k}.csv" for k in range(admix.k_min, admix.k_max + 1)},
        "project_q_files": {k: d / f"project.{k}.csv" for k in range(admix.k_min, admix.k_max + 1)},
    }


def embedding_output_paths(io: IOConfig, emb: EmbeddingConfig) -> Dict[str, Path]:
    # The dimensionality is in the filename, so a 3-D run does not overwrite a
    # 2-D one and neither is mislabelled. n_components defaults to 2, which
    # keeps the historical `<method>_2d.csv` name the checkpoint logic and the
    # example scripts depend on.
    n_components = int(emb.params.get("n_components", 2))
    d = io.output_dir / "embeddings"
    paths = {"embedding": d / f"{emb.method}_{n_components}d.csv"}
    if emb.input_mode == "both":
        paths["fit_embedding"] = d / f"{emb.method}_fit_{n_components}d.csv"
    return paths


def embedding_grid_output_paths(io: IOConfig) -> Dict[str, Path]:
    """Where the comparison grids keep their embeddings and figures.

    Separate from ``embeddings/<method>_<n>d.csv`` so a grid panel never
    overwrites, or is mistaken for, the run's own embedding.
    """
    return {
        "embeddings": io.output_dir / "embeddings" / "grids",
        "figures": io.output_dir / "figures" / "embeddings" / "grids",
    }


def metrics_output_paths(io: IOConfig) -> Dict[str, Path]:
    d = io.output_dir / "metrics"
    return {"geographic": d / "geographic.json", "admixture": d / "admixture.json"}


def figure_output_paths(io: IOConfig) -> Dict[str, Path]:
    """Where each figure family is written. Pure; creates nothing."""
    root = io.output_dir / "figures"
    admixture = root / "admixture"
    return {
        "root": root,
        "pca": root / "pca",
        "embeddings": root / "embeddings",
        "admixture": admixture,
        "admixture_bars": admixture / "project_bars.png",
        "admixture_colored_embedding": admixture / "project_admixture_colored_embedding.png",
        "admixture_colored_embedding_3d": admixture / "project_admixture_colored_embedding_3d.html",
    }
