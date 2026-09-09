"""Pure output-path helpers — the single source of truth for where each pipeline
step writes. No I/O; deterministic from the config arguments.

These paths are a contract: downstream example scripts and the pipeline's
checkpoint logic depend on them (spec constraint B).
"""

from pathlib import Path
from typing import Dict

from ..config import AdmixtureConfig, EmbeddingConfig, IOConfig, PCAConfig


def pca_output_paths(io: IOConfig, pca: PCAConfig) -> Dict[str, Path]:
    d = io.output_dir / "pca"
    return {
        "fit_pca": d / f"fit_pca_{pca.n_pcs}.csv",
        "project_pca": d / f"project_pca_{pca.n_pcs}.csv",
        "flashpca_dir": d / "flashpca_outputs",
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
    d = io.output_dir / "embeddings"
    paths = {"embedding": d / f"{emb.method}_2d.csv"}
    if emb.input_mode == "both":
        paths["fit_embedding"] = d / f"{emb.method}_fit_2d.csv"
    return paths


def metrics_output_paths(io: IOConfig) -> Dict[str, Path]:
    d = io.output_dir / "metrics"
    return {"geographic": d / "geographic.json", "admixture": d / "admixture.json"}
