"""Tests for the pure output-path helpers.

These paths are a contract (spec constraint B): every downstream example script
and the pipeline's own checkpoint logic depend on them. If a path here changes,
that is a deliberate, breaking decision.
"""

from pathlib import Path

from manifold_genetics.pipeline.config import (
    AdmixtureConfig,
    EmbeddingConfig,
    IOConfig,
    PCAConfig,
)
from manifold_genetics.pipeline.steps.paths import (
    admixture_output_paths,
    embedding_output_paths,
    metrics_output_paths,
    pca_output_paths,
)

OUT = Path("/work/results")


def _io():
    return IOConfig(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=OUT,
        fit_labels=Path("fl.csv"),
        project_labels=Path("pl.csv"),
        fit_colormap=Path("fc.json"),
        project_colormap=Path("pc.json"),
    )


def test_pca_paths_match_documented_layout():
    p = pca_output_paths(_io(), PCAConfig(n_pcs=50))
    assert p["fit_pca"] == OUT / "pca" / "fit_pca_50.csv"
    assert p["project_pca"] == OUT / "pca" / "project_pca_50.csv"
    assert p["flashpca_dir"] == OUT / "pca" / "flashpca_outputs"


def test_pca_paths_track_n_pcs():
    assert pca_output_paths(_io(), PCAConfig(n_pcs=20))["project_pca"] == (
        OUT / "pca" / "project_pca_20.csv"
    )


def test_admixture_paths_match_documented_layout():
    p = admixture_output_paths(_io(), AdmixtureConfig(k_min=2, k_max=5))
    assert p["dir"] == OUT / "admixture"
    assert p["checkpoints_dir"] == OUT / "admixture" / "checkpoints"
    assert p["fit_prefix"] == OUT / "admixture" / "fit"
    assert p["project_prefix"] == OUT / "admixture" / "project"


def test_admixture_per_k_csv_derives_from_project_prefix():
    p = admixture_output_paths(_io(), AdmixtureConfig(k_min=2, k_max=3))
    prefix = p["project_prefix"]
    k3 = prefix.parent / f"{prefix.name}.3.csv"
    assert k3 == OUT / "admixture" / "project.3.csv"


def test_embedding_paths_both_mode_has_fit_embedding():
    p = embedding_output_paths(_io(), EmbeddingConfig(method="phate", input_mode="both"))
    assert p["embedding"] == OUT / "embeddings" / "phate_2d.csv"
    assert p["fit_embedding"] == OUT / "embeddings" / "phate_fit_2d.csv"


def test_embedding_paths_single_mode_has_no_fit_embedding():
    for mode in ("fit", "project"):
        p = embedding_output_paths(_io(), EmbeddingConfig(method="umap", input_mode=mode))
        assert p["embedding"] == OUT / "embeddings" / "umap_2d.csv"
        assert "fit_embedding" not in p


def test_metrics_paths_match_documented_layout():
    p = metrics_output_paths(_io())
    assert p["geographic"] == OUT / "metrics" / "geographic.json"
    assert p["admixture"] == OUT / "metrics" / "admixture.json"


def test_helpers_do_no_io(tmp_path):
    io = IOConfig(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=tmp_path / "never_created",
        fit_labels=Path("fl.csv"),
        project_labels=Path("pl.csv"),
        fit_colormap=Path("fc.json"),
        project_colormap=Path("pc.json"),
    )
    pca_output_paths(io, PCAConfig())
    admixture_output_paths(io, AdmixtureConfig())
    embedding_output_paths(io, EmbeddingConfig())
    metrics_output_paths(io)
    assert not (tmp_path / "never_created").exists()
