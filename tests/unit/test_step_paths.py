"""Tests for the pure output-path helpers.

These paths are a contract (spec constraint B): every downstream example script
and the pipeline's own checkpoint logic depend on them. If a path here changes,
that is a deliberate, breaking decision.
"""

import json
from pathlib import Path

import pandas as pd

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


class TestCrossCohortFixtures:
    def test_fixture_provides_five_readable_paths(self, cross_cohort_fixtures):
        assert set(cross_cohort_fixtures) == {
            "fit_labels",
            "project_labels",
            "fit_colormap",
            "project_colormap",
            "geographic",
        }
        for p in cross_cohort_fixtures.values():
            assert p.exists(), p

    def test_labels_have_sample_id_and_distinct_group_columns(self, cross_cohort_fixtures):
        fit = pd.read_csv(cross_cohort_fixtures["fit_labels"])
        proj = pd.read_csv(cross_cohort_fixtures["project_labels"])
        assert "sample_id" in fit.columns and "sample_id" in proj.columns
        # fit uses "Population"; project uses "self_described_ancestry" — mirrors the
        # HGDP -> UKBB projection example.
        assert "Population" in fit.columns
        assert "self_described_ancestry" in proj.columns
        assert len(fit) == 50 and len(proj) == 50

    def test_colormaps_key_on_their_cohort_label_column(self, cross_cohort_fixtures):
        fit_cmap = json.loads(cross_cohort_fixtures["fit_colormap"].read_text())
        proj_cmap = json.loads(cross_cohort_fixtures["project_colormap"].read_text())
        assert "Population" in fit_cmap
        assert "self_described_ancestry" in proj_cmap

    def test_geographic_has_coords_for_project_samples(self, cross_cohort_fixtures):
        geo = pd.read_csv(cross_cohort_fixtures["geographic"])
        assert {"sample_id", "latitude", "longitude"} <= set(geo.columns)
        assert len(geo) == 50
