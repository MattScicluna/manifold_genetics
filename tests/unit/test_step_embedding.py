"""Tests for the in-process embedding step (pipeline/steps/embedding.py).

This step replaces a `subprocess.run(["manifold-genetics", "embed", ...])` call,
so these assert behaviour — which CSV is fitted, which is transformed, which
file each result lands in, and exactly which kwargs reach the model — rather
than an argv list.

Two behaviours are load-bearing and pinned below:
  * `n_landmark` is always passed to PHATE, as None when unset, because PHATE's
    own default is 2000 and the CLI path has always overridden it.
  * Modes 'fit' and 'project' call fit() then transform() on the same input;
    they do not call fit_transform().
"""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.config import EmbeddingConfig, IOConfig
from manifold_genetics.pipeline.steps.embedding import (
    EmbeddingStepResult,
    build_embedding_model,
    run_embedding,
    run_embedding_step,
)
from manifold_genetics.pipeline.steps.paths import embedding_output_paths
from manifold_genetics.pipeline.steps.pca import PCAStepResult

MODULE = "manifold_genetics.pipeline.steps.embedding"


def coords(n_rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n_rows)],
            "dim_1": [0.1] * n_rows,
            "dim_2": [0.2] * n_rows,
        }
    )


class FakeEmbed:
    """Stand-in for PHATE/UMAP/TSNE/DiffusionMap. Records construction and calls."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeEmbed.instances.append(self)

    def fit(self, X):
        self.calls.append(("fit", str(X)))
        return self

    def transform(self, X):
        self.calls.append(("transform", str(X)))
        return coords()

    def fit_transform(self, *a, **k):
        raise AssertionError("fit_transform must not be used by the embedding step")


@pytest.fixture
def fake_models(monkeypatch):
    FakeEmbed.instances = []
    for name in ("PHATE", "UMAP", "TSNE", "DiffusionMap"):
        monkeypatch.setattr(f"{MODULE}.{name}", FakeEmbed)
    return FakeEmbed


@pytest.fixture
def stub_validate_embedding_csv(monkeypatch):
    """No-op the seam's input validation so fake, non-existent paths can be
    used to test fit/transform call sequencing without touching disk."""
    monkeypatch.setattr(f"{MODULE}.validate_embedding_csv", lambda *a, **k: None)


def make_io(tmp_path) -> IOConfig:
    return IOConfig(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=tmp_path / "out",
        fit_labels=Path("fl.csv"),
        project_labels=Path("pl.csv"),
        fit_colormap=Path("fc.json"),
        project_colormap=Path("pc.json"),
    )


def make_pca_result(tmp_path) -> PCAStepResult:
    d = tmp_path / "out" / "pca"
    d.mkdir(parents=True, exist_ok=True)
    fit_pca, project_pca = d / "fit_pca_10.csv", d / "project_pca_10.csv"
    coords().to_csv(fit_pca, index=False)
    coords().to_csv(project_pca, index=False)
    return PCAStepResult(fit_pca=fit_pca, project_pca=project_pca)


# ---------------------------------------------------------------------------
# build_embedding_model — method string + params -> model instance
# ---------------------------------------------------------------------------


class TestBuildEmbeddingModel:
    """The single place a method name becomes a model. If the kwargs it passes
    drift from what the CLI used to pass, every embedding silently changes."""

    def test_phate_always_passes_n_landmark_even_when_absent_from_params(self, fake_models):
        """PHATE's own default is n_landmark=2000. The CLI path has always passed
        n_landmark explicitly (None when unset), disabling landmarking. Omitting
        it here would silently switch landmarking on for every pipeline run."""
        build_embedding_model("phate", {"knn": 25})

        (inst,) = fake_models.instances
        assert "n_landmark" in inst.kwargs, "n_landmark must be passed explicitly"
        assert inst.kwargs["n_landmark"] is None

    def test_phate_forwards_all_supported_params(self, fake_models):
        build_embedding_model(
            "phate",
            {
                "knn": 7,
                "t": 3,
                "n_landmark": 200,
                "random_landmarking": True,
                "embed_batch_size": 64,
            },
        )
        k = fake_models.instances[0].kwargs
        assert k["n_components"] == 2
        assert k["knn"] == 7
        assert k["t"] == 3
        assert k["n_landmark"] == 200
        assert k["random_landmarking"] is True
        assert k["embed_batch_size"] == 64

    def test_phate_defaults_match_the_cli_defaults(self, fake_models):
        """Drift here changes results between `manifold-genetics embed` and the pipeline."""
        build_embedding_model("phate", {})
        k = fake_models.instances[0].kwargs
        assert k["knn"] == 25
        assert k["t"] == "auto"
        assert k["n_landmark"] is None
        assert k["random_landmarking"] is False
        assert k["embed_batch_size"] is None

    def test_umap_params_and_defaults(self, fake_models):
        build_embedding_model("umap", {})
        k = fake_models.instances[0].kwargs
        assert k["n_components"] == 2
        assert k["n_neighbors"] == 15
        assert k["min_dist"] == 0.1

        build_embedding_model("umap", {"n_neighbors": 40, "min_dist": 0.3})
        k = fake_models.instances[1].kwargs
        assert k["n_neighbors"] == 40
        assert k["min_dist"] == 0.3

    def test_tsne_params_and_defaults(self, fake_models):
        build_embedding_model("tsne", {})
        assert fake_models.instances[0].kwargs["perplexity"] == 30
        build_embedding_model("tsne", {"perplexity": 12})
        assert fake_models.instances[1].kwargs["perplexity"] == 12

    def test_diffusion_map_params_and_defaults(self, fake_models):
        build_embedding_model("diffusion_map", {})
        assert fake_models.instances[0].kwargs["knn"] == 25
        build_embedding_model("diffusion_map", {"knn": 9})
        assert fake_models.instances[1].kwargs["knn"] == 9

    def test_unknown_method_raises_valueerror(self, fake_models):
        with pytest.raises(ValueError, match="[Uu]nknown"):
            build_embedding_model("wavelet", {})

    def test_none_params_treated_as_empty(self, fake_models):
        build_embedding_model("tsne", None)
        assert fake_models.instances[0].kwargs["perplexity"] == 30


# ---------------------------------------------------------------------------
# run_embedding — the seam shared with `manifold-genetics embed`
# ---------------------------------------------------------------------------


class TestRunEmbedding:
    def test_fits_on_fit_input_and_transforms_project_input(
        self, tmp_path, fake_models, stub_validate_embedding_csv
    ):
        out = tmp_path / "emb.csv"
        run_embedding("fit.csv", "proj.csv", project_output=out, method="umap")

        (inst,) = fake_models.instances
        assert inst.calls == [("fit", "fit.csv"), ("transform", "proj.csv")]
        assert out.exists()

    def test_writes_fit_output_when_requested(
        self, tmp_path, fake_models, stub_validate_embedding_csv
    ):
        fit_out, proj_out = tmp_path / "fit.csv", tmp_path / "proj.csv"
        run_embedding(
            "fit_in.csv", "proj_in.csv", fit_output=fit_out, project_output=proj_out, method="umap"
        )

        (inst,) = fake_models.instances
        assert inst.calls == [
            ("fit", "fit_in.csv"),
            ("transform", "fit_in.csv"),
            ("transform", "proj_in.csv"),
        ]
        assert fit_out.exists() and proj_out.exists()

    def test_absent_project_input_reuses_fit_input(
        self, tmp_path, fake_models, stub_validate_embedding_csv
    ):
        """Modes 'fit'/'project': fit and transform the SAME csv — not fit_transform.
        FakeEmbed.fit_transform raises, so a regression to fit_transform fails here."""
        out = tmp_path / "emb.csv"
        run_embedding("only.csv", None, project_output=out, method="umap")

        (inst,) = fake_models.instances
        assert inst.calls == [("fit", "only.csv"), ("transform", "only.csv")]

    def test_creates_parent_directories(self, tmp_path, fake_models, stub_validate_embedding_csv):
        out = tmp_path / "nested" / "deeper" / "emb.csv"
        run_embedding("a.csv", None, project_output=out, method="umap")
        assert out.exists()

    def test_returns_the_projected_coordinates(
        self, tmp_path, fake_models, stub_validate_embedding_csv
    ):
        df = run_embedding("a.csv", None, project_output=tmp_path / "e.csv", method="umap")
        assert list(df.columns) == ["sample_id", "dim_1", "dim_2"]

    def test_params_reach_the_model(self, tmp_path, fake_models, stub_validate_embedding_csv):
        run_embedding(
            "a.csv",
            None,
            project_output=tmp_path / "e.csv",
            method="tsne",
            params={"perplexity": 5},
        )
        assert fake_models.instances[0].kwargs["perplexity"] == 5


# ---------------------------------------------------------------------------
# run_embedding_step — the three --embedding-input modes
# ---------------------------------------------------------------------------


class TestRunEmbeddingStep:
    """embedding_input='fit'|'project'|'both' decides which PCA file is fitted and
    which is projected. Getting it wrong embeds the wrong cohort, or embeds the
    project cohort with its own geometry instead of the fit cohort's."""

    def test_mode_both_fits_fit_pca_and_projects_project_pca(self, tmp_path, fake_models):
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="both", params={"knn": 5})

        result = run_embedding_step(io, emb, pca=pca)

        (inst,) = fake_models.instances
        assert inst.calls == [
            ("fit", str(pca.fit_pca)),
            ("transform", str(pca.fit_pca)),
            ("transform", str(pca.project_pca)),
        ]
        paths = embedding_output_paths(io, emb)
        assert result.embedding_file == paths["embedding"]
        assert result.fit_embedding_file == paths["fit_embedding"]
        assert result.embedding_file.exists()
        assert result.fit_embedding_file.exists()

    def test_mode_fit_uses_fit_pca_only(self, tmp_path, fake_models):
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="fit", params={"knn": 5})

        result = run_embedding_step(io, emb, pca=pca)

        (inst,) = fake_models.instances
        assert inst.calls == [("fit", str(pca.fit_pca)), ("transform", str(pca.fit_pca))]
        assert result.fit_embedding_file is None, "no separate fit embedding in single-input modes"
        assert result.embedding_file.exists()

    def test_mode_project_uses_project_pca_only(self, tmp_path, fake_models):
        """The project PCA must be the FITTED input here. Passing the fit PCA
        instead would embed project samples using fit-cohort geometry."""
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="project", params={"knn": 5})

        result = run_embedding_step(io, emb, pca=pca)

        (inst,) = fake_models.instances
        assert inst.calls == [("fit", str(pca.project_pca)), ("transform", str(pca.project_pca))]
        assert result.fit_embedding_file is None
        assert result.embedding_file.exists()

    def test_writes_the_documented_output_layout(self, tmp_path, fake_models):
        """Spec constraint B: downstream example scripts read these exact paths."""
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="umap", input_mode="both", params={})

        run_embedding_step(io, emb, pca=pca)

        d = io.output_dir / "embeddings"
        assert (d / "umap_2d.csv").exists()
        assert (d / "umap_fit_2d.csv").exists()

    def test_result_carries_coords_and_is_not_skipped(self, tmp_path, fake_models):
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="umap", input_mode="fit", params={})

        result = run_embedding_step(io, emb, pca=pca)

        assert isinstance(result, EmbeddingStepResult)
        assert isinstance(result.coords_df, pd.DataFrame)
        assert result.skipped is False

    def test_config_params_reach_the_model(self, tmp_path, fake_models):
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="fit", params={"knn": 3, "n_landmark": 50})

        run_embedding_step(io, emb, pca=pca)

        k = fake_models.instances[0].kwargs
        assert k["knn"] == 3
        assert k["n_landmark"] == 50

    def test_mode_fit_with_missing_fit_pca_raises(self, tmp_path, fake_models):
        """Under --skip-pca with only the project PCA on disk, pca.fit_pca is
        None. Mode 'fit' needs it and must fail loudly, not with a KeyError deep
        inside the embedder."""
        io = make_io(tmp_path)
        pca = PCAStepResult(fit_pca=None, project_pca=Path("project.csv"))
        emb = EmbeddingConfig(method="phate", input_mode="fit", params={})

        with pytest.raises(RuntimeError, match="mode 'fit' requires the fit PCA"):
            run_embedding_step(io, emb, pca=pca)

    def test_mode_project_with_missing_project_pca_raises(self, tmp_path, fake_models):
        io = make_io(tmp_path)
        pca = PCAStepResult(fit_pca=Path("fit.csv"), project_pca=None)
        emb = EmbeddingConfig(method="phate", input_mode="project", params={})

        with pytest.raises(RuntimeError, match="mode 'project' requires the project PCA"):
            run_embedding_step(io, emb, pca=pca)

    def test_mode_both_with_missing_project_pca_raises(self, tmp_path, fake_models):
        io = make_io(tmp_path)
        pca = PCAStepResult(fit_pca=Path("fit.csv"), project_pca=None)
        emb = EmbeddingConfig(method="phate", input_mode="both", params={})

        with pytest.raises(RuntimeError, match="mode 'both' requires the project PCA"):
            run_embedding_step(io, emb, pca=pca)

    def test_validates_the_pca_coordinates_it_is_handed(self, tmp_path, fake_models, monkeypatch):
        """Regression: Pipeline.run() no longer shells out to `manifold-genetics
        embed`, so cmd_embed's input validation no longer runs on the pipeline
        path. run_embedding_step must validate the PCA coordinates itself (via
        the shared run_embedding() seam) so a malformed cached PCA CSV — e.g.
        under --skip-pca — fails with a clean validation error instead of deep
        inside PHATE."""
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="both", params={})

        validated = []
        monkeypatch.setattr(f"{MODULE}.validate_embedding_csv", lambda path: validated.append(path))

        run_embedding_step(io, emb, pca=pca)

        assert validated == [pca.fit_pca, pca.project_pca]
