"""Tests for pipeline/orchestrator.py — Pipeline init routing, embedding input
modes, admixture CLI flags, in-process PCA dispatch, and _get_embedding_model
dispatch.

Strategy: PCA now runs in-process, so it is stubbed via `stub_pca_step()`, which
replaces `run_pca_step()` with a fake that writes real CSVs to tmp_path (the
orchestrator reads these back for post-run result loading). Admixture and
embedding still shell out, so `subprocess.run` is mocked at the module level to
capture CLI commands without executing them. This exercises real pandas and path
logic without needing any external binaries.

Tests are organised around failure modes: each test documents what would break in
production if the assertion failed.
"""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.embeddings import PHATE, TSNE, UMAP, DiffusionMap
from manifold_genetics.pipeline.orchestrator import Pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pipeline(tmp_path, **kwargs) -> Pipeline:
    defaults = dict(
        fit_plink_prefix="fit",
        project_plink_prefix="project",
        labels="labels.csv",
        colormap="cmap.json",
        output_dir=tmp_path / "out",
    )
    defaults.update(kwargs)
    return Pipeline(**defaults)


def write_pca_csv(path: Path, n_dims: int, n_rows: int = 2) -> None:
    cols = {"sample_id": [f"s{i}" for i in range(n_rows)]}
    cols.update({f"dim_{i}": [float(i)] * n_rows for i in range(1, n_dims + 1)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(path, index=False)


def capture_subprocess(monkeypatch, on_call=None):
    """Patch subprocess.run; call on_call(cmd) after recording if provided."""
    calls = []
    import manifold_genetics.pipeline.orchestrator as orch_mod

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if on_call:
            on_call(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(orch_mod.subprocess, "run", fake_run)
    return calls


def stub_pca_step(monkeypatch, n_pcs):
    """Replace the in-process PCA step with a fake that writes both CSVs.

    Returns the list of (io, pca_config) it was called with, so tests can assert
    what the orchestrator handed the step.
    """
    import manifold_genetics.pipeline.orchestrator as orch
    from manifold_genetics.pipeline.steps.pca import PCAStepResult

    calls = []

    def fake_run_pca_step(io, pca):
        calls.append((io, pca))
        paths = orch.pca_output_paths(io, pca)
        write_pca_csv(paths["fit_pca"], pca.n_pcs)
        write_pca_csv(paths["project_pca"], pca.n_pcs)
        return PCAStepResult(
            fit_pca=paths["fit_pca"],
            project_pca=paths["project_pca"],
            coords_df=pd.read_csv(paths["project_pca"]),
        )

    monkeypatch.setattr(orch, "run_pca_step", fake_run_pca_step)
    return calls


def stub_metrics_steps(monkeypatch, geo=None, admix=None):
    """Stub the metric computations and validators inside steps.metrics so the
    real step code (JSON write + reload) runs against fake inputs."""
    import manifold_genetics.pipeline.steps.metrics as m

    for name in (
        "validate_embedding_csv",
        "validate_geographic_csv",
        "validate_admixture_csv",
        "validate_sample_id_overlap",
    ):
        monkeypatch.setattr(m, name, lambda *a, **k: None)

    monkeypatch.setattr(
        m, "compute_geographic_preservation", lambda **k: geo or {"correlation": 0.9}
    )
    # int keys on purpose: the JSON round-trip inside the step is what turns
    # them into the string keys run_pipeline() has always returned.
    monkeypatch.setattr(
        m, "compute_admixture_preservation", lambda **k: admix or {2: {"correlation": 0.5}}
    )


def pca_calls(calls):
    return [c for c in calls if "manifold-genetics" in c and "pca" in c]


def admix_calls(calls):
    return [c for c in calls if "manifold-genetics" in c and "admixture" in c]


def embed_calls(calls):
    return [c for c in calls if "manifold-genetics" in c and "embed" in c]


# ---------------------------------------------------------------------------
# TestPipelineInit — label and colormap routing
# ---------------------------------------------------------------------------


class TestPipelineInit:
    """__init__ routes labels/colormaps to fit and project datasets.

    Getting this wrong means the wrong labels are used for the wrong cohort,
    which silently corrupts all downstream figures and metrics.
    """

    def test_shared_labels_assign_to_both_fit_and_project(self, tmp_path):
        """Single 'labels' arg must become both fit_labels and project_labels."""
        p = make_pipeline(tmp_path, labels="shared.csv")
        assert p.fit_labels == Path("shared.csv")
        assert p.project_labels == Path("shared.csv")

    def test_fit_labels_override_does_not_affect_project(self, tmp_path):
        """fit_labels overrides the shared label for fit only.
        project_labels must still resolve to the shared path."""
        p = make_pipeline(tmp_path, labels="shared.csv", fit_labels="fit_specific.csv")
        assert p.fit_labels == Path("fit_specific.csv")
        assert p.project_labels == Path("shared.csv")

    def test_project_labels_override_does_not_affect_fit(self, tmp_path):
        p = make_pipeline(tmp_path, labels="shared.csv", project_labels="proj_specific.csv")
        assert p.project_labels == Path("proj_specific.csv")
        assert p.fit_labels == Path("shared.csv")

    def test_separate_labels_without_shared_are_independent(self, tmp_path):
        """Separate fit/project labels with no shared: labels attr must be None."""
        p = Pipeline(
            fit_plink_prefix="fit",
            project_plink_prefix="project",
            fit_labels="fit.csv",
            project_labels="project.csv",
            colormap="cmap.json",
            output_dir=tmp_path / "out",
        )
        assert p.fit_labels == Path("fit.csv")
        assert p.project_labels == Path("project.csv")
        assert p.labels is None

    def test_fit_labels_without_project_labels_raises(self, tmp_path):
        """fit_labels alone (no shared, no project_labels) must raise immediately.
        Silently defaulting project_labels to None would crash in the visualisation step
        with a confusing AttributeError rather than a clear upfront message.
        """
        with pytest.raises(ValueError, match="fit_labels|project_labels"):
            Pipeline(
                fit_plink_prefix="fit",
                project_plink_prefix="project",
                fit_labels="fit.csv",
                colormap="cmap.json",
                output_dir=tmp_path / "out",
            )

    def test_project_colormap_without_fit_colormap_raises(self, tmp_path):
        """project_colormap alone (no shared colormap) must raise.
        Downstream code calls read_colormap(self.fit_colormap) for PCA visualisation;
        a None path would crash with a misleading FileNotFoundError.
        """
        with pytest.raises(ValueError, match="colormap"):
            Pipeline(
                fit_plink_prefix="fit",
                project_plink_prefix="project",
                labels="labels.csv",
                project_colormap="project.json",
                output_dir=tmp_path / "out",
            )

    def test_output_dir_created_on_init(self, tmp_path):
        """output_dir.mkdir(parents=True) must be called in __init__.
        run() immediately writes files there; if missing it would fail with FileNotFoundError.
        """
        out = tmp_path / "nested" / "output"
        assert not out.exists()
        make_pipeline(tmp_path, output_dir=out)
        assert out.is_dir()


# ---------------------------------------------------------------------------
# TestPipelineRunEmbeddingInputMode — which PCA file feeds the embedder
# ---------------------------------------------------------------------------


def stub_embedding_step(monkeypatch):
    """Replace the in-process embedding step with a fake that writes its outputs.

    Returns the list of (io, emb, pca) it was called with.
    """
    import manifold_genetics.pipeline.orchestrator as orch
    from manifold_genetics.pipeline.steps.embedding import EmbeddingStepResult
    from manifold_genetics.pipeline.steps.paths import embedding_output_paths

    calls = []

    def fake_run_embedding_step(io, emb, *, pca):
        calls.append((io, emb, pca))
        paths = embedding_output_paths(io, emb)
        write_pca_csv(paths["embedding"], 2)
        fit_file = paths.get("fit_embedding")
        if fit_file is not None:
            write_pca_csv(fit_file, 2)
        return EmbeddingStepResult(
            embedding_file=paths["embedding"],
            fit_embedding_file=fit_file,
            coords_df=pd.read_csv(paths["embedding"]),
        )

    monkeypatch.setattr(orch, "run_embedding_step", fake_run_embedding_step)
    return calls


class TestPipelineRunEmbeddingInputMode:
    """embedding_input='fit'|'project'|'both' controls which PCA coordinates the
    embedding step receives.

    'fit':     embed the fit cohort's PCA standalone
    'project': embed the project cohort's PCA standalone
    'both':    fit on the fit PCA, apply to the project PCA, write both files

    Wrong mode silently embeds the wrong dataset.
    """

    def _setup(self, pipeline, n_pcs):
        pca_dir = pipeline.output_dir / "pca"
        fit_file = pca_dir / f"fit_pca_{n_pcs}.csv"
        project_file = pca_dir / f"project_pca_{n_pcs}.csv"
        write_pca_csv(fit_file, n_pcs)
        write_pca_csv(project_file, n_pcs)
        return fit_file, project_file

    def _run(self, pipeline, monkeypatch, n_pcs, mode, method="phate"):
        calls = stub_embedding_step(monkeypatch)
        capture_subprocess(monkeypatch)
        pipeline.run(
            n_pcs=n_pcs,
            embedding=method,
            embedding_params={"knn": 5},
            embedding_input=mode,
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )
        assert len(calls) == 1, f"Expected one embedding step call, got {len(calls)}"
        return calls[0]

    def test_mode_both_passes_both_pca_files(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        fit_file, project_file = self._setup(pipeline, n_pcs)

        io, emb, pca = self._run(pipeline, monkeypatch, n_pcs, "both")

        assert emb.input_mode == "both"
        assert pca.fit_pca == fit_file
        assert pca.project_pca == project_file

    def test_mode_fit_passes_the_fit_pca(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        fit_file, _ = self._setup(pipeline, n_pcs)

        io, emb, pca = self._run(pipeline, monkeypatch, n_pcs, "fit")

        assert emb.input_mode == "fit"
        assert pca.fit_pca == fit_file

    def test_mode_project_passes_the_project_pca(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        _, project_file = self._setup(pipeline, n_pcs)

        io, emb, pca = self._run(pipeline, monkeypatch, n_pcs, "project")

        assert emb.input_mode == "project"
        assert pca.project_pca == project_file

    def test_method_and_params_reach_the_step(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)

        io, emb, pca = self._run(pipeline, monkeypatch, n_pcs, "both", method="umap")

        assert emb.method == "umap"
        assert emb.params["knn"] == 5

    def test_embedding_no_longer_shells_out(self, tmp_path, monkeypatch):
        """Re-introducing a subprocess hop would restore the API -> CLI -> API
        inversion this refactor removed."""
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)

        stub_embedding_step(monkeypatch)
        calls = capture_subprocess(monkeypatch)
        pipeline.run(
            n_pcs=n_pcs,
            embedding="phate",
            embedding_params={"knn": 5},
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )
        assert embed_calls(calls) == [], f"Embedding must run in-process, got: {calls}"

    def test_results_expose_embedding_paths(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)
        stub_embedding_step(monkeypatch)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=n_pcs,
            embedding="phate",
            embedding_params={"knn": 5},
            embedding_input="both",
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        emb_dir = pipeline.output_dir / "embeddings"
        assert results["embedding_file"] == emb_dir / "phate_2d.csv"
        assert results["fit_embedding_file"] == emb_dir / "phate_fit_2d.csv"
        assert isinstance(results["embedding_coords"], pd.DataFrame)

    def test_single_input_mode_reports_no_fit_embedding_file(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)
        stub_embedding_step(monkeypatch)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=n_pcs,
            embedding="phate",
            embedding_params={"knn": 5},
            embedding_input="fit",
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )
        assert "fit_embedding_file" not in results


# ---------------------------------------------------------------------------
# TestPipelineRunMissingPCA — guard against silent incomplete results
# ---------------------------------------------------------------------------


class TestPipelineRunMissingPCA:

    def test_embedding_raises_runtime_error_when_pca_files_absent(self, tmp_path):
        """skip_pca=True with no cached PCA files must raise RuntimeError immediately.

        Without this guard the pipeline returns silently with no embedding output
        and no indication of what went wrong — the caller would have to inspect
        the results dict to notice something is missing.
        """
        pipeline = make_pipeline(tmp_path)

        with pytest.raises(RuntimeError, match="[Pp][Cc][Aa]"):
            pipeline.run(
                skip_pca=True,
                skip_admixture=True,
                skip_pca_visualization=True,
                skip_metrics=True,
            )

    def test_embedding_succeeds_when_cached_pca_files_present(self, tmp_path, monkeypatch):
        """skip_pca=True with existing PCA files on disk must not raise."""
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        fit_pca = pipeline.output_dir / "pca" / f"fit_pca_{n_pcs}.csv"
        project_pca = pipeline.output_dir / "pca" / f"project_pca_{n_pcs}.csv"
        write_pca_csv(fit_pca, n_pcs)
        write_pca_csv(project_pca, n_pcs)

        stub_embedding_step(monkeypatch)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=n_pcs,
            embedding="phate",
            embedding_params={"knn": 5},
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )
        assert "embedding_file" in results


# ---------------------------------------------------------------------------
# TestPipelineRunAdmixtureFlags — optional CLI args
# ---------------------------------------------------------------------------


class TestPipelineRunAdmixtureFlags:
    """Optional admixture args are appended to the CLI command only when provided.

    The critical case is admix_gpus=0: zero is falsy but is a valid, meaningful
    value (explicit CPU-only execution). Using `if admix_gpus:` instead of
    `if admix_gpus is not None:` would silently drop it, causing neural-admixture
    to auto-detect and potentially use GPUs against the caller's intent.
    """

    def _admix_cmd(self, pipeline, monkeypatch, **run_kwargs):
        calls = capture_subprocess(monkeypatch)
        pipeline.run(
            skip_pca=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_admixture_visualization=True,
            skip_metrics=True,
            **run_kwargs,
        )
        cmds = admix_calls(calls)
        assert len(cmds) == 1, f"Expected one admixture command, got: {cmds}"
        return cmds[0]

    def test_threads_flag_present_when_set(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        cmd = self._admix_cmd(pipeline, monkeypatch, admix_threads=4)
        assert "--threads" in cmd
        assert cmd[cmd.index("--threads") + 1] == "4"

    def test_threads_flag_absent_when_none(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        cmd = self._admix_cmd(pipeline, monkeypatch, admix_threads=None)
        assert "--threads" not in cmd

    def test_gpus_zero_included_in_command(self, tmp_path, monkeypatch):
        """admix_gpus=0 explicitly requests CPU-only mode — must not be silently dropped."""
        pipeline = make_pipeline(tmp_path)
        cmd = self._admix_cmd(pipeline, monkeypatch, admix_gpus=0)
        assert "--num-gpus" in cmd, (
            "admix_gpus=0 must appear as --num-gpus 0. "
            "Dropping it would cause neural-admixture to auto-detect GPUs."
        )
        assert cmd[cmd.index("--num-gpus") + 1] == "0"

    def test_gpus_flag_absent_when_none(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        cmd = self._admix_cmd(pipeline, monkeypatch, admix_gpus=None)
        assert "--num-gpus" not in cmd

    def test_batch_size_flag_present_when_set(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        cmd = self._admix_cmd(pipeline, monkeypatch, admix_batch_size=256)
        assert "--neuraladmixture-batch-size" in cmd
        assert cmd[cmd.index("--neuraladmixture-batch-size") + 1] == "256"

    def test_batch_size_flag_absent_when_none(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        cmd = self._admix_cmd(pipeline, monkeypatch, admix_batch_size=None)
        assert "--neuraladmixture-batch-size" not in cmd


# ---------------------------------------------------------------------------
# TestGetEmbeddingModel — method dispatch and parameter forwarding
# ---------------------------------------------------------------------------


class TestGetEmbeddingModel:
    """_get_embedding_model maps a method string to an embedding class instance.

    Parameter forwarding matters: silently ignoring user-supplied knn or
    n_neighbors would produce embeddings with wrong neighbourhood scale,
    which is undetectable from the output shape alone.

    Dead code: _get_embedding_model is no longer called by the live path
    (superseded by pipeline.steps.embedding.build_embedding_model) and is kept
    only until a later PR deletes it. Its defaults deliberately differ from the
    live path — see test_empty_params_uses_defaults below, which pins PHATE's
    own n_landmark=2000 default, not the live path's explicit None.
    """

    @pytest.fixture
    def pipeline(self, tmp_path):
        return make_pipeline(tmp_path)

    def test_unknown_method_raises_valueerror(self, pipeline):
        with pytest.raises(ValueError, match="Unknown embedding method"):
            pipeline._get_embedding_model("foo")

    def test_phate_returns_phate_instance(self, pipeline):
        assert isinstance(pipeline._get_embedding_model("phate"), PHATE)

    def test_umap_returns_umap_instance(self, pipeline):
        assert isinstance(pipeline._get_embedding_model("umap"), UMAP)

    def test_tsne_returns_tsne_instance(self, pipeline):
        assert isinstance(pipeline._get_embedding_model("tsne"), TSNE)

    def test_diffusion_map_returns_diffusion_map_instance(self, pipeline):
        assert isinstance(pipeline._get_embedding_model("diffusion_map"), DiffusionMap)

    def test_phate_knn_param_overrides_default(self, pipeline):
        """User-supplied knn=100 must override the built-in default of 25."""
        model = pipeline._get_embedding_model("phate", {"knn": 100})
        assert model.knn == 100, (
            f"Expected knn=100, got {model.knn}. "
            "Silently ignoring user params produces wrong-scale embeddings."
        )

    def test_umap_n_neighbors_param_overrides_default(self, pipeline):
        model = pipeline._get_embedding_model("umap", {"n_neighbors": 50})
        assert model.n_neighbors == 50

    def test_tsne_perplexity_param_overrides_default(self, pipeline):
        model = pipeline._get_embedding_model("tsne", {"perplexity": 100})
        assert model.perplexity == 100

    def test_empty_params_uses_defaults(self, pipeline):
        """Empty params dict must not crash — defaults apply."""
        model = pipeline._get_embedding_model("phate", {})
        assert model.knn == 25  # built-in default


# ---------------------------------------------------------------------------
# TestPipelineRunFullFlow — the step bodies run() executes when nothing skipped
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Minimal AdmixtureBackend stand-in for the backend branch of run()."""

    def __init__(self):
        self.calls = []

    def fit(self, *a, **k):
        self.calls.append("fit")

    def fit_transform(self, *a, **k):
        self.calls.append("fit_transform")
        return {}

    def transform(self, *a, **k):
        self.calls.append("transform")
        return {}


class TestPipelineRunFullFlow:
    """PCA viz, admixture (backend + CLI), bar plot, embedding viz, admixture-
    colored embedding, and metrics all run when their skip flags are False.
    All plotting and I/O is stubbed; subprocess writes the CSV/JSON the
    orchestrator reads back."""

    def _stub_plots(self, monkeypatch):
        import manifold_genetics.pipeline.orchestrator as orch

        for name in (
            "visualize",
            "plot_admixture_bar_grid",
            "plot_admixture_embedding_grid",
            "plot_projection",
        ):
            monkeypatch.setattr(orch, name, lambda *a, **k: ["fig.png"])
        monkeypatch.setattr(
            "manifold_genetics.visualization.plot_pca_pairs",
            lambda **k: Path("pca_pairs.png"),
        )
        monkeypatch.setattr(
            "manifold_genetics.utils.io.read_colormap",
            lambda p: {"Population": {"A": "#000000"}},
        )

    def test_full_run_with_backend_and_metrics(self, tmp_path, monkeypatch):
        self._stub_plots(monkeypatch)
        backend = _FakeBackend()
        pipeline = make_pipeline(
            tmp_path,
            geographic_coords="geo.csv",
            projection_plot_fit_column="Population",
            projection_plot_project_column="Population",
            fit_labels="fl.csv",
            project_labels="pl.csv",
            fit_colormap="fc.json",
            project_colormap="pc.json",
            admixture_backend=backend,
        )
        n_pcs = 3
        calls = capture_subprocess(monkeypatch)

        stub_pca_step(monkeypatch, n_pcs)
        stub_embedding_step(monkeypatch)
        stub_metrics_steps(monkeypatch)
        results = pipeline.run(
            n_pcs=n_pcs,
            k_min=2,
            k_max=3,
            embedding="phate",
            embedding_params={"knn": 5},
            embedding_input="both",
        )

        assert backend.calls == ["fit", "fit_transform", "transform"]
        assert "pca_figures" in results
        assert "admixture_figures" in results
        assert "embedding_figures" in results
        assert "projection_plot" in results
        assert results["metrics"]["geographic"]["correlation"] == 0.9
        assert results["metrics"]["admixture"]["2"]["correlation"] == 0.5
        # no admixture CLI call — backend was used
        assert admix_calls(calls) == []
        # Metrics ran in-process and landed at the documented paths (constraint B)
        assert (pipeline.output_dir / "metrics" / "geographic.json").exists()
        assert (pipeline.output_dir / "metrics" / "admixture.json").exists()
        assert [c for c in calls if "metrics-geographic" in c] == []
        assert [c for c in calls if "metrics-admixture" in c] == []

    def test_metrics_skipped_when_no_geographic_coords(self, tmp_path, monkeypatch):
        """Without geographic_coords the geographic metric must not run, but the
        admixture metric still must."""
        self._stub_plots(monkeypatch)
        stub_metrics_steps(monkeypatch)
        pipeline = make_pipeline(tmp_path, admixture_backend=_FakeBackend())
        n_pcs = 3
        stub_pca_step(monkeypatch, n_pcs)
        stub_embedding_step(monkeypatch)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=n_pcs,
            k_min=2,
            k_max=3,
            embedding="phate",
            embedding_params={"knn": 5},
        )

        assert "geographic" not in results["metrics"]
        assert not (pipeline.output_dir / "metrics" / "geographic.json").exists()
        assert results["metrics"]["admixture"]["2"]["correlation"] == 0.5

    def test_full_run_via_cli_admixture(self, tmp_path, monkeypatch):
        self._stub_plots(monkeypatch)
        pipeline = make_pipeline(tmp_path)
        n_pcs = 3
        calls = capture_subprocess(monkeypatch)

        stub_pca_step(monkeypatch, n_pcs)
        stub_embedding_step(monkeypatch)
        pipeline.run(
            n_pcs=n_pcs,
            k_min=2,
            k_max=3,
            embedding="phate",
            embedding_params={"knn": 5},
            skip_metrics=True,
        )
        admix = admix_calls(calls)
        assert len(admix) == 1
        assert "--k-min" in admix[0] and "--k-max" in admix[0]

    def test_phate_landmark_and_batch_flags_forwarded(self, tmp_path, monkeypatch):
        self._stub_plots(monkeypatch)
        pipeline = make_pipeline(tmp_path)
        n_pcs = 3
        capture_subprocess(monkeypatch)

        stub_pca_step(monkeypatch, n_pcs)
        embed_step_calls = stub_embedding_step(monkeypatch)
        pipeline.run(
            n_pcs=n_pcs,
            embedding="phate",
            embedding_params={
                "knn": 5,
                "n_landmark": 200,
                "random_landmarking": True,
                "embed_batch_size": 64,
            },
            skip_admixture=True,
            skip_metrics=True,
        )
        io, emb, pca = embed_step_calls[0]
        assert emb.params["n_landmark"] == 200
        assert emb.params["random_landmarking"] is True
        assert emb.params["embed_batch_size"] == 64

    def test_pca_viz_warns_when_pca_file_absent(self, tmp_path, monkeypatch):
        self._stub_plots(monkeypatch)
        pipeline = make_pipeline(tmp_path)
        # subprocess writes nothing -> project_pca file never appears
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=3,
            skip_pca=True,
            skip_admixture=True,
            skip_embedding=True,
            skip_metrics=True,
        )
        assert "pca_figures" not in results


# ---------------------------------------------------------------------------
# TestPipelineRunPCAInProcess — PR 2 cutover: no subprocess for PCA
# ---------------------------------------------------------------------------


class TestPipelineRunPCAInProcess:
    """run() calls run_pca_step() directly. Re-introducing a subprocess hop
    would restore the API → CLI → API inversion this refactor removed, along
    with its cold-start cost and buried tracebacks."""

    def test_pca_no_longer_shells_out(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        stub_pca_step(monkeypatch, 3)
        calls = capture_subprocess(monkeypatch)

        pipeline.run(
            n_pcs=3,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        assert pca_calls(calls) == [], f"PCA must run in-process, got: {calls}"

    def test_step_receives_pipeline_io_and_n_pcs(self, tmp_path, monkeypatch):
        """The step's IOConfig must carry the pipeline's own prefixes; passing
        the wrong cohort would fit PCA on the projection set."""
        pipeline = make_pipeline(
            tmp_path, fit_plink_prefix="fitset", project_plink_prefix="projectset"
        )
        step_calls = stub_pca_step(monkeypatch, 7)
        capture_subprocess(monkeypatch)

        pipeline.run(
            n_pcs=7,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        assert len(step_calls) == 1
        io, pca_cfg = step_calls[0]
        assert io.fit_plink == Path("fitset")
        assert io.project_plink == Path("projectset")
        assert io.output_dir == pipeline.output_dir
        assert pca_cfg.n_pcs == 7

    def test_results_expose_pca_paths_and_coords(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        stub_pca_step(monkeypatch, 4)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=4,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        pca_dir = pipeline.output_dir / "pca"
        assert results["fit_pca_file"] == pca_dir / "fit_pca_4.csv"
        assert results["project_pca_file"] == pca_dir / "project_pca_4.csv"
        assert results["pca_file"] == results["project_pca_file"]
        assert isinstance(results["pca_coords"], pd.DataFrame)

    def test_skip_pca_still_resolves_existing_outputs(self, tmp_path, monkeypatch):
        """Spec constraint C: a skipped step still supplies its paths downstream."""
        pipeline = make_pipeline(tmp_path)
        pca_dir = pipeline.output_dir / "pca"
        write_pca_csv(pca_dir / "fit_pca_4.csv", 4)
        write_pca_csv(pca_dir / "project_pca_4.csv", 4)

        step_calls = stub_pca_step(monkeypatch, 4)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=4,
            skip_pca=True,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        assert step_calls == [], "skip_pca must not run the step"
        assert results["fit_pca_file"] == pca_dir / "fit_pca_4.csv"
        assert results["project_pca_file"] == pca_dir / "project_pca_4.csv"
