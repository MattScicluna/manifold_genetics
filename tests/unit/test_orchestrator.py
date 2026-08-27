"""Tests for pipeline/orchestrator.py — Pipeline init routing, PCA force logic,
embedding input modes, admixture CLI flags, and _get_embedding_model dispatch.

Strategy: mock subprocess.run at the module level to capture CLI commands without
executing them. Real CSV files are written to tmp_path where the orchestrator reads
them back (PCA component-count check, post-run result loading). This exercises real
pandas and path logic without needing any external binaries.

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
        transform_plink_prefix="project",
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
            transform_plink_prefix="project",
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
                transform_plink_prefix="project",
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
                transform_plink_prefix="project",
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
# TestPipelineRunPCAForce — --force flag on component count mismatch
# ---------------------------------------------------------------------------


class TestPipelineRunPCAForce:
    """The PCA step inspects the cached transform_pca_{n}.csv before running.
    If the existing file has a different number of dim_ columns than n_pcs,
    --force is appended to recompute from scratch.

    Silent reuse of stale PCA would propagate wrong-dimensionality coordinates
    to all downstream embedding and metric computations.
    """

    def _run_pca_step(self, pipeline, n_pcs, monkeypatch, transform_pca_path):
        """Run only the PCA step; fake subprocess writes the expected output file."""

        def on_call(cmd):
            if "pca" in cmd:
                write_pca_csv(transform_pca_path, n_pcs)

        return capture_subprocess(monkeypatch, on_call=on_call)

    def test_force_flag_added_when_existing_pca_has_wrong_component_count(
        self, tmp_path, monkeypatch
    ):
        """Existing file has 10 dim_ columns; n_pcs=50 → --force must appear."""
        pipeline = make_pipeline(tmp_path)
        n_pcs = 50
        transform_pca = pipeline.output_dir / "pca" / f"transform_pca_{n_pcs}.csv"
        write_pca_csv(transform_pca, n_dims=10)  # 10 != 50

        calls = self._run_pca_step(pipeline, n_pcs, monkeypatch, transform_pca)
        pipeline.run(
            n_pcs=n_pcs,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        cmd = pca_calls(calls)
        assert len(cmd) == 1
        assert (
            "--force" in cmd[0]
        ), f"Expected --force when existing PCA has 10 dims but n_pcs=50. Got: {cmd[0]}"

    def test_no_force_flag_when_component_count_matches(self, tmp_path, monkeypatch):
        """Existing file already has 50 dim_ columns; n_pcs=50 → no --force.
        Adding --force unnecessarily would recompute expensive PCA on every run.
        """
        pipeline = make_pipeline(tmp_path)
        n_pcs = 50
        transform_pca = pipeline.output_dir / "pca" / f"transform_pca_{n_pcs}.csv"
        write_pca_csv(transform_pca, n_dims=50)  # correct count

        calls = capture_subprocess(monkeypatch)
        pipeline.run(
            n_pcs=n_pcs,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )
        # Fake run didn't write the file after subprocess; read_csv will re-read original.
        # Patch pd.read_csv for the post-run load only.

        cmd = pca_calls(calls)
        assert len(cmd) == 1
        assert "--force" not in cmd[0], (
            f"--force must not appear when existing PCA already has {n_pcs} components. "
            f"Got: {cmd[0]}"
        )

    def test_no_force_flag_on_fresh_run_no_existing_file(self, tmp_path, monkeypatch):
        """No cached file → first-time run must not include --force."""
        pipeline = make_pipeline(tmp_path)
        n_pcs = 50
        transform_pca = pipeline.output_dir / "pca" / f"transform_pca_{n_pcs}.csv"

        def on_call(cmd):
            if "pca" in cmd:
                write_pca_csv(transform_pca, n_pcs)

        calls = capture_subprocess(monkeypatch, on_call=on_call)
        pipeline.run(
            n_pcs=n_pcs,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        cmd = pca_calls(calls)
        assert len(cmd) == 1
        assert "--force" not in cmd[0]


# ---------------------------------------------------------------------------
# TestPipelineRunEmbeddingInputMode — which PCA file feeds the embedder
# ---------------------------------------------------------------------------


class TestPipelineRunEmbeddingInputMode:
    """embedding_input='fit'|'project'|'both' controls which PCA files are passed
    to the embedding CLI command.

    'fit':    embed fit PCA standalone (fit_transform only, no --project-input)
    'project': embed project PCA standalone (using transform PCA as --input)
    'both':   fit embedding on fit PCA, project embedding on project PCA
              (--project-input present, separate --fit-output written)

    Wrong mode silently embeds the wrong dataset.
    """

    def _setup(self, pipeline, n_pcs):
        """Write both PCA files and return (fit_path, transform_path)."""
        pca_dir = pipeline.output_dir / "pca"
        fit_file = pca_dir / f"fit_pca_{n_pcs}.csv"
        transform_file = pca_dir / f"transform_pca_{n_pcs}.csv"
        write_pca_csv(fit_file, n_pcs)
        write_pca_csv(transform_file, n_pcs)
        return fit_file, transform_file

    def _run_embedding_only(self, pipeline, monkeypatch, n_pcs, mode, method="phate"):
        embedding_dir = pipeline.output_dir / "embeddings"
        embedding_file = embedding_dir / f"{method}_2d.csv"
        fit_embedding_file = embedding_dir / f"{method}_fit_2d.csv"

        def on_call(cmd):
            if "embed" in cmd:
                write_pca_csv(embedding_file, 2)
                write_pca_csv(fit_embedding_file, 2)

        calls = capture_subprocess(monkeypatch, on_call=on_call)
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
        return embed_calls(calls)

    def test_mode_both_includes_project_input_and_fit_output(self, tmp_path, monkeypatch):
        """Mode 'both': --project-input (transform PCA) and --fit-output must both appear.
        Without --project-input, only the fit dataset gets embedded; project data is lost.
        """
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        fit_file, transform_file = self._setup(pipeline, n_pcs)

        cmds = self._run_embedding_only(pipeline, monkeypatch, n_pcs, "both")
        assert len(cmds) == 1, f"Expected one embed call, got {len(cmds)}"
        cmd = cmds[0]

        assert (
            "--project-input" in cmd
        ), f"Mode 'both' must include --project-input for the transform dataset. Got: {cmd}"
        assert (
            "--fit-output" in cmd
        ), f"Mode 'both' must include --fit-output to save fit embedding. Got: {cmd}"

        input_val = cmd[cmd.index("--input") + 1]
        assert (
            "fit_pca" in input_val
        ), f"Mode 'both': primary --input should be fit PCA, got: {input_val}"

        project_input_val = cmd[cmd.index("--project-input") + 1]
        assert (
            "transform_pca" in project_input_val
        ), f"Mode 'both': --project-input should be transform PCA, got: {project_input_val}"

    def test_mode_fit_omits_project_input(self, tmp_path, monkeypatch):
        """Mode 'fit': standalone fit_transform — no --project-input flag."""
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)

        cmds = self._run_embedding_only(pipeline, monkeypatch, n_pcs, "fit")
        assert len(cmds) == 1
        cmd = cmds[0]

        assert (
            "--project-input" not in cmd
        ), f"Mode 'fit' must not include --project-input. Got: {cmd}"
        input_val = cmd[cmd.index("--input") + 1]
        assert "fit_pca" in input_val

    def test_mode_project_uses_transform_pca_as_primary_input(self, tmp_path, monkeypatch):
        """Mode 'project': the transform/project PCA feeds --input (not fit PCA).
        If the fit PCA were passed instead, project samples would be embedded using
        fit-cohort geometry.
        """
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)

        cmds = self._run_embedding_only(pipeline, monkeypatch, n_pcs, "project")
        assert len(cmds) == 1
        cmd = cmds[0]

        input_val = cmd[cmd.index("--input") + 1]
        assert (
            "transform_pca" in input_val
        ), f"Mode 'project': --input should be transform PCA, got: {input_val}"
        assert "--project-input" not in cmd


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
        transform_pca = pipeline.output_dir / "pca" / f"transform_pca_{n_pcs}.csv"
        write_pca_csv(fit_pca, n_pcs)
        write_pca_csv(transform_pca, n_pcs)

        embedding_file = pipeline.output_dir / "embeddings" / "phate_2d.csv"
        fit_embedding_file = pipeline.output_dir / "embeddings" / "phate_fit_2d.csv"

        def on_call(cmd):
            if "embed" in cmd:
                write_pca_csv(embedding_file, 2)
                write_pca_csv(fit_embedding_file, 2)

        capture_subprocess(monkeypatch, on_call=on_call)

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

    def _writer(self, pipeline, n_pcs, method="phate"):
        pca_dir = pipeline.output_dir / "pca"
        emb_dir = pipeline.output_dir / "embeddings"
        metrics_dir = pipeline.output_dir / "metrics"

        def on_call(cmd):
            if "pca" in cmd:
                write_pca_csv(pca_dir / f"fit_pca_{n_pcs}.csv", n_pcs)
                write_pca_csv(pca_dir / f"transform_pca_{n_pcs}.csv", n_pcs)
            elif "embed" in cmd:
                write_pca_csv(emb_dir / f"{method}_2d.csv", 2)
                write_pca_csv(emb_dir / f"{method}_fit_2d.csv", 2)
            elif "metrics-geographic" in cmd:
                (metrics_dir).mkdir(parents=True, exist_ok=True)
                (metrics_dir / "geographic.json").write_text('{"correlation": 0.9}')
            elif "metrics-admixture" in cmd:
                (metrics_dir).mkdir(parents=True, exist_ok=True)
                (metrics_dir / "admixture.json").write_text('{"2": {"correlation": 0.5}}')

        return on_call

    def test_full_run_with_backend_and_metrics(self, tmp_path, monkeypatch):
        self._stub_plots(monkeypatch)
        backend = _FakeBackend()
        pipeline = make_pipeline(
            tmp_path,
            geographic_coords="geo.csv",
            projection_plot_fit_column="Population",
            projection_plot_transform_column="Population",
            fit_labels="fl.csv",
            project_labels="pl.csv",
            fit_colormap="fc.json",
            project_colormap="pc.json",
            admixture_backend=backend,
        )
        n_pcs = 3
        calls = capture_subprocess(monkeypatch, on_call=self._writer(pipeline, n_pcs))

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

    def test_full_run_via_cli_admixture(self, tmp_path, monkeypatch):
        self._stub_plots(monkeypatch)
        pipeline = make_pipeline(tmp_path)
        n_pcs = 3
        calls = capture_subprocess(monkeypatch, on_call=self._writer(pipeline, n_pcs))

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
        calls = capture_subprocess(monkeypatch, on_call=self._writer(pipeline, n_pcs))

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
        cmd = embed_calls(calls)[0]
        assert cmd[cmd.index("--n-landmark") + 1] == "200"
        assert "--random-landmarking" in cmd
        assert cmd[cmd.index("--embed-batch-size") + 1] == "64"

    def test_pca_viz_warns_when_pca_file_absent(self, tmp_path, monkeypatch):
        self._stub_plots(monkeypatch)
        pipeline = make_pipeline(tmp_path)
        # subprocess writes nothing -> transform_pca file never appears
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=3,
            skip_pca=True,
            skip_admixture=True,
            skip_embedding=True,
            skip_metrics=True,
        )
        assert "pca_figures" not in results
