"""Tests for the in-process visualization steps (pipeline/steps/viz.py).

These four steps are extracted verbatim from inline blocks in Pipeline.run().
The tests assert the extraction is faithful: same plotting function, same
kwargs, same figure paths. Every plotting function is stubbed — nothing here
renders a figure.

Note the asymmetry with the compute steps: visualization is NON-FATAL by spec
constraint D. These step functions may raise; the orchestrator's _run_viz
wrapper is what catches. That contract is tested in test_orchestrator.py.
"""

import json
from pathlib import Path

import pytest

from manifold_genetics import cli as mg_cli
from manifold_genetics.pipeline.config import (
    AdmixtureConfig,
    EmbeddingConfig,
    IOConfig,
    VizConfig,
)
from manifold_genetics.pipeline.steps.admixture import AdmixtureStepResult
from manifold_genetics.pipeline.steps.embedding import EmbeddingStepResult
from manifold_genetics.pipeline.steps.paths import figure_output_paths
from manifold_genetics.pipeline.steps.viz import (
    VizStepResult,
    run_admixture_embedding_viz_step,
    run_admixture_viz_step,
    run_embedding_viz_step,
    run_pca_viz_step,
)

MODULE = "manifold_genetics.pipeline.steps.viz"

COLORMAP = {"Population": {"A": "#000000"}, "Region": {"X": "#ffffff"}}


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


@pytest.fixture
def stub_colormap(monkeypatch):
    monkeypatch.setattr(f"{MODULE}.read_colormap", lambda p: dict(COLORMAP))


@pytest.fixture
def calls(monkeypatch):
    """Record every plotting call as (name, kwargs)."""
    recorded = []

    def rec(name, ret):
        def _fn(**kwargs):
            recorded.append((name, kwargs))
            return ret

        return _fn

    monkeypatch.setattr(f"{MODULE}.plot_pca_pairs", rec("plot_pca_pairs", Path("pca.png")))
    monkeypatch.setattr(f"{MODULE}.visualize", rec("visualize", [Path("emb.png")]))
    monkeypatch.setattr(f"{MODULE}.plot_projection", rec("plot_projection", Path("proj.png")))
    monkeypatch.setattr(f"{MODULE}.plot_admixture_bar_grid", rec("plot_admixture_bar_grid", None))
    monkeypatch.setattr(
        f"{MODULE}.plot_admixture_embedding_grid",
        rec("plot_admixture_embedding_grid", None),
    )
    return recorded


# ---------------------------------------------------------------------------
# figure_output_paths — the figures/** layout contract
# ---------------------------------------------------------------------------


def test_figure_output_paths_match_the_documented_layout(tmp_path):
    """Spec constraint B. Downstream scripts and users look for these by name."""
    io = make_io(tmp_path)
    p = figure_output_paths(io)
    root = io.output_dir / "figures"
    assert p["root"] == root
    assert p["pca"] == root / "pca"
    assert p["embeddings"] == root / "embeddings"
    assert p["admixture"] == root / "admixture"
    assert p["admixture_bars"] == root / "admixture" / "project_bars.png"
    assert (
        p["admixture_colored_embedding"]
        == root / "admixture" / "project_admixture_colored_embedding.png"
    )


def test_figure_output_paths_is_pure(tmp_path):
    """No directory may be created by asking where figures go."""
    io = make_io(tmp_path)
    figure_output_paths(io)
    assert not (io.output_dir / "figures").exists()


# ---------------------------------------------------------------------------
# run_pca_viz_step
# ---------------------------------------------------------------------------


class TestPcaVizStep:
    def test_plots_one_grid_per_colormap_column(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        pca_file = tmp_path / "project_pca_10.csv"

        result = run_pca_viz_step(io, VizConfig(), pca_file=pca_file, n_pcs=10)

        names = [c[0] for c in calls]
        assert names == ["plot_pca_pairs", "plot_pca_pairs"], "one grid per colormap column"
        cols = [c[1]["label_column"] for c in calls]
        assert cols == ["Population", "Region"]
        assert isinstance(result, VizStepResult)
        assert result.failed is False
        assert len(result.figures) == 2

    def test_forwards_the_documented_kwargs(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        pca_file = tmp_path / "project_pca_10.csv"

        run_pca_viz_step(io, VizConfig(), pca_file=pca_file, n_pcs=10)

        _, kw = calls[0]
        assert kw["pca_coords"] == pca_file
        assert kw["labels"] == io.project_labels
        assert kw["colormap"] == COLORMAP
        assert kw["n_pcs"] == 10
        assert kw["output_path"] == figure_output_paths(io)["pca"] / "pca_pairs_by_Population.png"
        assert kw["title"] == "PCA Pairs by Population"

    def test_creates_the_figures_directory(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_pca_viz_step(io, VizConfig(), pca_file=tmp_path / "p.csv", n_pcs=3)
        assert figure_output_paths(io)["pca"].is_dir()


def test_cmd_plot_pca_uses_the_same_seam_as_the_pipeline_step(tmp_path, monkeypatch):
    """``manifold-genetics plot-pca`` and ``run_pca_viz_step`` both go through
    ``plot_pca_pair_grids`` now, so patching ``plot_pca_pairs`` at its one
    definition intercepts both callers, and the CLI's output filenames match
    the naming the pipeline step already uses.
    """
    for name in [
        "validate_embedding_csv",
        "validate_labels_csv",
        "validate_colormap_json",
        "validate_labels_colormap_match",
        "validate_sample_id_overlap",
    ]:
        monkeypatch.setattr(mg_cli, name, lambda *a, **k: None)

    produced = []

    def fake_plot_pca_pairs(**kwargs):
        produced.append(kwargs["output_path"])
        return kwargs["output_path"]

    monkeypatch.setattr(f"{MODULE}.plot_pca_pairs", fake_plot_pca_pairs)

    cmap_path = tmp_path / "cmap.json"
    cmap_path.write_text(json.dumps(COLORMAP))
    out_dir = tmp_path / "figs"

    rc = mg_cli.main(
        [
            "plot-pca",
            "--input",
            "pca.csv",
            "--labels",
            "labels.csv",
            "--colormap",
            str(cmap_path),
            "--output",
            str(out_dir),
            "--n-pcs",
            "4",
        ]
    )

    assert rc == 0
    assert [p.name for p in produced] == [f"pca_pairs_by_{col}.png" for col in COLORMAP]


# ---------------------------------------------------------------------------
# run_embedding_viz_step
# ---------------------------------------------------------------------------


class TestEmbeddingVizStep:
    def _emb(self, tmp_path, with_fit=True):
        return EmbeddingStepResult(
            embedding_file=tmp_path / "phate_2d.csv",
            fit_embedding_file=(tmp_path / "phate_fit_2d.csv") if with_fit else None,
        )

    def test_project_only_when_there_is_no_fit_embedding(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_embedding_viz_step(
            io, VizConfig(), embedding=self._emb(tmp_path, with_fit=False), method="phate"
        )

        vis = [c for c in calls if c[0] == "visualize"]
        assert len(vis) == 1, "single-input modes produce only the project figures"
        assert vis[0][1]["dataset_prefix"] == "project_"

    def test_fit_and_project_when_both_embeddings_exist(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_embedding_viz_step(io, VizConfig(), embedding=self._emb(tmp_path), method="phate")

        vis = [c for c in calls if c[0] == "visualize"]
        assert [v[1]["dataset_prefix"] for v in vis] == ["fit_", "project_"]
        assert vis[0][1]["labels"] == io.fit_labels
        assert vis[0][1]["colormap"] == io.fit_colormap
        assert vis[1][1]["labels"] == io.project_labels
        assert vis[1][1]["colormap"] == io.project_colormap

    def test_projection_plot_only_when_both_columns_configured(
        self, tmp_path, calls, stub_colormap
    ):
        io = make_io(tmp_path)

        run_embedding_viz_step(io, VizConfig(), embedding=self._emb(tmp_path), method="phate")
        assert not [c for c in calls if c[0] == "plot_projection"]

        calls.clear()
        viz = VizConfig(
            projection_plot_fit_column="Population", projection_plot_project_column="Region"
        )
        run_embedding_viz_step(io, viz, embedding=self._emb(tmp_path), method="phate")
        proj = [c for c in calls if c[0] == "plot_projection"]
        assert len(proj) == 1
        assert proj[0][1]["fit_label_column"] == "Population"
        assert proj[0][1]["project_label_column"] == "Region"
        assert proj[0][1]["output_path"].name == (
            "phate_projection_fit_Population_project_Region.png"
        )

    def test_projection_plot_failure_does_not_lose_the_other_figures(
        self, tmp_path, calls, stub_colormap, monkeypatch
    ):
        """The projection plot keeps its own try/except: losing it must not lose
        the fit and project figures already produced."""

        def boom(**kwargs):
            raise RuntimeError("projection exploded")

        monkeypatch.setattr(f"{MODULE}.plot_projection", boom)
        io = make_io(tmp_path)
        viz = VizConfig(
            projection_plot_fit_column="Population", projection_plot_project_column="Region"
        )

        result = run_embedding_viz_step(io, viz, embedding=self._emb(tmp_path), method="phate")

        assert result.figures, "fit/project figures must survive a projection-plot failure"

    def test_split_fields_populated_with_fit_embedding_and_projection_columns(
        self, tmp_path, calls, stub_colormap
    ):
        """VizStepResult carries fit_figures/project_figures/projection_plot
        separately from the flat `figures` aggregate, so the orchestrator can
        reconstruct the three distinct `results` keys the old inline block set."""
        io = make_io(tmp_path)
        viz = VizConfig(
            projection_plot_fit_column="Population", projection_plot_project_column="Region"
        )

        result = run_embedding_viz_step(io, viz, embedding=self._emb(tmp_path), method="phate")

        assert result.fit_figures == (Path("emb.png"),)
        assert result.project_figures == (Path("emb.png"),)
        assert result.projection_plot is not None
        assert result.projection_plot.name == ("phate_projection_fit_Population_project_Region.png")
        assert result.figures == (
            result.fit_figures + result.project_figures + (result.projection_plot,)
        )

    def test_split_fields_without_fit_embedding(self, tmp_path, calls, stub_colormap):
        """No fit embedding means fit_figures is empty and projection_plot is
        None, even when both projection columns are configured — there is no
        fit embedding to pair the project embedding with."""
        io = make_io(tmp_path)
        viz = VizConfig(
            projection_plot_fit_column="Population", projection_plot_project_column="Region"
        )

        result = run_embedding_viz_step(
            io, viz, embedding=self._emb(tmp_path, with_fit=False), method="phate"
        )

        assert result.fit_figures == ()
        assert result.project_figures == (Path("emb.png"),)
        assert result.projection_plot is None
        assert result.figures == result.project_figures


# ---------------------------------------------------------------------------
# run_admixture_viz_step
# ---------------------------------------------------------------------------


class TestAdmixtureVizStep:
    def _admix(self, tmp_path):
        d = tmp_path / "out" / "admixture"
        return AdmixtureStepResult(
            q_prefix=d / "project",
            k_values=(2, 3),
            dir=d,
            checkpoints_dir=d / "checkpoints",
            fit_prefix=d / "fit",
        )

    def test_bar_grid_kwargs_and_path(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_admixture_viz_step(io, VizConfig(), admixture=self._admix(tmp_path))

        ((name, kw),) = [c for c in calls if c[0] == "plot_admixture_bar_grid"]
        assert kw["q_prefix"] == self._admix(tmp_path).q_prefix
        assert kw["labels"] == io.project_labels
        assert kw["colormap"] == COLORMAP
        assert kw["subsample_per_group"] == 300
        assert list(kw["k_values"]) == [2, 3]
        assert kw["output_path"] == figure_output_paths(io)["admixture_bars"]

    def test_group_column_defaults_to_first_colormap_key(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_admixture_viz_step(
            io, VizConfig(admix_group_column=None), admixture=self._admix(tmp_path)
        )
        assert calls[0][1]["group_column"] == "Population"

    def test_explicit_group_column_wins(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_admixture_viz_step(
            io, VizConfig(admix_group_column="Region"), admixture=self._admix(tmp_path)
        )
        assert calls[0][1]["group_column"] == "Region"

    def test_within_group_order_forwarded(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_admixture_viz_step(
            io, VizConfig(admix_within_group_order="tree"), admixture=self._admix(tmp_path)
        )
        assert calls[0][1]["within_group_order"] == "tree"


# ---------------------------------------------------------------------------
# run_admixture_embedding_viz_step
# ---------------------------------------------------------------------------


class TestAdmixtureEmbeddingVizStep:
    def test_grid_kwargs_and_path(self, tmp_path, calls):
        io = make_io(tmp_path)
        d = tmp_path / "out" / "admixture"
        admix = AdmixtureStepResult(
            q_prefix=d / "project",
            k_values=(2, 3, 4),
            dir=d,
            checkpoints_dir=d / "checkpoints",
            fit_prefix=d / "fit",
        )
        emb = EmbeddingStepResult(embedding_file=tmp_path / "phate_2d.csv")

        result = run_admixture_embedding_viz_step(io, embedding=emb, admixture=admix)

        ((name, kw),) = [c for c in calls if c[0] == "plot_admixture_embedding_grid"]
        assert kw["embedding"] == emb.embedding_file
        assert kw["q_prefix"] == admix.q_prefix
        assert list(kw["k_values"]) == [2, 3, 4]
        assert kw["output_path"] == figure_output_paths(io)["admixture_colored_embedding"]
        assert result.figures == (figure_output_paths(io)["admixture_colored_embedding"],)
