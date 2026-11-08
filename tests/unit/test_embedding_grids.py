"""Tests for the embedding comparison grids.

Two figures a pipeline run can add: ``method_grid`` (one column per method,
each with its own parameters) and ``phate_grid`` (PHATE over one or two
parameters). The step tests stub the embedding and the plotting so they check
the layout and the parameters each panel receives; one test renders a real grid.
"""

from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd
import pytest
import yaml

from manifold_genetics.pipeline.config import (
    IOConfig,
    VizConfig,
    build_configs,
    validate_method_grid,
    validate_phate_grid,
)
from manifold_genetics.pipeline.configfile import ConfigFileError, load_config
from manifold_genetics.pipeline.steps.pca import PCAStepResult
from manifold_genetics.pipeline.steps.viz import run_embedding_grid_viz_step
from manifold_genetics.visualization import plot_embedding_grid

MODULE = "manifold_genetics.pipeline.steps.viz"


# ---------------------------------------------------------------------------
# plot_embedding_grid -- real rendering
# ---------------------------------------------------------------------------


class TestPlotEmbeddingGrid:
    def test_writes_a_figure_for_a_two_by_three_grid(
        self, temp_dir, small_embedding_data, labels_csv, colormap_json
    ):
        grid = [[small_embedding_data] * 3, [small_embedding_data] * 3]

        out = plot_embedding_grid(
            grid,
            labels_csv,
            colormap_json,
            temp_dir / "grid.png",
            label_column="Population",
            row_titles=["a", "b"],
            col_titles=["PHATE", "UMAP", "t-SNE"],
            axis_names=[["PHATE", "UMAP", "t-SNE"]] * 2,
            title="Comparison",
        )

        assert out.exists()
        height, width = mpimg.imread(out).shape[:2]
        assert width > height, "three columns should be wider than two rows are tall"

    def test_a_missing_panel_is_drawn_empty_rather_than_failing(
        self, temp_dir, small_embedding_data, labels_csv, colormap_json
    ):
        out = plot_embedding_grid(
            [[small_embedding_data, None]], labels_csv, colormap_json, temp_dir / "grid.png"
        )

        assert out.exists()

    def test_ragged_rows_are_rejected(
        self, temp_dir, small_embedding_data, labels_csv, colormap_json
    ):
        with pytest.raises(ValueError, match="rectangular"):
            plot_embedding_grid(
                [[small_embedding_data], [small_embedding_data] * 2],
                labels_csv,
                colormap_json,
                temp_dir / "grid.png",
            )

    def test_an_unknown_label_column_is_rejected(
        self, temp_dir, small_embedding_data, labels_csv, colormap_json
    ):
        with pytest.raises(ValueError, match="Ancestry"):
            plot_embedding_grid(
                [[small_embedding_data]],
                labels_csv,
                colormap_json,
                temp_dir / "grid.png",
                label_column="Ancestry",
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_a_well_formed_method_grid_passes(self):
        validate_method_grid({"phate": {"knn": 5}, "umap": None, "tsne": {"perplexity": 10}})

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="pca"):
            validate_method_grid({"pca": {}})

    def test_a_parameter_of_another_method_is_rejected(self):
        with pytest.raises(ValueError, match="perplexity"):
            validate_method_grid({"umap": {"perplexity": 30}})

    def test_n_components_is_not_settable_in_a_grid(self):
        with pytest.raises(ValueError, match="n_components"):
            validate_method_grid({"phate": {"n_components": 3}})

    def test_a_phate_grid_sweeps_one_or_two_parameters(self):
        validate_phate_grid({"knn": [5, 25]})
        validate_phate_grid({"knn": [5, 25], "t": [3, "auto"], "gamma": 0})

    @pytest.mark.parametrize(
        "grid",
        [{"knn": 5}, {"knn": [5], "t": [3], "gamma": [0, 1]}],
        ids=["none-swept", "three-swept"],
    )
    def test_a_phate_grid_must_sweep_one_or_two(self, grid):
        with pytest.raises(ValueError, match="one or two"):
            validate_phate_grid(grid)

    def test_an_empty_sweep_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            validate_phate_grid({"knn": []})

    def test_build_configs_validates_and_carries_the_grids(self):
        common = dict(fit_plink="f", project_plink="p", output_dir="o", labels="l", colormap="c")

        pc = build_configs(**common, method_grid={"tsne": {}}, phate_grid={"t": [1, 2]})
        assert pc.viz.method_grid == {"tsne": {}}
        assert pc.viz.phate_grid == {"t": [1, 2]}

        with pytest.raises(ValueError):
            build_configs(**common, phate_grid={"nope": [1]})

    def test_the_config_file_visualization_section_carries_the_grids(self, tmp_path):
        config = {
            "data": {
                "fit_plink": "f",
                "project_plink": "p",
                "labels": "l.csv",
                "colormap": "c.json",
                "output_dir": "o",
            },
            "visualization": {
                "method_grid": {"phate": {"knn": 25}, "umap": {"n_neighbors": 15}},
                "phate_grid": {"knn": [5, 25], "t": [3, "auto"]},
            },
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config))

        kwargs = load_config(path)

        assert kwargs["method_grid"] == {"phate": {"knn": 25}, "umap": {"n_neighbors": 15}}
        assert kwargs["phate_grid"] == {"knn": [5, 25], "t": [3, "auto"]}


# ---------------------------------------------------------------------------
# run_embedding_grid_viz_step -- layout and parameters, nothing computed
# ---------------------------------------------------------------------------


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


PCA = PCAStepResult(fit_pca=Path("fit_pca.csv"), project_pca=Path("project_pca.csv"))


@pytest.fixture
def stubs(monkeypatch):
    """Record every embedding and plotting call; fail embeddings named in ``fail``."""
    rec = {"embeds": [], "plots": [], "fail": set()}

    def fake_run_embedding(fit_input, project_input, *, project_output, method, params):
        rec["embeds"].append(
            dict(fit=fit_input, project=project_input, out=project_output, method=method, **params)
        )
        if method in rec["fail"]:
            raise ValueError("boom")
        return pd.DataFrame()

    def fake_plot(**kwargs):
        rec["plots"].append(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(f"{MODULE}.run_embedding", fake_run_embedding)
    monkeypatch.setattr(f"{MODULE}.plot_embedding_grid", fake_plot)
    monkeypatch.setattr(f"{MODULE}.read_colormap", lambda p: {"Population": {}, "Region": {}})
    return rec


class TestEmbeddingGridStep:
    def test_nothing_configured_does_nothing(self, tmp_path, stubs):
        result = run_embedding_grid_viz_step(
            make_io(tmp_path), VizConfig(), pca=PCA, input_mode="both"
        )

        assert result.figures == ()
        assert stubs["embeds"] == []

    def test_method_grid_is_one_row_one_column_per_method(self, tmp_path, stubs):
        viz = VizConfig(method_grid={"phate": {"knn": 7}, "umap": None, "tsne": {"perplexity": 9}})

        result = run_embedding_grid_viz_step(make_io(tmp_path), viz, pca=PCA, input_mode="both")

        assert [e["method"] for e in stubs["embeds"]] == ["phate", "umap", "tsne"]
        assert stubs["embeds"][0]["knn"] == 7
        assert stubs["embeds"][2]["perplexity"] == 9
        # Same inputs as the run's own embedding in "both" mode.
        assert {(e["fit"], e["project"]) for e in stubs["embeds"]} == {
            (PCA.fit_pca, PCA.project_pca)
        }
        assert all(e["n_components"] == 2 for e in stubs["embeds"])

        # One figure per colormap column, each with the full 1x3 grid.
        assert [p["label_column"] for p in stubs["plots"]] == ["Population", "Region"]
        assert len(stubs["plots"][0]["embeddings"]) == 1
        assert len(stubs["plots"][0]["embeddings"][0]) == 3
        assert stubs["plots"][0]["col_titles"] == ["PHATE", "UMAP", "t-SNE"]
        assert [f.name for f in result.figures] == [
            "method_grid_by_Population.png",
            "method_grid_by_Region.png",
        ]

    def test_phate_grid_rows_are_the_first_parameter_columns_the_second(self, tmp_path, stubs):
        viz = VizConfig(phate_grid={"knn": [5, 50], "t": [3, 10, "auto"], "gamma": 0})

        run_embedding_grid_viz_step(
            make_io(tmp_path),
            viz,
            pca=PCA,
            input_mode="both",
            run_params={"knn": 100, "n_landmark": 2000, "n_components": 3},
        )

        grid = stubs["plots"][0]["embeddings"]
        assert len(grid) == 2 and all(len(row) == 3 for row in grid)
        assert stubs["plots"][0]["row_titles"] == ["knn = 5", "knn = 50"]
        assert stubs["plots"][0]["col_titles"] == ["t = 3", "t = 10", "t = auto"]

        first = stubs["embeds"][0]
        assert (first["knn"], first["t"]) == (5, 3)
        # The run's own settings carry over; the grid's scalars override them;
        # and every panel is 2-D whatever the run's dimensionality.
        assert first["n_landmark"] == 2000
        assert first["gamma"] == 0
        assert first["n_components"] == 2
        assert first["out"].name == "phate_grid_knn-5_t-3.csv"

    def test_one_swept_parameter_is_a_single_row(self, tmp_path, stubs):
        viz = VizConfig(phate_grid={"t": [3, 10]})

        run_embedding_grid_viz_step(make_io(tmp_path), viz, pca=PCA, input_mode="project")

        assert len(stubs["plots"][0]["embeddings"]) == 1
        assert stubs["plots"][0]["row_titles"] is None
        assert stubs["embeds"][0]["fit"] == PCA.project_pca
        assert stubs["embeds"][0]["out"].name == "phate_grid_t-3.csv"

    def test_fit_mode_colours_by_the_fit_cohort(self, tmp_path, stubs):
        run_embedding_grid_viz_step(
            make_io(tmp_path), VizConfig(phate_grid={"t": [3]}), pca=PCA, input_mode="fit"
        )

        assert stubs["plots"][0]["labels"] == Path("fl.csv")
        assert stubs["plots"][0]["colormap"] == Path("fc.json")

    def test_a_failed_panel_is_left_empty_and_reported(self, tmp_path, stubs):
        stubs["fail"].add("tsne")
        viz = VizConfig(method_grid={"phate": {}, "tsne": {}})

        result = run_embedding_grid_viz_step(make_io(tmp_path), viz, pca=PCA, input_mode="both")

        row = stubs["plots"][0]["embeddings"][0]
        assert row[0] is not None and row[1] is None
        assert result.failed_substeps == ("method_grid_tsne",)
        assert len(result.figures) == 2

    def test_missing_pca_raises_for_the_orchestrator_to_catch(self, tmp_path, stubs):
        with pytest.raises(RuntimeError, match="PCA"):
            run_embedding_grid_viz_step(
                make_io(tmp_path),
                VizConfig(phate_grid={"t": [3]}),
                pca=PCAStepResult(fit_pca=None, project_pca=None),
                input_mode="both",
            )


# ---------------------------------------------------------------------------
# Review fixes: colours, landmarking, config normalisation, failure handling
# ---------------------------------------------------------------------------


@pytest.fixture
def scatter_calls(monkeypatch):
    """Every ``Axes.scatter`` call as (number of points, colour), still drawing."""
    from matplotlib.axes import Axes

    calls = []
    real = Axes.scatter

    def spy(self, x, y, *args, **kwargs):
        calls.append((len(x), kwargs.get("color")))
        return real(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", spy)
    return calls


class TestGridColours:
    def test_a_label_missing_from_the_colormap_is_drawn_grey_not_dropped(
        self, temp_dir, scatter_calls
    ):
        embedding = pd.DataFrame(
            {"sample_id": ["a", "b", "c", "d"], "dim_1": [0, 1, 2, 3], "dim_2": [0, 1, 0, 1]}
        )
        labels = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "pop": ["X", "X", "Y", "Z"]})

        plot_embedding_grid([[embedding]], labels, {"pop": {"X": "#ff0000"}}, temp_dir / "g.png")

        assert sum(n for n, _ in scatter_calls) == 4, "every sample must be drawn"
        assert (2, "lightgray") in scatter_calls

    def test_numeric_looking_labels_match_the_colormaps_string_keys(self, temp_dir, scatter_calls):
        embedding = pd.DataFrame({"sample_id": ["a", "b", "c"], "dim_1": [0, 1, 2], "dim_2": 0})
        # read_csv without dtype=str gives integers, and JSON colormap keys are strings.
        labels = pd.DataFrame({"sample_id": ["a", "b", "c"], "k": [1, 2, 2]})

        plot_embedding_grid(
            [[embedding]], labels, {"k": {"1": "#ff0000", "2": "#0000ff"}}, temp_dir / "g.png"
        )

        assert sorted(scatter_calls) == [(1, "#ff0000"), (2, "#0000ff")]


class TestGridLandmarking:
    def test_method_grid_phate_panel_inherits_the_runs_landmarking(self, tmp_path, stubs):
        viz = VizConfig(method_grid={"phate": {"knn": 7}, "umap": None})

        run_embedding_grid_viz_step(
            make_io(tmp_path),
            viz,
            pca=PCA,
            input_mode="both",
            run_method="umap",
            run_params={"n_landmark": 10000, "random_landmarking": True, "n_neighbors": 30},
        )

        phate, umap = stubs["embeds"]
        assert phate["n_landmark"] == 10000
        assert phate["random_landmarking"] is True
        assert phate["knn"] == 7
        assert "n_landmark" not in umap and "n_neighbors" not in umap

    def test_phate_grid_on_a_non_phate_run_takes_only_the_landmarking(self, tmp_path, stubs):
        run_embedding_grid_viz_step(
            make_io(tmp_path),
            VizConfig(phate_grid={"t": [3]}),
            pca=PCA,
            input_mode="both",
            run_method="diffusion_map",
            run_params={"knn": 99, "n_landmark": 500, "random_landmarking": True},
        )

        (panel,) = stubs["embeds"]
        assert panel["n_landmark"] == 500 and panel["random_landmarking"] is True
        assert "knn" not in panel, "diffusion_map's knn is not a PHATE setting"

    def test_a_grid_override_wins_and_an_exact_panel_drops_random_landmarking(
        self, tmp_path, stubs
    ):
        run_embedding_grid_viz_step(
            make_io(tmp_path),
            VizConfig(phate_grid={"n_landmark": [None, 200]}),
            pca=PCA,
            input_mode="both",
            run_params={"n_landmark": 10000, "random_landmarking": True},
        )

        exact, landmarked = stubs["embeds"]
        assert exact["n_landmark"] is None and "random_landmarking" not in exact
        assert landmarked["n_landmark"] == 200 and landmarked["random_landmarking"] is True


def write_grid_config(tmp_path, visualization, embedding=None):
    config = {
        "data": {
            "fit_plink": "f",
            "project_plink": "p",
            "labels": "l.csv",
            "colormap": "c.json",
            "output_dir": "o",
        },
        "visualization": visualization,
    }
    if embedding is not None:
        config["embedding"] = embedding
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class TestGridConfigNormalisation:
    def test_yaml_none_is_none_in_both_grids(self, tmp_path):
        path = write_grid_config(
            tmp_path,
            {
                "method_grid": {"phate": {"n_landmark": "none"}},
                "phate_grid": {"n_landmark": ["none", 200]},
            },
        )

        kwargs = load_config(path)

        assert kwargs["method_grid"] == {"phate": {"n_landmark": None}}
        assert kwargs["phate_grid"] == {"n_landmark": [None, 200]}

    @pytest.mark.parametrize(
        "visualization, embedding",
        [
            ({"phate_grid": {"t": [3], "random_landmarking": True}}, None),
            ({"phate_grid": {"n_landmark": ["none", 50], "random_landmarking": True}}, None),
            ({"method_grid": {"phate": {"random_landmarking": True}}}, {"n_landmark": "none"}),
        ],
        ids=["run-has-none", "swept-none", "method-grid"],
    )
    def test_random_landmarking_without_n_landmark_is_refused(
        self, tmp_path, visualization, embedding
    ):
        path = write_grid_config(tmp_path, visualization, embedding)

        with pytest.raises(ConfigFileError, match="random_landmarking is set"):
            load_config(path)

    def test_random_landmarking_with_the_runs_n_landmark_is_accepted(self, tmp_path):
        path = write_grid_config(
            tmp_path,
            {"phate_grid": {"t": [3, 10], "random_landmarking": True}},
            {"n_landmark": 2000},
        )

        assert load_config(path)["phate_grid"]["random_landmarking"] is True

    @pytest.mark.parametrize(
        "grid, message",
        [
            ({"phate_grid": {"knn": ["five"]}}, "knn must be a positive integer"),
            ({"phate_grid": {"t": [3, "sometimes"]}}, "t must be a positive integer or 'auto'"),
            ({"method_grid": {"tsne": {"perplexity": -1}}}, "perplexity must be a positive"),
            ({"method_grid": {"phate": {"knn": [5, 10]}}}, "one value per parameter"),
            ({"method_grid": {"phate": {"random_landmarking": "yes"}}}, "true or false"),
        ],
        ids=["knn", "t", "perplexity", "list-in-method-grid", "bool"],
    )
    def test_a_wrongly_typed_value_is_refused(self, tmp_path, grid, message):
        with pytest.raises(ConfigFileError, match=message):
            load_config(write_grid_config(tmp_path, grid))

    def test_build_configs_checks_against_the_runs_parameters(self):
        common = dict(fit_plink="f", project_plink="p", output_dir="o", labels="l", colormap="c")

        with pytest.raises(ValueError, match="random_landmarking"):
            build_configs(**common, phate_grid={"t": [3], "random_landmarking": True})
        build_configs(
            **common,
            embedding_params={"n_landmark": 100},
            phate_grid={"t": [3], "random_landmarking": True},
        )


class TestGridCellFailures:
    def test_out_of_memory_is_not_recorded_as_one_failed_panel(self, tmp_path, stubs, monkeypatch):
        def oom(*args, **kwargs):
            raise MemoryError

        monkeypatch.setattr(f"{MODULE}.run_embedding", oom)

        with pytest.raises(MemoryError):
            run_embedding_grid_viz_step(
                make_io(tmp_path), VizConfig(phate_grid={"t": [3]}), pca=PCA, input_mode="both"
            )

    def test_a_failed_panel_leaves_no_stale_csv_from_an_earlier_run(self, tmp_path, stubs):
        io = make_io(tmp_path)
        stale = io.output_dir / "embeddings" / "grids" / "method_grid_tsne.csv"
        stale.parent.mkdir(parents=True)
        stale.write_text("sample_id,dim_1,dim_2\nold,0,0\n")
        stubs["fail"].add("tsne")

        result = run_embedding_grid_viz_step(
            io, VizConfig(method_grid={"tsne": {}}), pca=PCA, input_mode="both"
        )

        assert result.failed_substeps == ("method_grid_tsne",)
        assert not stale.exists()


# ---------------------------------------------------------------------------
# End to end: a real, tiny grid through run_pipeline
# ---------------------------------------------------------------------------


def test_a_real_grid_runs_end_to_end_through_run_pipeline(tmp_path):
    """PLINK in, then PCA, the run's embedding and both grids, plotted for real.

    Small enough to stay fast: 90 samples and landmarked PHATE. t-SNE's
    perplexity is set above the cohort size, so that one panel fails and must be
    reported while the rest of the figure is still written.
    """
    import json

    import numpy as np

    from manifold_genetics.pipeline.runner import run_pipeline
    from manifold_genetics.scaffold import write_bed

    rng = np.random.default_rng(0)
    groups = np.repeat(["A", "B", "C"], 30)
    freqs = {g: rng.uniform(0.05, 0.95, 300) for g in "ABC"}
    dosages = np.stack([rng.binomial(2, freqs[g]) for g in groups])
    ids = [f"S{i:03d}" for i in range(len(groups))]
    data = tmp_path / "data"
    data.mkdir()
    write_bed(data / "cohort", dosages, ids)
    # Group C has no colour, so it must come out grey rather than vanish.
    pd.DataFrame({"sample_id": ids, "group": groups}).to_csv(data / "labels.csv", index=False)
    (data / "colormap.json").write_text(json.dumps({"group": {"A": "#d62728", "B": "#1f77b4"}}))

    result = run_pipeline(
        fit_plink=data / "cohort",
        project_plink=data / "cohort",
        output_dir=tmp_path / "out",
        labels=data / "labels.csv",
        colormap=data / "colormap.json",
        n_pcs=5,
        embedding_input="project",
        embedding_params={"knn": 5, "t": 5, "n_landmark": 40, "random_landmarking": True},
        method_grid={"phate": {}, "umap": {"n_neighbors": 10}, "tsne": {"perplexity": 500}},
        phate_grid={"knn": [5, 10], "t": [5, 10]},
        skip_admixture=True,
        skip_metrics=True,
        skip_pca_visualization=True,
    )

    assert result.failed_steps == ("method_grid_tsne",)
    grids = tmp_path / "out" / "figures" / "embeddings" / "grids"
    assert sorted(p.name for p in grids.iterdir()) == [
        "method_grid_by_group.png",
        "phate_grid_by_group.png",
    ]
    assert set(result.embedding_grid_figures) == set(grids.iterdir())
    panels = tmp_path / "out" / "embeddings" / "grids"
    assert sorted(p.name for p in panels.iterdir()) == [
        "method_grid_phate.csv",
        "method_grid_umap.csv",
        "phate_grid_knn-10_t-10.csv",
        "phate_grid_knn-10_t-5.csv",
        "phate_grid_knn-5_t-10.csv",
        "phate_grid_knn-5_t-5.csv",
    ]
    assert len(pd.read_csv(panels / "method_grid_phate.csv")) == 90
