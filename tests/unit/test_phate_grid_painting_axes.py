"""phate_grid's painting axes: genotype_weight and painting_knn as grid rows and columns.

Validation is tested directly and through a config file. The grid runs through
the orchestrator with the painting and PCA steps replaced by fakes and the
embeddings real, so each panel can be compared with the run it stands for.
"""

import pandas as pd
import pytest

from manifold_genetics.embeddings import PHATE
from manifold_genetics.embeddings.phate import GenotypePaintingPHATE
from manifold_genetics.pipeline.config import (
    GENOTYPE_KERNEL_PARAM,
    check_grid_painting_axes,
    grid_genotype_weights,
    validate_method_grid,
    validate_phate_grid,
)
from manifold_genetics.pipeline.configfile import ConfigFileError, load_config

from .test_painting_genotype_weight import _config, _run, cohorts  # noqa: F401

GRID = {"genotype_weight": [0, 0.5, 1], "painting_knn": [3, 5]}
RUN = dict(knn=10, t=5, n_landmark=None)


# --- validation ---------------------------------------------------------------------


def test_the_painting_axes_are_phate_grid_parameters():
    validate_phate_grid({"genotype_weight": [0, 0.25, 0.5, 0.75], "painting_knn": [3, 5, 10]})
    validate_phate_grid({"genotype_weight": [0, 1], "knn": [30, 100], "painting_knn": 5})
    validate_phate_grid({"painting_knn": [3, 5], "t": 10})


@pytest.mark.parametrize(
    "grid, match",
    [
        ({"genotype_weight": [0, 1.5]}, "genotype_weight must be a number from 0 to 1"),
        ({"genotype_weight": ["half"]}, "genotype_weight must be a number from 0 to 1"),
        ({"genotype_weight": [True]}, "genotype_weight must be a number from 0 to 1"),
        ({"painting_knn": [0, 5]}, "painting_knn must be a positive integer"),
        ({"painting_knn": [2.5]}, "painting_knn must be a positive integer"),
        ({"genotype_weight": [0, 1], "painting_knn": [3], "knn": [5]}, "one or two parameters"),
        ({"genotype_weight": [0], "paintng_knn": [3]}, "Unknown phate parameter"),
    ],
)
def test_bad_painting_axes_are_refused(grid, match):
    with pytest.raises(ValueError, match=match):
        validate_phate_grid(grid)


def test_method_grid_does_not_take_the_painting_axes():
    with pytest.raises(ValueError, match="Unknown phate parameter"):
        validate_method_grid({"phate": {"genotype_weight": 0.5}})


def test_the_grid_weights_are_the_grids_else_the_runs():
    assert grid_genotype_weights({"genotype_weight": [0, 0.5]}, 0.25) == [0.0, 0.5]
    assert grid_genotype_weights({"genotype_weight": 1}, 0.25) == [1.0]
    assert grid_genotype_weights({"painting_knn": [3, 5]}, 0.25) == [0.25]


def test_grids_without_painting_axes_are_not_checked():
    check_grid_painting_axes({"knn": [5, 10]}, False, 0, "both", {"n_landmark": 2000})
    check_grid_painting_axes(None, False, 0, "both")


def test_the_painting_axes_need_painting():
    with pytest.raises(ValueError, match="genotype_weight and painting_knn need painting"):
        check_grid_painting_axes(GRID, False, 0, "project")
    with pytest.raises(ValueError, match="painting_knn need painting"):
        check_grid_painting_axes({"painting_knn": [3, 5]}, False, 0, "project")


@pytest.mark.parametrize(
    "input_mode, params, grid, match",
    [
        ("both", {}, GRID, "input_mode both"),
        ("project", {"n_landmark": 2000}, GRID, "n_landmark to none"),
        ("project", {}, {**GRID, "n_landmark": [None, 50]}, "phate_grid cannot set n_landmark"),
    ],
)
def test_a_mixed_panel_inherits_the_mixing_refusals(input_mode, params, grid, match):
    with pytest.raises(ValueError, match=match):
        check_grid_painting_axes(grid, True, 0, input_mode, params)


def test_panels_at_zero_and_one_run_like_any_run():
    grid = {"genotype_weight": [0, 1], "knn": [5, 10]}
    check_grid_painting_axes(grid, True, 0, "both", {"n_landmark": 2000})


def test_the_grids_n_landmark_none_replaces_the_runs():
    check_grid_painting_axes({**GRID, "n_landmark": None}, True, 0, "project", {"n_landmark": 2000})


def test_a_painting_knn_axis_takes_the_runs_mixing_weight():
    with pytest.raises(ValueError, match="input_mode both"):
        check_grid_painting_axes({"painting_knn": [3, 5]}, True, 0.5, "both")


def test_a_run_at_weight_one_does_not_paint_for_its_grid():
    with pytest.raises(ValueError, match="painting genotype_weight 1 skips"):
        check_grid_painting_axes({"genotype_weight": [0.5, 1]}, True, 1, "project")
    check_grid_painting_axes({"genotype_weight": [1], "knn": [5, 10]}, True, 1, "project")


@pytest.mark.parametrize(
    "grid, weight",
    [({"genotype_weight": [0, 1], "painting_knn": [3, 5]}, 0), ({"painting_knn": [3, 5]}, 0)],
)
def test_painting_knn_without_a_mixed_panel_is_reported(caplog, grid, weight):
    check_grid_painting_axes(grid, True, weight, "project")
    assert "phate_grid painting_knn has no effect" in caplog.text


def test_painting_knn_with_a_mixed_panel_is_not_reported(caplog):
    check_grid_painting_axes(GRID, True, 0, "project")
    assert "no effect" not in caplog.text


def test_a_config_file_is_checked_when_read(tmp_path):
    viz = {"phate_grid": {"genotype_weight": [0, 0.5], "painting_knn": [3, 5]}}
    kwargs = load_config(_config(tmp_path, {"enabled": True}, {"input_mode": "project"}, None, viz))
    assert kwargs["phate_grid"] == viz["phate_grid"]

    with pytest.raises(ConfigFileError, match="need painting"):
        load_config(_config(tmp_path, {"enabled": False}, {"input_mode": "project"}, None, viz))
    with pytest.raises(ConfigFileError, match="input_mode both"):
        load_config(_config(tmp_path, {"enabled": True}, None, None, viz))
    with pytest.raises(ConfigFileError, match="Unknown phate parameter"):
        load_config(
            _config(tmp_path, {"enabled": True}, None, None, {"phate_grid": {"weight": [0, 1]}})
        )


def test_a_run_is_checked_before_it_starts(tmp_path, cohorts, monkeypatch):  # noqa: F811
    with pytest.raises(ValueError, match="need painting"):
        _run(tmp_path, cohorts, monkeypatch, painting=False, phate_grid=GRID)


# --- the grid -----------------------------------------------------------------------


@pytest.fixture
def grid_run(tmp_path, cohorts, monkeypatch):  # noqa: F811
    """Run a grid with real embeddings, recording each embedding, kernel build and plot."""
    import manifold_genetics.pipeline.steps.viz as viz

    rec = {"embeds": [], "plots": [], "kernels": []}
    real_embedding, real_kernel = viz.run_embedding, GenotypePaintingPHATE.view_kernel

    def embedding(fit, project, *, project_output, method, params):
        rec["embeds"].append(dict(fit=fit, out=project_output.name, **params))
        return real_embedding(
            fit, project, project_output=project_output, method=method, params=params
        )

    def kernel(self, X, knn):
        rec["kernels"].append((X.shape[1], knn))
        return real_kernel(self, X, knn)

    monkeypatch.setattr(viz, "run_embedding", embedding)
    monkeypatch.setattr(GenotypePaintingPHATE, "view_kernel", kernel)
    monkeypatch.setattr(viz, "plot_embedding_grid", lambda **kw: rec["plots"].append(kw))
    monkeypatch.setattr(viz, "read_colormap", lambda p: {"Population": {}})

    def run(**kwargs):
        kwargs.setdefault("skip_embedding", True)
        kwargs.setdefault("embedding_params", dict(RUN))
        rec["seen"] = _run(
            tmp_path, cohorts, monkeypatch, skip_visualization=False, real_embedding=True, **kwargs
        )
        return rec

    rec["dir"] = tmp_path / "out" / "embeddings" / "grids"
    return run


def _panel(rec, name):
    return pd.read_csv(rec["dir"] / f"{name}.csv", dtype={"sample_id": str})


def test_the_grid_has_a_panel_per_weight_and_painting_knn(grid_run):
    rec = grid_run(phate_grid=GRID)

    (plot,) = rec["plots"]
    assert [len(row) for row in plot["embeddings"]] == [2, 2, 2]
    assert plot["row_titles"] == [
        "genotype_weight = 0",
        "genotype_weight = 0.5",
        "genotype_weight = 1",
    ]
    assert plot["col_titles"] == ["painting_knn = 3", "painting_knn = 5"]
    assert plot["panel_notes"] == [
        ["painting only, painting_knn unused"] * 2,
        [None, None],
        ["genotype PCA only, painting_knn unused"] * 2,
    ]
    assert all(cell is not None for row in plot["embeddings"] for cell in row)
    names = sorted(p.name for p in rec["dir"].glob("*.csv"))
    assert names == sorted(
        f"phate_grid_genotype_weight-{w}_painting_knn-{k}.csv" for w in (0, 0.5, 1) for k in (3, 5)
    )


def test_each_panel_runs_as_a_run_with_its_values(grid_run, cohorts):  # noqa: F811
    geno, paint = cohorts
    rec = grid_run(phate_grid=GRID)

    # Weights 0 and 1 ignore painting_knn, so each of those rows is one embedding.
    by_name = {e["out"]: e for e in rec["embeds"]}
    assert sorted(by_name) == [
        "phate_grid_genotype_weight-0.5_painting_knn-3.csv",
        "phate_grid_genotype_weight-0.5_painting_knn-5.csv",
        "phate_grid_genotype_weight-0_painting_knn-3.csv",
        "phate_grid_genotype_weight-1_painting_knn-3.csv",
    ]
    w0 = by_name["phate_grid_genotype_weight-0_painting_knn-3.csv"]
    assert w0["fit"] == paint.project_pca and GENOTYPE_KERNEL_PARAM not in w0
    assert (w0["knn"], w0["t"]) == (10, 5)
    w1 = by_name["phate_grid_genotype_weight-1_painting_knn-3.csv"]
    assert w1["fit"].name == "project_genotype_pca_top10.csv"
    assert GENOTYPE_KERNEL_PARAM not in w1
    for k in (3, 5):
        mixed = by_name[f"phate_grid_genotype_weight-0.5_painting_knn-{k}.csv"]
        kernel = mixed[GENOTYPE_KERNEL_PARAM]
        assert mixed["fit"] == paint.project_pca
        assert (kernel.genotype, kernel.weight, kernel.painting_knn) == (geno.project_pca, 0.5, k)
        assert (mixed["knn"], mixed["t"]) == (10, 5)
    # The genotype PCA ran once, for the grid alone.
    assert rec["seen"]["pca_runs"] == 1


def test_the_weight_zero_panel_is_painting_only_phate(grid_run, cohorts):  # noqa: F811
    _, paint = cohorts
    rec = grid_run(phate_grid=GRID)

    expected = PHATE(**RUN).fit_transform(paint.project_pca)
    for k in (3, 5):
        got = _panel(rec, f"phate_grid_genotype_weight-0_painting_knn-{k}")
        pd.testing.assert_frame_equal(got, expected)


def test_the_weight_one_and_mixed_panels_match_their_runs(grid_run, cohorts):  # noqa: F811
    geno, paint = cohorts
    rec = grid_run(phate_grid=GRID)

    top = pd.read_csv(geno.project_pca, dtype={"sample_id": str}).iloc[:, :11]
    expected = PHATE(**RUN).fit_transform(top)
    pd.testing.assert_frame_equal(
        _panel(rec, "phate_grid_genotype_weight-1_painting_knn-5"), expected
    )
    mixed = GenotypePaintingPHATE(
        genotype=geno.project_pca, genotype_weight=0.5, painting_knn=3, **RUN
    ).fit_transform(paint.project_pca)
    got = _panel(rec, "phate_grid_genotype_weight-0.5_painting_knn-3")
    pd.testing.assert_frame_equal(got, mixed)


def test_each_kernel_is_built_once(grid_run):
    rec = grid_run(phate_grid={"genotype_weight": [0.25, 0.5, 0.75], "painting_knn": [3, 5, 10]})

    # Nine mixed panels: one genotype kernel (10 PCs, knn 10) and one painting
    # kernel (12 coordinates) per painting_knn, rather than eighteen kernels.
    assert sorted(rec["kernels"]) == [(10, 10), (12, 3), (12, 5), (12, 10)]
    assert len(rec["embeds"]) == 9


def test_the_grid_reuses_the_runs_kernels(grid_run):
    rec = grid_run(
        skip_embedding=False,
        painting_genotype_weight=0.5,
        painting_knn=5,
        phate_grid={"painting_knn": [5, 10]},
    )

    assert sorted(rec["kernels"]) == [(10, 10), (12, 5), (12, 10)]
    (plot,) = rec["plots"]
    assert plot["row_titles"] is None
    assert plot["col_titles"] == ["painting_knn = 5", "painting_knn = 10"]
    assert [e[GENOTYPE_KERNEL_PARAM].weight for e in rec["embeds"]] == [0.5, 0.5]


def test_a_weight_one_panel_drops_the_painting_t_default(grid_run):
    rec = grid_run(
        embedding_params={"knn": 10, "n_landmark": None}, phate_grid={"genotype_weight": [0, 1]}
    )

    by_name = {e["out"]: e for e in rec["embeds"]}
    assert by_name["phate_grid_genotype_weight-0.csv"]["t"] == 10
    assert "t" not in by_name["phate_grid_genotype_weight-1.csv"]
    assert rec["plots"][0]["panel_notes"] == [["painting only", "genotype PCA only"]]
