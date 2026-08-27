"""Tests for plot_projection and plot_admixture_embedding_grid.

plotting.py forces the Agg backend at import. Rather than only asserting a
PNG lands (which cannot tell a correct plot from a wrong one), these record
every Axes.scatter call and assert what was actually drawn: which colormap
coloured which dataset, which marker, which z-order layer, and that the
missing-data layer is present. The real scatter still runs, so genuine
rendering crashes are also caught.
"""

import json

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import LinearSegmentedColormap

from manifold_genetics.visualization.plotting import (
    plot_admixture_embedding_grid,
    plot_projection,
)


@pytest.fixture
def scatter_calls(monkeypatch):
    """Record kwargs + point count of every Axes.scatter call; still render."""
    calls = []
    real = matplotlib.axes.Axes.scatter

    def recording_scatter(self, x, y=None, *args, **kwargs):
        n = len(x) if hasattr(x, "__len__") else 1
        calls.append({"n": n, "c": kwargs.get("c"), **kwargs})
        return real(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
    return calls


def _emb(ids, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "sample_id": [str(i) for i in ids],
            "dim_1": rng.normal(size=len(ids)),
            "dim_2": rng.normal(size=len(ids)),
        }
    )


# ---------------------------------------------------------------------------
# plot_projection
# ---------------------------------------------------------------------------

# Distinct colormaps so "fit coloured by project map" is detectable.
FIT_CMAP = {"Region": {"A": "#aa0000", "B": "#00aa00", "C": "#0000aa"}}
PROJECT_CMAP = {"Region": {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}}
FIT_HEXES = set(FIT_CMAP["Region"].values())
PROJECT_HEXES = set(PROJECT_CMAP["Region"].values())


@pytest.fixture
def projection_inputs():
    fit_emb = _emb(range(6), seed=1)
    proj_emb = _emb(range(100, 106), seed=2)
    fit_labels = pd.DataFrame(
        {"sample_id": [str(i) for i in range(6)], "Region": ["A", "A", "B", "B", "C", "C"]}
    )
    proj_labels = pd.DataFrame(
        {
            "sample_id": [str(i) for i in range(100, 106)],
            "Region": ["A", "B", "A", "B", "C", "C"],
        }
    )
    return fit_emb, proj_emb, fit_labels, proj_labels


def _run_projection(inputs, out, **overrides):
    fit_emb, proj_emb, fit_labels, proj_labels = inputs
    kwargs = dict(
        fit_embedding=fit_emb,
        project_embedding=proj_emb,
        fit_labels=fit_labels,
        project_labels=proj_labels,
        fit_colormap=FIT_CMAP,
        project_colormap=PROJECT_CMAP,
        output_path=out,
        fit_label_column="Region",
        project_label_column="Region",
    )
    kwargs.update(overrides)
    return plot_projection(**kwargs)


def test_projection_colors_each_dataset_with_its_own_colormap(
    projection_inputs, scatter_calls, tmp_path
):
    _run_projection(projection_inputs, tmp_path / "p.png")

    fit_group_calls = [c for c in scatter_calls if c.get("marker") == "^" and c.get("zorder") == 2]
    proj_group_calls = [c for c in scatter_calls if c.get("marker") == "o" and c.get("zorder") == 2]

    assert fit_group_calls, "no coloured fit-group scatter calls recorded"
    assert proj_group_calls, "no coloured project-group scatter calls recorded"

    # Every fit group must use a colour from the FIT colormap (not the project one)
    for c in fit_group_calls:
        assert (
            c["edgecolors"] in FIT_HEXES
        ), f"fit group drawn with non-fit colour {c['edgecolors']}"
    for c in proj_group_calls:
        assert c["edgecolors"] in PROJECT_HEXES


def test_projection_uses_distinct_markers_and_layers(projection_inputs, scatter_calls, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels = projection_inputs
    fit_labels.loc[0, "Region"] = np.nan  # force the missing-data layer to draw
    _run_projection((fit_emb, proj_emb, fit_labels, proj_labels), tmp_path / "p.png")
    markers = {c.get("marker") for c in scatter_calls}
    assert "^" in markers and "o" in markers
    zorders = {c.get("zorder") for c in scatter_calls if c.get("marker")}
    assert zorders == {1, 2}  # missing-data layer (1) below coloured groups (2)
    # coloured groups must sit above the missing-data layer
    assert all(
        c["zorder"] == 2 for c in scatter_calls if c.get("edgecolors") in FIT_HEXES | PROJECT_HEXES
    )


def test_projection_draws_missing_data_layer(projection_inputs, scatter_calls, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels = projection_inputs
    fit_labels.loc[0, "Region"] = np.nan
    proj_labels.loc[0, "Region"] = np.nan
    _run_projection((fit_emb, proj_emb, fit_labels, proj_labels), tmp_path / "p.png")
    gray_calls = [
        c for c in scatter_calls if c.get("edgecolors") == "lightgray" and c.get("zorder") == 1
    ]
    assert gray_calls, "missing-label samples were not drawn as a lightgray layer"
    assert sum(c["n"] for c in gray_calls) >= 2  # one missing fit + one missing project


def test_projection_plots_groups_in_reverse_colormap_order(
    projection_inputs, scatter_calls, tmp_path
):
    _run_projection(projection_inputs, tmp_path / "p.png")
    fit_colors_in_call_order = [
        c["edgecolors"] for c in scatter_calls if c.get("marker") == "^" and c.get("zorder") == 2
    ]
    # colormap order is A,B,C (#aa0000,#00aa00,#0000aa); code iterates reversed()
    assert fit_colors_in_call_order == ["#0000aa", "#00aa00", "#aa0000"]


def test_projection_no_legend_skips_legend(projection_inputs, monkeypatch, tmp_path):
    import matplotlib.axes

    legend_calls = []
    real_legend = matplotlib.axes.Axes.legend
    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "legend",
        lambda self, *a, **k: legend_calls.append(1) or real_legend(self, *a, **k),
    )
    _run_projection(projection_inputs, tmp_path / "p.png", show_legend=False)
    assert legend_calls == []


def test_projection_bad_fit_column_raises(projection_inputs, tmp_path):
    with pytest.raises(ValueError, match="Fit column 'Nope' not in fit colormap"):
        _run_projection(projection_inputs, tmp_path / "x.png", fit_label_column="Nope")


def test_projection_bad_project_column_raises(projection_inputs, tmp_path):
    with pytest.raises(ValueError, match="Project column 'Nope'"):
        _run_projection(projection_inputs, tmp_path / "x.png", project_label_column="Nope")


def test_projection_reads_from_paths(projection_inputs, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels = projection_inputs
    paths = {}
    for name, df in [
        ("fe", fit_emb),
        ("pe", proj_emb),
        ("fl", fit_labels),
        ("pl", proj_labels),
    ]:
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    fc = tmp_path / "fc.json"
    pc = tmp_path / "pc.json"
    fc.write_text(json.dumps(FIT_CMAP))
    pc.write_text(json.dumps(PROJECT_CMAP))
    out = tmp_path / "proj_paths.png"
    plot_projection(
        fit_embedding=paths["fe"],
        project_embedding=paths["pe"],
        fit_labels=paths["fl"],
        project_labels=paths["pl"],
        fit_colormap=fc,
        project_colormap=pc,
        output_path=out,
        fit_label_column="Region",
        project_label_column="Region",
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# plot_admixture_embedding_grid
# ---------------------------------------------------------------------------


def _write_q_files(tmp_path, ids, ks=(2, 3), seed=0):
    rng = np.random.default_rng(seed)
    prefix = tmp_path / "q" / "transform"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frames = {}
    for k in ks:
        q = rng.dirichlet(np.ones(k), len(ids))
        df = pd.DataFrame(q, columns=[f"component_{i + 1}" for i in range(k)])
        df.insert(0, "sample_id", [str(i) for i in ids])
        df.to_csv(f"{prefix}.{k}.csv", index=False)
        frames[k] = df
    return prefix, frames


def test_admixture_grid_colors_points_by_component_value(scatter_calls, tmp_path):
    ids = list(range(15))
    prefix, frames = _write_q_files(tmp_path, ids, ks=(2,))
    plot_admixture_embedding_grid(
        embedding=_emb(ids),
        q_prefix=prefix,
        k_values=[2],
        output_path=tmp_path / "grid.png",
    )
    value_calls = [c for c in scatter_calls if c.get("c") is not None]
    assert value_calls, "no component-coloured scatter calls"
    # default (no component_colormap) uses the seismic colormap on a 0..1 scale
    assert all(c.get("cmap") == "seismic" for c in value_calls)
    assert all(c.get("vmin") == 0 and c.get("vmax") == 1 for c in value_calls)
    # the colour array for a K=2 subplot must be one of the real component columns
    comp_values = {
        tuple(np.round(frames[2][col].to_numpy(), 6)) for col in ("component_1", "component_2")
    }
    for c in value_calls:
        assert tuple(np.round(np.asarray(c["c"], dtype=float), 6)) in comp_values


def test_admixture_grid_component_colormap_uses_gradient(scatter_calls, tmp_path):
    ids = list(range(15))
    prefix, _ = _write_q_files(tmp_path, ids, ks=(2,))
    cmap_json = tmp_path / "components.json"
    cmap_json.write_text(
        json.dumps(
            {
                "2": {
                    "component_1": {"lineage": 0, "color": "#e41a1c"},
                    "component_2": {"lineage": 1, "color": "#377eb8"},
                }
            }
        )
    )
    plot_admixture_embedding_grid(
        embedding=_emb(ids),
        q_prefix=prefix,
        k_values=[2],
        output_path=tmp_path / "grid.png",
        component_colormap=cmap_json,
    )
    value_calls = [c for c in scatter_calls if c.get("c") is not None]
    assert value_calls
    assert all(isinstance(c.get("cmap"), LinearSegmentedColormap) for c in value_calls)


def test_admixture_grid_subsample_limits_points(scatter_calls, tmp_path):
    ids = list(range(40))
    prefix, _ = _write_q_files(tmp_path, ids, ks=(2,))
    plot_admixture_embedding_grid(
        embedding=_emb(ids),
        q_prefix=prefix,
        k_values=[2],
        output_path=tmp_path / "grid.png",
        subsample=10,
    )
    value_calls = [c for c in scatter_calls if c.get("c") is not None]
    assert value_calls and all(c["n"] == 10 for c in value_calls)


def test_admixture_grid_no_files_raises(tmp_path):
    with pytest.raises(ValueError, match="No admixture CSVs found"):
        plot_admixture_embedding_grid(
            embedding=_emb(range(5)),
            q_prefix=tmp_path / "missing",
            k_values=[2, 3],
            output_path=tmp_path / "x.png",
        )


def test_admixture_grid_bad_pc_columns_raises(tmp_path):
    ids = list(range(10))
    prefix, _ = _write_q_files(tmp_path, ids, ks=(2,))
    with pytest.raises(ValueError, match="PC columns"):
        plot_admixture_embedding_grid(
            embedding=_emb(ids),
            q_prefix=prefix,
            k_values=[2],
            output_path=tmp_path / "x.png",
            pc_x=5,
            pc_y=6,
        )
