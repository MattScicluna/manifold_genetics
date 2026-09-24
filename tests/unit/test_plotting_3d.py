"""Coverage for the interactive 3-D embedding plots.

Same approach as test_plotting_coverage.py: record the traces handed to plotly
and assert colour / grouping / hover content where it matters, exercise the
error paths, and only fall back to "an HTML file was written" for the branches
with no observable output.

plotly is an optional dependency, so the tests that need it skip when it is
absent; the one test that asserts the missing-plotly message runs regardless.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.visualization.plotting import (
    UNKNOWN_LABEL,
    plot_admixture_embedding_3d,
    plot_embedding_3d,
    visualize_3d,
)

CMAP = {"Region": {"A": "#111111", "B": "#222222", "C": "#333333"}}


def _emb(ids, dims=3, seed=0):
    rng = np.random.default_rng(seed)
    d = {"sample_id": [str(i) for i in ids]}
    d.update({f"dim_{i}": rng.normal(size=len(ids)) for i in range(1, dims + 1)})
    return pd.DataFrame(d)


@pytest.fixture
def emb_inputs():
    ids = list(range(9))
    emb = _emb(ids, seed=1)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": list("AABBCCAAB")})
    return emb, labels


@pytest.fixture
def traces(monkeypatch):
    """Record every Scatter3d plotly is asked to build."""
    go = pytest.importorskip("plotly.graph_objects")
    recorded = []
    real = go.Scatter3d

    def rec(**kwargs):
        recorded.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(go, "Scatter3d", rec)
    return recorded


# ---------------------------------------------------------------------------
# plot_embedding_3d
# ---------------------------------------------------------------------------


def test_writes_one_trace_per_present_group_with_colormap_colours(emb_inputs, traces, tmp_path):
    emb, labels = emb_inputs
    out = tmp_path / "e.html"

    plot_embedding_3d(emb, labels, CMAP, out)

    assert out.exists()
    assert [t["name"] for t in traces] == ["A", "B", "C"]
    assert [t["marker"]["color"] for t in traces] == ["#111111", "#222222", "#333333"]
    # Every point lands in exactly one trace.
    assert sum(len(t["x"]) for t in traces) == len(emb)


def test_uses_all_three_dimensions(emb_inputs, traces, tmp_path):
    """A 3-D plot that silently drops dim_3 would look like a valid 2-D one."""
    emb, labels = emb_inputs

    plot_embedding_3d(emb, labels, CMAP, tmp_path / "e.html")

    group_a = emb[emb["sample_id"].isin(["0", "1", "6", "7"])]
    trace_a = next(t for t in traces if t["name"] == "A")
    assert sorted(trace_a["z"]) == sorted(group_a["dim_3"])


def test_unlabelled_samples_become_a_grey_background_trace(traces, tmp_path):
    ids = list(range(6))
    emb = _emb(ids, seed=2)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": ["A", None] * 3})

    plot_embedding_3d(emb, labels, CMAP, tmp_path / "e.html")

    # Unknown is emitted first so it sits behind the coloured groups.
    assert traces[0]["name"] == UNKNOWN_LABEL
    assert traces[0]["marker"]["color"] == "lightgray"
    assert len(traces[0]["x"]) == 3


def test_hover_carries_sample_id(emb_inputs, traces, tmp_path):
    emb, labels = emb_inputs

    plot_embedding_3d(emb, labels, CMAP, tmp_path / "e.html")

    trace_a = next(t for t in traces if t["name"] == "A")
    assert set(trace_a["text"]) == {"0", "1", "6", "7"}


def test_hover_sample_id_false_writes_no_identifiers(emb_inputs, traces, tmp_path):
    """The figure of a controlled-access cohort has to be shareable without
    carrying its participant identifiers into the HTML."""
    emb, labels = emb_inputs
    out = tmp_path / "e.html"

    plot_embedding_3d(emb, labels, CMAP, out, hover_sample_id=False)

    assert all(t["text"] is None for t in traces)
    assert all("%{text}" not in t["hovertemplate"] for t in traces)

    html = out.read_text()
    for sample_id in emb["sample_id"]:
        assert f'"{sample_id}"' not in html


def test_missing_third_dimension_raises_with_a_remedy(tmp_path):
    ids = list(range(4))
    emb = _emb(ids, dims=2, seed=3)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": list("AABB")})

    with pytest.raises(ValueError, match="n_components=3"):
        plot_embedding_3d(emb, labels, CMAP, tmp_path / "e.html")


def test_unknown_label_column_raises(emb_inputs, tmp_path):
    emb, labels = emb_inputs

    with pytest.raises(ValueError, match="colormap"):
        plot_embedding_3d(emb, labels, CMAP, tmp_path / "e.html", label_column="Nope")


def test_default_view_is_face_on_orthographic_with_matched_aspect(emb_inputs, tmp_path):
    """The default must open in the plane of the 2-D figure. A PHATE manifold is
    a curved sheet; plotly's oblique perspective default shows it edge-on."""
    go = pytest.importorskip("plotly.graph_objects")
    captured = {}
    real = go.Figure.update_layout

    def rec(self, *a, **k):
        captured.update(k)
        return real(self, *a, **k)

    emb, labels = emb_inputs
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(go.Figure, "update_layout", rec)
        plot_embedding_3d(emb, labels, CMAP, tmp_path / "e.html")
    scene = captured["scene"]
    assert scene["camera"]["eye"] == dict(x=0, y=0, z=2.0)
    assert scene["camera"]["projection"] == dict(type="orthographic")
    assert scene["aspectmode"] == "manual"
    ext = emb[["dim_1", "dim_2", "dim_3"]].max() - emb[["dim_1", "dim_2", "dim_3"]].min()
    assert scene["aspectratio"]["x"] == 1 and scene["aspectratio"]["y"] == 1
    assert abs(scene["aspectratio"]["z"] - ext["dim_3"] / ext["dim_1"]) < 1e-9

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(go.Figure, "update_layout", rec)
        plot_embedding_3d(emb, labels, CMAP, tmp_path / "t.html", aspect="true")
    assert captured["scene"]["aspectmode"] == "data"

    with pytest.raises(ValueError, match="aspect"):
        plot_embedding_3d(emb, labels, CMAP, tmp_path / "x.html", aspect="cube")


def test_max_points_caps_the_file_and_is_reproducible(traces, tmp_path):
    ids = list(range(50))
    emb = _emb(ids, seed=4)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": ["A"] * 50})

    plot_embedding_3d(emb, labels, CMAP, tmp_path / "a.html", max_points=10)
    first = sorted(traces[0]["text"])

    traces.clear()
    plot_embedding_3d(emb, labels, CMAP, tmp_path / "b.html", max_points=10)

    assert len(first) == 10
    assert sorted(traces[0]["text"]) == first


def test_max_points_none_keeps_everything(traces, tmp_path):
    ids = list(range(50))
    emb = _emb(ids, seed=5)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": ["A"] * 50})

    plot_embedding_3d(emb, labels, CMAP, tmp_path / "e.html", max_points=None)

    assert len(traces[0]["x"]) == 50


def test_missing_plotly_raises_with_install_instructions(monkeypatch):
    """The optional dependency must fail with an instruction, not a traceback."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("no plotly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    # Valid inputs, so the missing dependency -- not input validation -- is
    # what raises. Input errors are reported before plotly is needed.
    ids = list(range(4))
    emb = _emb(ids, seed=9)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": list("AABB")})
    with pytest.raises(ImportError, match=r"interactive"):
        plot_embedding_3d(emb, labels, CMAP, "x.html")


def test_input_errors_do_not_need_plotly(monkeypatch, tmp_path):
    """A 2-D file must get 'run with n_components=3', not 'install plotly',
    whether or not the optional dependency is installed. CI runs without it."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("no plotly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    ids = list(range(4))
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": list("AABB")})
    with pytest.raises(ValueError, match="n_components=3"):
        plot_embedding_3d(_emb(ids, dims=2, seed=3), labels, CMAP, tmp_path / "e.html")
    with pytest.raises(ValueError, match="colormap"):
        plot_embedding_3d(_emb(ids, seed=3), labels, CMAP, tmp_path / "e.html", label_column="Nope")


# ---------------------------------------------------------------------------
# visualize_3d
# ---------------------------------------------------------------------------


def test_visualize_3d_writes_one_html_per_label_column(tmp_path):
    pytest.importorskip("plotly")
    ids = list(range(6))
    emb = _emb(ids, seed=6)
    labels = pd.DataFrame(
        {
            "sample_id": [str(i) for i in ids],
            "Region": list("AABBAA"),
            "Cohort": list("XXYYXX"),
        }
    )
    cmap = {"Region": {"A": "#111111", "B": "#222222"}, "Cohort": {"X": "#333333", "Y": "#444444"}}

    paths = visualize_3d(emb, labels, cmap, output_dir=tmp_path)

    assert [p.name for p in paths] == [
        "embedding_3d_by_Region.html",
        "embedding_3d_by_Cohort.html",
    ]
    assert all(Path(p).exists() for p in paths)


def test_visualize_3d_honours_dataset_prefix(tmp_path):
    pytest.importorskip("plotly")
    ids = list(range(4))
    emb = _emb(ids, seed=7)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": list("AABB")})

    paths = visualize_3d(
        emb, labels, {"Region": CMAP["Region"]}, output_dir=tmp_path, dataset_prefix="fit_"
    )

    assert paths[0].name == "fit_embedding_3d_by_Region.html"


# ---------------------------------------------------------------------------
# plot_admixture_embedding_3d
# ---------------------------------------------------------------------------


def _write_q(tmp_path, ids, ks, seed=0):
    rng = np.random.default_rng(seed)
    prefix = tmp_path / "q" / "t"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for k in ks:
        q = rng.dirichlet(np.ones(k), len(ids))
        df = pd.DataFrame(q, columns=[f"component_{i + 1}" for i in range(k)])
        df.insert(0, "sample_id", [str(i) for i in ids])
        df.to_csv(f"{prefix}.{k}.csv", index=False)
    return prefix


@pytest.fixture
def admix_inputs(tmp_path):
    ids = list(range(12))
    return _emb(ids, seed=7), _write_q(tmp_path, ids, (2, 3))


@pytest.fixture
def layout(monkeypatch):
    """Capture the kwargs of the final ``update_layout`` call."""
    go = pytest.importorskip("plotly.graph_objects")
    captured = {}
    real = go.Figure.update_layout

    def rec(self, *a, **k):
        captured.update(k)
        return real(self, *a, **k)

    monkeypatch.setattr(go.Figure, "update_layout", rec)
    return captured


def _buttons(layout):
    (menu,) = layout["updatemenus"]
    return menu["buttons"]


def test_admixture_3d_is_one_trace_coloured_by_the_first_component(admix_inputs, traces, tmp_path):
    emb, prefix = admix_inputs
    out = plot_admixture_embedding_3d(emb, prefix, [2, 3], tmp_path / "a.html")

    assert out.exists()
    assert len(traces) == 1
    (t,) = traces
    q2 = pd.read_csv(f"{prefix}.2.csv")
    assert list(t["x"]) == list(emb["dim_1"])
    assert list(t["z"]) == list(emb["dim_3"])
    np.testing.assert_allclose(t["marker"]["color"], q2["component_1"], atol=5e-5)
    assert t["marker"]["cmin"] == 0 and t["marker"]["cmax"] == 1


def test_admixture_3d_dropdown_has_one_entry_per_k_and_component(
    admix_inputs, traces, layout, tmp_path
):
    """One file, and the viewer picks the K and component from a menu."""
    emb, prefix = admix_inputs
    plot_admixture_embedding_3d(emb, prefix, [2, 3], tmp_path / "a.html")

    buttons = _buttons(layout)
    assert [b["label"] for b in buttons] == [
        "K=2 · Comp 1",
        "K=2 · Comp 2",
        "K=3 · Comp 1",
        "K=3 · Comp 2",
        "K=3 · Comp 3",
    ]
    q3 = pd.read_csv(f"{prefix}.3.csv")
    restyle = buttons[4]["args"][0]
    np.testing.assert_allclose(restyle["marker.color"][0], q3["component_3"], atol=5e-5)


def test_admixture_3d_component_colormap_sets_gradient_and_lineage_order(
    admix_inputs, traces, layout, tmp_path
):
    """With the colormap exported by plot-admixture, each entry is a white-to-
    component-colour gradient and components are listed by lineage, as in the
    2-D grid."""
    emb, prefix = admix_inputs
    cmap = {
        "2": {
            "component_1": {"color": "#ff0000", "lineage": 1},
            "component_2": {"color": "#0000ff", "lineage": 0},
        },
        "3": {
            "component_1": {"color": "#ff0000", "lineage": 1},
            "component_2": {"color": "#0000ff", "lineage": 0},
            "component_3": {"color": "#00ff00", "lineage": 2},
        },
    }
    plot_admixture_embedding_3d(emb, prefix, [2, 3], tmp_path / "a.html", component_colormap=cmap)

    q2 = pd.read_csv(f"{prefix}.2.csv")
    (t,) = traces
    # first entry is K=2, lineage 0 -> component_2, blue
    np.testing.assert_allclose(t["marker"]["color"], q2["component_2"], atol=5e-5)
    assert t["marker"]["colorscale"] == [[0, "white"], [1, "#0000ff"]]
    labels = [b["label"] for b in _buttons(layout)]
    assert labels[:2] == ["K=2 · Comp 1", "K=2 · Comp 2"]
    restyle = _buttons(layout)[1]["args"][0]
    assert restyle["marker.colorscale"] == [[[0, "white"], [1, "#ff0000"]]]


def test_admixture_3d_hover_ids_can_be_withheld(admix_inputs, traces, tmp_path):
    emb, prefix = admix_inputs
    plot_admixture_embedding_3d(emb, prefix, [2], tmp_path / "a.html")
    assert list(traces[0]["text"]) == list(emb["sample_id"])

    traces.clear()
    plot_admixture_embedding_3d(emb, prefix, [2], tmp_path / "b.html", hover_sample_id=False)
    assert traces[0]["text"] is None
    assert "sample" not in (tmp_path / "b.html").read_text().lower()


def test_admixture_3d_shares_the_face_on_view(admix_inputs, layout, tmp_path):
    emb, prefix = admix_inputs
    plot_admixture_embedding_3d(emb, prefix, [2], tmp_path / "a.html")
    scene = layout["scene"]
    assert scene["camera"]["projection"] == dict(type="orthographic")
    assert scene["aspectmode"] == "manual"


def test_admixture_3d_max_points_subsamples_embedding_and_q_together(
    admix_inputs, traces, tmp_path
):
    emb, prefix = admix_inputs
    plot_admixture_embedding_3d(emb, prefix, [2], tmp_path / "a.html", max_points=5)
    (t,) = traces
    assert len(t["x"]) == 5
    q2 = pd.read_csv(f"{prefix}.2.csv", dtype={"sample_id": str}).set_index("sample_id")
    np.testing.assert_allclose(
        t["marker"]["color"], q2.loc[list(t["text"]), "component_1"], atol=5e-5
    )


def test_admixture_3d_requires_three_dimensions(tmp_path):
    ids = list(range(6))
    prefix = _write_q(tmp_path, ids, (2,))
    with pytest.raises(ValueError, match="n_components=3"):
        plot_admixture_embedding_3d(_emb(ids, dims=2), prefix, [2], tmp_path / "a.html")


def test_admixture_3d_no_q_files_raises(tmp_path):
    with pytest.raises(ValueError, match="No admixture CSVs"):
        plot_admixture_embedding_3d(_emb(range(4)), tmp_path / "nothing", [2], tmp_path / "a.html")
