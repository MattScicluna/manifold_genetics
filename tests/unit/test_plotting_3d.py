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

    with pytest.raises(ImportError, match=r"interactive"):
        plot_embedding_3d(pd.DataFrame(), pd.DataFrame(), CMAP, "x.html")


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
