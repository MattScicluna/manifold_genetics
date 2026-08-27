"""Branch coverage for plot_embedding, plot_pca_pairs, plot_admixture_bar_grid
and plot_knn_composition.

Same approach as test_plotting_projection.py: record Axes.scatter calls and
assert colour / marker / z-order where it matters, exercise the error paths
and the ordering options, and only fall back to "a PNG was written" for the
branches with no observable output.
"""

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

from manifold_genetics.visualization.plotting import (
    plot_admixture_bar_grid,
    plot_embedding,
    plot_knn_composition,
    plot_pca_pairs,
)


@pytest.fixture
def scatter_calls(monkeypatch):
    calls = []
    real = matplotlib.axes.Axes.scatter

    def rec(self, x, y=None, *args, **kwargs):
        n = len(x) if hasattr(x, "__len__") else 1
        calls.append({"n": n, **kwargs})
        return real(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", rec)
    return calls


def _emb(ids, dims=2, seed=0):
    rng = np.random.default_rng(seed)
    d = {"sample_id": [str(i) for i in ids]}
    d.update({f"dim_{i}": rng.normal(size=len(ids)) for i in range(1, dims + 1)})
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# plot_embedding
# ---------------------------------------------------------------------------

CMAP = {"Region": {"A": "#111111", "B": "#222222", "C": "#333333"}}


@pytest.fixture
def emb_inputs():
    ids = list(range(9))
    emb = _emb(ids, seed=1)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": list("AABBCCAAB")})
    return emb, labels


def test_plot_embedding_colours_groups_and_layers(emb_inputs, scatter_calls, tmp_path):
    emb, labels = emb_inputs
    plot_embedding(emb, labels, CMAP, tmp_path / "e.png")
    group_calls = [c for c in scatter_calls if c.get("zorder") == 2]
    assert group_calls
    for c in group_calls:
        assert c["color"] in set(CMAP["Region"].values())


def test_plot_embedding_missing_data_layer_and_unknown_legend(emb_inputs, scatter_calls, tmp_path):
    emb, labels = emb_inputs
    labels.loc[0, "Region"] = np.nan
    plot_embedding(emb, labels, CMAP, tmp_path / "e.png")
    gray = [c for c in scatter_calls if c.get("color") == "lightgray" and c.get("zorder") == 1]
    assert gray and sum(c["n"] for c in gray) == 1


def test_plot_embedding_skips_column_absent_from_labels(
    emb_inputs, scatter_calls, tmp_path, caplog
):
    emb, labels = emb_inputs
    cmap = {"Region": CMAP["Region"], "NotThere": {"x": "#000000"}}
    plot_embedding(emb, labels, cmap, tmp_path / "e.png")
    assert "NotThere" in caplog.text


def test_plot_embedding_no_legend(emb_inputs, monkeypatch, tmp_path):
    emb, labels = emb_inputs
    legend = []
    real = matplotlib.axes.Axes.legend
    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "legend",
        lambda self, *a, **k: legend.append(1) or real(self, *a, **k),
    )
    plot_embedding(emb, labels, CMAP, tmp_path / "e.png", show_legend=False)
    assert legend == []


# ---------------------------------------------------------------------------
# plot_pca_pairs
# ---------------------------------------------------------------------------


@pytest.fixture
def pca_inputs():
    ids = list(range(12))
    pca = _emb(ids, dims=6, seed=2)
    labels = pd.DataFrame({"sample_id": [str(i) for i in ids], "Region": list("AAAABBBBCCCC")})
    return pca, labels


def test_plot_pca_pairs_basic(pca_inputs, scatter_calls, tmp_path):
    pca, labels = pca_inputs
    plot_pca_pairs(pca, labels, CMAP, tmp_path / "p.png", label_column="Region", n_pcs=6)
    group_calls = [c for c in scatter_calls if c.get("zorder") == 2]
    # 3 PC pairs x 3 groups
    assert len(group_calls) == 9
    for c in group_calls:
        assert c["color"] in set(CMAP["Region"].values())


def test_plot_pca_pairs_default_colours_when_column_not_in_colormap(pca_inputs, tmp_path, caplog):
    pca, labels = pca_inputs
    cmap = {"Other": {"z": "#000000"}}
    plot_pca_pairs(pca, labels, cmap, tmp_path / "p.png", label_column="Region", n_pcs=6)
    assert "not in colormap" in caplog.text
    assert (tmp_path / "p.png").exists()


def test_plot_pca_pairs_missing_column_raises(pca_inputs, tmp_path):
    pca, labels = pca_inputs
    with pytest.raises(ValueError, match="not found in labels data"):
        plot_pca_pairs(pca, labels, CMAP, tmp_path / "p.png", label_column="Nope", n_pcs=6)


def test_plot_pca_pairs_needs_two_pcs(pca_inputs, tmp_path):
    pca, labels = pca_inputs
    one_pc = pca[["sample_id", "dim_1"]]
    with pytest.raises(ValueError, match="at least 2 PCs"):
        plot_pca_pairs(one_pc, labels, CMAP, tmp_path / "p.png", label_column="Region", n_pcs=6)


def test_plot_pca_pairs_missing_data_layer(pca_inputs, scatter_calls, tmp_path):
    pca, labels = pca_inputs
    labels.loc[0, "Region"] = np.nan
    plot_pca_pairs(pca, labels, CMAP, tmp_path / "p.png", label_column="Region", n_pcs=6)
    gray = [c for c in scatter_calls if c.get("color") == "lightgray" and c.get("zorder") == 1]
    assert gray


# ---------------------------------------------------------------------------
# plot_admixture_bar_grid
# ---------------------------------------------------------------------------


def _write_q(tmp_path, ids, ks, seed=0):
    rng = np.random.default_rng(seed)
    prefix = tmp_path / "q" / "t"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for k in ks:
        q = rng.dirichlet(np.ones(k), len(ids))
        df = pd.DataFrame(q, columns=[f"component_{i+1}" for i in range(k)])
        df.insert(0, "sample_id", [str(i) for i in ids])
        df.to_csv(f"{prefix}.{k}.csv", index=False)
    return prefix


@pytest.fixture
def bar_inputs(tmp_path):
    ids = list(range(24))
    prefix = _write_q(tmp_path, ids, (2, 3))
    labels = pd.DataFrame(
        {"sample_id": [str(i) for i in ids], "Region": (["A"] * 8 + ["B"] * 8 + ["C"] * 8)}
    )
    return prefix, labels


@pytest.mark.parametrize("order", ["chron", "tree", None])
def test_bar_grid_within_group_order_variants(bar_inputs, tmp_path, order):
    prefix, labels = bar_inputs
    out = tmp_path / f"bars_{order}.png"
    plot_admixture_bar_grid(
        q_prefix=prefix,
        labels=labels,
        group_column="Region",
        k_values=[2, 3],
        output_path=out,
        within_group_order=order,
    )
    assert out.exists()


def test_bar_grid_group_order_from_colormap(bar_inputs, tmp_path):
    prefix, labels = bar_inputs
    cmap = {"Region": {"C": "#c", "B": "#b", "A": "#a"}}  # reversed order
    out = tmp_path / "bars.png"
    plot_admixture_bar_grid(
        q_prefix=prefix,
        labels=labels,
        group_column="Region",
        k_values=[2],
        output_path=out,
        colormap=cmap,
        within_group_order=None,
    )
    assert out.exists()


def test_bar_grid_missing_group_column_raises(bar_inputs, tmp_path):
    prefix, labels = bar_inputs
    with pytest.raises(ValueError, match="Group column 'Nope' not found"):
        plot_admixture_bar_grid(
            q_prefix=prefix,
            labels=labels,
            group_column="Nope",
            k_values=[2],
            output_path=tmp_path / "x.png",
            within_group_order=None,
        )


def test_bar_grid_no_shared_samples_across_k_raises(tmp_path):
    prefix = tmp_path / "q" / "t"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for k, offset in ((2, 0), (3, 100)):
        ids = list(range(offset, offset + 10))
        q = rng.dirichlet(np.ones(k), 10)
        df = pd.DataFrame(q, columns=[f"component_{i+1}" for i in range(k)])
        df.insert(0, "sample_id", [str(i) for i in ids])
        df.to_csv(f"{prefix}.{k}.csv", index=False)
    labels = pd.DataFrame({"sample_id": [str(i) for i in range(110)], "Region": ["A"] * 110})
    with pytest.raises(ValueError, match="No shared sample_id"):
        plot_admixture_bar_grid(
            q_prefix=prefix,
            labels=labels,
            group_column="Region",
            k_values=[2, 3],
            output_path=tmp_path / "x.png",
            within_group_order=None,
        )


# ---------------------------------------------------------------------------
# plot_knn_composition
# ---------------------------------------------------------------------------

FIT_CMAP = {"Population": {p: f"#{i}{i}{i}{i}{i}{i}" for i, p in enumerate("PQRS", start=1)}}


@pytest.fixture
def knn_inputs():
    fit_ids = list(range(40))
    proj_ids = list(range(100, 130))
    fe = _emb(fit_ids, seed=3)
    pe = _emb(proj_ids, seed=4)
    fl = pd.DataFrame(
        {"sample_id": [str(i) for i in fit_ids], "Population": ["P", "Q", "R", "S"] * 10}
    )
    pl = pd.DataFrame(
        {"sample_id": [str(i) for i in proj_ids], "Region": (["X"] * 15 + ["Y"] * 15)}
    )
    return fe, pe, fl, pl


def _run_knn(inputs, out, **kw):
    fe, pe, fl, pl = inputs
    return plot_knn_composition(
        fit_embedding=fe,
        project_embedding=pe,
        fit_labels=fl,
        project_labels=pl,
        fit_colormap=FIT_CMAP,
        fit_label_column="Population",
        project_label_column="Region",
        output_path=out,
        **kw,
    )


def test_knn_basic_two_panels(knn_inputs, tmp_path):
    out = _run_knn(knn_inputs, tmp_path / "knn.png", k=5)
    assert out.exists()


def test_knn_subsample_and_no_dominant_sort(knn_inputs, tmp_path):
    out = _run_knn(
        knn_inputs, tmp_path / "knn.png", k=5, subsample_per_group=8, sort_by_dominant=False
    )
    assert out.exists()


def test_knn_project_label_subset(knn_inputs, tmp_path):
    out = _run_knn(knn_inputs, tmp_path / "knn.png", k=5, project_label_subset=["X"])
    assert out.exists()


def test_knn_k_larger_than_fit_raises(knn_inputs, tmp_path):
    with pytest.raises(ValueError, match="must be <= number of fit samples"):
        _run_knn(knn_inputs, tmp_path / "knn.png", k=999)
