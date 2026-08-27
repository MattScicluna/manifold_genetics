"""Smoke tests for plot_projection and plot_admixture_embedding_grid.

plotting.py forces the Agg backend at import, so these just build tiny
DataFrames / CSV fixtures, render, and assert a PNG landed. They cover the
large branch-heavy bodies (missing-data layers, legend variants, the
lineage-sorted colormap path) without real embeddings or admixture output.
"""

import json

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.visualization.plotting import (
    plot_admixture_embedding_grid,
    plot_projection,
)


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
    cmap = {"Region": {"A": "#e41a1c", "B": "#377eb8", "C": "#4daf4a"}}
    return fit_emb, proj_emb, fit_labels, proj_labels, cmap


def test_plot_projection_basic(projection_inputs, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels, cmap = projection_inputs
    out = tmp_path / "proj.png"
    result = plot_projection(
        fit_embedding=fit_emb,
        project_embedding=proj_emb,
        fit_labels=fit_labels,
        project_labels=proj_labels,
        fit_colormap=cmap,
        project_colormap=cmap,
        output_path=out,
        fit_label_column="Region",
        project_label_column="Region",
    )
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_plot_projection_handles_missing_labels(projection_inputs, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels, cmap = projection_inputs
    fit_labels.loc[0, "Region"] = np.nan
    proj_labels.loc[0, "Region"] = np.nan
    out = tmp_path / "proj_missing.png"
    plot_projection(
        fit_embedding=fit_emb,
        project_embedding=proj_emb,
        fit_labels=fit_labels,
        project_labels=proj_labels,
        fit_colormap=cmap,
        project_colormap=cmap,
        output_path=out,
        fit_label_column="Region",
        project_label_column="Region",
    )
    assert out.exists()


def test_plot_projection_without_legend(projection_inputs, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels, cmap = projection_inputs
    out = tmp_path / "proj_nolegend.png"
    plot_projection(
        fit_embedding=fit_emb,
        project_embedding=proj_emb,
        fit_labels=fit_labels,
        project_labels=proj_labels,
        fit_colormap=cmap,
        project_colormap=cmap,
        output_path=out,
        fit_label_column="Region",
        project_label_column="Region",
        show_legend=False,
    )
    assert out.exists()


def test_plot_projection_bad_fit_column_raises(projection_inputs, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels, cmap = projection_inputs
    with pytest.raises(ValueError, match="Fit column 'Nope' not in fit colormap"):
        plot_projection(
            fit_embedding=fit_emb,
            project_embedding=proj_emb,
            fit_labels=fit_labels,
            project_labels=proj_labels,
            fit_colormap=cmap,
            project_colormap=cmap,
            output_path=tmp_path / "x.png",
            fit_label_column="Nope",
            project_label_column="Region",
        )


def test_plot_projection_bad_project_column_raises(projection_inputs, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels, cmap = projection_inputs
    with pytest.raises(ValueError, match="Project column 'Nope'"):
        plot_projection(
            fit_embedding=fit_emb,
            project_embedding=proj_emb,
            fit_labels=fit_labels,
            project_labels=proj_labels,
            fit_colormap=cmap,
            project_colormap=cmap,
            output_path=tmp_path / "x.png",
            fit_label_column="Region",
            project_label_column="Nope",
        )


def test_plot_projection_reads_from_paths(projection_inputs, tmp_path):
    fit_emb, proj_emb, fit_labels, proj_labels, cmap = projection_inputs
    fe = tmp_path / "fe.csv"
    pe = tmp_path / "pe.csv"
    fl = tmp_path / "fl.csv"
    pl = tmp_path / "pl.csv"
    cm = tmp_path / "cm.json"
    fit_emb.to_csv(fe, index=False)
    proj_emb.to_csv(pe, index=False)
    fit_labels.to_csv(fl, index=False)
    proj_labels.to_csv(pl, index=False)
    cm.write_text(json.dumps(cmap))
    out = tmp_path / "proj_paths.png"
    plot_projection(
        fit_embedding=fe,
        project_embedding=pe,
        fit_labels=fl,
        project_labels=pl,
        fit_colormap=cm,
        project_colormap=cm,
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
    for k in ks:
        q = rng.dirichlet(np.ones(k), len(ids))
        df = pd.DataFrame(q, columns=[f"component_{i + 1}" for i in range(k)])
        df.insert(0, "sample_id", [str(i) for i in ids])
        df.to_csv(f"{prefix}.{k}.csv", index=False)
    return prefix


def test_admixture_embedding_grid_seismic(tmp_path):
    ids = list(range(20))
    prefix = _write_q_files(tmp_path, ids)
    out = tmp_path / "grid.png"
    result = plot_admixture_embedding_grid(
        embedding=_emb(ids),
        q_prefix=prefix,
        k_values=[2, 3],
        output_path=out,
    )
    assert result == out
    assert out.exists()


def test_admixture_embedding_grid_with_component_colormap(tmp_path):
    ids = list(range(20))
    prefix = _write_q_files(tmp_path, ids, ks=(2,))
    cmap = tmp_path / "components.json"
    cmap.write_text(
        json.dumps(
            {
                "2": {
                    "component_1": {"lineage": 0, "color": "#e41a1c"},
                    "component_2": {"lineage": 1, "color": "#377eb8"},
                }
            }
        )
    )
    out = tmp_path / "grid_cmap.png"
    plot_admixture_embedding_grid(
        embedding=_emb(ids),
        q_prefix=prefix,
        k_values=[2],
        output_path=out,
        component_colormap=cmap,
    )
    assert out.exists()


def test_admixture_embedding_grid_subsample(tmp_path):
    ids = list(range(40))
    prefix = _write_q_files(tmp_path, ids, ks=(2,))
    out = tmp_path / "grid_sub.png"
    plot_admixture_embedding_grid(
        embedding=_emb(ids),
        q_prefix=prefix,
        k_values=[2],
        output_path=out,
        subsample=10,
    )
    assert out.exists()


def test_admixture_embedding_grid_no_files_raises(tmp_path):
    with pytest.raises(ValueError, match="No admixture CSVs found"):
        plot_admixture_embedding_grid(
            embedding=_emb(range(5)),
            q_prefix=tmp_path / "missing",
            k_values=[2, 3],
            output_path=tmp_path / "x.png",
        )


def test_admixture_embedding_grid_bad_pc_columns_raises(tmp_path):
    ids = list(range(10))
    prefix = _write_q_files(tmp_path, ids, ks=(2,))
    with pytest.raises(ValueError, match="PC columns"):
        plot_admixture_embedding_grid(
            embedding=_emb(ids),
            q_prefix=prefix,
            k_values=[2],
            output_path=tmp_path / "x.png",
            pc_x=5,
            pc_y=6,
        )
