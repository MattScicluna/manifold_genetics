"""Tests for visualization functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.plotting._core as plotting_core
import pytest

from manifold_genetics.visualization import (
    plot_admixture_bar_grid,
    plot_embedding,
    plot_knn_composition,
    plot_pca_pairs,
    visualize,
)


class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_embedding_basic(self, small_embedding_csv, labels_csv, colormap_json, temp_dir):
        """Test basic plot_embedding functionality."""
        output_file = temp_dir / "test_plot.png"

        result = plot_embedding(
            embedding=small_embedding_csv,
            labels=labels_csv,
            colormap=colormap_json,
            output_path=output_file,
        )

        assert output_file.exists()
        assert result == output_file

    def test_visualize_multiple_plots(
        self, small_embedding_csv, labels_csv, colormap_json, temp_dir
    ):
        """Test visualize function that creates multiple plots."""
        figure_paths = visualize(
            embedding=small_embedding_csv,
            labels=labels_csv,
            colormap=colormap_json,
            output_dir=temp_dir,
            output_prefix="test",
        )

        # Should create at least one figure
        assert len(figure_paths) > 0
        assert all(isinstance(p, Path) for p in figure_paths)
        assert all(p.exists() for p in figure_paths)

    def test_embedding_labels_alignment(
        self, small_embedding_data, labels_data, colormap_data, temp_dir
    ):
        """Test that embedding and labels are correctly aligned by row order."""
        output_file = temp_dir / "alignment_test.png"

        # Both should be in same order (row 0 = SAMPLE_000, etc.)
        result = plot_embedding(
            embedding=small_embedding_data,
            labels=labels_data.set_index("sample_id"),
            colormap=colormap_data,
            output_path=output_file,
        )

        assert output_file.exists()

    def test_visualization_with_dataframes(
        self, small_embedding_data, labels_data, colormap_data, temp_dir
    ):
        """Test visualization with DataFrame inputs instead of files."""
        output_file = temp_dir / "df_test.png"

        result = plot_embedding(
            embedding=small_embedding_data,
            labels=labels_data.set_index("sample_id"),
            colormap=colormap_data,
            output_path=output_file,
        )

        assert output_file.exists()

    def test_plot_pca_pairs_with_numeric_sample_ids(
        self, numeric_pca_csv, numeric_labels_csv, colormap_json, temp_dir
    ):
        """plot_pca_pairs must not raise a merge error when PCA CSV has int sample_ids."""
        output_file = temp_dir / "pca_pairs_numeric.png"
        result = plot_pca_pairs(
            pca_coords=numeric_pca_csv,
            labels=numeric_labels_csv,
            colormap=colormap_json,
            output_path=output_file,
            label_column="Population",
        )
        assert output_file.exists()
        assert result == output_file

    def test_plot_admixture_bar_grid_with_numeric_sample_ids(self, temp_dir):
        """plot_admixture_bar_grid must not fail when admixture CSVs have int sample_ids."""
        np.random.seed(42)
        n_samples = 50
        int_ids = np.arange(n_samples, dtype=np.int64)
        str_ids = [str(i) for i in int_ids]

        q_prefix = temp_dir / "admixture" / "transform"
        q_prefix.parent.mkdir(parents=True, exist_ok=True)

        for k in [2, 3]:
            q_data = np.random.dirichlet(np.ones(k), n_samples)
            df = pd.DataFrame(q_data, columns=[f"component_{i+1}" for i in range(k)])
            df.insert(0, "sample_id", int_ids)  # int64, mimicking pre-fix CSVs
            df.to_csv(f"{q_prefix}.{k}.csv", index=False)

        labels = pd.DataFrame({"sample_id": str_ids, "Population": ["PopA"] * 25 + ["PopB"] * 25})

        output_file = temp_dir / "admixture_numeric.png"
        plot_admixture_bar_grid(
            q_prefix=q_prefix,
            labels=labels,
            group_column="Population",
            k_values=[2, 3],
            output_path=output_file,
            within_group_order=None,
        )
        assert output_file.exists()

    def test_admixture_bar_grid_aligns_component_colors_across_k(self, monkeypatch, temp_dir):
        """Stable components should keep their color even if component columns permute."""
        rng = np.random.default_rng(42)
        n_samples = 120
        sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
        groups = np.where(np.arange(n_samples) < n_samples // 2, "PopA", "PopB")

        # Build a stable A/B ancestry axis with group structure.
        a = np.where(
            groups == "PopA",
            rng.uniform(0.75, 0.95, n_samples),
            rng.uniform(0.05, 0.25, n_samples),
        )
        b = 1.0 - a
        c = rng.uniform(0.0, 0.08, n_samples)  # New component introduced at K=3.

        q_prefix = temp_dir / "admixture" / "transform"
        q_prefix.parent.mkdir(parents=True, exist_ok=True)

        k2 = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "component_1": a,
                "component_2": b,
            }
        )
        # Permute columns so the stable A component moves from comp_1 -> comp_3.
        k3 = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "component_1": c,
                "component_2": b * (1.0 - c),
                "component_3": a * (1.0 - c),
            }
        )

        k2.to_csv(f"{q_prefix}.2.csv", index=False)
        k3.to_csv(f"{q_prefix}.3.csv", index=False)
        labels = pd.DataFrame({"sample_id": sample_ids, "Population": groups})

        recorded_colors = []
        original_call = plotting_core.PlotAccessor.__call__

        def wrapped_plot_call(self, *args, **kwargs):
            if kwargs.get("kind") == "bar" and kwargs.get("stacked"):
                recorded_colors.append(list(kwargs.get("color", [])))
            return original_call(self, *args, **kwargs)

        monkeypatch.setattr(plotting_core.PlotAccessor, "__call__", wrapped_plot_call)

        output_file = temp_dir / "admixture.png"
        plot_admixture_bar_grid(
            q_prefix=q_prefix,
            labels=labels,
            group_column="Population",
            k_values=[2, 3],
            output_path=output_file,
            within_group_order=None,
        )

        assert output_file.exists()
        assert len(recorded_colors) == 2

        k2_colors, k3_colors = recorded_colors
        # Stable A: K2 component_1 matches K3 component_3.
        assert k2_colors[0] == k3_colors[2]
        # Stable B: K2 component_2 matches K3 component_2.
        assert k2_colors[1] == k3_colors[1]
        # New component should get a distinct color.
        assert k3_colors[0] not in {k3_colors[1], k3_colors[2]}

    def test_plot_knn_composition_basic(self, temp_dir):
        """plot_knn_composition runs without error on small synthetic data."""
        import json

        rng = np.random.default_rng(42)
        n_fit, n_proj, d = 60, 30, 2
        fit_label_arr = np.where(np.arange(n_fit) < 30, "PopA", "PopB")
        proj_label_arr = np.where(np.arange(n_proj) < 15, "GroupX", "GroupY")

        fit_emb = pd.DataFrame(rng.standard_normal((n_fit, d)), columns=["dim_1", "dim_2"])
        fit_emb.insert(0, "sample_id", [f"F{i}" for i in range(n_fit)])
        proj_emb = pd.DataFrame(rng.standard_normal((n_proj, d)), columns=["dim_1", "dim_2"])
        proj_emb.insert(0, "sample_id", [f"P{i}" for i in range(n_proj)])

        fit_lbl = pd.DataFrame({"sample_id": [f"F{i}" for i in range(n_fit)], "Pop": fit_label_arr})
        proj_lbl = pd.DataFrame(
            {"sample_id": [f"P{i}" for i in range(n_proj)], "Group": proj_label_arr}
        )
        cmap = {"Pop": {"PopA": "#FF0000", "PopB": "#0000FF"}}

        for df, name in [(fit_emb, "fe"), (proj_emb, "pe"), (fit_lbl, "fl"), (proj_lbl, "pl")]:
            df.to_csv(temp_dir / f"{name}.csv", index=False)
        (temp_dir / "cmap.json").write_text(json.dumps(cmap))

        out = temp_dir / "knn_comp.png"
        result = plot_knn_composition(
            fit_embedding=temp_dir / "fe.csv",
            project_embedding=temp_dir / "pe.csv",
            fit_labels=temp_dir / "fl.csv",
            project_labels=temp_dir / "pl.csv",
            fit_colormap=temp_dir / "cmap.json",
            fit_label_column="Pop",
            project_label_column="Group",
            output_path=out,
            k=5,
        )
        assert out.exists()
        assert result == out

    def test_plot_knn_composition_empty_fit_merge_raises(self, temp_dir):
        """plot_knn_composition raises ValueError with actionable message when fit merge is empty."""
        import json

        rng = np.random.default_rng(0)
        n_fit, n_proj, d = 10, 5, 2

        fit_emb = pd.DataFrame(rng.standard_normal((n_fit, d)), columns=["dim_1", "dim_2"])
        fit_emb.insert(0, "sample_id", [f"FIT_{i}" for i in range(n_fit)])
        proj_emb = pd.DataFrame(rng.standard_normal((n_proj, d)), columns=["dim_1", "dim_2"])
        proj_emb.insert(0, "sample_id", [f"P{i}" for i in range(n_proj)])

        # Deliberately mismatched sample_ids for fit labels
        fit_lbl = pd.DataFrame(
            {"sample_id": [f"NOMATCH_{i}" for i in range(n_fit)], "Pop": ["A"] * n_fit}
        )
        proj_lbl = pd.DataFrame(
            {"sample_id": [f"P{i}" for i in range(n_proj)], "Group": ["X"] * n_proj}
        )
        cmap = {"Pop": {"A": "#FF0000"}}

        for df, name in [(fit_emb, "fe"), (proj_emb, "pe"), (fit_lbl, "fl"), (proj_lbl, "pl")]:
            df.to_csv(temp_dir / f"{name}.csv", index=False)
        (temp_dir / "cmap.json").write_text(json.dumps(cmap))

        with pytest.raises(ValueError, match="fit_emb_df.*fit_labels_df.*sample_id.*zero rows"):
            plot_knn_composition(
                fit_embedding=temp_dir / "fe.csv",
                project_embedding=temp_dir / "pe.csv",
                fit_labels=temp_dir / "fl.csv",
                project_labels=temp_dir / "pl.csv",
                fit_colormap=temp_dir / "cmap.json",
                fit_label_column="Pop",
                project_label_column="Group",
                output_path=temp_dir / "out.png",
                k=3,
            )

    def test_plot_knn_composition_empty_proj_merge_raises(self, temp_dir):
        """plot_knn_composition raises ValueError with actionable message when proj merge is empty."""
        import json

        rng = np.random.default_rng(1)
        n_fit, n_proj, d = 10, 5, 2

        fit_emb = pd.DataFrame(rng.standard_normal((n_fit, d)), columns=["dim_1", "dim_2"])
        fit_emb.insert(0, "sample_id", [f"F{i}" for i in range(n_fit)])
        proj_emb = pd.DataFrame(rng.standard_normal((n_proj, d)), columns=["dim_1", "dim_2"])
        proj_emb.insert(0, "sample_id", [f"PROJ_{i}" for i in range(n_proj)])

        fit_lbl = pd.DataFrame({"sample_id": [f"F{i}" for i in range(n_fit)], "Pop": ["A"] * n_fit})
        # Deliberately mismatched sample_ids for proj labels
        proj_lbl = pd.DataFrame(
            {"sample_id": [f"NOMATCH_{i}" for i in range(n_proj)], "Group": ["X"] * n_proj}
        )
        cmap = {"Pop": {"A": "#FF0000"}}

        for df, name in [(fit_emb, "fe"), (proj_emb, "pe"), (fit_lbl, "fl"), (proj_lbl, "pl")]:
            df.to_csv(temp_dir / f"{name}.csv", index=False)
        (temp_dir / "cmap.json").write_text(json.dumps(cmap))

        with pytest.raises(ValueError, match="proj_emb_df.*proj_labels_df.*sample_id.*zero rows"):
            plot_knn_composition(
                fit_embedding=temp_dir / "fe.csv",
                project_embedding=temp_dir / "pe.csv",
                fit_labels=temp_dir / "fl.csv",
                project_labels=temp_dir / "pl.csv",
                fit_colormap=temp_dir / "cmap.json",
                fit_label_column="Pop",
                project_label_column="Group",
                output_path=temp_dir / "out.png",
                k=3,
            )

    def test_plot_knn_composition_float_sample_ids(self, temp_dir):
        """plot_knn_composition handles float64 whole-number sample_id values without error."""
        import json

        rng = np.random.default_rng(42)
        n_fit, n_proj, d = 60, 30, 2
        fit_label_arr = np.where(np.arange(n_fit) < 30, "PopA", "PopB")
        proj_label_arr = np.where(np.arange(n_proj) < 15, "GroupX", "GroupY")

        # Use float64 whole-number sample IDs to exercise type-coercion behavior.
        fit_ids = np.arange(1, n_fit + 1, dtype=float)
        proj_ids = np.arange(1, n_proj + 1, dtype=float)

        fit_emb = pd.DataFrame(rng.standard_normal((n_fit, d)), columns=["dim_1", "dim_2"])
        fit_emb.insert(0, "sample_id", fit_ids)
        proj_emb = pd.DataFrame(rng.standard_normal((n_proj, d)), columns=["dim_1", "dim_2"])
        proj_emb.insert(0, "sample_id", proj_ids)

        fit_lbl = pd.DataFrame({"sample_id": fit_ids, "Pop": fit_label_arr})
        proj_lbl = pd.DataFrame({"sample_id": proj_ids, "Group": proj_label_arr})
        cmap = {"Pop": {"PopA": "#FF0000", "PopB": "#0000FF"}}

        for df, name in [
            (fit_emb, "fe_f"),
            (proj_emb, "pe_f"),
            (fit_lbl, "fl_f"),
            (proj_lbl, "pl_f"),
        ]:
            df.to_csv(temp_dir / f"{name}.csv", index=False)
        (temp_dir / "cmap_f.json").write_text(json.dumps(cmap))

        out = temp_dir / "knn_comp_float_ids.png"
        result = plot_knn_composition(
            fit_embedding=temp_dir / "fe_f.csv",
            project_embedding=temp_dir / "pe_f.csv",
            fit_labels=temp_dir / "fl_f.csv",
            project_labels=temp_dir / "pl_f.csv",
            fit_colormap=temp_dir / "cmap_f.json",
            fit_label_column="Pop",
            project_label_column="Group",
            output_path=out,
            k=5,
        )
        assert out.exists()
        assert result == out
