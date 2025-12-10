"""Tests for visualization functions."""

import pytest
from pathlib import Path

from manifold_genetics.visualization import visualize, plot_embedding


class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_embedding_basic(self, small_embedding_csv, labels_csv, colormap_json, temp_dir):
        """Test basic plot_embedding functionality."""
        output_file = temp_dir / "test_plot.png"

        result = plot_embedding(
            embedding=small_embedding_csv,
            labels=labels_csv,
            colormap=colormap_json,
            output_path=output_file
        )

        assert output_file.exists()
        assert result == output_file

    def test_visualize_multiple_plots(self, small_embedding_csv, labels_csv, colormap_json, temp_dir):
        """Test visualize function that creates multiple plots."""
        figure_paths = visualize(
            embedding=small_embedding_csv,
            labels=labels_csv,
            colormap=colormap_json,
            output_dir=temp_dir,
            output_prefix="test"
        )

        # Should create at least one figure
        assert len(figure_paths) > 0
        assert all(isinstance(p, Path) for p in figure_paths)
        assert all(p.exists() for p in figure_paths)

    def test_embedding_labels_alignment(self, small_embedding_data, labels_data, colormap_data, temp_dir):
        """Test that embedding and labels are correctly aligned by row order."""
        output_file = temp_dir / "alignment_test.png"

        # Both should be in same order (row 0 = SAMPLE_000, etc.)
        result = plot_embedding(
            embedding=small_embedding_data,
            labels=labels_data.set_index("sample_id"),
            colormap=colormap_data,
            output_path=output_file
        )

        assert output_file.exists()

    def test_visualization_with_dataframes(self, small_embedding_data, labels_data, colormap_data, temp_dir):
        """Test visualization with DataFrame inputs instead of files."""
        output_file = temp_dir / "df_test.png"

        result = plot_embedding(
            embedding=small_embedding_data,
            labels=labels_data.set_index("sample_id"),
            colormap=colormap_data,
            output_path=output_file
        )

        assert output_file.exists()
