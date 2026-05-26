"""Tests for embedding methods (PHATE, UMAP, t-SNE, Diffusion Maps)."""

import numpy as np
import pytest

from manifold_genetics.embeddings import PHATE, TSNE, UMAP, DiffusionMap


class TestPHATE:
    """Tests for PHATE embedding."""

    def test_fit_transform(self, small_pca_data):
        """Test PHATE fit_transform on small dataset."""
        phate = PHATE(n_components=2, knn=5, n_landmark=None)
        embedding = phate.fit_transform(small_pca_data)

        assert embedding.shape == (50, 3)  # sample_id + 2 dimensions
        assert list(embedding.columns) == ["sample_id", "dim_1", "dim_2"]
        assert "sample_id" in embedding.columns

    def test_with_csv_input(self, small_pca_csv, temp_dir):
        """Test PHATE with CSV file input."""
        phate = PHATE(n_components=2, knn=5, n_landmark=None)
        output_file = temp_dir / "phate_output.csv"

        embedding = phate.fit_transform(small_pca_csv, output_path=output_file)

        assert embedding.shape == (50, 3)  # sample_id + 2 dimensions
        assert output_file.exists()

    def test_custom_parameters(self, small_pca_data):
        """Test PHATE with custom knn and t parameters."""
        phate = PHATE(n_components=2, knn=10, t=3, n_landmark=None)
        embedding = phate.fit_transform(small_pca_data)

        assert embedding.shape == (50, 3)  # sample_id + 2 dimensions

    def test_no_landmarking(self, small_pca_data):
        """Test that n_landmark=None uses all samples."""
        phate = PHATE(n_components=2, knn=5, n_landmark=None)
        embedding = phate.fit_transform(small_pca_data)

        # Should have same number of samples (no subsampling)
        assert len(embedding) == len(small_pca_data)

    @pytest.mark.parametrize("n_components", [2, 3, 5])
    def test_different_dimensions(self, small_pca_data, n_components):
        """Test PHATE with different output dimensions."""
        phate = PHATE(n_components=n_components, knn=5, n_landmark=None)
        embedding = phate.fit_transform(small_pca_data)

        assert embedding.shape[1] == n_components + 1  # +1 for sample_id
        expected_cols = ["sample_id"] + [f"dim_{i+1}" for i in range(n_components)]
        assert list(embedding.columns) == expected_cols


class TestUMAP:
    """Tests for UMAP embedding."""

    def test_fit_transform(self, small_pca_data):
        """Test UMAP fit_transform."""
        umap = UMAP(n_components=2, n_neighbors=5, min_dist=0.1)
        embedding = umap.fit_transform(small_pca_data)

        assert embedding.shape == (50, 3)  # sample_id + 2 dimensions
        assert list(embedding.columns) == ["sample_id", "dim_1", "dim_2"]

    def test_with_output_file(self, small_pca_data, temp_dir):
        """Test UMAP with output file."""
        umap = UMAP(n_components=2, n_neighbors=5)
        output_file = temp_dir / "umap_output.csv"

        embedding = umap.fit_transform(small_pca_data, output_path=output_file)

        assert output_file.exists()
        assert embedding.shape == (50, 3)  # sample_id + 2 dimensions


class TestTSNE:
    """Tests for t-SNE embedding."""

    def test_fit_transform(self, small_pca_data):
        """Test t-SNE fit_transform."""
        tsne = TSNE(n_components=2, perplexity=15)
        embedding = tsne.fit_transform(small_pca_data)

        assert embedding.shape == (50, 3)  # sample_id + 2 dimensions
        assert list(embedding.columns) == ["sample_id", "dim_1", "dim_2"]


class TestDiffusionMap:
    """Tests for Diffusion Map embedding."""

    def test_fit_transform(self, small_pca_data):
        """Test Diffusion Map fit_transform."""
        dm = DiffusionMap(n_components=2, knn=5)
        embedding = dm.fit_transform(small_pca_data)

        assert embedding.shape == (50, 3)  # sample_id + 2 dimensions
        assert list(embedding.columns) == ["sample_id", "dim_1", "dim_2"]


class TestEmbeddingOutputFormat:
    """Tests for embedding output format consistency."""

    def test_includes_sample_id_column(self, small_pca_data):
        """Test that embeddings include sample_id column."""
        phate = PHATE(n_components=2, knn=5, n_landmark=None)
        embedding = phate.fit_transform(small_pca_data)

        assert "sample_id" in embedding.columns
        assert embedding.columns[0] == "sample_id"  # First column

    def test_sample_id_plus_dim_columns(self, small_pca_data):
        """Test that embeddings have sample_id plus dim_ columns."""
        phate = PHATE(n_components=3, knn=5, n_landmark=None)
        embedding = phate.fit_transform(small_pca_data)

        expected_cols = ["sample_id", "dim_1", "dim_2", "dim_3"]
        assert list(embedding.columns) == expected_cols
