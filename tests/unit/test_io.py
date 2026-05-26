"""Tests for I/O utilities."""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.utils.io import (
    read_admixture_csv,
    read_embedding_csv,
    write_embedding_csv,
    read_labels_csv,
    read_colormap,
    _coerce_ids_to_str,
)


class TestEmbeddingIO:
    """Tests for embedding CSV I/O."""

    def test_write_and_read_embedding(self, temp_dir):
        """Test writing and reading embedding CSV."""
        # Create test data with sample_id column
        n_samples, n_dims = 100, 5
        embedding_data = np.random.randn(n_samples, n_dims)
        sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
        df = pd.DataFrame(embedding_data, columns=[f"dim_{i+1}" for i in range(n_dims)])
        df.insert(0, "sample_id", sample_ids)
        output_file = temp_dir / "test.csv"

        # Write
        write_embedding_csv(df, output_file)
        assert output_file.exists()

        # Read
        df_read = read_embedding_csv(output_file)
        assert df_read.shape == (n_samples, n_dims + 1)  # +1 for sample_id
        expected_cols = ["sample_id"] + [f"dim_{i+1}" for i in range(n_dims)]
        assert list(df_read.columns) == expected_cols
        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)

    def test_embedding_format_includes_sample_id(self, temp_dir):
        """Test that embedding CSV has sample_id column."""
        embedding_data = np.random.randn(20, 3)
        sample_ids = [f"SAMPLE_{i:03d}" for i in range(20)]
        df = pd.DataFrame(embedding_data, columns=[f"dim_{i+1}" for i in range(3)])
        df.insert(0, "sample_id", sample_ids)
        output_file = temp_dir / "test.csv"

        write_embedding_csv(df, output_file)
        df_read = pd.read_csv(output_file)

        # Should have sample_id as first column
        assert "sample_id" in df_read.columns
        assert df_read.columns[0] == "sample_id"
        expected_cols = ["sample_id", "dim_1", "dim_2", "dim_3"]
        assert list(df_read.columns) == expected_cols

    def test_write_dataframe(self, temp_dir, small_pca_data):
        """Test writing DataFrame to embedding CSV."""
        output_file = temp_dir / "test.csv"

        write_embedding_csv(small_pca_data, output_file)
        df = read_embedding_csv(output_file)

        pd.testing.assert_frame_equal(small_pca_data, df, check_dtype=False)

    def test_read_embedding_csv_coerces_numeric_sample_id_to_str(self, numeric_pca_csv):
        """read_embedding_csv must coerce int sample_ids to str dtype."""
        df = read_embedding_csv(numeric_pca_csv)
        assert "sample_id" in df.columns
        assert pd.api.types.is_string_dtype(
            df["sample_id"]
        ), f"Expected str dtype for sample_id, got {df['sample_id'].dtype}"


class TestLabelsIO:
    """Tests for labels CSV I/O."""

    def test_read_labels_csv(self, labels_csv):
        """Test reading labels CSV."""
        df = read_labels_csv(labels_csv)

        assert df.index.name == "sample_id"
        assert "Population" in df.columns
        assert "Region" in df.columns

    def test_labels_index_is_sample_id(self, labels_csv):
        """Test that sample_id becomes the index."""
        df = read_labels_csv(labels_csv)

        # Index should be sample IDs
        assert df.index[0] == "SAMPLE_000"
        assert len(df.index) == 50

    def test_read_labels_csv_coerces_numeric_sample_id_to_str(self, numeric_labels_csv):
        """read_labels_csv must coerce int sample_ids to str index dtype."""
        df = read_labels_csv(numeric_labels_csv)
        assert pd.api.types.is_string_dtype(
            df.index
        ), f"Expected str dtype for index, got {df.index.dtype}"
        assert df.index[0] == "0"


class TestAdmixtureIO:
    """Tests for admixture CSV I/O."""

    def test_read_admixture_csv_basic(self, temp_dir):
        """read_admixture_csv returns DataFrame with sample_id column and component columns."""
        df = pd.DataFrame(
            {
                "sample_id": [f"SAMPLE_{i:03d}" for i in range(20)],
                "component_1": np.random.dirichlet([1, 1], 20)[:, 0],
                "component_2": np.random.dirichlet([1, 1], 20)[:, 1],
            }
        )
        path = temp_dir / "admix.2.csv"
        df.to_csv(path, index=False)

        result = read_admixture_csv(path)
        assert "sample_id" in result.columns
        assert "component_1" in result.columns
        assert pd.api.types.is_string_dtype(result["sample_id"])

    def test_read_admixture_csv_coerces_numeric_sample_id_to_str(self, temp_dir):
        """read_admixture_csv must coerce int sample_ids to str dtype."""
        df = pd.DataFrame(
            {
                "sample_id": np.arange(20, dtype=np.int64),
                "component_1": np.random.rand(20),
                "component_2": np.random.rand(20),
            }
        )
        path = temp_dir / "admix_numeric.2.csv"
        df.to_csv(path, index=False)

        result = read_admixture_csv(path)
        assert pd.api.types.is_string_dtype(
            result["sample_id"]
        ), f"Expected str dtype, got {result['sample_id'].dtype}"
        assert result["sample_id"].iloc[0] == "0"


class TestColormapIO:
    """Tests for colormap JSON I/O."""

    def test_read_colormap(self, colormap_json):
        """Test reading colormap JSON."""
        colormap = read_colormap(colormap_json)

        assert "Population" in colormap
        assert "Region" in colormap
        assert colormap["Population"]["PopA"] == "#FF0000"

    def test_colormap_hex_colors(self, colormap_json):
        """Test that colors are valid hex codes."""
        colormap = read_colormap(colormap_json)

        for label_col, colors in colormap.items():
            for color_value in colors.values():
                assert color_value.startswith("#")
                assert len(color_value) == 7  # #RRGGBB


class TestCoerceIdsToStr:
    """Tests for _coerce_ids_to_str helper.

    The core problem: when a CSV has integer sample IDs, pandas may load them
    as float64 if any NaNs are present (legacy pandas behaviour). A naive
    .astype(str) then yields "123456.0" instead of "123456", breaking merges.
    """

    def test_int64_unchanged(self):
        """int64 series → plain string conversion, no '.0' suffix."""
        s = pd.Series([0, 1, 123456], dtype=np.int64)
        result = _coerce_ids_to_str(s)
        assert list(result) == ["0", "1", "123456"]

    def test_float64_whole_numbers(self):
        """float64 whole numbers (e.g. from NaN-containing int column) → no '.0' suffix."""
        s = pd.Series([0.0, 1.0, 123456.0], dtype=np.float64)
        result = _coerce_ids_to_str(s)
        assert list(result) == ["0", "1", "123456"]

    def test_float64_non_integer_values_preserved(self):
        """float64 non-integer values must NOT be truncated to int."""
        s = pd.Series([1.2, 3.7, 99.9], dtype=np.float64)
        result = _coerce_ids_to_str(s)
        assert list(result) == ["1.2", "3.7", "99.9"]

    def test_float64_whole_numbers_with_nan(self):
        """float64 whole numbers with NaN → integer strings, NaN preserved as 'nan'."""
        s = pd.Series([0.0, np.nan, 2.0], dtype=np.float64)
        result = _coerce_ids_to_str(s)
        assert result.iloc[0] == "0"
        assert result.iloc[2] == "2"
        # NaN position should not crash and should be some string
        assert isinstance(result.iloc[1], str)

    def test_object_string_series_unchanged(self):
        """String (object dtype) series passes through unchanged."""
        s = pd.Series(["SAMPLE_001", "SAMPLE_002", "IND_999"], dtype=object)
        result = _coerce_ids_to_str(s)
        assert list(result) == ["SAMPLE_001", "SAMPLE_002", "IND_999"]

    def test_mixed_integer_ids_merge_after_coercion(self, temp_dir):
        """End-to-end: PCA CSV with float64 IDs merges correctly with string labels.

        Simulates the real bug: FlashPCA writes integer IDs → pandas reads as
        float64 when NaNs exist → naive str coercion breaks the merge.
        """
        # PCA file: sample_ids stored as float64 (as if loaded from a CSV with NaNs elsewhere)
        pca_ids_float = pd.Series([100.0, 200.0, 300.0], dtype=np.float64)
        pca_df = pd.DataFrame(
            {
                "sample_id": pca_ids_float,
                "dim_1": [0.1, 0.2, 0.3],
                "dim_2": [0.4, 0.5, 0.6],
            }
        )
        pca_path = temp_dir / "pca_float_ids.csv"
        pca_df.to_csv(pca_path, index=False)

        # Labels file: same IDs stored as plain integers / strings
        labels_df = pd.DataFrame(
            {
                "sample_id": ["100", "200", "300"],
                "Population": ["A", "B", "C"],
            }
        )
        labels_path = temp_dir / "labels_str_ids.csv"
        labels_df.to_csv(labels_path, index=False)

        pca_read = read_embedding_csv(pca_path)
        labels_read = read_labels_csv(labels_path)
        labels_read = labels_read.reset_index()  # bring sample_id back as column

        merged = pca_read.merge(labels_read, on="sample_id", how="inner")
        assert len(merged) == 3, (
            f"Expected all 3 samples to merge, got {len(merged)}. "
            f"PCA IDs: {pca_read['sample_id'].tolist()}, "
            f"Labels IDs: {labels_read['sample_id'].tolist()}"
        )
