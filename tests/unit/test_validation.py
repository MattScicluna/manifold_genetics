"""Tests for input validation utilities."""

import json
import logging

import pandas as pd
import pytest

from manifold_genetics.utils.validation import (
    ValidationError,
    validate_admixture_csv,
    validate_colormap_json,
    validate_embedding_csv,
    validate_geographic_csv,
    validate_labels_colormap_match,
    validate_labels_csv,
    validate_sample_id_overlap,
)


# ---------------------------------------------------------------------------
# validate_embedding_csv
# ---------------------------------------------------------------------------


class TestValidateEmbeddingCsv:
    def test_valid_embedding_passes(self, small_embedding_csv):
        validate_embedding_csv(small_embedding_csv)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_embedding_csv(tmp_path / "nonexistent.csv")

    def test_missing_sample_id_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"dim_1": [1.0], "dim_2": [2.0]}).to_csv(path, index=False)
        with pytest.raises(ValidationError, match="sample_id"):
            validate_embedding_csv(path)

    def test_no_dim_columns_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"sample_id": ["s1"], "foo": [1.0]}).to_csv(path, index=False)
        with pytest.raises(ValidationError, match="dim_"):
            validate_embedding_csv(path)

    def test_non_numeric_dim_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"sample_id": ["s1"], "dim_1": ["abc"]}).to_csv(path, index=False)
        with pytest.raises(ValidationError, match="non-numeric"):
            validate_embedding_csv(path)


# ---------------------------------------------------------------------------
# validate_labels_csv
# ---------------------------------------------------------------------------


class TestValidateLabelsCsv:
    def test_valid_labels_passes(self, labels_csv):
        validate_labels_csv(labels_csv)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_labels_csv(tmp_path / "nonexistent.csv")

    def test_missing_sample_id_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"Population": ["A"], "Region": ["North"]}).to_csv(
            path, index=False
        )
        with pytest.raises(ValidationError, match="sample_id"):
            validate_labels_csv(path)


# ---------------------------------------------------------------------------
# validate_colormap_json
# ---------------------------------------------------------------------------


class TestValidateColormapJson:
    def test_valid_colormap_passes(self, colormap_json):
        validate_colormap_json(colormap_json)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_colormap_json(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{broken json")
        with pytest.raises(ValidationError, match="Invalid JSON"):
            validate_colormap_json(path)

    def test_non_dict_top_level_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValidationError, match="JSON object"):
            validate_colormap_json(path)

    def test_non_dict_inner_value_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"Population": "not_a_dict"}))
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_colormap_json(path)


# ---------------------------------------------------------------------------
# validate_admixture_csv
# ---------------------------------------------------------------------------


class TestValidateAdmixtureCsv:
    def _write_q_files(self, tmp_path, prefix_name="q", k_values=(2, 3)):
        prefix = tmp_path / prefix_name
        for k in k_values:
            cols = {"sample_id": ["s1", "s2"]}
            for i in range(1, k + 1):
                cols[f"component_{i}"] = [1.0 / k] * 2
            pd.DataFrame(cols).to_csv(f"{prefix}.{k}.csv", index=False)
        return str(prefix)

    def test_valid_admixture_passes(self, tmp_path):
        prefix = self._write_q_files(tmp_path)
        validate_admixture_csv(prefix, [2, 3])

    def test_missing_file_raises(self, tmp_path):
        prefix = tmp_path / "q"
        with pytest.raises(ValidationError, match="not found"):
            validate_admixture_csv(str(prefix), [2, 3])

    def test_missing_sample_id_raises(self, tmp_path):
        prefix = tmp_path / "q"
        pd.DataFrame({"component_1": [0.5], "component_2": [0.5]}).to_csv(
            f"{prefix}.2.csv", index=False
        )
        with pytest.raises(ValidationError, match="sample_id"):
            validate_admixture_csv(str(prefix), [2])

    def test_no_component_columns_raises(self, tmp_path):
        prefix = tmp_path / "q"
        pd.DataFrame({"sample_id": ["s1"], "foo": [0.5]}).to_csv(
            f"{prefix}.2.csv", index=False
        )
        with pytest.raises(ValidationError, match="component_"):
            validate_admixture_csv(str(prefix), [2])


# ---------------------------------------------------------------------------
# validate_geographic_csv
# ---------------------------------------------------------------------------


class TestValidateGeographicCsv:
    def test_valid_geographic_passes(self, tmp_path):
        path = tmp_path / "geo.csv"
        pd.DataFrame(
            {"sample_id": ["s1"], "longitude": [1.0], "latitude": [2.0]}
        ).to_csv(path, index=False)
        validate_geographic_csv(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_geographic_csv(tmp_path / "nonexistent.csv")

    def test_missing_lat_lon_raises(self, tmp_path):
        path = tmp_path / "geo.csv"
        pd.DataFrame({"sample_id": ["s1"], "x": [1.0], "y": [2.0]}).to_csv(
            path, index=False
        )
        with pytest.raises(ValidationError, match="longitude"):
            validate_geographic_csv(path)

    def test_custom_column_names(self, tmp_path):
        path = tmp_path / "geo.csv"
        pd.DataFrame({"sample_id": ["s1"], "lon": [1.0], "lat": [2.0]}).to_csv(
            path, index=False
        )
        validate_geographic_csv(path, longitude_col="lon", latitude_col="lat")

    def test_non_numeric_coords_raises(self, tmp_path):
        path = tmp_path / "geo.csv"
        pd.DataFrame(
            {"sample_id": ["s1"], "longitude": ["abc"], "latitude": ["def"]}
        ).to_csv(path, index=False)
        with pytest.raises(ValidationError, match="non-numeric"):
            validate_geographic_csv(path)


# ---------------------------------------------------------------------------
# validate_labels_colormap_match
# ---------------------------------------------------------------------------


class TestValidateLabelsColormapMatch:
    def test_matching_passes(self, labels_csv, colormap_json):
        validate_labels_colormap_match(labels_csv, colormap_json)

    def test_misspelled_column_raises_with_suggestion(self, tmp_path):
        labels_path = tmp_path / "labels.csv"
        pd.DataFrame(
            {"sample_id": ["s1"], "Population": ["A"], "Region": ["N"]}
        ).to_csv(labels_path, index=False)

        cmap_path = tmp_path / "cmap.json"
        cmap_path.write_text(json.dumps({"Populaton": {"A": "#FF0000"}}))

        with pytest.raises(ValidationError, match="Did you mean.*Population"):
            validate_labels_colormap_match(labels_path, cmap_path)

    def test_missing_label_values_warns(self, tmp_path, caplog):
        labels_path = tmp_path / "labels.csv"
        pd.DataFrame(
            {"sample_id": ["s1", "s2", "s3"], "Pop": ["A", "B", "C"]}
        ).to_csv(labels_path, index=False)

        cmap_path = tmp_path / "cmap.json"
        cmap_path.write_text(json.dumps({"Pop": {"A": "#FF0000", "B": "#00FF00"}}))

        with caplog.at_level(logging.WARNING):
            validate_labels_colormap_match(labels_path, cmap_path)

        assert "C" in caplog.text


# ---------------------------------------------------------------------------
# validate_sample_id_overlap
# ---------------------------------------------------------------------------


class TestValidateSampleIdOverlap:
    def test_full_overlap_passes(self, tmp_path):
        path1 = tmp_path / "a.csv"
        path2 = tmp_path / "b.csv"
        pd.DataFrame({"sample_id": ["s1", "s2"]}).to_csv(path1, index=False)
        pd.DataFrame({"sample_id": ["s1", "s2"]}).to_csv(path2, index=False)
        validate_sample_id_overlap(path1, path2, "file1", "file2")

    def test_zero_overlap_raises(self, tmp_path):
        path1 = tmp_path / "a.csv"
        path2 = tmp_path / "b.csv"
        pd.DataFrame({"sample_id": ["s1", "s2"]}).to_csv(path1, index=False)
        pd.DataFrame({"sample_id": ["s3", "s4"]}).to_csv(path2, index=False)
        with pytest.raises(ValidationError, match="No overlapping"):
            validate_sample_id_overlap(path1, path2, "file1", "file2")

    def test_low_overlap_warns(self, tmp_path, caplog):
        path1 = tmp_path / "a.csv"
        path2 = tmp_path / "b.csv"
        pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4", "s5"]}).to_csv(
            path1, index=False
        )
        pd.DataFrame({"sample_id": ["s1", "x2", "x3", "x4", "x5"]}).to_csv(
            path2, index=False
        )
        with caplog.at_level(logging.WARNING):
            validate_sample_id_overlap(path1, path2, "file1", "file2")
        assert "Low sample_id overlap" in caplog.text

    def test_partial_overlap_passes(self, tmp_path):
        path1 = tmp_path / "a.csv"
        path2 = tmp_path / "b.csv"
        pd.DataFrame({"sample_id": ["s1", "s2", "s3"]}).to_csv(path1, index=False)
        pd.DataFrame({"sample_id": ["s1", "s2", "s4"]}).to_csv(path2, index=False)
        # 2/3 = 67% overlap, should pass without error
        validate_sample_id_overlap(path1, path2, "file1", "file2")
