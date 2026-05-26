"""Tests for preservation metrics (geographic and admixture)."""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.metrics import (
    compute_admixture_preservation,
    compute_geographic_preservation,
)


class TestGeographicPreservation:
    """Tests for geographic preservation metrics."""

    @pytest.fixture
    def geographic_coords(self, temp_dir):
        """Create fake geographic coordinates (lat/lon)."""
        np.random.seed(42)
        n_samples = 50

        # Create realistic-looking coordinates
        # Simulate samples spread across a region
        lats = 40 + np.random.randn(n_samples) * 5  # ~40°N ± 5°
        lons = -100 + np.random.randn(n_samples) * 10  # ~100°W ± 10°

        geo_df = pd.DataFrame(
            {
                "sample_id": [f"SAMPLE_{i:03d}" for i in range(n_samples)],
                "latitude": lats,  # Changed to lowercase to match function default
                "longitude": lons,  # Changed to lowercase to match function default
            }
        )

        geo_file = temp_dir / "geo_coords.csv"
        geo_df.to_csv(geo_file, index=False)
        return geo_file

    def test_geographic_preservation_basic(self, small_embedding_csv, geographic_coords):
        """Test basic geographic preservation computation."""
        result = compute_geographic_preservation(
            embedding=small_embedding_csv,
            geographic_coords=geographic_coords,
        )

        # Check result structure
        assert "correlation" in result
        assert "p_value" in result
        assert "n_samples" in result
        assert "n_pairs" in result

        # Check types
        assert isinstance(result["correlation"], float)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["n_samples"], int)
        assert isinstance(result["n_pairs"], int)

        # Correlation should be between -1 and 1
        assert -1 <= result["correlation"] <= 1

    def test_geographic_preservation_with_dataframes(self, small_embedding_data):
        """Test with DataFrame inputs instead of files."""
        np.random.seed(42)
        n_samples = 50

        # Create geographic coordinates DataFrame with matching sample IDs
        sample_ids = small_embedding_data["sample_id"].tolist()
        geo_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "latitude": 40 + np.random.randn(n_samples) * 5,  # Changed to lowercase
                "longitude": -100 + np.random.randn(n_samples) * 10,  # Changed to lowercase
            }
        )
        # Set sample_id as index (like read_labels_csv does)
        geo_df = geo_df.set_index("sample_id")

        result = compute_geographic_preservation(
            embedding=small_embedding_data,
            geographic_coords=geo_df,
        )

        assert "correlation" in result
        assert result["n_samples"] == n_samples


class TestAdmixturePreservation:
    """Tests for admixture preservation metrics."""

    @pytest.fixture
    def fake_q_files(self, temp_dir):
        """Create fake admixture Q files in CSV format with sample_id."""
        np.random.seed(42)
        n_samples = 50
        q_files = {}

        for k in [2, 3, 4]:
            # Generate random admixture proportions that sum to 1
            q_data = np.random.dirichlet(np.ones(k), size=n_samples)

            # Create CSV with sample_id and component columns
            sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
            component_cols = [f"component_{i + 1}" for i in range(k)]

            df = pd.DataFrame(q_data, columns=component_cols)
            df.insert(0, "sample_id", sample_ids)

            # Save as CSV file
            q_file = temp_dir / f"admixture_k{k}.csv"
            df.to_csv(q_file, index=False)
            q_files[k] = q_file

        return q_files

    def test_admixture_preservation_basic(self, small_embedding_csv, fake_q_files):
        """Test basic admixture preservation computation."""
        result = compute_admixture_preservation(
            embedding=small_embedding_csv,
            q_files=fake_q_files,
        )

        # Should have results for each K value
        assert 2 in result
        assert 3 in result
        assert 4 in result

        # Check structure for each K
        for k, metrics in result.items():
            assert "correlation" in metrics
            assert "p_value" in metrics
            assert "n_samples" in metrics
            assert "n_pairs" in metrics

            # Correlation should be between -1 and 1
            assert -1 <= metrics["correlation"] <= 1

    def test_admixture_preservation_single_k(self, small_embedding_csv, fake_q_files):
        """Test with specific K value."""
        result = compute_admixture_preservation(
            embedding=small_embedding_csv,
            q_files=fake_q_files,
            k_value=3,
        )

        # Should only have results for K=3
        assert 3 in result
        assert 2 not in result
        assert 4 not in result

    def test_admixture_preservation_with_dataframe(self, small_embedding_data, fake_q_files):
        """Test with DataFrame embedding input."""
        result = compute_admixture_preservation(
            embedding=small_embedding_data,
            q_files=fake_q_files,
        )

        assert len(result) == 3  # K=2,3,4
        assert all(isinstance(metrics, dict) for metrics in result.values())

    def test_admixture_preservation_with_numeric_sample_ids(self, temp_dir):
        """compute_admixture_preservation must not fail when Q files have int sample_ids."""
        np.random.seed(42)
        n_samples = 50
        # Embedding with string-form integer sample_ids (as read_embedding_csv produces)
        embedding_df = pd.DataFrame(np.random.randn(n_samples, 2), columns=["dim_1", "dim_2"])
        embedding_df.insert(0, "sample_id", [str(i) for i in range(n_samples)])
        embedding_file = temp_dir / "embedding.csv"
        embedding_df.to_csv(embedding_file, index=False)

        # Q file written with raw int64 sample_ids (pre-fix scenario)
        q_data = np.random.dirichlet([1, 1], n_samples)
        q_df = pd.DataFrame(q_data, columns=["component_1", "component_2"])
        q_df.insert(0, "sample_id", np.arange(n_samples, dtype=np.int64))
        q_file = temp_dir / "admix.2.csv"
        q_df.to_csv(q_file, index=False)

        result = compute_admixture_preservation(
            embedding=embedding_file,
            q_files={2: q_file},
        )
        assert 2 in result
        assert result[2]["n_samples"] == n_samples


class TestMetricsIntegration:
    """Integration tests for metrics with realistic scenarios."""

    def test_positive_control_perfect_preservation(self, temp_dir):
        """Positive control: Geographic coords == embedding coords → correlation ≈ 1.0"""
        np.random.seed(42)
        n_samples = 100

        # Create 2D embedding with sample_id column
        embedding = np.random.randn(n_samples, 2) * 10
        sample_ids = [f"S{i}" for i in range(n_samples)]
        embedding_df = pd.DataFrame(embedding, columns=["dim_1", "dim_2"])
        embedding_df.insert(0, "sample_id", sample_ids)
        embedding_file = temp_dir / "embedding.csv"
        embedding_df.to_csv(embedding_file, index=False)

        # POSITIVE CONTROL: Use embedding coords as geographic coords
        geo_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "latitude": embedding[:, 0],  # dim_1 = latitude (changed to lowercase)
                "longitude": embedding[:, 1],  # dim_2 = longitude (changed to lowercase)
            }
        )
        geo_file = temp_dir / "geo.csv"
        geo_df.to_csv(geo_file, index=False)

        result = compute_geographic_preservation(embedding_file, geo_file)

        # Should have near-perfect correlation
        assert result["correlation"] > 0.95
        assert result["p_value"] < 0.001

    def test_negative_control_random_data(self, temp_dir):
        """Negative control: Random uncorrelated data → correlation ≈ 0.0"""
        np.random.seed(42)
        n_samples = 100

        # Create random embedding with sample_id
        embedding = np.random.randn(n_samples, 2)
        sample_ids = [f"S{i}" for i in range(n_samples)]
        embedding_df = pd.DataFrame(embedding, columns=["dim_1", "dim_2"])
        embedding_df.insert(0, "sample_id", sample_ids)
        embedding_file = temp_dir / "embedding.csv"
        embedding_df.to_csv(embedding_file, index=False)

        # NEGATIVE CONTROL: Completely different random coordinates
        np.random.seed(123)  # Different seed
        geo_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "latitude": np.random.randn(n_samples) * 50,  # Changed to lowercase
                "longitude": np.random.randn(n_samples) * 50,  # Changed to lowercase
            }
        )
        geo_file = temp_dir / "geo.csv"
        geo_df.to_csv(geo_file, index=False)

        result = compute_geographic_preservation(embedding_file, geo_file)

        # Correlation should be weak (close to 0)
        assert abs(result["correlation"]) < 0.3

    def test_missing_geographic_data(self, temp_dir):
        """Test with missing geographic coordinates for some samples."""
        np.random.seed(42)
        n_samples = 100

        # Create embedding for all samples with sample_id
        embedding = np.random.randn(n_samples, 2)
        sample_ids = [f"S{i}" for i in range(n_samples)]
        embedding_df = pd.DataFrame(embedding, columns=["dim_1", "dim_2"])
        embedding_df.insert(0, "sample_id", sample_ids)
        embedding_file = temp_dir / "embedding.csv"
        embedding_df.to_csv(embedding_file, index=False)

        # Create geographic data with 20 samples missing (NaN values)
        lats = np.random.randn(n_samples) * 10
        lons = np.random.randn(n_samples) * 10

        # Set 20 random samples to NaN
        missing_indices = np.random.choice(n_samples, 20, replace=False)
        lats[missing_indices] = np.nan
        lons[missing_indices] = np.nan

        geo_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "latitude": lats,  # Changed to lowercase
                "longitude": lons,  # Changed to lowercase
            }
        )
        geo_file = temp_dir / "geo.csv"
        geo_df.to_csv(geo_file, index=False)

        result = compute_geographic_preservation(embedding_file, geo_file, ignore_missing=True)

        # Should use only 80 samples (those with coordinates)
        assert result["n_samples"] == 80

    def test_partial_admixture_data(self, temp_dir):
        """Test admixture preservation when Q file has fewer samples than embedding."""
        np.random.seed(42)
        n_embedding_samples = 100
        n_admixture_samples = 80  # Only 80 have admixture data

        # Create full embedding with sample_id
        embedding = np.random.randn(n_embedding_samples, 2)
        embedding_sample_ids = [f"S{i}" for i in range(n_embedding_samples)]
        embedding_df = pd.DataFrame(embedding, columns=["dim_1", "dim_2"])
        embedding_df.insert(0, "sample_id", embedding_sample_ids)
        embedding_file = temp_dir / "embedding.csv"
        embedding_df.to_csv(embedding_file, index=False)

        # Create admixture CSV file with only first 80 samples
        q_data = np.random.dirichlet(np.ones(3), size=n_admixture_samples)
        admixture_sample_ids = [f"S{i}" for i in range(n_admixture_samples)]

        q_df = pd.DataFrame(q_data, columns=["component_1", "component_2", "component_3"])
        q_df.insert(0, "sample_id", admixture_sample_ids)

        q_file = temp_dir / "admixture_k3.csv"
        q_df.to_csv(q_file, index=False)

        q_files = {3: q_file}

        result = compute_admixture_preservation(
            embedding=embedding_file,
            q_files=q_files,
        )

        # Should use only samples that have admixture data
        assert result[3]["n_samples"] == n_admixture_samples

    def test_admixture_positive_control(self, temp_dir):
        """Positive control: Embedding distances = admixture distances."""
        np.random.seed(42)
        n_samples = 100

        # Create admixture proportions
        k = 3
        q_data = np.random.dirichlet(np.ones(k), size=n_samples)
        sample_ids = [f"S{i}" for i in range(n_samples)]

        # POSITIVE CONTROL: Use admixture proportions as embedding
        embedding_df = pd.DataFrame(q_data, columns=["dim_1", "dim_2", "dim_3"])
        embedding_df.insert(0, "sample_id", sample_ids)
        embedding_file = temp_dir / "embedding.csv"
        embedding_df.to_csv(embedding_file, index=False)

        # Save Q file in new CSV format
        q_df = pd.DataFrame(q_data, columns=["component_1", "component_2", "component_3"])
        q_df.insert(0, "sample_id", sample_ids)
        q_file = temp_dir / "admixture_k3.csv"
        q_df.to_csv(q_file, index=False)
        q_files = {3: q_file}

        result = compute_admixture_preservation(embedding_file, q_files)

        # Should have very high correlation (distances match exactly)
        assert result[3]["correlation"] > 0.95
        assert result[3]["p_value"] < 0.001
