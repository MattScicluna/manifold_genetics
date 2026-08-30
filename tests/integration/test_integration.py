"""
Integration tests for manifold-genetics pipeline.

These tests verify end-to-end workflows using small test data.
They use precomputed admixture fixtures to avoid slow neural-admixture computation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
def test_fixtures_exist():
    """Verify test fixtures are available."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"

    # Check admixture fixtures
    admix_dir = fixtures_dir / "admixture"
    assert (admix_dir / "fit.2.csv").exists(), "Missing fit.2.csv fixture"
    assert (admix_dir / "fit.3.csv").exists(), "Missing fit.3.csv fixture"
    assert (admix_dir / "project.2.csv").exists(), "Missing project.2.csv fixture"
    assert (admix_dir / "project.3.csv").exists(), "Missing project.3.csv fixture"

    # Verify fixture format
    fit_2 = pd.read_csv(admix_dir / "fit.2.csv")
    assert "sample_id" in fit_2.columns
    assert "component_1" in fit_2.columns
    assert "component_2" in fit_2.columns
    assert len(fit_2) == 50, "Expected 50 samples in fixtures"

    # Verify components sum to 1.0
    components = fit_2[["component_1", "component_2"]].values
    sums = components.sum(axis=1)
    assert np.allclose(sums, 1.0, rtol=1e-5), "Components should sum to 1.0"


@pytest.mark.integration
def test_precomputed_admixture_format():
    """Verify precomputed admixture files have correct format."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "admixture"

    for k in [2, 3]:
        for subset in ["fit", "project"]:
            filepath = fixtures_dir / f"{subset}.{k}.csv"
            df = pd.read_csv(filepath)

            # Check structure
            assert "sample_id" in df.columns
            assert len(df.columns) == k + 1  # sample_id + K components

            # Check component names
            for i in range(1, k + 1):
                assert f"component_{i}" in df.columns

            # Check values
            components = df[[f"component_{i}" for i in range(1, k + 1)]].values
            assert np.all(components >= 0), "All components should be non-negative"
            assert np.all(components <= 1), "All components should be <= 1.0"

            # Check sums
            sums = components.sum(axis=1)
            assert np.allclose(sums, 1.0, rtol=1e-5), f"Components in {filepath} should sum to 1.0"


@pytest.mark.integration
def test_sample_ids_consistent():
    """Verify sample IDs are consistent across fixtures."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "admixture"

    # Load all fixture files
    fit_2 = pd.read_csv(fixtures_dir / "fit.2.csv")
    fit_3 = pd.read_csv(fixtures_dir / "fit.3.csv")
    project_2 = pd.read_csv(fixtures_dir / "project.2.csv")
    project_3 = pd.read_csv(fixtures_dir / "project.3.csv")

    # Check fit subset has consistent sample IDs
    assert list(fit_2["sample_id"]) == list(
        fit_3["sample_id"]
    ), "Fit subset should have same sample IDs across K values"

    # Check project subset has consistent sample IDs
    assert list(project_2["sample_id"]) == list(
        project_3["sample_id"]
    ), "Project subset should have same sample IDs across K values"

    # Check expected sample ID format
    expected_ids = [f"SAMPLE_{i:03d}" for i in range(50)]
    assert list(fit_2["sample_id"]) == expected_ids
    assert list(project_2["sample_id"]) == expected_ids


# TODO: Add integration test with actual pipeline run once admixture backend interface is implemented
# @pytest.mark.integration
# def test_generic_pipeline_with_precomputed_admixture(tmp_path):
#     """
#     Test complete pipeline with precomputed admixture.
#
#     This test will:
#     1. Use tiny PLINK files (50 samples, 100 SNPs)
#     2. Use PrecomputedAdmixtureBackend to bypass slow computation
#     3. Run full pipeline: PCA → Admix (fake) → Embedding → Viz
#     4. Validate output structure
#     """
#     pass
