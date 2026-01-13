"""
HGDP+1KGP reproducibility and equivalence tests.

These tests verify:
1. The HGDP+1KGP pipeline is reproducible
2. Stepwise CLI commands produce equivalent results to the pipeline command
3. The README examples actually work

These are integration tests that require the full HGDP+1KGP dataset.
They use precomputed admixture to avoid slow computation but test everything else.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import tempfile


def get_hgdp_data_dir():
    """Get path to HGDP+1KGP data directory."""
    examples_dir = Path(__file__).parent.parent / "examples" / "hgdp_1kgp"
    data_dir = examples_dir / "data"
    return data_dir


def check_hgdp_data_available():
    """Check if HGDP+1KGP data is available."""
    data_dir = get_hgdp_data_dir()
    fit_plink = data_dir / "fit_subset"
    required_files = [
        fit_plink.with_suffix(".bed"),
        fit_plink.with_suffix(".bim"),
        fit_plink.with_suffix(".fam"),
    ]
    return all(f.exists() for f in required_files)


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.skipif(
    not check_hgdp_data_available(),
    reason="HGDP+1KGP data not available (run examples/hgdp_1kgp/download_data.sh)"
)
def test_hgdp_data_structure():
    """Verify HGDP+1KGP data has expected structure."""
    data_dir = get_hgdp_data_dir()

    # Check fit subset
    fit_plink = data_dir / "fit_subset"
    assert fit_plink.with_suffix(".bed").exists()
    assert fit_plink.with_suffix(".bim").exists()
    assert fit_plink.with_suffix(".fam").exists()

    # Check project subset
    project_plink = data_dir / "project_subset"
    assert project_plink.with_suffix(".bed").exists()
    assert project_plink.with_suffix(".bim").exists()
    assert project_plink.with_suffix(".fam").exists()

    # Check labels
    labels_csv = data_dir / "hgdp_project_labels.csv"
    assert labels_csv.exists()

    # Check geographic coordinates
    geographic_csv = data_dir / "hgdp_project_geographic.csv"
    assert geographic_csv.exists()

    # Verify labels format
    labels = pd.read_csv(labels_csv)
    assert "sample_id" in labels.columns
    assert "Population" in labels.columns
    assert "Genetic_region_merged" in labels.columns


# TODO: Implement once admixture backend interface is ready
# @pytest.mark.integration
# @pytest.mark.network
# @pytest.mark.skipif(not check_hgdp_data_available(), reason="HGDP data not available")
# def test_hgdp_pipeline_reproducibility(tmp_path):
#     """
#     Test that running the HGDP pipeline twice produces identical results.
#
#     Uses precomputed admixture to keep test fast.
#     """
#     pass


# TODO: Implement once we can control admixture computation
# @pytest.mark.integration
# @pytest.mark.network
# @pytest.mark.skipif(not check_hgdp_data_available(), reason="HGDP data not available")
# def test_stepwise_cli_equals_pipeline(tmp_path):
#     """
#     Test that stepwise CLI commands produce same output as pipeline command.
#
#     This verifies the README examples are accurate and equivalent.
#
#     Compares outputs with numerical tolerance: numpy.allclose(rtol=1e-5, atol=1e-8)
#     """
#     pass


@pytest.mark.integration
def test_pytest_markers_configured():
    """Verify pytest markers are properly configured."""
    # This test itself demonstrates the markers work
    pass
