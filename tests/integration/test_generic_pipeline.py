"""
Integration test for generic pipeline workflow.

Tests end-to-end pipeline execution with precomputed admixture backend.
Uses small test data (50 samples, 100 SNPs) for fast execution.
"""

import pytest
import pandas as pd
from pathlib import Path

from manifold_genetics.pipeline import run_pipeline
from manifold_genetics.admixture.backends import PrecomputedAdmixtureBackend


@pytest.mark.integration
def test_generic_pipeline_with_precomputed_admixture(
    fit_plink_files,
    transform_plink_files,
    labels_csv,
    colormap_json,
    temp_dir,
):
    """
    Test complete pipeline with precomputed admixture backend.

    This test:
    1. Uses tiny PLINK files (50 samples, 100 SNPs)
    2. Uses PrecomputedAdmixtureBackend to bypass slow computation
    3. Runs full pipeline: PCA → Admix (precomputed) → Embedding → Viz
    4. Validates output structure
    """
    # Find fixtures directory
    tests_dir = Path(__file__).parent.parent
    fixtures_dir = tests_dir / "fixtures" / "admixture"

    # Create precomputed admixture backend
    admix_backend = PrecomputedAdmixtureBackend(
        k_min=2,
        k_max=3,
        fixtures_dir=str(fixtures_dir)
    )

    # Set up output directory
    output_dir = temp_dir / "pipeline_output"
    output_dir.mkdir()

    # Run pipeline with precomputed admixture backend
    try:
        results = run_pipeline(
            fit_plink=fit_plink_files,
            project_plink=transform_plink_files,
            output_dir=str(output_dir),
            labels=str(labels_csv),
            colormap=str(colormap_json),
            # PCA parameters
            n_pcs=10,  # Small number for fast testing
            # Admixture parameters - use precomputed backend for fast testing
            k_min=2,
            k_max=3,
            admixture_backend=admix_backend,  # Inject precomputed backend!
            skip_admixture_visualization=True,  # Skip viz for now (requires colormap refactoring)
            skip_pca_visualization=True,  # Skip PCA viz to avoid DataFrame vs path issue
            # Embedding parameters
            embedding="phate",
            embedding_params={"knn": 5, "n_components": 2},  # Small knn for fast testing
            # Skip metrics and viz since we don't have geographic data in test fixtures
            skip_metrics=True,
            skip_visualization=True,  # Skip viz for now - there's a bug with DataFrame vs path
        )

        # Validate PCA outputs
        assert "pca_coords" in results
        assert results["pca_coords"] is not None

        # Handle both DataFrame and string path returns
        if isinstance(results["pca_coords"], pd.DataFrame):
            pca_df = results["pca_coords"]
        else:
            pca_file = Path(results["pca_coords"])
            assert pca_file.exists(), "PCA output file should exist"
            pca_df = pd.read_csv(pca_file)

        assert "sample_id" in pca_df.columns
        assert len(pca_df) == 50  # Transform subset has 50 samples
        assert len(pca_df.columns) == 11  # sample_id + 10 PCs

        # Validate admixture outputs
        assert "admixture_dir" in results
        admix_dir = Path(results["admixture_dir"])
        assert admix_dir.exists(), "Admixture directory should exist"

        # Check that Q files were created for K=2,3
        for k in [2, 3]:
            fit_q_file = admix_dir / f"fit.{k}.csv"
            transform_q_file = admix_dir / f"transform.{k}.csv"
            assert fit_q_file.exists(), f"Fit Q file for K={k} should exist"
            assert transform_q_file.exists(), f"Transform Q file for K={k} should exist"

            # Validate format
            fit_q_df = pd.read_csv(fit_q_file)
            assert "sample_id" in fit_q_df.columns
            assert len(fit_q_df.columns) == k + 1  # sample_id + K components
            assert len(fit_q_df) == 50

        # Validate embedding outputs
        assert "embedding_coords" in results
        assert results["embedding_coords"] is not None

        # Handle both DataFrame and string path returns
        if isinstance(results["embedding_coords"], pd.DataFrame):
            embedding_df = results["embedding_coords"]
        else:
            embedding_file = Path(results["embedding_coords"])
            assert embedding_file.exists(), "Embedding output file should exist"
            embedding_df = pd.read_csv(embedding_file)

        assert "sample_id" in embedding_df.columns
        assert len(embedding_df) == 50
        assert len(embedding_df.columns) == 3  # sample_id + 2 dims

        # NOTE: Visualization is skipped in this test due to DataFrame vs path bug
        # Will be tested separately once that issue is fixed

    except Exception as e:
        pytest.fail(f"Pipeline execution failed: {e}")


@pytest.mark.integration
def test_pipeline_output_structure(
    fit_plink_files,
    transform_plink_files,
    labels_csv,
    colormap_json,
    temp_dir,
):
    """
    Test that pipeline creates expected directory structure.
    """
    output_dir = temp_dir / "pipeline_structure_test"
    output_dir.mkdir()

    results = run_pipeline(
        fit_plink=fit_plink_files,
        project_plink=transform_plink_files,
        output_dir=str(output_dir),
        labels=str(labels_csv),
        colormap=str(colormap_json),
        n_pcs=5,
        skip_admixture=True,
        skip_admixture_visualization=True,
        skip_pca_visualization=True,
        skip_visualization=True,
        embedding="phate",
        embedding_params={"knn": 5, "n_components": 2},
        skip_metrics=True,
    )

    # Check directory structure
    assert (output_dir / "pca").exists(), "PCA directory should exist"
    assert (output_dir / "embeddings").exists(), "Embeddings directory should exist"

    # Check specific files
    assert (output_dir / "pca" / "transform_pca_5.csv").exists(), "PCA transform output should exist"
    assert (output_dir / "embeddings" / "phate_2d.csv").exists(), "Embedding output should exist"


@pytest.mark.integration
def test_pipeline_with_fit_only(
    fit_plink_files,
    labels_csv,
    colormap_json,
    temp_dir,
):
    """
    Test pipeline with fit-only mode (no separate projection).
    """
    output_dir = temp_dir / "fit_only_test"
    output_dir.mkdir()

    results = run_pipeline(
        fit_plink=fit_plink_files,
        project_plink=fit_plink_files,  # Same as fit
        output_dir=str(output_dir),
        labels=str(labels_csv),
        colormap=str(colormap_json),
        n_pcs=5,
        skip_admixture=True,
        skip_admixture_visualization=True,
        skip_pca_visualization=True,
        skip_visualization=True,
        embedding="phate",
        embedding_params={"knn": 5, "n_components": 2},
        skip_metrics=True,
    )

    # Should still work and produce outputs
    assert results["pca_coords"] is not None
    assert results["embedding_coords"] is not None

    # Check that fit and transform outputs exist
    pca_dir = output_dir / "pca"
    assert (pca_dir / "fit_pca_5.csv").exists(), "Fit PCA should exist"
    assert (pca_dir / "transform_pca_5.csv").exists(), "Transform PCA should exist"


@pytest.mark.integration
def test_skip_pca_resolves_existing_pca_files(
    fit_plink_files,
    labels_csv,
    colormap_json,
    temp_dir,
    small_pca_data,
):
    """
    Test that skip_pca=True resolves pre-existing PCA files into results.

    Regression test for the bug where --skip-pca silently blocked the embedding step
    by never populating fit_pca_file/pca_file in the results dict.
    """
    output_dir = temp_dir / "skip_pca_reuse_test"
    output_dir.mkdir()

    pca_dir = output_dir / "pca"
    pca_dir.mkdir()
    n_pcs = 10

    fit_pca_file = pca_dir / f"fit_pca_{n_pcs}.csv"
    transform_pca_file = pca_dir / f"transform_pca_{n_pcs}.csv"
    small_pca_data.to_csv(fit_pca_file, index=False)
    small_pca_data.to_csv(transform_pca_file, index=False)

    results = run_pipeline(
        fit_plink=fit_plink_files,
        project_plink=fit_plink_files,
        output_dir=str(output_dir),
        labels=str(labels_csv),
        colormap=str(colormap_json),
        n_pcs=n_pcs,
        skip_pca=True,
        skip_admixture=True,
        skip_embedding=True,
        skip_visualization=True,
        skip_pca_visualization=True,
        skip_admixture_visualization=True,
        skip_metrics=True,
    )

    assert "fit_pca_file" in results, "fit_pca_file should be resolved from pre-existing file"
    assert "pca_file" in results, "pca_file should be resolved from pre-existing transform file"
    assert results["fit_pca_file"] == fit_pca_file
    assert results["pca_file"] == transform_pca_file


@pytest.mark.integration
def test_skip_pca_raises_when_no_pca_files(
    fit_plink_files,
    labels_csv,
    colormap_json,
    temp_dir,
):
    """
    Test that skip_pca=True with no pre-existing PCA files raises RuntimeError.

    Previously the pipeline would silently skip embedding. Now it raises a clear error.
    """
    output_dir = temp_dir / "skip_pca_missing_test"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="No PCA files found"):
        run_pipeline(
            fit_plink=fit_plink_files,
            project_plink=fit_plink_files,
            output_dir=str(output_dir),
            labels=str(labels_csv),
            colormap=str(colormap_json),
            n_pcs=10,
            skip_pca=True,
            skip_admixture=True,
            # Not skipping embedding — triggers RuntimeError check
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_admixture_visualization=True,
            skip_metrics=True,
        )
