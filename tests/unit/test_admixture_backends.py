"""
Unit tests for admixture backends.

Tests the interface and behavior of FakeAdmixtureBackend, PrecomputedAdmixtureBackend,
and (optionally) NeuralAdmixtureBackend.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.admixture.backends import (
    AdmixtureBackend,
    FakeAdmixtureBackend,
    PrecomputedAdmixtureBackend,
)


class TestFakeAdmixtureBackend:
    """Tests for FakeAdmixtureBackend (fast random Q matrix generation)."""

    def test_interface_compliance(self):
        """Verify FakeAdmixtureBackend implements AdmixtureBackend interface."""
        backend = FakeAdmixtureBackend(k_min=2, k_max=5)
        assert isinstance(backend, AdmixtureBackend)
        assert hasattr(backend, "fit")
        assert hasattr(backend, "transform")
        assert hasattr(backend, "fit_transform")

    def test_fit_transform_generates_files(self, dummy_plink_files, temp_dir):
        """Test that fit_transform generates Q files for all K values."""
        backend = FakeAdmixtureBackend(k_min=2, k_max=4, random_seed=42)

        output_prefix = temp_dir / "out"

        # Run fit_transform
        q_files = backend.fit_transform(dummy_plink_files, str(output_prefix))

        # Check that we got files for K=2,3,4
        assert len(q_files) == 3
        assert 2 in q_files
        assert 3 in q_files
        assert 4 in q_files

        # Check files exist
        for k, filepath in q_files.items():
            assert Path(filepath).exists()

    def test_output_format(self, dummy_plink_files, temp_dir):
        """Test that generated Q files have correct CSV format."""
        backend = FakeAdmixtureBackend(k_min=2, k_max=3, random_seed=42)

        output_prefix = temp_dir / "out"

        q_files = backend.fit_transform(dummy_plink_files, str(output_prefix))

        for k, filepath in q_files.items():
            df = pd.read_csv(filepath)

            # Check columns
            assert "sample_id" in df.columns
            assert len(df.columns) == k + 1  # sample_id + K components

            for i in range(1, k + 1):
                assert f"component_{i}" in df.columns

            # Check values
            components = df[[f"component_{i}" for i in range(1, k + 1)]].values
            assert np.all(components >= 0), "All components should be non-negative"
            assert np.all(components <= 1), "All components should be <= 1.0"

            # Check sums (should be 1.0 for valid Q matrix)
            sums = components.sum(axis=1)
            assert np.allclose(sums, 1.0, rtol=1e-5), "Components should sum to 1.0"

            # Check sample count matches PLINK file (50 samples)
            assert len(df) == 50

    def test_reproducibility_with_seed(self, dummy_plink_files, temp_dir):
        """Test that same seed produces identical Q matrices."""
        backend1 = FakeAdmixtureBackend(k_min=2, k_max=2, random_seed=42)
        backend2 = FakeAdmixtureBackend(k_min=2, k_max=2, random_seed=42)

        output_prefix1 = temp_dir / "out1"
        output_prefix2 = temp_dir / "out2"

        q_files1 = backend1.fit_transform(dummy_plink_files, str(output_prefix1))
        q_files2 = backend2.fit_transform(dummy_plink_files, str(output_prefix2))

        df1 = pd.read_csv(q_files1[2])
        df2 = pd.read_csv(q_files2[2])

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_results(self, dummy_plink_files, temp_dir):
        """Test that different seeds produce different Q matrices."""
        backend1 = FakeAdmixtureBackend(k_min=2, k_max=2, random_seed=42)
        backend2 = FakeAdmixtureBackend(k_min=2, k_max=2, random_seed=123)

        output_prefix1 = temp_dir / "out1"
        output_prefix2 = temp_dir / "out2"

        q_files1 = backend1.fit_transform(dummy_plink_files, str(output_prefix1))
        q_files2 = backend2.fit_transform(dummy_plink_files, str(output_prefix2))

        df1 = pd.read_csv(q_files1[2])
        df2 = pd.read_csv(q_files2[2])

        # Sample IDs should match
        assert list(df1["sample_id"]) == list(df2["sample_id"])

        # But component values should differ
        components1 = df1[["component_1", "component_2"]].values
        components2 = df2[["component_1", "component_2"]].values
        assert not np.allclose(
            components1, components2
        ), "Different seeds should produce different Q matrices"


class TestPrecomputedAdmixtureBackend:
    """Tests for PrecomputedAdmixtureBackend (loads fixtures)."""

    def test_interface_compliance(self):
        """Verify PrecomputedAdmixtureBackend implements AdmixtureBackend interface."""
        backend = PrecomputedAdmixtureBackend(k_min=2, k_max=3)
        assert isinstance(backend, AdmixtureBackend)
        assert hasattr(backend, "fit")
        assert hasattr(backend, "transform")
        assert hasattr(backend, "fit_transform")

    def test_loads_existing_fixtures(self, fit_plink_files, temp_dir):
        """Test that backend successfully loads precomputed fixtures."""
        # Find fixtures directory
        tests_dir = Path(__file__).parent.parent
        fixtures_dir = tests_dir / "fixtures" / "admixture"

        assert fixtures_dir.exists(), f"Fixtures not found at {fixtures_dir}"

        backend = PrecomputedAdmixtureBackend(k_min=2, k_max=3, fixtures_dir=str(fixtures_dir))

        output_prefix = temp_dir / "out"
        q_files = backend.fit_transform(fit_plink_files, str(output_prefix))

        # Check that we got files for K=2,3
        assert len(q_files) == 2
        assert 2 in q_files
        assert 3 in q_files

        # Check files exist and have correct format
        for k, filepath in q_files.items():
            assert Path(filepath).exists()
            df = pd.read_csv(filepath)
            assert "sample_id" in df.columns
            assert len(df.columns) == k + 1

    def test_fit_and_transform_separately(self, fit_plink_files, transform_plink_files, temp_dir):
        """Test that fit() and transform() work separately."""
        tests_dir = Path(__file__).parent.parent
        fixtures_dir = tests_dir / "fixtures" / "admixture"

        backend = PrecomputedAdmixtureBackend(k_min=2, k_max=3, fixtures_dir=str(fixtures_dir))

        # Fit on "fit" subset
        output_dir = temp_dir / "models"
        output_dir.mkdir()
        backend.fit(fit_plink_files, str(output_dir), "model")

        # Transform on "transform" subset
        output_prefix = temp_dir / "transform_out"
        q_files = backend.transform(transform_plink_files, str(output_prefix))

        # Check transform output
        assert len(q_files) == 2
        for k, filepath in q_files.items():
            assert Path(filepath).exists()
            df = pd.read_csv(filepath)

            # Verify it's the transform subset (50 samples)
            assert len(df) == 50

    def test_missing_fixtures_raises_error(self, dummy_plink_files, temp_dir):
        """Test that missing fixtures raise helpful error."""
        # Create empty fixtures directory
        empty_fixtures = temp_dir / "empty_fixtures"
        empty_fixtures.mkdir()

        backend = PrecomputedAdmixtureBackend(k_min=2, k_max=3, fixtures_dir=str(empty_fixtures))

        # Should raise error when trying to load missing fixtures
        with pytest.raises(FileNotFoundError):
            backend.fit_transform(dummy_plink_files, str(temp_dir / "out"))


# Optionally test NeuralAdmixtureBackend if marked as slow
# These tests are skipped by default unless --run-slow is passed
@pytest.mark.slow
@pytest.mark.skipif(
    not shutil.which("neural-admixture"),
    reason="neural-admixture binary not in PATH — install manifold-genetics[admixture]",
)
class TestNeuralAdmixtureBackend:
    """Tests for NeuralAdmixtureBackend (real computation).

    These tests are marked as @pytest.mark.slow and are skipped by default.
    Run with: pytest -m slow
    """

    def test_interface_compliance(self):
        """Verify NeuralAdmixtureBackend implements AdmixtureBackend interface."""
        from manifold_genetics.admixture.backends import NeuralAdmixtureBackend

        backend = NeuralAdmixtureBackend(k_min=2, k_max=3)
        assert isinstance(backend, AdmixtureBackend)
        assert hasattr(backend, "fit")
        assert hasattr(backend, "transform")
        assert hasattr(backend, "fit_transform")

    # Additional slow tests would go here
    # These would require actual PLINK data and run neural-admixture


class TestNeuralAdmixtureBackendLazyResolution:
    """The backend must be constructible without the neural-admixture binary.

    Pipelines that inject a different backend, and tests that mock _train/_infer,
    need to build NeuralAdmixture (which constructs a NeuralAdmixtureBackend by
    default) with only the core install - no `admixture` extra, no binary.
    """

    def test_construction_does_not_resolve_binary(self, monkeypatch):
        from manifold_genetics.admixture.backends import NeuralAdmixtureBackend
        from manifold_genetics.utils import tools

        def _boom(self):
            raise tools.ToolNotFoundError("neural-admixture not found")

        monkeypatch.setattr(tools.ToolResolver, "resolve_neural_admixture", _boom)

        backend = NeuralAdmixtureBackend(k_min=2, k_max=3, num_gpus=0)
        assert backend._nadm_exec is None

        # Resolution (and its failure) happens only on first real use.
        with pytest.raises(tools.ToolNotFoundError):
            _ = backend.nadm_exec

    def test_explicit_path_bypasses_resolver(self, monkeypatch):
        from manifold_genetics.admixture.backends import NeuralAdmixtureBackend
        from manifold_genetics.utils import tools

        def _boom(self):
            raise AssertionError("resolver must not be called when a path is given")

        monkeypatch.setattr(tools.ToolResolver, "resolve_neural_admixture", _boom)

        backend = NeuralAdmixtureBackend(
            k_min=2, k_max=3, neural_admixture_path="/usr/bin/true", num_gpus=0
        )
        assert backend.nadm_exec == "/usr/bin/true"
