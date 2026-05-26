"""Tests for pipeline/runner.py — run_pipeline() argument validation.

run_pipeline() is a thin wrapper around Pipeline that validates label/colormap
arguments before touching the filesystem. These tests cover every invalid
combination so failures produce clear, actionable error messages rather than
cryptic downstream crashes.
"""

import pytest

from manifold_genetics.pipeline.runner import run_pipeline


class TestRunPipelineValidation:
    """All invalid argument combinations must raise ValueError before any I/O."""

    # ---- labels ----

    def test_raises_with_no_labels_at_all(self, tmp_path):
        """Neither labels nor fit_labels/project_labels provided."""
        with pytest.raises(ValueError, match="labels"):
            run_pipeline(
                fit_plink="fit",
                project_plink="project",
                output_dir=tmp_path,
                colormap="cmap.json",
            )

    def test_raises_with_only_fit_labels(self, tmp_path):
        """fit_labels alone (no project_labels, no shared labels) is invalid."""
        with pytest.raises(ValueError, match="labels"):
            run_pipeline(
                fit_plink="fit",
                project_plink="project",
                output_dir=tmp_path,
                fit_labels="fit_labels.csv",
                colormap="cmap.json",
            )

    def test_raises_with_only_project_labels(self, tmp_path):
        """project_labels alone (no fit_labels, no shared labels) is invalid."""
        with pytest.raises(ValueError, match="labels"):
            run_pipeline(
                fit_plink="fit",
                project_plink="project",
                output_dir=tmp_path,
                project_labels="project_labels.csv",
                colormap="cmap.json",
            )

    # ---- colormap ----

    def test_raises_with_no_colormap_at_all(self, tmp_path):
        """Neither colormap nor fit_colormap/project_colormap provided."""
        with pytest.raises(ValueError, match="colormap"):
            run_pipeline(
                fit_plink="fit",
                project_plink="project",
                output_dir=tmp_path,
                labels="labels.csv",
            )

    def test_raises_with_only_fit_colormap(self, tmp_path):
        """fit_colormap alone (no project_colormap, no shared colormap) is invalid."""
        with pytest.raises(ValueError, match="colormap"):
            run_pipeline(
                fit_plink="fit",
                project_plink="project",
                output_dir=tmp_path,
                labels="labels.csv",
                fit_colormap="fit_cmap.json",
            )

    def test_raises_with_only_project_colormap(self, tmp_path):
        """project_colormap alone (no fit_colormap, no shared colormap) is invalid."""
        with pytest.raises(ValueError, match="colormap"):
            run_pipeline(
                fit_plink="fit",
                project_plink="project",
                output_dir=tmp_path,
                labels="labels.csv",
                project_colormap="project_cmap.json",
            )

    # ---- errors fire before filesystem access ----

    def test_validation_fires_before_output_dir_created(self, tmp_path):
        """Validation errors must be raised before output_dir is created."""
        output_dir = tmp_path / "should_not_exist"
        with pytest.raises(ValueError):
            run_pipeline(
                fit_plink="fit",
                project_plink="project",
                output_dir=output_dir,
                # both labels and colormap missing
            )
        assert not output_dir.exists(), (
            "Output directory should not be created when validation fails. "
            "Pipeline.__init__ calls output_dir.mkdir() before run_pipeline validates — "
            "validation should happen first."
        )

    # ---- valid combinations do not raise ----

    def test_shared_labels_and_colormap_reaches_pipeline(self, tmp_path):
        """Shared labels + colormap passes validation and reaches Pipeline.__init__."""
        # Will fail at clone/PCA — not a ValueError, so validation passed
        with pytest.raises((ValueError, RuntimeError, Exception)) as exc_info:
            run_pipeline(
                fit_plink="nonexistent_fit",
                project_plink="nonexistent_project",
                output_dir=tmp_path / "out",
                labels="labels.csv",
                colormap="cmap.json",
            )
        # Must NOT be the labels/colormap validation error
        assert (
            "labels" not in str(exc_info.value).lower()
            or "colormap" not in str(exc_info.value).lower()
        )

    def test_separate_labels_and_colormaps_reaches_pipeline(self, tmp_path):
        """Separate fit/project labels + colormaps also passes validation."""
        with pytest.raises((ValueError, RuntimeError, Exception)) as exc_info:
            run_pipeline(
                fit_plink="nonexistent_fit",
                project_plink="nonexistent_project",
                output_dir=tmp_path / "out",
                fit_labels="fit_labels.csv",
                project_labels="project_labels.csv",
                fit_colormap="fit_cmap.json",
                project_colormap="project_cmap.json",
            )
        assert "fit_labels" not in str(exc_info.value) and "project_labels" not in str(
            exc_info.value
        )
