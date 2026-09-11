"""Tests for the in-process PCA step (pipeline/steps/pca.py).

The step replaces a `subprocess.run(["manifold-genetics", "pca", ...])` call, so
these tests assert *behaviour* — which PLINK prefix is fitted, which files are
written, when a stale cache forces recomputation — rather than an argv list.

The real `PCA` class shells out to the flashpca binary; it is faked here.
"""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.config import IOConfig, PCAConfig
from manifold_genetics.pipeline.steps.paths import pca_output_paths
from manifold_genetics.pipeline.steps.pca import PCAStepResult, run_pca, run_pca_step


def coords_frame(n_dims: int, n_rows: int = 3) -> pd.DataFrame:
    cols = {"sample_id": [f"s{i}" for i in range(n_rows)]}
    cols.update({f"dim_{i}": [float(i)] * n_rows for i in range(1, n_dims + 1)})
    return pd.DataFrame(cols)


def write_pca_csv(path: Path, n_dims: int, n_rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coords_frame(n_dims, n_rows).to_csv(path, index=False)


class FakePCA:
    """Stand-in for manifold_genetics.pca.PCA. Records calls, writes real CSVs."""

    instances = []

    def __init__(self, n_components, force=False, backend="flashpca"):
        self.n_components = n_components
        self.force = force
        self.backend = backend
        self.calls = []
        FakePCA.instances.append(self)

    def fit(self, plink_prefix, output_dir=None):
        self.calls.append(("fit", str(plink_prefix), str(output_dir)))

    def project(self, plink_prefix, output_path=None):
        self.calls.append(("project", str(plink_prefix), str(output_path)))
        df = coords_frame(self.n_components)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        return df

    def fit_transform(self, plink_prefix, output_path=None):
        self.calls.append(("fit_transform", str(plink_prefix), str(output_path)))
        df = coords_frame(self.n_components)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        return df


@pytest.fixture
def fake_pca(monkeypatch):
    FakePCA.instances = []
    monkeypatch.setattr("manifold_genetics.pipeline.steps.pca.PCA", FakePCA)
    return FakePCA


def make_io(tmp_path) -> IOConfig:
    return IOConfig(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=tmp_path / "out",
        fit_labels=Path("fl.csv"),
        project_labels=Path("pl.csv"),
        fit_colormap=Path("fc.json"),
        project_colormap=Path("pc.json"),
    )


# ---------------------------------------------------------------------------
# run_pca — the seam shared with `manifold-genetics pca`
# ---------------------------------------------------------------------------


class TestRunPca:
    """run_pca is called by both cmd_pca and run_pca_step. If the two ever
    disagree about which dataset is fitted vs. projected, cross-cohort results
    are silently wrong — that is the drift this seam exists to prevent."""

    def test_fits_on_fit_prefix_and_projects_project_prefix(self, tmp_path, fake_pca):
        out = tmp_path / "project.csv"
        run_pca("fitset", "projectset", project_output=out, n_pcs=4)

        (inst,) = fake_pca.instances
        assert inst.calls[0] == ("fit", "fitset", "None")
        # No --fit-output requested, so the only projection is the project set.
        assert [c[:2] for c in inst.calls] == [("fit", "fitset"), ("project", "projectset")]

    def test_writes_fit_output_when_requested(self, tmp_path, fake_pca):
        fit_out = tmp_path / "fit.csv"
        proj_out = tmp_path / "project.csv"
        run_pca("fitset", "projectset", fit_output=fit_out, project_output=proj_out, n_pcs=4)

        assert fit_out.exists(), "--fit-output equivalent must be written"
        assert proj_out.exists()
        (inst,) = fake_pca.instances
        assert [c[:2] for c in inst.calls] == [
            ("fit", "fitset"),
            ("project", "fitset"),
            ("project", "projectset"),
        ]

    def test_single_dataset_uses_fit_transform(self, tmp_path, fake_pca):
        """project_plink=None is the `manifold-genetics pca --input X` shape."""
        out = tmp_path / "all.csv"
        run_pca("allsamples", None, project_output=out, n_pcs=4)

        (inst,) = fake_pca.instances
        assert [c[0] for c in inst.calls] == ["fit_transform"]
        assert out.exists()

    def test_flashpca_dir_forwarded_to_fit(self, tmp_path, fake_pca):
        d = tmp_path / "flashpca_outputs"
        run_pca("fitset", "projectset", project_output=tmp_path / "p.csv", flashpca_dir=d)

        (inst,) = fake_pca.instances
        assert inst.calls[0] == ("fit", "fitset", str(d))

    def test_force_forwarded_to_pca_constructor(self, tmp_path, fake_pca):
        run_pca("fitset", "projectset", project_output=tmp_path / "p.csv", force=True)
        assert fake_pca.instances[0].force is True

    def test_returns_project_coordinates(self, tmp_path, fake_pca):
        df = run_pca("fitset", "projectset", project_output=tmp_path / "p.csv", n_pcs=3)
        assert list(df.columns) == ["sample_id", "dim_1", "dim_2", "dim_3"]


# ---------------------------------------------------------------------------
# run_pca_step — config in, typed result out
# ---------------------------------------------------------------------------


class TestRunPcaStep:
    def test_writes_the_documented_output_layout(self, tmp_path, fake_pca):
        """Spec constraint B: downstream example scripts read these exact paths."""
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)

        run_pca_step(io, cfg)

        paths = pca_output_paths(io, cfg)
        assert paths["fit_pca"].exists()
        assert paths["project_pca"].exists()
        assert paths["flashpca_dir"].is_dir()

    def test_result_carries_paths_and_coordinates(self, tmp_path, fake_pca):
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)

        result = run_pca_step(io, cfg)

        paths = pca_output_paths(io, cfg)
        assert isinstance(result, PCAStepResult)
        assert result.fit_pca == paths["fit_pca"]
        assert result.project_pca == paths["project_pca"]
        assert result.skipped is False
        assert isinstance(result.coords_df, pd.DataFrame)
        assert len(result.coords_df.columns) == 6  # sample_id + 5 dims

    def test_fits_on_fit_plink_and_projects_project_plink(self, tmp_path, fake_pca):
        io = make_io(tmp_path)
        run_pca_step(io, PCAConfig(n_pcs=5))

        (inst,) = fake_pca.instances
        assert [c[:2] for c in inst.calls] == [
            ("fit", "data/fit"),
            ("project", "data/fit"),
            ("project", "data/project"),
        ]

    def test_fresh_run_does_not_force(self, tmp_path, fake_pca):
        """Forcing on every run would recompute expensive flashpca needlessly."""
        run_pca_step(make_io(tmp_path), PCAConfig(n_pcs=5))
        assert fake_pca.instances[0].force is False

    def test_matching_cached_dim_count_does_not_force(self, tmp_path, fake_pca):
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)
        write_pca_csv(pca_output_paths(io, cfg)["project_pca"], n_dims=5)

        run_pca_step(io, cfg)
        assert fake_pca.instances[0].force is False

    def test_mismatched_cached_dim_count_forces_recompute(self, tmp_path, fake_pca):
        """A cached 10-PC file with n_pcs=50 requested must not be silently reused —
        stale dimensionality would propagate into every downstream embedding."""
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=50)
        write_pca_csv(pca_output_paths(io, cfg)["project_pca"], n_dims=10)

        run_pca_step(io, cfg)
        assert fake_pca.instances[0].force is True

    def test_unreadable_cached_file_forces_recompute(self, tmp_path, fake_pca):
        """An empty/corrupt cache file must trigger recomputation, not a crash."""
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)
        project_pca = pca_output_paths(io, cfg)["project_pca"]
        project_pca.parent.mkdir(parents=True, exist_ok=True)
        project_pca.write_text("")  # pandas raises EmptyDataError

        run_pca_step(io, cfg)
        assert fake_pca.instances[0].force is True

    def test_config_force_wins_over_matching_cache(self, tmp_path, fake_pca):
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5, force=True)
        write_pca_csv(pca_output_paths(io, cfg)["project_pca"], n_dims=5)

        run_pca_step(io, cfg)
        assert fake_pca.instances[0].force is True

    def test_second_call_is_idempotent(self, tmp_path, fake_pca):
        """Checkpointing: re-running the step on valid outputs re-uses them and
        does not force. PCA(force=False) skips the flashpca work internally."""
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)

        run_pca_step(io, cfg)
        second = run_pca_step(io, cfg)

        assert fake_pca.instances[1].force is False
        assert second.project_pca.exists()
        assert second.project_pca == pca_output_paths(io, cfg)["project_pca"]
