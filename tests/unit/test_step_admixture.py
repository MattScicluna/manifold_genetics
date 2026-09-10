"""Tests for the in-process admixture step (pipeline/steps/admixture.py).

This step replaces BOTH a `subprocess.run(["manifold-genetics", "admixture", ...])`
call and a special-case branch that called an injected backend directly. The two
were not equivalent — the backend branch used fit_transform() for the fit cohort
where the CLI path used transform(). These tests pin the unified sequence:
fit() once, then transform() on each cohort.
"""

from pathlib import Path

import pytest

from manifold_genetics.pipeline.config import AdmixtureConfig, IOConfig
from manifold_genetics.pipeline.steps.admixture import (
    AdmixtureStepResult,
    run_admixture,
    run_admixture_step,
)
from manifold_genetics.pipeline.steps.paths import admixture_output_paths

MODULE = "manifold_genetics.pipeline.steps.admixture"


class FakeBackend:
    """Records the call sequence an injected backend observes."""

    def __init__(self):
        self.calls = []

    def fit(self, plink_prefix, output_dir, model_name="fit"):
        self.calls.append(("fit", str(plink_prefix), str(output_dir), model_name))

    def transform(self, plink_prefix, output_prefix):
        self.calls.append(("transform", str(plink_prefix), str(output_prefix)))
        return {2: Path(f"{output_prefix}.2.csv")}

    def fit_transform(self, plink_prefix, output_prefix):
        self.calls.append(("fit_transform", str(plink_prefix), str(output_prefix)))
        return {2: Path(f"{output_prefix}.2.csv")}


class FakeNeuralAdmixture:
    """Stand-in for the NeuralAdmixture wrapper. Records construction kwargs."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeNeuralAdmixture.instances.append(self)

    def fit(self, plink_prefix, output_dir=None, model_name="fit"):
        self.calls.append(("fit", str(plink_prefix), str(output_dir), model_name))

    def transform(self, plink_prefix, output_prefix=None):
        self.calls.append(("transform", str(plink_prefix), str(output_prefix)))
        return {2: Path(f"{output_prefix}.2.csv")}

    def fit_transform(self, plink_prefix, output_prefix=None):
        raise AssertionError("the admixture step must not call fit_transform")


@pytest.fixture
def fake_admixture(monkeypatch):
    FakeNeuralAdmixture.instances = []
    monkeypatch.setattr(f"{MODULE}.NeuralAdmixture", FakeNeuralAdmixture)
    return FakeNeuralAdmixture


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
# run_admixture — the seam shared with `manifold-genetics admixture`
# ---------------------------------------------------------------------------


class TestRunAdmixture:
    def test_fits_once_then_transforms_each_cohort(self, tmp_path, fake_admixture):
        """The unified sequence. Using fit_transform for the fit cohort — as the
        old backend branch did — would retrain on a real backend."""
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            k_min=2,
            k_max=3,
        )

        (inst,) = fake_admixture.instances
        assert [c[0] for c in inst.calls] == ["fit", "transform", "transform"]
        assert inst.calls[0][1] == "fitset"
        assert inst.calls[1][1] == "fitset"
        assert inst.calls[2][1] == "projectset"

    def test_returns_both_q_file_maps(self, tmp_path, fake_admixture):
        fit_q, project_q = run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
        )
        assert fit_q == {2: Path(f"{tmp_path / 'fit'}.2.csv")}
        assert project_q == {2: Path(f"{tmp_path / 'project'}.2.csv")}

    def test_creates_checkpoints_and_output_directories(self, tmp_path, fake_admixture):
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "deep" / "ckpt",
            fit_output=tmp_path / "nested" / "fit",
            project_output=tmp_path / "nested" / "project",
        )
        assert (tmp_path / "deep" / "ckpt").is_dir()
        assert (tmp_path / "nested").is_dir()

    def test_construction_kwargs_forwarded(self, tmp_path, fake_admixture):
        """Constraint E: detect_cluster.sh feeds threads/gpus/batch size here."""
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            k_min=3,
            k_max=7,
            force=True,
            threads=8,
            num_gpus=2,
            batch_size=256,
        )
        k = fake_admixture.instances[0].kwargs
        assert k["k_min"] == 3
        assert k["k_max"] == 7
        assert k["force"] is True
        assert k["threads"] == 8
        assert k["num_gpus"] == 2
        assert k["batch_size"] == 256

    def test_zero_gpus_is_forwarded_not_dropped(self, tmp_path, fake_admixture):
        """num_gpus=0 explicitly requests CPU-only. The old argv builder handled
        this correctly; passing values directly must not regress it."""
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            num_gpus=0,
        )
        assert fake_admixture.instances[0].kwargs["num_gpus"] == 0

    def test_zero_threads_is_forwarded_not_dropped(self, tmp_path, fake_admixture):
        """The old argv builder used `if admix_threads:` and silently dropped 0."""
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            threads=0,
        )
        assert fake_admixture.instances[0].kwargs["threads"] == 0

    def test_model_name_defaults_to_fit_and_is_overridable(self, tmp_path, fake_admixture):
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
        )
        assert fake_admixture.instances[0].calls[0][3] == "fit"

        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            model_name="custom",
        )
        assert fake_admixture.instances[1].calls[0][3] == "custom"

    def test_injected_backend_is_passed_to_the_wrapper(self, tmp_path, fake_admixture):
        """The unification point: an injected backend goes through the SAME
        NeuralAdmixture wrapper the real path uses, not a separate branch."""
        backend = FakeBackend()
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            backend=backend,
        )
        assert fake_admixture.instances[0].kwargs["backend"] is backend


# ---------------------------------------------------------------------------
# run_admixture_step — config in, typed result out
# ---------------------------------------------------------------------------


class TestRunAdmixtureStep:
    def test_writes_to_the_documented_layout(self, tmp_path, fake_admixture):
        """Spec constraint B: downstream scripts read these exact paths."""
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=2, k_max=3)

        run_admixture_step(io, cfg)

        paths = admixture_output_paths(io, cfg)
        assert paths["dir"].is_dir()
        assert paths["checkpoints_dir"].is_dir()
        (inst,) = fake_admixture.instances
        assert inst.calls[1][2] == str(paths["fit_prefix"])
        assert inst.calls[2][2] == str(paths["project_prefix"])

    def test_result_carries_the_documented_fields(self, tmp_path, fake_admixture):
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=2, k_max=4)

        result = run_admixture_step(io, cfg)

        paths = admixture_output_paths(io, cfg)
        assert isinstance(result, AdmixtureStepResult)
        assert result.q_prefix == paths["project_prefix"]
        assert result.fit_prefix == paths["fit_prefix"]
        assert result.dir == paths["dir"]
        assert result.checkpoints_dir == paths["checkpoints_dir"]
        assert result.k_values == (2, 3, 4)
        assert result.fit_q_files == paths["fit_q_files"]
        assert result.project_q_files == paths["project_q_files"]
        assert result.skipped is False

    def test_fits_on_fit_cohort_and_transforms_both(self, tmp_path, fake_admixture):
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=2, k_max=3)

        run_admixture_step(io, cfg)

        (inst,) = fake_admixture.instances
        assert [c[0] for c in inst.calls] == ["fit", "transform", "transform"]
        assert inst.calls[0][1] == "data/fit"
        assert inst.calls[1][1] == "data/fit"
        assert inst.calls[2][1] == "data/project"

    def test_config_values_reach_the_wrapper(self, tmp_path, fake_admixture):
        io = make_io(tmp_path)
        cfg = AdmixtureConfig(k_min=2, k_max=5, threads=4, num_gpus=0, batch_size=400)

        run_admixture_step(io, cfg)

        k = fake_admixture.instances[0].kwargs
        assert k["k_min"] == 2
        assert k["k_max"] == 5
        assert k["threads"] == 4
        assert k["num_gpus"] == 0
        assert k["batch_size"] == 400

    def test_injected_backend_takes_the_same_path(self, tmp_path, fake_admixture):
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=2, k_max=3)
        backend = FakeBackend()

        run_admixture_step(io, cfg, backend=backend)

        assert fake_admixture.instances[0].kwargs["backend"] is backend

    def test_q_file_maps_track_the_k_range(self, tmp_path, fake_admixture):
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=3, k_max=5)

        result = run_admixture_step(io, cfg)

        assert sorted(result.project_q_files) == [3, 4, 5]
        assert result.project_q_files[4].name == "project.4.csv"
