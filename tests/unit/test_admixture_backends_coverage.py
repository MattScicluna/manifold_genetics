"""Branch coverage for the three admixture backends.

Interface / happy-path behaviour is in test_admixture_backends.py and the
subprocess-command construction in test_subprocess_wrappers.py. These cover
the resolver branches, the checkpoint-skip short-circuits, and the
fixture/validation error paths.
"""

from pathlib import Path

import pytest

from manifold_genetics.admixture.backends import (
    FakeAdmixtureBackend,
    NeuralAdmixtureBackend,
    PrecomputedAdmixtureBackend,
)
from manifold_genetics.admixture.backends import neural as neural_mod

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "admixture"


# ---------------------------------------------------------------------------
# FakeAdmixtureBackend
# ---------------------------------------------------------------------------


def test_fake_backend_skips_when_outputs_present(dummy_plink_files, temp_dir):
    backend = FakeAdmixtureBackend(k_min=2, k_max=3, random_seed=1)
    prefix = temp_dir / "out"
    for k in (2, 3):
        Path(f"{prefix}.{k}.csv").write_text("sentinel")

    result = backend.transform(dummy_plink_files, str(prefix))
    assert set(result) == {2, 3}
    # untouched — the skip branch returned the existing paths
    assert Path(f"{prefix}.2.csv").read_text() == "sentinel"


# ---------------------------------------------------------------------------
# PrecomputedAdmixtureBackend
# ---------------------------------------------------------------------------


def test_precomputed_missing_fixtures_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Fixtures directory not found"):
        PrecomputedAdmixtureBackend(k_min=2, k_max=3, fixtures_dir=tmp_path / "nope")


def test_precomputed_fixture_file_missing_raises(fit_plink_files, temp_dir):
    # 'fit' fixtures exist for K=2,3 but not K=4
    backend = PrecomputedAdmixtureBackend(k_min=4, k_max=4, fixtures_dir=FIXTURES)
    with pytest.raises(FileNotFoundError, match="Fixture not found"):
        backend.transform(fit_plink_files, str(temp_dir / "out"))


def test_precomputed_transform_skips_when_outputs_present(fit_plink_files, temp_dir):
    backend = PrecomputedAdmixtureBackend(k_min=2, k_max=3, fixtures_dir=FIXTURES)
    prefix = temp_dir / "out"
    for k in (2, 3):
        Path(f"{prefix}.{k}.csv").write_text("sentinel")
    result = backend.transform(fit_plink_files, str(prefix))
    assert set(result) == {2, 3}
    assert Path(f"{prefix}.2.csv").read_text() == "sentinel"


def test_precomputed_fit_transform_skips_when_outputs_present(fit_plink_files, temp_dir):
    backend = PrecomputedAdmixtureBackend(k_min=2, k_max=3, fixtures_dir=FIXTURES)
    prefix = temp_dir / "ft"
    for k in (2, 3):
        Path(f"{prefix}.{k}.csv").write_text("sentinel")
    result = backend.fit_transform(fit_plink_files, str(prefix))
    assert Path(f"{prefix}.3.csv").read_text() == "sentinel"
    assert set(result) == {2, 3}


# ---------------------------------------------------------------------------
# NeuralAdmixtureBackend — resolver branches
# ---------------------------------------------------------------------------


def _backend(**kw):
    kw.setdefault("neural_admixture_path", "/fake/neural-admixture")
    kw.setdefault("num_gpus", 0)  # skip the torch.cuda probe unless a test opts in
    return NeuralAdmixtureBackend(k_min=2, k_max=3, **kw)


def test_resolve_num_gpus_explicit_values():
    assert _backend(num_gpus=3).num_gpus == 3
    assert _backend(num_gpus=-5).num_gpus == 0


@pytest.mark.parametrize("cuda, expected", [(True, 1), (False, 0)])
def test_resolve_num_gpus_auto_uses_cuda(monkeypatch, cuda, expected):
    monkeypatch.setattr(neural_mod, "_TORCH_AVAILABLE", True, raising=False)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return cuda

    monkeypatch.setattr(neural_mod, "torch", type("T", (), {"cuda": _FakeCuda}), raising=False)
    assert _backend(num_gpus=None).num_gpus == expected


def test_resolve_num_gpus_no_torch(monkeypatch):
    monkeypatch.setattr(neural_mod, "_TORCH_AVAILABLE", False, raising=False)
    assert _backend(num_gpus=None).num_gpus == 0


def test_resolve_threads_explicit_and_slurm(monkeypatch):
    assert _backend(threads=6).threads == 6
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    assert _backend(threads=None).threads == 7
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-an-int")
    # falls through to affinity / cpu_count, which is a positive int
    assert _backend(threads=None).threads >= 1


# ---------------------------------------------------------------------------
# NeuralAdmixtureBackend — fit / transform orchestration (internals mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def no_validate(monkeypatch):
    monkeypatch.setattr(neural_mod, "validate_plink_files", lambda p: Path(p))


def test_neural_fit_calls_train_and_sets_state(no_validate, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(NeuralAdmixtureBackend, "_train", lambda self, *a, **k: calls.append(a))
    b = _backend()
    b.fit("ref", tmp_path / "models", model_name="fit")
    assert b._model_dir == tmp_path / "models"
    assert b._model_name == "fit"
    assert calls  # _train was invoked


def test_neural_transform_requires_fit(no_validate, tmp_path):
    with pytest.raises(RuntimeError, match="fit\\(\\) must be called before transform"):
        _backend().transform("target", tmp_path / "out")


def test_neural_transform_skips_when_csv_present(no_validate, monkeypatch, tmp_path):
    monkeypatch.setattr(
        NeuralAdmixtureBackend,
        "_infer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not infer")),
    )
    b = _backend()
    b._model_dir = tmp_path
    prefix = tmp_path / "out"
    for k in (2, 3):
        Path(f"{prefix}.{k}.csv").write_text("x")
    result = b.transform("target", str(prefix))
    assert set(result) == {2, 3}


def test_neural_transform_infers_and_cleans_up_raw_q(no_validate, monkeypatch, tmp_path):
    raw = {k: tmp_path / f"raw.{k}.Q" for k in (2, 3)}
    for p in raw.values():
        p.write_text("0.5")
    csvs = {k: tmp_path / f"out.{k}.csv" for k in (2, 3)}

    monkeypatch.setattr(NeuralAdmixtureBackend, "_infer", lambda self, *a, **k: raw)
    monkeypatch.setattr(
        NeuralAdmixtureBackend, "_convert_q_files_to_csv", lambda self, *a, **k: csvs
    )
    b = _backend()
    b._model_dir = tmp_path
    result = b.transform("target", str(tmp_path / "out"))
    assert result == csvs
    assert not any(p.exists() for p in raw.values())  # raw Q files removed
