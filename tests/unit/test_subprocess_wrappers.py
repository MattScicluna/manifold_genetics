"""Tests for the external-binary wrappers in pca/flashpca.py and
admixture/backends/neural.py.

`subprocess.run` is monkeypatched to a stub that records the command and
creates the output files the caller expects, so the full command-building,
checkpoint-skip, and error-handling logic runs without flashpca or
neural-admixture installed. No torch, no binaries, no real genotype data.
"""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.admixture.backends.neural import NeuralAdmixtureBackend
from manifold_genetics.pca.flashpca import PCA


class FakeCompleted:
    def __init__(self):
        self.stdout = ""
        self.stderr = ""
        self.returncode = 0


def make_runner(record, *, creates=(), rc=0, sigkill=False):
    """Build a fake subprocess.run: records argv, optionally creates files or fails."""

    def _run(cmd, **kwargs):
        record.append(list(cmd))
        for path in creates:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("stub\n")
        if sigkill:
            raise subprocess.CalledProcessError(-9, cmd, output="", stderr="killed")
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd, output="out", stderr="err")
        return FakeCompleted()

    return _run


# ---------------------------------------------------------------------------
# flashpca._run_flashpca_fit
# ---------------------------------------------------------------------------


def _pca():
    return PCA(n_components=3, flashpca_path="/fake/flashpca", force=True)


def test_flashpca_fit_builds_command_and_returns_outputs(monkeypatch, tmp_path):
    calls = []
    prefix = tmp_path / "out" / "fit"
    outs = [f"{prefix}.{s}" for s in ("PC", "loadings", "meansd", "eigenvec", "eigenval")]
    monkeypatch.setattr(subprocess, "run", make_runner(calls, creates=outs))

    result = _pca()._run_flashpca_fit(Path("data/ref"), prefix)

    assert calls[0][0] == "/fake/flashpca"
    assert "--bfile" in calls[0] and "data/ref" in calls[0]
    assert calls[0][calls[0].index("-d") + 1] == "3"
    assert set(result) == {"pc", "loadings", "meansd", "eigenvec", "eigenval"}


def test_flashpca_fit_skips_when_checkpoint_present(monkeypatch, tmp_path):
    calls = []
    prefix = tmp_path / "fit"
    for s in ("PC", "loadings", "meansd", "eigenvec", "eigenval"):
        (tmp_path / f"fit.{s}").write_text("x")
    monkeypatch.setattr(subprocess, "run", make_runner(calls))

    pca = PCA(n_components=3, flashpca_path="/fake/flashpca", force=False)
    pca._run_flashpca_fit(Path("data/ref"), prefix)
    assert calls == []  # binary never invoked


def test_flashpca_fit_wraps_subprocess_error(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", make_runner([], rc=1))
    with pytest.raises(RuntimeError, match="FlashPCA fit failed"):
        _pca()._run_flashpca_fit(Path("data/ref"), tmp_path / "fit")


def test_flashpca_fit_tolerates_missing_outputs(monkeypatch, tmp_path):
    # subprocess "succeeds" but creates nothing -> warn, still return dict
    monkeypatch.setattr(subprocess, "run", make_runner([]))
    result = _pca()._run_flashpca_fit(Path("data/ref"), tmp_path / "fit")
    assert set(result) == {"pc", "loadings", "meansd", "eigenvec", "eigenval"}


# ---------------------------------------------------------------------------
# flashpca._run_flashpca_project
# ---------------------------------------------------------------------------


def test_flashpca_project_builds_command(monkeypatch, tmp_path):
    calls = []
    prefix = tmp_path / "transform_ds"
    monkeypatch.setattr(subprocess, "run", make_runner(calls, creates=[f"{prefix}.PC"]))
    pca = _pca()
    pca._loadings_path = tmp_path / "fit.loadings"
    pca._meansd_path = tmp_path / "fit.meansd"

    pc = pca._run_flashpca_project(Path("data/target"), prefix)

    assert "--project" in calls[0]
    assert str(pca._loadings_path) in calls[0]
    assert pc == Path(f"{prefix}.PC")


def test_flashpca_project_skips_when_checkpoint_present(monkeypatch, tmp_path):
    calls = []
    prefix = tmp_path / "transform_ds"
    (tmp_path / "transform_ds.PC").write_text("x")
    monkeypatch.setattr(subprocess, "run", make_runner(calls))

    pca = PCA(n_components=3, flashpca_path="/fake/flashpca", force=False)
    pca._loadings_path = tmp_path / "l"
    pca._meansd_path = tmp_path / "m"
    pca._run_flashpca_project(Path("data/target"), prefix)
    assert calls == []


def test_flashpca_project_raises_if_output_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", make_runner([]))  # succeeds, no file
    pca = _pca()
    pca._loadings_path = tmp_path / "l"
    pca._meansd_path = tmp_path / "m"
    with pytest.raises(FileNotFoundError):
        pca._run_flashpca_project(Path("data/target"), tmp_path / "transform_ds")


def test_flashpca_project_wraps_subprocess_error(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", make_runner([], rc=2))
    pca = _pca()
    pca._loadings_path = tmp_path / "l"
    pca._meansd_path = tmp_path / "m"
    with pytest.raises(RuntimeError, match="FlashPCA project failed"):
        pca._run_flashpca_project(Path("data/target"), tmp_path / "transform_ds")


# ---------------------------------------------------------------------------
# flashpca._convert_pc_to_csv
# ---------------------------------------------------------------------------


def test_flashpca_convert_pc_to_csv(tmp_path):
    pc_file = tmp_path / "fit.PC"
    pc_file.write_text("FID IID PC1 PC2 PC3\n0 s1 0.1 0.2 0.3\n0 s2 0.4 0.5 0.6\n")
    df = _pca()._convert_pc_to_csv(pc_file, Path("data/ref"))
    assert list(df.columns) == ["sample_id", "dim_1", "dim_2", "dim_3"]
    assert list(df["sample_id"]) == ["s1", "s2"]


# ---------------------------------------------------------------------------
# neural backend: _train
# ---------------------------------------------------------------------------


def _backend(**kw):
    kw.setdefault("neural_admixture_path", "/fake/neural-admixture")
    kw.setdefault("num_gpus", 0)
    return NeuralAdmixtureBackend(k_min=2, k_max=3, **kw)


def test_neural_train_builds_command_per_k(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", make_runner(calls))
    _backend(force=True)._train(Path("data/ref"), tmp_path, model_name="fit")

    assert len(calls) == 2  # k=2 and k=3
    assert calls[0][:2] == ["/fake/neural-admixture", "train"]
    assert "--data_path" in calls[0]
    ks = {c[c.index("--k") + 1] for c in calls}
    assert ks == {"2", "3"}


def test_neural_train_skips_when_models_present(monkeypatch, tmp_path):
    calls = []
    for k in (2, 3):
        (tmp_path / f"fit_k{k}.pt").write_text("model")
    monkeypatch.setattr(subprocess, "run", make_runner(calls))
    _backend(force=False)._train(Path("data/ref"), tmp_path, model_name="fit")
    assert calls == []


def test_neural_train_passes_batch_size(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", make_runner(calls))
    _backend(force=True, batch_size=128)._train(Path("d/ref"), tmp_path, model_name="fit")
    assert "--batch_size" in calls[0]
    assert calls[0][calls[0].index("--batch_size") + 1] == "128"


def test_neural_train_reraises_subprocess_error(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", make_runner([], rc=1))
    with pytest.raises(subprocess.CalledProcessError):
        _backend(force=True)._train(Path("d/ref"), tmp_path, model_name="fit")


# ---------------------------------------------------------------------------
# neural backend: _infer
# ---------------------------------------------------------------------------


def test_neural_infer_requires_fit_first(tmp_path):
    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        _backend()._infer(Path("d/t"), tmp_path, "proj")


def test_neural_infer_builds_cpu_only_command(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", make_runner(calls))
    monkeypatch.setattr(
        "manifold_genetics.admixture.backends.neural.get_sample_ids_from_plink",
        lambda p: ["s1", "s2"],
    )
    b = _backend(force=True)
    b._model_dir = tmp_path
    b._model_name = "fit"
    b._infer(Path("d/target"), tmp_path / "q", "proj")

    assert len(calls) == 2
    assert calls[0][:2] == ["/fake/neural-admixture", "infer"]
    assert calls[0][calls[0].index("--num_gpus") + 1] == "0"


def test_neural_infer_skips_when_q_files_present(monkeypatch, tmp_path):
    calls = []
    qdir = tmp_path / "q"
    qdir.mkdir()
    for k in (2, 3):
        (qdir / f"proj.{k}.Q").write_text("0.5")
    monkeypatch.setattr(subprocess, "run", make_runner(calls))
    b = _backend(force=False)
    b._model_dir = tmp_path
    b._model_name = "fit"
    b._infer(Path("d/target"), qdir, "proj")
    assert calls == []


def test_neural_infer_reraises_on_sigkill(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", make_runner([], sigkill=True))
    monkeypatch.setattr(
        "manifold_genetics.admixture.backends.neural.get_sample_ids_from_plink",
        lambda p: ["s1"],
    )
    b = _backend(force=True)
    b._model_dir = tmp_path
    b._model_name = "fit"
    with pytest.raises(subprocess.CalledProcessError):
        b._infer(Path("d/target"), tmp_path / "q", "proj")


# ---------------------------------------------------------------------------
# neural backend: _infer_on_training_data / _convert_q_files_to_csv
# ---------------------------------------------------------------------------


def test_infer_on_training_data_collects_q_files(tmp_path):
    for k in (2, 3):
        (tmp_path / f"fit_k{k}.{k}.Q").write_text("0.5 0.5")
    result = _backend()._infer_on_training_data(tmp_path)
    assert set(result) == {2, 3}


def test_infer_on_training_data_raises_when_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        _backend()._infer_on_training_data(tmp_path)


def test_convert_q_files_to_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "manifold_genetics.admixture.backends.neural.get_sample_ids_from_plink",
        lambda p: ["s1", "s2"],
    )
    q2 = tmp_path / "fit.2.Q"
    q2.write_text("0.9 0.1\n0.2 0.8\n")
    csv_files = _backend()._convert_q_files_to_csv({2: q2}, Path("d/ref"), tmp_path / "out")
    df = pd.read_csv(csv_files[2])
    assert list(df.columns) == ["sample_id", "component_1", "component_2"]
    assert list(df["sample_id"].astype(str)) == ["s1", "s2"]
