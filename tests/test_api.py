"""Python API contract tests using lightweight fakes.

These tests avoid running external binaries (flashpca, neural-admixture) by
monkeypatching internal helpers, and check that fit/project flows produce the
expected outputs/paths.
"""

from pathlib import Path

import pandas as pd

from manifold_genetics.pca import PCA
from manifold_genetics.admixture import NeuralAdmixture


def test_python_api_pca_fit_and_project(monkeypatch, tmp_path):
    calls = []

    def fake_run_fit(self, plink_prefix, output_prefix):
        calls.append(("fit", Path(plink_prefix), Path(output_prefix)))
        # Return paths to satisfy downstream checks
        return {
            "pc": tmp_path / "fit.PC",
            "loadings": tmp_path / "fit.loadings",
            "meansd": tmp_path / "fit.meansd",
            "eigenvec": tmp_path / "fit.eigenvec",
            "eigenval": tmp_path / "fit.eigenval",
        }

    def fake_run_project(self, plink_prefix, output_prefix):
        calls.append(("project", Path(plink_prefix), Path(output_prefix)))
        pc = Path(output_prefix).with_suffix(".PC")
        pc.parent.mkdir(parents=True, exist_ok=True)
        pc.write_text("s1 0.1\n")
        return pc

    def fake_convert(self, pc_file, plink_prefix):
        return pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1], "dim_2": [0.2]})

    # Patch the heavy pieces
    monkeypatch.setattr(PCA, "_run_flashpca_fit", fake_run_fit, raising=False)
    monkeypatch.setattr(PCA, "_run_flashpca_project", fake_run_project, raising=False)
    monkeypatch.setattr(PCA, "_convert_pc_to_csv", fake_convert, raising=False)
    monkeypatch.setattr(
        "manifold_genetics.pca.flashpca.validate_plink_files",
        lambda p: Path(p),
    )

    pca = PCA(n_components=2, force=True)

    # Fit + project separately
    pca.fit("fit_prefix", output_dir=tmp_path)
    proj_out = tmp_path / "proj.csv"
    df_proj = pca.project("project_prefix", output_path=proj_out)

    # Fit+transform in one call
    both_out = tmp_path / "both.csv"
    df_both = pca.fit_transform("fit_prefix", output_path=both_out)

    assert proj_out.exists()
    assert both_out.exists()
    assert list(df_proj.columns)[:2] == ["sample_id", "dim_1"]
    assert list(df_both.columns)[:2] == ["sample_id", "dim_1"]
    assert any(call[0] == "fit" and call[1] == Path("fit_prefix") for call in calls)
    # Check that project was called with project_prefix (comparing just the stem/name)
    assert any(call[0] == "project" and Path(call[1]).name == "project_prefix" for call in calls)


def test_python_api_admixture_fit_and_project(monkeypatch, tmp_path):
    calls = []

    # Bypass filesystem checks and heavy binaries
    # Patch in multiple places since backend calls it too
    def fake_validate(prefix):
        return Path(prefix)

    monkeypatch.setattr(
        "manifold_genetics.utils.io.validate_plink_files",
        fake_validate,
    )
    monkeypatch.setattr(
        "manifold_genetics.admixture.backends.neural.validate_plink_files",
        fake_validate,
    )

    def fake_get_sample_ids(prefix):
        return ["s1", "s2"]

    monkeypatch.setattr(
        "manifold_genetics.utils.io.get_sample_ids_from_plink",
        fake_get_sample_ids,
    )
    monkeypatch.setattr(
        "manifold_genetics.admixture.backends.neural.get_sample_ids_from_plink",
        fake_get_sample_ids,
    )

    def fake_train(self, plink_prefix, output_dir, model_name):
        calls.append(("train", Path(plink_prefix), Path(output_dir), model_name))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def fake_infer(self, plink_prefix, output_dir, out_name):
        calls.append(("infer", Path(plink_prefix), Path(output_dir), out_name))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        q_files = {}
        for k in (2, 3):
            q = output_dir / f"{out_name}.{k}.Q"
            # Write k columns per line
            row = " ".join(["0.5"] * k)
            q.write_text(f"{row}\n{row}\n")
            q_files[k] = q
        return q_files

    def fake_infer_train(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        q_files = {}
        for k in (2, 3):
            q = output_dir / f"fit_k{k}.{k}.Q"
            row = " ".join(["0.5"] * k)
            q.write_text(f"{row}\n{row}\n")
            q_files[k] = q
        return q_files

    # Patch backend methods (NeuralAdmixture now uses backend)
    from manifold_genetics.admixture.backends import NeuralAdmixtureBackend

    monkeypatch.setattr(NeuralAdmixtureBackend, "_train", fake_train, raising=False)
    monkeypatch.setattr(NeuralAdmixtureBackend, "_infer", fake_infer, raising=False)
    monkeypatch.setattr(
        NeuralAdmixtureBackend, "_infer_on_training_data", fake_infer_train, raising=False
    )

    admix = NeuralAdmixture(k_min=2, k_max=3, force=True, threads=1, num_gpus=0)

    # Fit + infer same data
    fit_outputs = admix.fit_transform("fit_prefix", output_prefix=tmp_path / "fit")
    # Fit then project another dataset
    admix.fit("fit_prefix", output_dir=tmp_path, model_name="fit")
    proj_outputs = admix.transform("project_prefix", output_prefix=tmp_path / "proj")

    assert (tmp_path / "fit.2.csv").exists()
    assert (tmp_path / "proj.3.csv").exists()
    assert set(fit_outputs.keys()) == {2, 3}
    assert set(proj_outputs.keys()) == {2, 3}
    assert any(call[0] == "train" for call in calls)
    assert any(call[0] == "infer" and call[2] == Path(tmp_path) for call in calls)
