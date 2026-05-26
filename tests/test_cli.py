"""CLI contract tests (PCA, admixture, embedding) using lightweight fakes."""

import argparse
from pathlib import Path

import pandas as pd

from manifold_genetics import cli as mg_cli


def test_cli_pca_fit_project(monkeypatch, tmp_path):
    """Ensure PCA CLI supports fit + project prefixes and writes outputs."""
    calls = []

    class FakePCA:
        def __init__(self, n_components, force):
            self.n_components = n_components
            self.force = force

        def fit(self, prefix, output_dir=None):
            calls.append(("fit", prefix, output_dir))
            return self

        def project(self, prefix, output_path=None):
            calls.append(("project", prefix, output_path))
            df = pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1]})
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
            return df

        def fit_transform(self, *args, **kwargs):
            raise AssertionError("fit_transform should not be called in fit+project mode")

    monkeypatch.setattr(mg_cli, "PCA", FakePCA)

    fit_out = tmp_path / "fit.csv"
    proj_out = tmp_path / "proj.csv"
    args = argparse.Namespace(
        fit_plink="fit_prefix",
        project_plink="project_prefix",
        fit_output=str(fit_out),
        project_output=str(proj_out),
        input=None,
        output=None,
        n_pcs=2,
        force=False,
        model_dir=None,
        flashpca_output_dir=None,  # Added missing attribute
        verbose=False,
    )

    mg_cli.cmd_pca(args)

    assert fit_out.exists()
    assert proj_out.exists()
    assert calls[:2] == [("fit", "fit_prefix", None), ("project", "fit_prefix", str(fit_out))]
    assert calls[2] == ("project", "project_prefix", str(proj_out))


def test_cli_admixture_fit_project(monkeypatch, tmp_path):
    """Ensure admixture CLI fits on one prefix and projects another."""
    calls = []

    class FakeAdmix:
        def __init__(self, k_min, k_max, force, threads, num_gpus, batch_size=None):
            calls.append(("init", k_min, k_max, force, threads, num_gpus, batch_size))

        def fit(self, prefix, output_dir=None, model_name=None):
            calls.append(("fit", prefix, output_dir, model_name))

        def transform(self, prefix, output_prefix=None):
            calls.append(("transform", prefix, output_prefix))
            output_prefix = Path(output_prefix)
            output_prefix.parent.mkdir(parents=True, exist_ok=True)
            paths = {}
            for k in (2, 3):
                csv = Path(f"{output_prefix}.{k}.csv")
                pd.DataFrame({"sample_id": ["s1"], f"component_{k}": [0.5]}).to_csv(
                    csv, index=False
                )
                paths[k] = csv
            return paths

        def fit_transform(self, *args, **kwargs):
            raise AssertionError("fit_transform should not be called in fit+project mode")

    monkeypatch.setattr(mg_cli, "NeuralAdmixture", FakeAdmix)

    out_dir = tmp_path / "admix"
    args = argparse.Namespace(
        input=None,
        fit_plink="fit_prefix",
        project_plink="project_prefix",
        output=str(out_dir),
        model_name="fit",
        fit_output=str(out_dir / "fit"),
        project_output=str(out_dir / "transform"),
        k_min=2,
        k_max=3,
        force=False,
        threads=None,
        num_gpus=None,
        neuraladmixture_output_dir=None,  # Added missing attribute
        neuraladmixture_batch_size=None,  # Added missing attribute
        verbose=False,
    )

    mg_cli.cmd_admixture(args)

    expected_fit = out_dir / "fit.2.csv"
    expected_proj = out_dir / "transform.3.csv"
    assert expected_fit.exists()
    assert expected_proj.exists()
    # Verify call order: init -> fit -> transform fit -> transform project
    assert calls[0][0] == "init"
    assert ("fit", "fit_prefix", out_dir, "fit") in calls
    assert ("transform", "fit_prefix", out_dir / "fit") in calls
    assert ("transform", "project_prefix", out_dir / "transform") in calls


def test_cli_embed_fit_project(monkeypatch, tmp_path):
    """Ensure embed CLI fits on one input and projects another, writing outputs."""
    calls = []

    class FakeEmbed:
        def __init__(self, n_components=2, **kwargs):
            calls.append(("init", kwargs))

        def fit(self, X):
            calls.append(("fit", X))
            return self

        def transform(self, X):
            calls.append(("transform", X))
            return pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1], "dim_2": [0.2]})

        def fit_transform(self, *args, **kwargs):
            raise AssertionError("fit_transform should not be called in fit+project mode")

    # Stub all embedding models to the same fake
    monkeypatch.setattr(mg_cli, "PHATE", FakeEmbed)
    monkeypatch.setattr(mg_cli, "UMAP", FakeEmbed)
    monkeypatch.setattr(mg_cli, "TSNE", FakeEmbed)
    monkeypatch.setattr(mg_cli, "DiffusionMap", FakeEmbed)

    fit_input = tmp_path / "fit.csv"
    proj_input = tmp_path / "proj.csv"
    pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1]}).to_csv(fit_input, index=False)
    pd.DataFrame({"sample_id": ["s2"], "dim_1": [0.2]}).to_csv(proj_input, index=False)

    proj_out = tmp_path / "out.csv"
    args = argparse.Namespace(
        input=None,
        fit_input=str(fit_input),
        project_input=str(proj_input),
        fit_output=None,
        project_output=str(proj_out),
        output=None,
        method="phate",
        knn=5,
        t=2,
        n_neighbors=15,
        min_dist=0.1,
        perplexity=30.0,
        n_landmark=None,  # Added missing attribute
        random_landmarking=False,  # Added missing attribute
        verbose=False,
    )

    mg_cli.cmd_embed(args)

    assert proj_out.exists()
    assert ("fit", str(fit_input)) in calls
    assert ("transform", str(proj_input)) in calls
