"""Tests for the optional embedding baselines (DREAMS, FloDR, autoencoder, popVAE)."""

import argparse
import inspect
import sys

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.cli import _embedding_params_from_args
from manifold_genetics.embeddings import DREAMS, Autoencoder, FloDR, PopVAE
from manifold_genetics.pipeline.steps.embedding import build_embedding_model


def _has_dreams_fork():
    try:
        import openTSNE
    except ImportError:
        return False
    return "reg_lambda" in inspect.signature(openTSNE.TSNE.__init__).parameters


@pytest.mark.parametrize(
    "method,cls",
    [("dreams", DREAMS), ("flodr", FloDR), ("autoencoder", Autoencoder), ("popvae", PopVAE)],
)
def test_build_embedding_model_knows_new_methods(method, cls):
    assert isinstance(build_embedding_model(method, {}), cls)


def test_build_embedding_model_passes_params_through():
    model = build_embedding_model("dreams", {"reg_lambda": 0.4, "n_components": 3})
    assert (model.reg_lambda, model.n_components) == (0.4, 3)
    with pytest.raises(TypeError):
        build_embedding_model("flodr", {"no_such_option": 1})


def test_cli_passes_only_flags_that_were_set():
    args = argparse.Namespace(
        method="dreams", perplexity=30, reg_lambda=None, seed=7, n_components=None
    )
    assert _embedding_params_from_args(args) == {"perplexity": 30, "random_state": 7}


def test_dreams_rejects_out_of_range_lambda():
    with pytest.raises(ValueError):
        DREAMS(reg_lambda=1.5)


@pytest.mark.skipif(not _has_dreams_fork(), reason="DREAMS openTSNE fork not installed")
def test_dreams_fit_transform(small_pca_data):
    out = DREAMS(perplexity=10, reg_lambda=0.5, n_jobs=1).fit_transform(small_pca_data)
    assert list(out.columns) == ["sample_id", "dim_1", "dim_2"]
    assert len(out) == len(small_pca_data)
    assert np.isfinite(out[["dim_1", "dim_2"]].values).all()


def test_flodr_fit_transform_and_project(small_pca_data):
    pytest.importorskip("flodr")
    model = FloDR(max_iter=50, device="cpu").fit(small_pca_data)
    fitted = model.transform(small_pca_data)
    assert fitted.shape == (len(small_pca_data), 3)
    assert np.isfinite(fitted[["dim_1", "dim_2"]].values).all()


def test_autoencoder_fit_transform_is_seeded(small_pca_data, temp_dir):
    pytest.importorskip("torch")
    a = Autoencoder(hidden=(16,), epochs=5, random_state=1).fit_transform(
        small_pca_data, output_path=temp_dir / "ae.csv"
    )
    b = Autoencoder(hidden=(16,), epochs=5, random_state=1).fit_transform(small_pca_data)
    assert list(a.columns) == ["sample_id", "dim_1", "dim_2"]
    assert (temp_dir / "ae.csv").exists()
    np.testing.assert_allclose(a[["dim_1", "dim_2"]].values, b[["dim_1", "dim_2"]].values)


def test_popvae_requires_genotypes(small_pca_data):
    with pytest.raises(ValueError, match="genotypes"):
        PopVAE(script="popvae.py").fit(small_pca_data)


def test_popvae_command_line():
    cmd = PopVAE(genotypes="g", python="py", script="popvae.py", random_state=3)._command(
        "in.zarr", "out"
    )
    assert cmd[:2] == ["py", "popvae.py"]
    assert cmd[cmd.index("--seed") + 1] == "3"
    assert cmd[cmd.index("--latent_dim") + 1] == "2"
    assert "--max_epochs" not in cmd  # popVAE's own default unless capped
    capped = PopVAE(script="popvae.py", max_epochs=100)._command("in.zarr", "out")
    assert capped[capped.index("--max_epochs") + 1] == "100"


def test_plink_to_popvae_zarr_layout(temp_dir):
    """Samples keep their order; allele counts become two calls; missing is -1."""
    bed_reader = pytest.importorskip("bed_reader")
    zarr = pytest.importorskip("zarr")
    from manifold_genetics.embeddings.popvae import plink_to_popvae_zarr

    G = np.array([[0, 1, 2], [2, np.nan, 0], [1, 1, 1]], dtype=float)
    prefix = temp_dir / "toy"
    bed_reader.to_bed(str(prefix) + ".bed", G, properties={"iid": ["a", "b", "c"]})
    out = plink_to_popvae_zarr(
        prefix, ["c", "a", "b"], temp_dir / "toy.zarr", python=sys.executable
    )
    root = zarr.open_group(str(out), mode="r")
    assert [str(s) for s in root["samples"][:]] == ["c", "a", "b"]
    gt = root["calldata/GT"][:]
    assert gt.shape == (3, 3, 2)
    assert (gt[:, 0, :].sum(axis=1) == [1, 1, 1]).all()  # sample c
    assert (gt[:, 1, :].sum(axis=1) == [0, 1, 2]).all()  # sample a
    assert (gt[1, 2] == [-1, -1]).all()  # sample b, SNP 2 missing


# --------------------------------------------------------------------------------------
# Error paths and projection
# --------------------------------------------------------------------------------------
def test_optional_import_names_the_install():
    from manifold_genetics.embeddings.base import optional_import

    with pytest.raises(ImportError, match="pip install something"):
        optional_import("no_such_module_for_manifold_genetics", "pip install something")


def test_dreams_refuses_upstream_opentsne(monkeypatch, small_pca_data):
    """The upstream openTSNE has no reg_lambda: say which package to install instead."""
    import types

    class UpstreamTSNE:
        def __init__(self, perplexity=30):
            pass

    monkeypatch.setitem(sys.modules, "openTSNE", types.SimpleNamespace(TSNE=UpstreamTSNE))
    with pytest.raises(ImportError, match="not the DREAMS fork"):
        DREAMS().fit(small_pca_data)


@pytest.mark.parametrize("cls", [DREAMS, FloDR, Autoencoder, PopVAE])
def test_transform_before_fit_raises(cls, small_pca_data):
    with pytest.raises(RuntimeError, match="fitted"):
        cls().transform(small_pca_data)


@pytest.mark.skipif(not _has_dreams_fork(), reason="DREAMS openTSNE fork not installed")
def test_dreams_projects_new_samples(small_pca_data):
    model = DREAMS(perplexity=5, reg_lambda=0.3, n_jobs=1).fit(small_pca_data.iloc[:40])
    same = model.transform(small_pca_data.iloc[:40])
    new = model.transform(small_pca_data.iloc[40:])
    assert same.shape == (40, 3)
    assert new.shape == (10, 3)
    assert new.sample_id.tolist() == small_pca_data.sample_id.iloc[40:].tolist()
    assert np.isfinite(new[["dim_1", "dim_2"]].values).all()
    assert model._embedding.gradient_descent_params["regularization"] is True  # restored


def test_flodr_fit_transform_writes_output_and_projects(small_pca_data, temp_dir):
    pytest.importorskip("flodr")
    out = temp_dir / "flodr.csv"
    model = FloDR(max_iter=50, device="cpu")
    fitted = model.fit_transform(small_pca_data.iloc[:40], output_path=out)
    assert out.exists() and fitted.shape == (40, 3)
    new = model.transform(small_pca_data.iloc[40:])
    assert new.sample_id.tolist() == small_pca_data.sample_id.iloc[40:].tolist()
    assert np.isfinite(new[["dim_1", "dim_2"]].values).all()


def test_autoencoder_projects_new_samples(small_pca_data):
    pytest.importorskip("torch")
    model = Autoencoder(hidden=(8,), epochs=3, random_state=0).fit(small_pca_data.iloc[:40])
    new = model.transform(small_pca_data.iloc[40:])
    assert new.shape == (10, 3)


def test_cli_embed_runs_an_optional_method(small_pca_csv, temp_dir):
    pytest.importorskip("torch")
    from manifold_genetics.cli import main

    out = temp_dir / "ae.csv"
    main(
        [
            "embed",
            "--input",
            str(small_pca_csv),
            "--output",
            str(out),
            "--method",
            "autoencoder",
            "--epochs",
            "2",
            "--seed",
            "1",
        ]
    )

    df = pd.read_csv(out)
    assert list(df.columns) == ["sample_id", "dim_1", "dim_2"] and len(df) == 50


# --------------------------------------------------------------------------------------
# popVAE end to end, with a stand-in for popvae.py
# --------------------------------------------------------------------------------------
FAKE_POPVAE = r"""
import argparse, json
import numpy as np, pandas as pd, zarr
p = argparse.ArgumentParser()
for a in ("--infile", "--out"):
    p.add_argument(a)
for a in ("--latent_dim", "--max_epochs", "--seed"):
    p.add_argument(a, type=int)
a = p.parse_args()
root = zarr.open_group(a.infile, mode="r")
samples = [str(s) for s in root["samples"][:]]
counts = root["calldata/GT"][:].clip(0).sum(axis=(0, 2))  # each sample's total allele count
cols = ["mean1", "mean2"] if a.latent_dim == 2 else [f"LD{i + 1}" for i in range(a.latent_dim)]
df = pd.DataFrame({c: counts + j for j, c in enumerate(cols)})
df["sampleID"] = samples
df.iloc[::-1].to_csv(a.out + "_latent_coords.txt", sep="\t", index=False)  # deliberately reversed
json.dump(vars(a), open(a.out + "_args.json", "w"))
"""


@pytest.fixture
def popvae_setup(temp_dir):
    bed_reader = pytest.importorskip("bed_reader")
    pytest.importorskip("zarr")
    rng = np.random.default_rng(3)
    G = rng.integers(0, 3, size=(12, 40)).astype(float)
    G[2, 5] = np.nan
    iid = [f"p{i}" for i in range(12)]
    prefix = temp_dir / "geno"
    bed_reader.to_bed(str(prefix) + ".bed", G, properties={"iid": iid})
    script = temp_dir / "popvae.py"
    script.write_text(FAKE_POPVAE)
    order = iid[::-1][:10]  # embed 10 of the 12, in an order unlike the .fam
    X = pd.DataFrame({"sample_id": order, "dim_1": np.arange(10.0)})
    counts = dict(zip(iid, np.nan_to_num(G).sum(axis=1)))
    return prefix, script, X, counts


def test_popvae_end_to_end_keeps_samples_and_genotypes_aligned(popvae_setup, temp_dir):
    import json

    prefix, script, X, counts = popvae_setup
    model = PopVAE(
        genotypes=prefix,
        python=sys.executable,
        script=str(script),
        max_epochs=7,
        workdir=temp_dir / "work",
        random_state=5,
    )
    out = model.fit_transform(X, output_path=temp_dir / "pv.csv")
    assert out.sample_id.tolist() == X.sample_id.tolist()
    assert out.dim_1.tolist() == [counts[s] for s in X.sample_id]  # right genotypes for each sample
    assert (out.dim_2 - out.dim_1 == 1).all()
    assert (temp_dir / "pv.csv").exists()
    args = json.loads((temp_dir / "work" / "popvae_seed5_args.json").read_text())
    assert (args["max_epochs"], args["seed"], args["latent_dim"]) == (7, 5, 2)
    assert model.transform(X).equals(out)
    with pytest.raises(NotImplementedError):
        model.transform(X.iloc[:5])


def test_popvae_reads_ld_columns_and_reuses_the_conversion(popvae_setup, temp_dir):
    prefix, script, X, counts = popvae_setup
    work = temp_dir / "work3"
    kw = dict(genotypes=prefix, python=sys.executable, script=str(script), workdir=work)
    out = PopVAE(n_components=3, **kw).fit_transform(X)
    assert list(out.columns) == ["sample_id", "dim_1", "dim_2", "dim_3"]
    stamp = (work / "input.zarr").stat().st_mtime_ns
    PopVAE(n_components=3, random_state=9, **kw).fit(X)
    assert (work / "input.zarr").stat().st_mtime_ns == stamp  # converted once per workdir


def test_popvae_needs_the_script(monkeypatch, popvae_setup, temp_dir):
    prefix, _, X, _ = popvae_setup
    monkeypatch.delenv("POPVAE_SCRIPT", raising=False)
    with pytest.raises(ValueError, match="POPVAE_SCRIPT"):
        PopVAE(genotypes=prefix, python=sys.executable, workdir=temp_dir / "w").fit(X)


def test_plink_conversion_reports_missing_samples(popvae_setup, temp_dir):
    import subprocess

    from manifold_genetics.embeddings.popvae import plink_to_popvae_zarr

    prefix, _, _, _ = popvae_setup
    with pytest.raises(subprocess.CalledProcessError):
        plink_to_popvae_zarr(prefix, ["p0", "nobody"], temp_dir / "bad.zarr", python=sys.executable)
