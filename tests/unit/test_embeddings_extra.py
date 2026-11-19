"""Tests for the optional embedding baselines (DREAMS, FloDR, autoencoder, popVAE)."""

import argparse
import inspect
import sys

import numpy as np
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
