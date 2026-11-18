"""Tests for the optional embedding baselines (DREAMS, FloDR, autoencoder, popVAE)."""

import inspect

import numpy as np
import pandas as pd
import pytest

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


def test_autoencoder_fit_transform_is_seeded(small_pca_data):
    pytest.importorskip("torch")
    a = Autoencoder(hidden=(16,), epochs=5, random_state=1).fit_transform(small_pca_data)
    b = Autoencoder(hidden=(16,), epochs=5, random_state=1).fit_transform(small_pca_data)
    assert list(a.columns) == ["sample_id", "dim_1", "dim_2"]
    np.testing.assert_allclose(a[["dim_1", "dim_2"]].values, b[["dim_1", "dim_2"]].values)


def test_popvae_requires_genotypes(small_pca_data):
    with pytest.raises(ValueError, match="genotypes"):
        PopVAE(script="popvae.py").fit(small_pca_data)


def test_popvae_command_line():
    cmd = PopVAE(genotypes="g", python="py", script="popvae.py", random_state=3, max_snps=100)._command(
        "in.popvae.hdf5", "out"
    )
    assert cmd[:2] == ["py", "popvae.py"]
    assert cmd[cmd.index("--seed") + 1] == "3"
    assert cmd[cmd.index("--max_SNPs") + 1] == "100"
    assert cmd[cmd.index("--latent_dim") + 1] == "2"


def test_plink_to_popvae_hdf5_preprocessing(temp_dir):
    """Singletons are dropped, missing calls imputed, dosages scaled to 0/0.5/1."""
    bed_reader = pytest.importorskip("bed_reader")
    h5py = pytest.importorskip("h5py")
    from manifold_genetics.embeddings.popvae import plink_to_popvae_hdf5

    rng = np.random.default_rng(0)
    G = rng.integers(0, 3, size=(20, 30)).astype(float)
    G[:, 0] = 0
    G[0, 0] = 1  # a singleton SNP: dropped
    G[3, 5] = np.nan  # a missing call: imputed
    prefix = temp_dir / "toy"
    iid = [f"s{i}" for i in range(20)]
    bed_reader.to_bed(str(prefix) + ".bed", G, properties={"iid": iid})
    order = iid[::-1]
    plink_to_popvae_hdf5(prefix, order, temp_dir / "toy.popvae.hdf5")
    with h5py.File(temp_dir / "toy.popvae.hdf5", "r") as h5:
        dc = np.array(h5["derived_counts"])
        samples = [s.decode() if isinstance(s, bytes) else s for s in h5["samples"][:]]
    assert samples == order
    assert dc.shape == (20, 29)
    assert set(np.unique(dc)) <= {0.0, 0.5, 1.0}
