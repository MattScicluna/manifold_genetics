"""Wrapper-logic tests for the embedding classes.

The real PHATE/UMAP/t-SNE/DiffusionMap compute is a third-party library and
is slow, so each class's underlying ``.model`` is replaced with a fake that
returns predictable arrays. What is tested is the code this repo owns: input
loading and type dispatch, the not-fitted guard, PHATE's batch-transform
splitting, and output formatting.
"""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.embeddings import PHATE, TSNE, UMAP, DiffusionMap
from manifold_genetics.embeddings.base import EmbeddingBase


class FakeModel:
    """Stand-in for phate.PHATE / umap.UMAP / etc."""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.fit_calls = []
        self.transform_calls = []

    def fit(self, X):
        self.fit_calls.append(np.asarray(X).shape)
        return self

    def transform(self, X):
        X = np.asarray(X)
        self.transform_calls.append(X.shape)
        # deterministic 2-D output: first two columns (padded)
        out = np.zeros((len(X), self.n_components))
        out[:, 0] = X[:, 0] if X.shape[1] else 0.0
        return out

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


@pytest.fixture
def phate_model(monkeypatch):
    m = PHATE(n_components=2, knn=3)
    fake = FakeModel(n_components=2)
    monkeypatch.setattr(m, "model", fake)
    return m, fake


# ---------------------------------------------------------------------------
# base._load_input_data — type dispatch
# ---------------------------------------------------------------------------


class _Concrete(EmbeddingBase):
    def fit(self, X):  # pragma: no cover - not exercised
        return self

    def transform(self, X):  # pragma: no cover
        return pd.DataFrame()

    def fit_transform(self, X, output_path=None):  # pragma: no cover
        return pd.DataFrame()


def test_load_input_data_from_numpy_generates_sample_ids():
    b = _Concrete()
    arr = np.arange(12).reshape(4, 3)
    X, ids = b._load_input_data(arr)
    assert X.shape == (4, 3)
    assert ids == ["sample_0", "sample_1", "sample_2", "sample_3"]


def test_load_input_data_from_dataframe_with_sample_id_column():
    b = _Concrete()
    df = pd.DataFrame({"sample_id": ["a", "b"], "dim_1": [1.0, 2.0], "dim_2": [3.0, 4.0]})
    X, ids = b._load_input_data(df)
    assert ids == ["a", "b"]
    assert X.shape == (2, 2)


def test_load_input_data_from_dataframe_with_sample_id_index():
    b = _Concrete()
    df = pd.DataFrame({"dim_1": [1.0, 2.0]}, index=pd.Index(["a", "b"], name="sample_id"))
    X, ids = b._load_input_data(df)
    assert ids == ["a", "b"]


def test_load_input_data_from_csv_path(tmp_path):
    b = _Concrete()
    p = tmp_path / "e.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "dim_1": [1.0, 2.0]}).to_csv(p, index=False)
    X, ids = b._load_input_data(p)
    assert ids == ["a", "b"]


def test_load_input_data_rejects_unsupported_type():
    b = _Concrete()
    with pytest.raises(TypeError, match="Unsupported input type"):
        b._load_input_data({"not": "supported"})


def test_format_output_shape_and_optional_write(tmp_path):
    b = _Concrete()
    out = tmp_path / "sub" / "o.csv"
    df = b._format_output(np.array([[0.1, 0.2], [0.3, 0.4]]), ["a", "b"], output_path=out)
    assert list(df.columns) == ["sample_id", "dim_1", "dim_2"]
    assert list(df["sample_id"]) == ["a", "b"]
    assert out.exists()


# ---------------------------------------------------------------------------
# PHATE wrapper
# ---------------------------------------------------------------------------


def test_phate_transform_before_fit_raises(phate_model):
    m, _ = phate_model
    with pytest.raises(RuntimeError, match="not fitted"):
        m.transform(np.random.rand(4, 3))


def test_phate_fit_then_transform(phate_model):
    m, fake = phate_model
    X = np.random.rand(6, 4)
    m.fit(X)
    assert m._is_fitted and fake.fit_calls == [(6, 4)]
    result = m.transform(X)
    assert list(result.columns) == ["sample_id", "dim_1", "dim_2"]
    assert len(result) == 6
    assert fake.transform_calls == [(6, 4)]  # single, unbatched call


def test_phate_transform_batches_when_batch_size_smaller_than_data(monkeypatch):
    m = PHATE(n_components=2, knn=3, embed_batch_size=2)
    fake = FakeModel(n_components=2)
    monkeypatch.setattr(m, "model", fake)
    X = np.random.rand(5, 3)
    m.fit(X)
    fake.transform_calls.clear()
    result = m.transform(X)
    # 5 samples, batch 2 -> slices of 2, 2, 1
    assert [shape[0] for shape in fake.transform_calls] == [2, 2, 1]
    assert len(result) == 5


def test_phate_transform_no_batch_when_data_smaller_than_batch_size(monkeypatch):
    m = PHATE(n_components=2, knn=3, embed_batch_size=100)
    fake = FakeModel(n_components=2)
    monkeypatch.setattr(m, "model", fake)
    X = np.random.rand(5, 3)
    m.fit(X)
    fake.transform_calls.clear()
    m.transform(X)
    assert fake.transform_calls == [(5, 3)]  # one call, no batching


def test_phate_fit_transform_writes_output(monkeypatch, tmp_path):
    m = PHATE(n_components=2, knn=3)
    fake = FakeModel(n_components=2)
    monkeypatch.setattr(m, "model", fake)
    out = tmp_path / "phate.csv"
    df = m.fit_transform(np.random.rand(4, 3), output_path=out)
    assert m._is_fitted
    assert len(fake.fit_calls) == 1
    assert out.exists()
    assert list(df.columns) == ["sample_id", "dim_1", "dim_2"]


# ---------------------------------------------------------------------------
# UMAP — model-backed wrapper, same guard/format as PHATE
# ---------------------------------------------------------------------------


def test_umap_guard_and_format(monkeypatch):
    m = UMAP(n_components=2)
    fake = FakeModel(n_components=2)
    monkeypatch.setattr(m, "model", fake)

    with pytest.raises(RuntimeError, match="not fitted"):
        m.transform(np.random.rand(4, 3))

    X = np.random.rand(5, 3)
    m.fit(X)
    result = m.transform(X)
    assert list(result.columns) == ["sample_id", "dim_1", "dim_2"]
    assert len(result) == 5


# ---------------------------------------------------------------------------
# t-SNE — transform() intentionally refits (no out-of-sample support)
# ---------------------------------------------------------------------------


def test_tsne_transform_refits_and_warns(monkeypatch, caplog):
    import logging

    m = TSNE(n_components=2)
    fake = FakeModel(n_components=2)
    monkeypatch.setattr(m, "model", fake)
    with caplog.at_level(logging.WARNING):
        result = m.transform(np.random.rand(5, 3))  # no prior fit — must not raise
    assert "does not support out-of-sample" in caplog.text
    assert len(result) == 5
    assert fake.fit_calls  # it refit


# ---------------------------------------------------------------------------
# DiffusionMap — real (tiny) eigendecomposition, no wrapped model object
# ---------------------------------------------------------------------------


def test_diffusion_map_guard_then_real_small_run():
    m = DiffusionMap(n_components=2, knn=3)
    with pytest.raises(RuntimeError, match="not fitted"):
        m.transform(np.random.rand(4, 3))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    df = m.fit_transform(X)
    assert m._is_fitted
    assert list(df.columns) == ["sample_id", "dim_1", "dim_2"]
    assert len(df) == 12
    assert np.isfinite(df[["dim_1", "dim_2"]].to_numpy()).all()
