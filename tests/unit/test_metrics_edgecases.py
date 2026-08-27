"""Edge-case coverage for metrics/.

The primary Spearman-preservation behaviour already has positive/negative
control tests in tests/test_metrics.py. These cover the branches those miss:
subsampling, the legacy Q format, and the error paths.
"""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.metrics.admixture import (
    _load_q_matrix,
    compute_admixture_preservation,
)
from manifold_genetics.metrics.geographic import compute_geographic_preservation


def _embedding(n, dims=2, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, dims)), columns=[f"dim_{i+1}" for i in range(dims)])
    df.insert(0, "sample_id", [f"s{i}" for i in range(n)])
    return df


def _geo_matching(embedding_df):
    """Geo coords identical to the embedding -> perfect preservation."""
    g = pd.DataFrame(
        {
            "sample_id": embedding_df["sample_id"],
            "latitude": embedding_df["dim_1"].to_numpy(),
            "longitude": embedding_df["dim_2"].to_numpy(),
        }
    ).set_index("sample_id")
    return g


# ---------------------------------------------------------------------------
# geographic
# ---------------------------------------------------------------------------


def test_geographic_subsamples_pairwise_distances():
    emb = _embedding(40)
    geo = _geo_matching(emb)
    result = compute_geographic_preservation(emb, geo, num_samples=50)
    # 40 samples -> 780 pairs, capped at 50
    assert result["n_pairs"] == 50
    assert result["n_samples"] == 40


def test_geographic_too_few_samples_raises():
    emb = _embedding(3)
    geo = _geo_matching(emb).iloc[:1]  # only one sample has coords
    with pytest.raises(ValueError, match="at least 2 samples"):
        compute_geographic_preservation(emb, geo)


# ---------------------------------------------------------------------------
# admixture
# ---------------------------------------------------------------------------


def _q_file(tmp_path, name, k, n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.dirichlet(np.ones(k), n)
    df = pd.DataFrame(q, columns=[f"component_{i+1}" for i in range(k)])
    df.insert(0, "sample_id", [f"s{i}" for i in range(n)])
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_admixture_preservation_all_k_values(tmp_path):
    emb = _embedding(30, dims=3)
    qf = {2: _q_file(tmp_path, "q2.csv", 2, 30), 3: _q_file(tmp_path, "q3.csv", 3, 30)}
    result = compute_admixture_preservation(emb, qf)
    assert set(result) == {2, 3}
    for k in (2, 3):
        assert set(result[k]) == {"correlation", "p_value", "n_samples", "n_pairs"}
        assert result[k]["n_samples"] == 30


def test_admixture_subsample_individuals(tmp_path):
    emb = _embedding(50)
    qf = {2: _q_file(tmp_path, "q2.csv", 2, 50)}
    result = compute_admixture_preservation(emb, qf, subsample=12)
    assert result[2]["n_samples"] == 12


def test_admixture_caps_pairwise_distances(tmp_path):
    emb = _embedding(40)
    qf = {2: _q_file(tmp_path, "q2.csv", 2, 40)}
    result = compute_admixture_preservation(emb, qf, num_samples=30)
    assert result[2]["n_pairs"] == 30


def test_admixture_k_value_not_in_q_files_is_skipped(tmp_path):
    emb = _embedding(20)
    qf = {2: _q_file(tmp_path, "q2.csv", 2, 20)}
    assert compute_admixture_preservation(emb, qf, k_value=9) == {}


def test_admixture_no_common_samples_raises(tmp_path):
    emb = _embedding(10)  # s0..s9
    rng = np.random.default_rng(1)
    q = pd.DataFrame(rng.dirichlet([1, 1], 10), columns=["component_1", "component_2"])
    q.insert(0, "sample_id", [f"other{i}" for i in range(10)])
    p = tmp_path / "q.csv"
    q.to_csv(p, index=False)
    with pytest.raises(ValueError, match="No common sample IDs"):
        compute_admixture_preservation(emb, {2: p})


def test_load_q_matrix_legacy_headerless_format(tmp_path):
    p = tmp_path / "legacy.Q"
    p.write_text("0.9 0.1\n0.3 0.7\n0.5 0.5\n")
    m = _load_q_matrix(p)
    assert m.shape == (3, 2)
    assert m.iloc[0, 0] == pytest.approx(0.9)


def test_load_q_matrix_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_q_matrix(tmp_path / "nope.Q")
