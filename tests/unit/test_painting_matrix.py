"""The pure half of haplotype painting: parsing pbwt output and building features.

The fixtures are hand-written in the exact layout pbwtPaint.c prints (commit
50fd6be): a ``RECIPIENT IND1 ...`` header, one ``IND<i>`` row per individual,
space-separated, zero diagonal.
"""

import numpy as np
import pytest

from manifold_genetics.painting.matrix import (
    build_features,
    centimorgan_weight,
    finestructure_features,
    hc_features,
    method_matrix,
    read_paint_matrix,
    read_pbwt_samples,
    sum_matrices,
)
from manifold_genetics.painting.reduce import painting_pca

CHUNKCOUNTS = """\
RECIPIENT IND1 IND2 IND3
IND1 0.0000 6.0000 2.0000
IND2 5.0000 0.0000 3.0000
IND3 1.0000 1.0000 0.0000
"""


def write(path, text):
    path.write_text(text)
    return path


def test_reads_pbwt_layout(tmp_path):
    labels, m = read_paint_matrix(write(tmp_path / "c.chunkcounts.out", CHUNKCOUNTS))

    assert labels == ["IND1", "IND2", "IND3"]
    np.testing.assert_array_equal(
        m, [[0, 6, 2], [5, 0, 3], [1, 1, 0]]
    )  # rows recipients, columns donors


def test_rejects_a_file_that_is_not_paint_output(tmp_path):
    with pytest.raises(ValueError, match="RECIPIENT"):
        read_paint_matrix(write(tmp_path / "x.out", "sample IND1\nIND1 0\n"))


def test_rejects_a_short_row(tmp_path):
    text = "RECIPIENT IND1 IND2\nIND1 0 1\nIND2 1\n"
    with pytest.raises(ValueError, match=":3 has 1 values"):
        read_paint_matrix(write(tmp_path / "x.out", text))


def test_rejects_rows_that_do_not_match_columns(tmp_path):
    text = "RECIPIENT IND1 IND2\nIND2 0 1\nIND1 1 0\n"
    with pytest.raises(ValueError, match="not a square"):
        read_paint_matrix(write(tmp_path / "x.out", text))


def test_reads_samples_and_ignores_pbwt_annotations(tmp_path):
    path = write(tmp_path / "s.samples", "HG001\nHG002\tPOP:GBR\n\nHG003\tFATHER:HG001\n")
    assert read_pbwt_samples(path) == ["HG001", "HG002", "HG003"]


def test_sum_matrices_adds_chromosomes():
    a, b = np.ones((2, 2)), np.full((2, 2), 2.0)
    np.testing.assert_array_equal(sum_matrices([a, b]), np.full((2, 2), 3.0))


def test_sum_matrices_refuses_different_sample_sets():
    with pytest.raises(ValueError, match="same samples"):
        sum_matrices([np.ones((2, 2)), np.ones((3, 3))])
    with pytest.raises(ValueError, match="No painting"):
        sum_matrices([])


def test_sum_matrices_applies_weights():
    a, b = np.ones((2, 2)), np.full((2, 2), 2.0)
    np.testing.assert_array_equal(sum_matrices([a, b], [3.0, 0.5]), np.full((2, 2), 4.0))
    with pytest.raises(ValueError, match="2 weights for 1"):
        sum_matrices([a], [1.0, 2.0])


def mypca_reference(x):
    """``mypca`` from the fineSTRUCTURE R library, transcribed line by line.

    mypcanorm: diag(x) <- rowSums(x)/(dim(x)[1]-1); x - rowMeans(x)
    mypca:     eigen(xN %*% t(xN))
    """
    x = np.array(x, dtype=float)
    np.fill_diagonal(x, x.sum(axis=1) / (x.shape[0] - 1))
    xn = x - x.mean(axis=1)[:, None]
    values, vectors = np.linalg.eigh(xn @ xn.T)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def test_finestructure_features_follow_mypcanorm():
    m = np.array([[0.0, 6.0, 2.0], [5.0, 0.0, 3.0], [1.0, 1.0, 0.0]])
    x = finestructure_features(m)

    # Diagonal set to the off-diagonal row mean, then rows centered, which
    # leaves the diagonal at exactly zero.
    np.testing.assert_allclose(np.diag(x), 0.0, atol=1e-12)
    np.testing.assert_allclose(x.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(x[0], [0.0, 2.0, -2.0])
    np.testing.assert_array_equal(m[0], [0.0, 6.0, 2.0])  # input untouched


def test_finestructure_pca_matches_mypca():
    rng = np.random.default_rng(0)
    m = rng.gamma(2.0, size=(12, 12)) + np.kron(np.eye(3), np.full((4, 4), 3.0))
    np.fill_diagonal(m, 0.0)
    ids = [f"S{i}" for i in range(12)]

    fit, _ = painting_pca(
        finestructure_features(m), ids, fit_rows=range(12), project_rows=[],
        n_components=3, method="finestructure",
    )  # fmt: skip
    values, vectors = mypca_reference(m)
    coords = fit[["dim_1", "dim_2", "dim_3"]].to_numpy()

    # Same components; ours are scaled by the singular value (scores, U D).
    for j in range(3):
        np.testing.assert_allclose(np.abs(coords[:, j]), np.abs(vectors[:, j]) * np.sqrt(values[j]))


def test_hc_is_the_uncentered_svd_of_log10_lengths():
    rng = np.random.default_rng(1)
    a = rng.gamma(2.0, size=(10, 10)) * 5
    np.fill_diagonal(a, 0.0)
    ids = [f"S{i}" for i in range(10)]
    feats = hc_features(a)
    np.testing.assert_allclose(feats, np.log10(a + 1))

    fit, project = painting_pca(
        feats, ids, fit_rows=range(10), project_rows=[3, 4], n_components=2, method="hc"
    )
    u, d, _ = np.linalg.svd(np.log10(a + 1))
    hcs = u[:, :2] * np.sqrt(d[:2])  # U sqrt(D), as in Yang et al. (2025)
    np.testing.assert_allclose(np.abs(fit[["dim_1", "dim_2"]].to_numpy()), np.abs(hcs))
    # Projecting a fitted row reproduces its fitted coordinates.
    np.testing.assert_allclose(
        project[["dim_1", "dim_2"]].to_numpy(), fit[["dim_1", "dim_2"]].to_numpy()[[3, 4]]
    )


def test_centimorgan_weight_reads_the_site_count_off_the_matrix():
    # pbwt scales each chunklengths row to n_sites * ploidy.
    lengths = np.array([[0.0, 150.0, 50.0], [100.0, 0.0, 100.0], [20.0, 180.0, 0.0]])
    assert centimorgan_weight(lengths, "22") == pytest.approx(74.10 / 100)
    assert centimorgan_weight(lengths, "chr21") == pytest.approx(62.79 / 100)
    with pytest.raises(ValueError, match="'X'"):
        centimorgan_weight(lengths, "X")


def test_build_features_by_method():
    counts = {"21": np.array([[0.0, 2.0], [1.0, 0.0]]), "22": np.array([[0.0, 1.0], [3.0, 0.0]])}
    summed, x = build_features(counts, "finestructure")
    np.testing.assert_array_equal(summed, [[0, 3], [4, 0]])
    np.testing.assert_allclose(x, finestructure_features(summed))

    lengths = {"21": np.array([[0.0, 4.0], [4.0, 0.0]]), "22": np.array([[0.0, 2.0], [2.0, 0.0]])}
    summed, x = build_features(lengths, "hc")
    # 2 sites on chr21, 1 on chr22: each row sums to 2 * (g_21 + g_22) cM.
    np.testing.assert_allclose(summed.sum(axis=1), 2 * (62.79 + 74.10))
    np.testing.assert_allclose(x, np.log10(summed + 1))

    assert method_matrix("hc") == "chunklengths"
    with pytest.raises(ValueError, match="Unknown painting method"):
        method_matrix("log")


def test_a_sample_that_copied_from_nobody_is_named():
    counts = {"1": np.array([[0.0, 1.0], [0.0, 0.0]])}
    with pytest.raises(ValueError, match="S2"):
        build_features(counts, "finestructure", labels=["S1", "S2"])
