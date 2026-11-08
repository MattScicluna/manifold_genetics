"""The pure half of haplotype painting: parsing pbwt output and building features.

The fixtures are hand-written in the exact layout pbwtPaint.c prints (commit
50fd6be): a ``RECIPIENT IND1 ...`` header, one ``IND<i>`` row per individual,
space-separated, zero diagonal.
"""

import numpy as np
import pytest

from manifold_genetics.painting.matrix import (
    coancestry_features,
    fill_self_copying,
    log_transform,
    read_paint_matrix,
    read_pbwt_samples,
    row_normalise,
    sum_matrices,
)

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


def test_fill_self_copying_uses_the_row_mean():
    m = np.array([[0.0, 6.0, 2.0], [5.0, 0.0, 3.0], [1.0, 1.0, 0.0]])
    filled = fill_self_copying(m)

    np.testing.assert_allclose(np.diag(filled), [4.0, 4.0, 1.0])
    np.testing.assert_array_equal(m[0], [0.0, 6.0, 2.0])  # input untouched


def test_row_normalise_sums_to_one():
    p = row_normalise(np.array([[1.0, 3.0], [2.0, 2.0]]))
    np.testing.assert_allclose(p, [[0.25, 0.75], [0.5, 0.5]])


def test_row_normalise_names_a_sample_that_copied_from_nobody():
    with pytest.raises(ValueError, match="S2"):
        row_normalise(np.array([[1.0, 1.0], [0.0, 0.0]]), labels=["S1", "S2"])


def test_log_transform_maps_uniform_copying_to_log_two():
    p = np.full((4, 4), 0.25)
    np.testing.assert_allclose(log_transform(p), np.log(2.0))
    assert np.isfinite(log_transform(np.array([[0.0, 1.0], [1.0, 0.0]]))).all()


def test_coancestry_features_composes_the_steps():
    m = np.array([[0.0, 6.0, 2.0], [5.0, 0.0, 3.0], [1.0, 1.0, 0.0]])

    plain = coancestry_features(m, log=False)
    np.testing.assert_allclose(plain.sum(axis=1), 1.0)
    np.testing.assert_allclose(plain[0], [4 / 12, 6 / 12, 2 / 12])

    np.testing.assert_allclose(coancestry_features(m, log=True), np.log1p(plain * 3))
