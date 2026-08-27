"""Tests for the smaller helpers and error paths in utils/io.py:
_append_extension, validate_plink_files, write_embedding_csv, read_fam_file,
get_sample_ids_from_plink, read_sample_indices, read_bim_file,
check_allele_compatibility, plus the missing-file / bad-format branches of
the readers.
"""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.utils import io

# ---------------------------------------------------------------------------
# _append_extension
# ---------------------------------------------------------------------------


def test_append_extension_adds_when_absent():
    assert str(io._append_extension("data/ref", ".bed")) == "data/ref.bed"


def test_append_extension_noop_when_present():
    assert str(io._append_extension("data/ref.bed", ".bed")) == "data/ref.bed"


def test_append_extension_preserves_dotted_prefix():
    result = io._append_extension("data/ref.noHLA.unrelated", ".fam")
    assert result.name == "ref.noHLA.unrelated.fam"


# ---------------------------------------------------------------------------
# validate_plink_files
# ---------------------------------------------------------------------------


def _make_plink(tmp_path, name="ds"):
    prefix = tmp_path / name
    for ext in (".bed", ".bim", ".fam"):
        (tmp_path / f"{name}{ext}").write_text("x")
    return prefix


def test_validate_plink_files_ok(tmp_path):
    prefix = _make_plink(tmp_path)
    assert io.validate_plink_files(prefix) == prefix


def test_validate_plink_files_missing_raises(tmp_path):
    (tmp_path / "ds.bed").write_text("x")  # only .bed
    with pytest.raises(FileNotFoundError, match=r"ds\.bim"):
        io.validate_plink_files(tmp_path / "ds")


# ---------------------------------------------------------------------------
# write_embedding_csv
# ---------------------------------------------------------------------------


def test_write_embedding_csv_from_numpy_with_ids(tmp_path):
    out = tmp_path / "sub" / "emb.csv"
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    io.write_embedding_csv(arr, out, sample_ids=["s1", "s2"])
    df = pd.read_csv(out)
    assert list(df.columns) == ["sample_id", "dim_1", "dim_2"]
    assert list(df["sample_id"].astype(str)) == ["s1", "s2"]


def test_write_embedding_csv_from_numpy_without_ids(tmp_path):
    out = tmp_path / "emb.csv"
    io.write_embedding_csv(np.zeros((3, 2)), out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["dim_1", "dim_2"]


def test_write_embedding_csv_dataframe_passthrough(tmp_path):
    out = tmp_path / "emb.csv"
    src = pd.DataFrame({"sample_id": ["a"], "dim_1": [1.0]})
    io.write_embedding_csv(src, out)
    assert pd.read_csv(out).equals(src)


# ---------------------------------------------------------------------------
# reader error paths
# ---------------------------------------------------------------------------


def test_read_embedding_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_embedding_csv(tmp_path / "nope.csv")


def test_read_embedding_csv_without_dim_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"sample_id": ["a"], "x": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="dim_"):
        io.read_embedding_csv(p)


def test_read_admixture_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_admixture_csv(tmp_path / "nope.csv")


def test_read_admixture_csv_missing_sample_id(tmp_path):
    p = tmp_path / "q.csv"
    pd.DataFrame({"component_1": [0.5], "component_2": [0.5]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="sample_id"):
        io.read_admixture_csv(p)


def test_read_labels_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_labels_csv(tmp_path / "nope.csv")


def test_read_labels_csv_without_sample_id_uses_first_column(tmp_path):
    p = tmp_path / "labels.csv"
    pd.DataFrame({"id": ["a", "b"], "Population": ["X", "Y"]}).to_csv(p, index=False)
    df = io.read_labels_csv(p)
    assert df.index.name == "id"


def test_read_colormap_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_colormap(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# fam / sample-id helpers
# ---------------------------------------------------------------------------


def test_read_fam_file_parses_columns(tmp_path):
    fam = tmp_path / "ds.fam"
    fam.write_text("F1 IID1 0 0 1 -9\nF2 IID2 0 0 2 -9\n")
    df = io.read_fam_file(fam)
    assert list(df["IID"]) == ["IID1", "IID2"]
    assert list(df.columns) == ["FID", "IID", "Father", "Mother", "Sex", "Phenotype"]


def test_read_fam_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_fam_file(tmp_path / "nope.fam")


def test_get_sample_ids_from_plink(tmp_path):
    prefix = _make_plink(tmp_path)
    (tmp_path / "ds.fam").write_text("F1 s1 0 0 1 -9\nF2 s2 0 0 1 -9\n")
    assert io.get_sample_ids_from_plink(prefix) == ["s1", "s2"]


def test_read_sample_indices_skips_blank_lines(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("s1\n\n  s2  \n\ns3\n")
    assert io.read_sample_indices(p) == ["s1", "s2", "s3"]


def test_read_sample_indices_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_sample_indices(tmp_path / "nope.txt")


# ---------------------------------------------------------------------------
# bim / allele compatibility
# ---------------------------------------------------------------------------


def test_read_bim_file_parses_and_skips_short_lines(tmp_path):
    bim = tmp_path / "ds.bim"
    bim.write_text("1\trs1\t0\t100\tA\tG\n1\trs2\t0\t200\tC\tT\nbad line\n")
    alleles = io.read_bim_file(bim)
    assert alleles == {"rs1": ("A", "G"), "rs2": ("C", "T")}


def test_read_bim_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_bim_file(tmp_path / "nope.bim")


def test_check_allele_compatibility_classifies_snps(tmp_path):
    b1 = tmp_path / "a.bim"
    b2 = tmp_path / "b.bim"
    b1.write_text(
        "1\trs_exact\t0\t1\tA\tG\n"
        "1\trs_flip\t0\t2\tA\tG\n"
        "1\trs_bad\t0\t3\tA\tG\n"
        "1\trs_only1\t0\t4\tA\tG\n"
    )
    b2.write_text("1\trs_exact\t0\t1\tA\tG\n" "1\trs_flip\t0\t2\tG\tA\n" "1\trs_bad\t0\t3\tC\tT\n")
    common = {"rs_exact", "rs_flip", "rs_bad", "rs_only1", "rs_only2"}
    exact, flip, incompatible = io.check_allele_compatibility(b1, b2, common)
    assert exact == {"rs_exact"}
    assert flip == {"rs_flip"}
    assert {"rs_bad", "rs_only1", "rs_only2"}.issubset(incompatible)
