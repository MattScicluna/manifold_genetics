"""Running pbwt per chromosome, against a fake binary with the real file layout."""

import os

import pytest

from manifold_genetics.painting.pbwt import (
    expand_phased_vcf,
    paint_chromosome,
    paint_outputs,
    parse_chromosomes,
)

from .fake_pbwt import calls, install_fake_pbwt, write_vcf


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, [str(c) for c in range(1, 23)]),
        (7, ["7"]),
        ([1, "2", "X"], ["1", "2", "X"]),
        ("1-3,7, X", ["1", "2", "3", "7", "X"]),
        ("21-22", ["21", "22"]),
    ],
)
def test_parse_chromosomes(value, expected):
    assert parse_chromosomes(value) == expected


@pytest.mark.parametrize("value", ["", "3-1", "1-X", "1,1", [], "1-2,2"])
def test_parse_chromosomes_rejects(value):
    with pytest.raises(ValueError):
        parse_chromosomes(value)


def test_expand_phased_vcf_substitutes_each_chromosome(tmp_path):
    paths = expand_phased_vcf(tmp_path / "chr{chr}.vcf.gz", ["1", "X"])
    assert paths == {"1": tmp_path / "chr1.vcf.gz", "X": tmp_path / "chrX.vcf.gz"}


def test_expand_phased_vcf_requires_the_placeholder():
    """pbwt paints only a file's first chromosome, so one file for all is refused."""
    with pytest.raises(ValueError, match="first chromosome"):
        expand_phased_vcf("data/all.vcf.gz", ["1", "2"])


def test_paint_chromosome_writes_final_files_only(tmp_path):
    pbwt = install_fake_pbwt(tmp_path / "bin")
    vcf = write_vcf(tmp_path / "chr1.vcf", ["A1", "A2", "B1"])

    out = paint_chromosome(str(pbwt), vcf, tmp_path / "out" / "chr1")

    assert out == paint_outputs(tmp_path / "out" / "chr1")
    assert out["chunkcounts"].read_text().startswith("RECIPIENT IND1 IND2 IND3")
    assert out["samples"].read_text().split() == ["A1", "A2", "B1"]
    assert not list((tmp_path / "out").glob("*.partial*"))
    assert (tmp_path / "out" / "chr1.log").exists()
    assert calls(tmp_path / "bin")[0].split()[:2] == ["-readVcfGT", str(vcf)]


def test_paint_chromosome_reuses_finished_output(tmp_path):
    pbwt = install_fake_pbwt(tmp_path / "bin")
    vcf = write_vcf(tmp_path / "chr1.vcf", ["A1", "B1"])
    root = tmp_path / "out" / "chr1"

    paint_chromosome(str(pbwt), vcf, root)
    paint_chromosome(str(pbwt), vcf, root)
    assert len(calls(tmp_path / "bin")) == 1

    paint_chromosome(str(pbwt), vcf, root, force=True)
    assert len(calls(tmp_path / "bin")) == 2


def test_paint_chromosome_repaints_when_the_vcf_is_newer(tmp_path):
    pbwt = install_fake_pbwt(tmp_path / "bin")
    vcf = write_vcf(tmp_path / "chr1.vcf", ["A1", "B1"])
    root = tmp_path / "out" / "chr1"
    paint_chromosome(str(pbwt), vcf, root)

    later = paint_outputs(root)["chunkcounts"].stat().st_mtime + 10
    os.utime(vcf, (later, later))
    paint_chromosome(str(pbwt), vcf, root)

    assert len(calls(tmp_path / "bin")) == 2


def test_paint_chromosome_reports_pbwt_failure_with_its_log(tmp_path, monkeypatch):
    pbwt = install_fake_pbwt(tmp_path / "bin")
    vcf = write_vcf(tmp_path / "chr1.vcf", ["A1", "B1"])
    monkeypatch.setenv("FAKE_PBWT_FAIL", "1")

    with pytest.raises(RuntimeError, match="forced failure"):
        paint_chromosome(str(pbwt), vcf, tmp_path / "out" / "chr1")
    assert not paint_outputs(tmp_path / "out" / "chr1")["chunkcounts"].exists()


def test_paint_chromosome_missing_vcf(tmp_path):
    pbwt = install_fake_pbwt(tmp_path / "bin")
    with pytest.raises(FileNotFoundError, match="Phased VCF"):
        paint_chromosome(str(pbwt), tmp_path / "nope.vcf", tmp_path / "out" / "chr1")
