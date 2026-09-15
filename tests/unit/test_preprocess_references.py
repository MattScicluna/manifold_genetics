"""`setup --preprocessing` fetches what the harmonise preset otherwise downloads
mid-run -- which a compute node without internet cannot do."""

import gzip
import zipfile
from pathlib import Path

import pytest

from manifold_genetics.preprocessing import references
from manifold_genetics.preprocessing.references import (  # noqa: F401
    HARMONISATION_REFERENCES,
    WRAYNER_URL,
    install_harmonisation_references,
)


def _fake_fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if url.endswith(".zip"):
        with zipfile.ZipFile(destination, "w") as zf:
            zf.writestr("HRC-1000G-check-bim.pl", '#!/usr/bin/perl\nprint SH "$plink --bfile";\n')
    elif ".bed.gz" in url:
        destination.write_bytes(gzip.compress(b"chr1\t1\t2\n"))
    else:
        destination.write_bytes(b"data")


def test_places_each_reference_where_the_shell_looks(tmp_path):
    placed = install_harmonisation_references(tmp_path, fetch=_fake_fetch)
    assert set(placed) == {"giab", "wrayner", "topmed"}
    assert placed["giab"] == tmp_path / "giab/GRCh38_alldifficultregions.bed"
    assert (
        placed["giab"].read_text() == "chr1\t1\t2\n"
    ), "the bed is stored gunzipped, as the shell writes it"
    assert placed["wrayner"] == tmp_path / "wrayner/HRC-1000G-check-bim.pl"
    assert placed["topmed"] == tmp_path / "topmed/bravo-dbsnp-all.hrc_format.tab.gz"


def test_is_idempotent(tmp_path):
    calls = []

    def counting(url, destination):
        calls.append(url)
        _fake_fetch(url, destination)

    install_harmonisation_references(tmp_path, fetch=counting)
    install_harmonisation_references(tmp_path, fetch=counting)
    assert len(calls) == 3


def test_wrayner_checker_is_patched_not_to_write_a_vcf(tmp_path):
    placed = install_harmonisation_references(tmp_path, fetch=_fake_fetch)
    assert placed["wrayner"].read_text().startswith("#!/usr/bin/perl")
    assert not (tmp_path / "wrayner/HRC-1000G-check-bim.zip").exists(), "no zip left behind"


def test_one_failed_download_does_not_stop_the_others(tmp_path):
    """The WRayner URL is 404 upstream; GIAB and TOPMed must still be placed,
    and the error must say where a hand-obtained checker goes."""

    def fetch(url, destination):
        if url == WRAYNER_URL:
            raise OSError("HTTP Error 404: Not Found")
        _fake_fetch(url, destination)

    with pytest.raises(RuntimeError) as excinfo:
        install_harmonisation_references(tmp_path, fetch=fetch)

    assert (tmp_path / "giab/GRCh38_alldifficultregions.bed").exists()
    assert (tmp_path / "topmed/bravo-dbsnp-all.hrc_format.tab.gz").exists()
    assert not (tmp_path / "wrayner/HRC-1000G-check-bim.pl").exists()
    message = str(excinfo.value)
    assert "wrayner" in message and WRAYNER_URL in message and "404" in message
    assert str(tmp_path / "wrayner/HRC-1000G-check-bim.pl") in message
    assert "giab" in message and "topmed" in message, "says what was placed"
    assert "Re-running" in message

    # And re-running with the URL back fetches only what is missing.
    calls = []

    def counting(url, destination):
        calls.append(url)
        _fake_fetch(url, destination)

    install_harmonisation_references(tmp_path, fetch=counting)
    assert calls == [WRAYNER_URL]


def test_an_interrupted_wrayner_install_leaves_no_checker_behind(tmp_path, monkeypatch):
    """Extract and patch happen in a scratch directory; only a finished, patched
    checker is moved into place. Half an install must not look like a whole one
    to the next run, which would then skip it."""
    original_extractall = zipfile.ZipFile.extractall

    def extract_then_die(self, path=None, *args, **kwargs):
        original_extractall(self, path, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(zipfile.ZipFile, "extractall", extract_then_die)
    with pytest.raises(KeyboardInterrupt):
        references._install_wrayner(
            WRAYNER_URL, tmp_path / "wrayner/HRC-1000G-check-bim.pl", _fake_fetch
        )

    assert not (tmp_path / "wrayner/HRC-1000G-check-bim.pl").exists()
    assert list((tmp_path / "wrayner").iterdir()) == [], "the scratch directory is removed"
