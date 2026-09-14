"""`setup --preprocessing` fetches what the harmonise preset otherwise downloads
mid-run -- which a compute node without internet cannot do."""

import gzip
import zipfile
from pathlib import Path

from manifold_genetics.preprocessing.references import (  # noqa: F401
    HARMONISATION_REFERENCES,
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
