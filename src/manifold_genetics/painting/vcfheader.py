"""Sample names from the header of a VCF or BCF, read without htslib.

Painting checks the samples of every phased VCF against the fit and project
cohorts before pbwt runs, because pbwt's own sample list only appears once a
chromosome is painted, which can be most of an hour into a genome-wide run.
Only the header is read, so the check takes a fraction of a second per file.

Three layouts are handled, which are the ones ``pbwt -readVcfGT`` reads through
htslib:

* plain text VCF,
* BGZF or gzip compressed VCF (``.vcf.gz``), which ``gzip`` reads as it is
  because BGZF is a series of gzip members,
* BCF, compressed or not. A BCF2 file opens with the magic ``BCF\\2\\x`` and a
  little-endian 32-bit length, followed by the header as plain VCF text.
"""

import gzip
import struct
from pathlib import Path
from typing import IO, Iterable, List, Union

__all__ = ["read_vcf_samples"]

PathLike = Union[str, Path]

_GZIP_MAGIC = b"\x1f\x8b"
_BCF_MAGIC = b"BCF"
_FIXED_COLUMNS = 9  # CHROM POS ID REF ALT QUAL FILTER INFO FORMAT


def _samples_from_lines(lines: Iterable[str], path: Path) -> List[str]:
    for line in lines:
        if line.startswith("#CHROM"):
            return line.rstrip("\r\n\0").split("\t")[_FIXED_COLUMNS:]
        if not line.startswith("#"):
            break
    raise ValueError(f"{path} has no #CHROM header line, so its samples cannot be read")


def _open(path: Path) -> IO[bytes]:
    with open(path, "rb") as raw:
        magic = raw.read(2)
    return gzip.open(path, "rb") if magic == _GZIP_MAGIC else open(path, "rb")


def read_vcf_samples(path: PathLike) -> List[str]:
    """The sample names of a VCF or BCF, in file order, from its header alone.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file is neither a VCF nor a BCF2, or has no ``#CHROM`` line.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Phased VCF not found: {path}")
    with _open(path) as fh:
        try:
            head = fh.read(5)
        except (OSError, EOFError) as exc:
            raise ValueError(f"Could not read the header of {path}: {exc}") from exc
        if head[:3] == _BCF_MAGIC:
            if head[3:4] != b"\x02":
                raise ValueError(f"{path} is BCF version {head[3]}, and only BCF2 is read")
            (length,) = struct.unpack("<I", fh.read(4))
            text = fh.read(length).decode("utf-8", errors="replace")
            return _samples_from_lines(text.split("\n"), path)
        if not head.startswith(b"#"):
            raise ValueError(f"{path} is neither a VCF nor a BCF: it does not start with '#'")

        def lines():
            first = head + fh.readline()
            yield first.decode("utf-8", errors="replace")
            for line in fh:
                yield line.decode("utf-8", errors="replace")

        return _samples_from_lines(lines(), path)
