"""Pure-Python reader for PLINK 1 binary genotype files.

Exists so that PCA needs no external binary: the package depends only on numpy,
which has wheels everywhere, rather than on a Linux-x86-64 ``flashpca``
executable. The PLINK 1 ``.bed`` format has been frozen since 2009 and is small
enough to read correctly in a few dozen lines.

Layout: a 3-byte magic header (``6c 1b 01``), then ``ceil(n_samples / 4)`` bytes
per variant, two bits per sample, with sample ``4i + k`` in bits ``[2k+1 : 2k]``
of byte ``i``. Codes are ``00`` homozygous A1, ``01`` missing, ``10``
heterozygous, ``11`` homozygous A2.

Dosage is the **count of the A1 allele** (``.bim`` column 5). That direction is
not a guess: checked against flashpca's own ``.meansd`` output over all 172,152
HGDP variants, counting A1 reproduced flashpca's per-variant mean to 5e-7, while
counting A2 was wrong by as much as 1.97.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "PlinkFormatError",
    "count_lines",
    "read_bed_dosages",
    "read_bim_variants",
    "read_fam_ids",
]

PathLike = Union[str, Path]

_MAGIC = b"\x6c\x1b"
_SNP_MAJOR = 0x01
_INDIVIDUAL_MAJOR = 0x00

# Code -> count of A1. Index by the 2-bit code; 0b01 (missing) becomes NaN.
_CODE_TO_A1 = np.array([2.0, np.nan, 1.0, 0.0], dtype=np.float64)


class PlinkFormatError(ValueError):
    """A .bed file is not the SNP-major PLINK 1 layout we can read."""


def read_bed_dosages(
    prefix: PathLike,
    *,
    n_samples: int,
    n_variants: int,
    variants: Optional[slice] = None,
) -> np.ndarray:
    """Read A1 dosages from ``<prefix>.bed``.

    Args:
        prefix: Path prefix; ``.bed`` is appended.
        n_samples: Sample count, from the ``.fam`` file.
        n_variants: Variant count in the file, from the ``.bim`` file.
        variants: Optional contiguous slice of variants to read. Only the bytes
            for those variants are read, so a large cohort can be processed in
            chunks without materialising the whole matrix.

    Returns:
        ``(n_samples, n_selected_variants)`` float64 array of A1 dosages, with
        NaN where the genotype is missing.

    Raises:
        PlinkFormatError: wrong magic number, individual-major layout, or a file
            shorter than ``n_samples``/``n_variants`` imply.
    """
    path = Path(f"{prefix}.bed")
    bytes_per_variant = (n_samples + 3) // 4

    sel = variants or slice(0, n_variants)
    start, stop, step = sel.indices(n_variants)
    if step != 1:
        raise ValueError(f"variants slice must be contiguous, got step={step}")
    n_selected = max(0, stop - start)

    with open(path, "rb") as fh:
        magic = fh.read(3)
        if len(magic) < 3 or magic[:2] != _MAGIC:
            raise PlinkFormatError(
                f"{path} does not start with the PLINK .bed magic number "
                f"(expected 6c 1b, got {magic[:2].hex(' ') or 'nothing'})"
            )
        if magic[2] == _INDIVIDUAL_MAJOR:
            raise PlinkFormatError(
                f"{path} is individual-major. Only SNP-major .bed files are supported; "
                "convert it with `plink --bfile <prefix> --make-bed --out <prefix>`."
            )
        if magic[2] != _SNP_MAJOR:
            raise PlinkFormatError(
                f"{path} has unknown layout byte 0x{magic[2]:02x}; expected 0x01 (SNP-major)"
            )

        fh.seek(3 + start * bytes_per_variant)
        raw = np.frombuffer(fh.read(bytes_per_variant * n_selected), dtype=np.uint8)

    if raw.size != bytes_per_variant * n_selected:
        raise PlinkFormatError(
            f"{path} is truncated: expected {bytes_per_variant * n_selected} bytes for "
            f"{n_selected} variants x {n_samples} samples, found {raw.size}"
        )

    raw = raw.reshape(n_selected, bytes_per_variant)

    # Unpack the four 2-bit codes per byte. Padding bits in the final byte decode
    # as 0b00 (homozygous A1) and must be dropped, not returned as real samples.
    codes = np.empty((n_selected, bytes_per_variant * 4), dtype=np.uint8)
    for k in range(4):
        codes[:, k::4] = (raw >> (2 * k)) & 0b11
    codes = codes[:, :n_samples]

    return _CODE_TO_A1[codes].T


def count_lines(path: PathLike) -> int:
    """Number of lines in a .bim or .fam file."""
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def read_fam_ids(prefix: PathLike) -> list:
    """Sample IIDs (column 2) from ``<prefix>.fam``, in file order."""
    with open(f"{prefix}.fam") as fh:
        return [line.split()[1] for line in fh if line.strip()]


def read_bim_variants(prefix: PathLike) -> Tuple[list, list]:
    """Variant IDs (column 2) and A1 alleles (column 5) from ``<prefix>.bim``."""
    ids, a1 = [], []
    with open(f"{prefix}.bim") as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.split()
            ids.append(parts[1])
            a1.append(parts[4])
    return ids, a1
