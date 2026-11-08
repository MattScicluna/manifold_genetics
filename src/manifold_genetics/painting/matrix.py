"""Pure functions that turn pbwt painting output into PCA-ready features.

``pbwt -paint <root>`` writes ``<root>.chunkcounts.out`` and
``<root>.chunklengths.out`` (plus two region files this package does not read).
Verified against pbwtPaint.c at commit 50fd6be:

* The matrices are already **per individual**, not per haplotype. pbwt maps
  haplotype ``h`` to individual ``h // ploidy`` while it accumulates, so an
  N-sample VCF gives an N x N matrix. No haplotype aggregation is needed here.
* Rows are **recipients** and columns are **donors**. The header is
  ``RECIPIENT IND1 IND2 ...`` and each row starts with its own ``IND<i>`` label,
  space-separated.
* Sample IDs never appear. ``IND<i>`` is the i-th sample of the VCF, so the
  names come from ``pbwt -writeSamples``, which writes them in the same order.
* The diagonal is zero: a haplotype never copies from its own individual.
* ``chunklengths`` rows are rescaled by pbwt to sum to ``n_sites * ploidy``, so
  summed across chromosomes each chromosome is weighted by its number of sites.

Nothing here touches the filesystem except the two ``read_*`` functions.
"""

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "MATRIX_KINDS",
    "coancestry_features",
    "fill_self_copying",
    "log_transform",
    "read_paint_matrix",
    "read_pbwt_samples",
    "row_normalise",
    "sum_matrices",
]

PathLike = Union[str, Path]

#: The two dense matrices ``pbwt -paint`` writes that can feed the PCA.
MATRIX_KINDS = ("chunkcounts", "chunklengths")


def read_paint_matrix(path: PathLike) -> Tuple[List[str], np.ndarray]:
    """Read one ``<root>.chunkcounts.out`` or ``<root>.chunklengths.out``.

    Returns:
        ``(labels, matrix)``: the ``IND<i>`` row labels in file order and the
        square recipient-by-donor matrix.

    Raises:
        ValueError: the header is not ``RECIPIENT ...``, the matrix is not
            square, a row has the wrong number of values, or the row labels do
            not match the column labels in order.
    """
    path = Path(path)
    with open(path) as fh:
        header = fh.readline().split()
        if not header or header[0] != "RECIPIENT":
            raise ValueError(
                f"{path} does not look like pbwt -paint output: its header should start "
                f"with RECIPIENT, but starts with {header[:1]}"
            )
        columns = header[1:]
        labels: List[str] = []
        rows: List[List[float]] = []
        for line_number, line in enumerate(fh, start=2):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != len(columns) + 1:
                raise ValueError(
                    f"{path}:{line_number} has {len(parts) - 1} values; the header names "
                    f"{len(columns)} donors"
                )
            labels.append(parts[0])
            rows.append([float(v) for v in parts[1:]])

    if labels != columns:
        raise ValueError(
            f"{path} is not a square recipient-by-donor matrix: its row labels do not "
            "match its column labels in order"
        )
    matrix = np.asarray(rows, dtype=np.float64).reshape(len(labels), len(columns))
    return labels, matrix


def read_pbwt_samples(path: PathLike) -> List[str]:
    """Sample IDs from a ``pbwt -writeSamples`` file, in VCF order.

    Each line is the sample name, optionally followed by tab-separated
    ``POP:``/``MOTHER:``/``FATHER:`` fields that pbwt adds when it knows them.
    """
    ids = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                ids.append(line.split("\t")[0].strip())
    return ids


def sum_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Element-wise sum of per-chromosome matrices of identical shape."""
    if not matrices:
        raise ValueError("No painting matrices to sum")
    shape = matrices[0].shape
    for i, m in enumerate(matrices):
        if m.shape != shape:
            raise ValueError(
                f"Painting matrix {i} has shape {m.shape}, the first has {shape}. Every "
                "chromosome must be painted on the same samples in the same order."
            )
    return np.sum(np.stack(matrices), axis=0)


def fill_self_copying(matrix: np.ndarray) -> np.ndarray:
    """Replace the structurally zero diagonal with each row's off-diagonal mean.

    pbwt never lets an individual copy from itself, so the diagonal is zero for
    a reason that has nothing to do with ancestry. Left in place, that zero is a
    feature only one sample has, which PCA would spend variance on. The row's
    own mean is the most neutral stand-in: it adds nothing to the row's profile.
    """
    m = np.array(matrix, dtype=np.float64, copy=True)
    n = m.shape[0]
    if n < 2:
        raise ValueError("Painting needs at least two samples")
    np.fill_diagonal(m, 0.0)
    np.fill_diagonal(m, m.sum(axis=1) / (n - 1))
    return m


def row_normalise(matrix: np.ndarray, labels: Sequence[str] = ()) -> np.ndarray:
    """Scale each row to sum to one.

    Rows then describe *how* a sample's genome is copied rather than how much of
    it was painted, so a sample with more matched sites does not sit farther out.

    Raises:
        ValueError: a row sums to zero (the sample copied from nobody, which
            usually means its genotypes were all missing). Named, because a
            silent row of NaNs would surface much later as a PCA failure.
    """
    totals = matrix.sum(axis=1)
    empty = np.flatnonzero(totals <= 0)
    if empty.size:
        names = [labels[i] if i < len(labels) else f"row {i}" for i in empty[:5]]
        raise ValueError(
            f"{empty.size} sample(s) copied from no donor in the painting, for example "
            f"{', '.join(names)}. Check their genotypes in the phased VCF."
        )
    return matrix / totals[:, None]


def log_transform(proportions: np.ndarray) -> np.ndarray:
    """``log(1 + n * p)`` for row-normalised proportions ``p`` over ``n`` donors.

    Copying is heavy-tailed: a few close relatives can dominate a row and with
    it the leading components. The log compresses that tail. Multiplying by
    ``n`` first makes uniform copying equal to one, so the transform does not
    change shape with cohort size, and the ``1 +`` keeps zeros finite.
    """
    n = proportions.shape[1]
    return np.log1p(proportions * n)


def coancestry_features(
    matrix: np.ndarray, *, log: bool = True, labels: Sequence[str] = ()
) -> np.ndarray:
    """Summed painting matrix to the features the PCA is fitted on.

    Fills the self-copying diagonal, row-normalises, then optionally logs.
    """
    features = row_normalise(fill_self_copying(matrix), labels)
    return log_transform(features) if log else features
