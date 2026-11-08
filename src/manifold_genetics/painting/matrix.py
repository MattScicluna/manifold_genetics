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
* ``chunklengths`` rows are rescaled by pbwt to sum to ``n_sites * ploidy``.

The features follow one of two published recipes, chosen by ``method``:

``finestructure``
    Lawson et al. (2012), PLoS Genet 8:e1002453, as implemented by ``mypca`` in
    the fineSTRUCTURE R library (``FinestructureLibrary.R``). The chunk counts
    are summed over chromosomes, the zero diagonal is set to the row's
    off-diagonal mean, each row is centered on its mean, and the PCA is the
    eigendecomposition of ``X X^T`` (no column centering, no log).
``hc``
    The haplotype components of Yang, Durbin, Iversen and Lawson (2025), Nat
    Commun 16:2742, the recipe pbwt's README points to for ``-paint`` output.
    Each chromosome's chunk lengths are converted from sites to centimorgans
    (weight ``g_c / K_c``), summed, transformed by ``log10(A + 1)`` and reduced
    by an uncentered SVD. The components are ``U sqrt(D)``.

Nothing here touches the filesystem except the two ``read_*`` functions.
"""

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "CHROMOSOME_CM",
    "METHODS",
    "MATRIX_FOR_METHOD",
    "build_features",
    "centimorgan_weight",
    "check_rows",
    "chromosome_weight",
    "features_from_sum",
    "finestructure_features",
    "hc_features",
    "method_matrix",
    "read_paint_matrix",
    "read_pbwt_samples",
    "sum_matrices",
]

PathLike = Union[str, Path]

#: The two feature recipes, by name. Each fixes which pbwt matrix it reads.
MATRIX_FOR_METHOD: Dict[str, str] = {"finestructure": "chunkcounts", "hc": "chunklengths"}
METHODS = tuple(MATRIX_FOR_METHOD)

#: Genetic length of each autosome in centimorgans, as used for the UK Biobank
#: haplotype components (SparsePainter painting-pipeline,
#: ``combine_chunklength.cpp``, commit e1d07b7).
CHROMOSOME_CM: Dict[str, float] = dict(
    zip(
        (str(c) for c in range(1, 23)),
        (
            286.28, 268.83, 223.26, 214.54, 204.09, 192.03, 187.16, 168.00,
            166.35, 180.91, 158.22, 174.68, 125.70, 120.20, 141.35, 134.05,
            128.78, 117.71, 107.73, 108.28, 62.79, 74.10,
        ),
    )
)  # fmt: skip


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
        n = len(columns)
        not_square = ValueError(
            f"{path} is not a square recipient-by-donor matrix: its row labels do not "
            "match its column labels in order"
        )
        # Filled row by row, so parsing holds one N x N array rather than
        # N x N Python floats, which cost several times as much.
        matrix = np.empty((n, n), dtype=np.float64)
        labels: List[str] = []
        for line_number, line in enumerate(fh, start=2):
            parts = line.split(None, 1)
            if not parts:
                continue
            values = np.array(parts[1].split() if len(parts) > 1 else [], dtype=np.float64)
            if values.size != n:
                raise ValueError(
                    f"{path}:{line_number} has {values.size} values; the header names "
                    f"{n} donors"
                )
            if len(labels) >= n:
                raise not_square
            matrix[len(labels)] = values
            labels.append(parts[0])

    if labels != columns:
        raise not_square
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


def sum_matrices(matrices: Sequence[np.ndarray], weights: Sequence[float] = ()) -> np.ndarray:
    """Element-wise sum of per-chromosome matrices of identical shape.

    ``weights``, when given, scales each matrix first (one weight per matrix).
    """
    if not matrices:
        raise ValueError("No painting matrices to sum")
    if weights and len(weights) != len(matrices):
        raise ValueError(f"{len(weights)} weights for {len(matrices)} painting matrices")
    shape = matrices[0].shape
    total = np.zeros(shape, dtype=np.float64)
    for i, m in enumerate(matrices):
        if m.shape != shape:
            raise ValueError(
                f"Painting matrix {i} has shape {m.shape}, the first has {shape}. Every "
                "chromosome must be painted on the same samples in the same order."
            )
        total += m * (weights[i] if weights else 1.0)
    return total


def centimorgan_weight(chunklengths: np.ndarray, chromosome: str, ploidy: int = 2) -> float:
    """Weight that converts one chromosome's chunk lengths from sites to centimorgans.

    This is ``g_c / K_c`` of Yang et al. (2025): the chromosome's genetic
    length over its number of sites. pbwt scales every ``chunklengths`` row to
    sum to ``K_c * ploidy``, so ``K_c`` is read off the matrix itself.

    Raises:
        ValueError: the chromosome has no genetic length in ``CHROMOSOME_CM``
            (only autosomes 1 to 22 do), or the matrix has no copying at all.
    """
    name = str(chromosome)
    for prefix in ("chr", "CHR", "Chr"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if name not in CHROMOSOME_CM:
        raise ValueError(
            f"No genetic length is known for chromosome {chromosome!r}, so its chunk "
            "lengths cannot be put in centimorgans. The hc method covers autosomes 1-22, so "
            "use method: finestructure for any other chromosome."
        )
    n_sites = float(np.mean(chunklengths.sum(axis=1))) / ploidy
    if n_sites <= 0:
        raise ValueError(f"The chunk lengths of chromosome {chromosome!r} are all zero")
    return CHROMOSOME_CM[name] / n_sites


def finestructure_features(counts: np.ndarray) -> np.ndarray:
    """fineSTRUCTURE's normalization of a chunk count matrix before PCA.

    Follows ``mypcanorm`` in ``FinestructureLibrary.R``: "setting the diagonal
    to be the average of the rows, then zero meaning by substracting the row
    mean (so the diagonal ends up as zero)". The PCA is then taken on
    ``X X^T`` without centering the columns.
    """
    x = np.array(counts, dtype=np.float64, copy=True)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Painting needs at least two samples")
    np.fill_diagonal(x, x.sum(axis=1) / (n - 1))
    return x - x.mean(axis=1, keepdims=True)


def hc_features(lengths_cm: np.ndarray) -> np.ndarray:
    """``log10(A + 1)`` of the summed chunk lengths in centimorgans (Yang et al. 2025).

    The diagonal stays zero, as in the sparse matrix the paper decomposes.
    """
    a = np.asarray(lengths_cm, dtype=np.float64)
    if (a < 0).any():
        raise ValueError("Chunk lengths cannot be negative")
    return np.log10(a + 1.0)


def method_matrix(method: str) -> str:
    """The pbwt matrix a method reads. Raises ``ValueError`` for an unknown method."""
    try:
        return MATRIX_FOR_METHOD[method]
    except KeyError:
        raise ValueError(
            f"Unknown painting method: {method!r}. Choose from: {', '.join(METHODS)}"
        ) from None


def check_rows(matrix: np.ndarray, labels: Sequence[str] = ()) -> None:
    """Raise if a sample copied from nobody, naming a few such samples.

    An all-zero row usually means the sample's genotypes were all missing. It
    is named here, because it would otherwise surface much later as an odd
    point in the PCA.
    """
    empty = np.flatnonzero(np.asarray(matrix).sum(axis=1) <= 0)
    if empty.size:
        names = [labels[i] if i < len(labels) else f"row {i}" for i in empty[:5]]
        raise ValueError(
            f"{empty.size} sample(s) copied from no donor in the painting, for example "
            f"{', '.join(names)}. Check their genotypes in the phased VCF."
        )


def chromosome_weight(method: str, matrix: np.ndarray, chromosome: str) -> float:
    """What one chromosome's matrix is multiplied by before the sum over chromosomes.

    One for ``finestructure``, which sums chunk counts as they are, and the
    centimorgan weight ``g_c / K_c`` for ``hc``.
    """
    method_matrix(method)
    return centimorgan_weight(matrix, chromosome) if method == "hc" else 1.0


def features_from_sum(summed: np.ndarray, method: str, labels: Sequence[str] = ()) -> np.ndarray:
    """The PCA features of a matrix already weighted and summed over chromosomes."""
    method_matrix(method)
    check_rows(summed, labels)
    return hc_features(summed) if method == "hc" else finestructure_features(summed)


def build_features(
    matrices: Mapping[str, np.ndarray], method: str, labels: Sequence[str] = ()
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-chromosome pbwt matrices to ``(summed, features)`` for ``method``.

    ``matrices`` maps chromosome to the matrix ``method_matrix(method)`` names.
    ``summed`` is the matrix written as the coancestry CSV: chunk counts for
    ``finestructure`` and chunk lengths in centimorgans for ``hc``. The
    pipeline sums one chromosome at a time instead, so that it never holds
    more than two N x N matrices while summing.
    """
    weights = [chromosome_weight(method, m, c) for c, m in matrices.items()]
    summed = sum_matrices(list(matrices.values()), weights)
    return summed, features_from_sum(summed, method, labels)
