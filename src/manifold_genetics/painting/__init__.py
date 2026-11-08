"""Haplotype painting with pbwt, as an alternative input to the embedding.

``matrix`` holds the pure parsing and feature functions, ``pbwt`` runs the
external binary, and ``reduce`` turns the coancestry matrix into coordinates.
"""

from .matrix import (
    MATRIX_KINDS,
    coancestry_features,
    fill_self_copying,
    log_transform,
    read_paint_matrix,
    read_pbwt_samples,
    row_normalise,
    sum_matrices,
)
from .pbwt import expand_phased_vcf, paint_chromosome, parse_chromosomes
from .reduce import match_cohort, painting_pca

__all__ = [
    "MATRIX_KINDS",
    "coancestry_features",
    "expand_phased_vcf",
    "fill_self_copying",
    "log_transform",
    "match_cohort",
    "paint_chromosome",
    "painting_pca",
    "parse_chromosomes",
    "read_paint_matrix",
    "read_pbwt_samples",
    "row_normalise",
    "sum_matrices",
]
