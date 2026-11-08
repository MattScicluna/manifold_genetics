"""Haplotype painting with pbwt, as an alternative input to the embedding.

``matrix`` holds the pure parsing and feature functions, ``pbwt`` runs the
external binary, and ``reduce`` turns the coancestry matrix into coordinates.
"""

from .matrix import (
    CHROMOSOME_CM,
    MATRIX_FOR_METHOD,
    METHODS,
    build_features,
    centimorgan_weight,
    chromosome_weight,
    features_from_sum,
    finestructure_features,
    hc_features,
    method_matrix,
    read_paint_matrix,
    read_pbwt_samples,
    sum_matrices,
)
from .pbwt import expand_phased_vcf, paint_chromosome, parse_chromosomes
from .reduce import MISSING_POLICIES, match_cohort, painting_pca
from .vcfheader import read_vcf_samples

__all__ = [
    "CHROMOSOME_CM",
    "MATRIX_FOR_METHOD",
    "METHODS",
    "MISSING_POLICIES",
    "build_features",
    "centimorgan_weight",
    "chromosome_weight",
    "features_from_sum",
    "expand_phased_vcf",
    "finestructure_features",
    "hc_features",
    "match_cohort",
    "method_matrix",
    "paint_chromosome",
    "painting_pca",
    "parse_chromosomes",
    "read_paint_matrix",
    "read_pbwt_samples",
    "read_vcf_samples",
    "sum_matrices",
]
