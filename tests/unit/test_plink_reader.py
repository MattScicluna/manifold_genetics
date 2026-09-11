"""Unit tests for the pure-Python PLINK .bed reader.

The byte layouts here are written by hand so the expected dosages can be read
off the test itself rather than trusted from a library. PLINK 1 .bed is
SNP-major: a 3-byte magic header, then ceil(n_samples / 4) bytes per variant,
two bits per sample, with sample 4i+k in bits [2k+1 : 2k] of byte i.

Codes are 00 = homozygous A1, 01 = missing, 10 = heterozygous, 11 = homozygous
A2. The dosage the pipeline needs is the **count of A1** -- verified against
flashpca's own ``.meansd`` output over all 172,152 HGDP variants, where counting
A1 matched flashpca's Mean to 5e-7 and counting A2 was wrong by up to 1.97.
"""

import numpy as np
import pytest

from manifold_genetics.pca.plink import PlinkFormatError, read_bed_dosages


def write_bed(tmp_path, variant_bytes, magic=b"\x6c\x1b\x01", name="t"):
    """Write a .bed file whose per-variant bytes are given literally."""
    path = tmp_path / f"{name}.bed"
    path.write_bytes(magic + bytes(variant_bytes))
    return tmp_path / name


# Variant 0: samples [hom A1, het, hom A2, missing] -> codes 00,10,11,01
#   byte = 01_11_10_00 = 0x78, A1 dosages [2, 1, 0, NaN]
# Variant 1: samples [hom A2, hom A2, het, hom A1] -> codes 11,11,10,00
#   byte = 00_10_11_11 = 0x2F, A1 dosages [0, 0, 1, 2]
TWO_VARIANTS = [0x78, 0x2F]


class TestReadBedDosages:
    def test_decodes_all_four_genotype_codes_as_a1_dosage(self, tmp_path):
        prefix = write_bed(tmp_path, TWO_VARIANTS)

        dosages = read_bed_dosages(prefix, n_samples=4, n_variants=2)

        expected = np.array(
            [
                [2.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [np.nan, 2.0],
            ]
        )
        np.testing.assert_array_equal(dosages, expected)

    def test_returns_samples_by_variants(self, tmp_path):
        prefix = write_bed(tmp_path, TWO_VARIANTS)

        dosages = read_bed_dosages(prefix, n_samples=4, n_variants=2)

        assert dosages.shape == (4, 2)

    def test_ignores_padding_bits_when_samples_do_not_fill_a_byte(self, tmp_path):
        # 3 samples still occupy a whole byte; the top 2 bits are padding and
        # PLINK sets them to zero, which is the code for "homozygous A1". Read
        # naively that invents a fourth sample with dosage 2.
        prefix = write_bed(tmp_path, [0b00_11_10_00])

        dosages = read_bed_dosages(prefix, n_samples=3, n_variants=1)

        assert dosages.shape == (3, 1)
        np.testing.assert_array_equal(dosages.ravel(), [2.0, 1.0, 0.0])

    def test_reads_a_slice_of_variants_without_reading_the_rest(self, tmp_path):
        prefix = write_bed(tmp_path, TWO_VARIANTS)

        dosages = read_bed_dosages(prefix, n_samples=4, n_variants=2, variants=slice(1, 2))

        assert dosages.shape == (4, 1)
        np.testing.assert_array_equal(dosages.ravel(), [0.0, 0.0, 1.0, 2.0])

    def test_rejects_individual_major_files(self, tmp_path):
        # Third magic byte 0x00 means individual-major. Decoding it as SNP-major
        # silently transposes the matrix rather than failing.
        prefix = write_bed(tmp_path, TWO_VARIANTS, magic=b"\x6c\x1b\x00")

        with pytest.raises(PlinkFormatError, match="individual-major"):
            read_bed_dosages(prefix, n_samples=4, n_variants=2)

    def test_rejects_a_file_without_the_plink_magic_number(self, tmp_path):
        prefix = write_bed(tmp_path, TWO_VARIANTS, magic=b"\x00\x00\x01")

        with pytest.raises(PlinkFormatError, match="magic"):
            read_bed_dosages(prefix, n_samples=4, n_variants=2)

    def test_rejects_a_truncated_file(self, tmp_path):
        # Declaring more variants than the file holds must fail loudly; numpy
        # would otherwise hand back a short buffer and the reshape would be wrong.
        prefix = write_bed(tmp_path, TWO_VARIANTS)

        with pytest.raises(PlinkFormatError, match="truncated"):
            read_bed_dosages(prefix, n_samples=4, n_variants=5)
