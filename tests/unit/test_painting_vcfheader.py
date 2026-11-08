"""Reading sample names from VCF and BCF headers without htslib."""

import gzip
import struct

import pytest

from manifold_genetics.painting.vcfheader import read_vcf_samples

SAMPLES = ["HG00096", "NA12878", "LP6005441-DNA_F08"]

HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=22,length=50818468>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(SAMPLES) + "\n"
)
RECORD = "22\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|1\t0|0\n"


def _bcf(header: str) -> bytes:
    text = header.encode() + b"\0"
    # A record follows the header in a real file; its bytes must not be read.
    return b"BCF\x02\x02" + struct.pack("<I", len(text)) + text + b"\x01\x02\x03\x04"


def test_plain_vcf(tmp_path):
    path = tmp_path / "chr22.vcf"
    path.write_text(HEADER + RECORD)
    assert read_vcf_samples(path) == SAMPLES


def test_gzipped_vcf_split_over_several_members(tmp_path):
    # BGZF is a series of gzip members, and the header may cross one.
    path = tmp_path / "chr22.vcf.gz"
    text = (HEADER + RECORD).encode()
    path.write_bytes(gzip.compress(text[:40]) + gzip.compress(text[40:]))
    assert read_vcf_samples(path) == SAMPLES


def test_compressed_bcf(tmp_path):
    path = tmp_path / "chr22.bcf"
    data = _bcf(HEADER)
    path.write_bytes(gzip.compress(data[:20]) + gzip.compress(data[20:]))
    assert read_vcf_samples(path) == SAMPLES


def test_uncompressed_bcf(tmp_path):
    path = tmp_path / "chr22.bcf"
    path.write_bytes(_bcf(HEADER))
    assert read_vcf_samples(path) == SAMPLES


def test_a_vcf_without_samples(tmp_path):
    path = tmp_path / "sites.vcf"
    path.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    assert read_vcf_samples(path) == []


def test_no_chrom_line_is_an_error(tmp_path):
    path = tmp_path / "bad.vcf"
    path.write_text("##fileformat=VCFv4.2\n" + RECORD)
    with pytest.raises(ValueError, match="no #CHROM header line"):
        read_vcf_samples(path)


def test_a_file_that_is_not_a_vcf(tmp_path):
    path = tmp_path / "chr22.bed"
    path.write_bytes(b"\x6c\x1b\x01\x00")
    with pytest.raises(ValueError, match="neither a VCF nor a BCF"):
        read_vcf_samples(path)


def test_bcf1_is_refused(tmp_path):
    path = tmp_path / "old.bcf"
    path.write_bytes(b"BCF\x04" + b"\0" * 8)
    with pytest.raises(ValueError, match="only BCF2"):
        read_vcf_samples(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Phased VCF not found"):
        read_vcf_samples(tmp_path / "chr1.vcf.gz")
