"""Tests for utils/filter_duplicates.py.

The functions under test parse PLINK-format text files and apply deduplication
logic. These tests are intentionally strict about format variants because real
PLINK1 and PLINK2 runs produce subtly different output that the current code
does not always handle correctly.

Bugs exposed by this suite:
- read_lmiss: per-cluster format (6 cols) shifts column index, reads N_GENO as
  F_MISS — silently wrong missingness values.
- read_frq: PLINK2 uses '.' for missing MAF, not 'NA' — current code crashes.
- filter_snps: chr-prefix vs numeric chromosome names are treated as distinct
  positions, so 'chr1:100' and '1:100' are never deduplicated.
"""

import textwrap
from pathlib import Path

import pytest

from manifold_genetics.utils.filter_duplicates import (
    filter_snps,
    read_bim,
    read_frq,
    read_lmiss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bim(tmp_path: Path, rows: list[tuple]) -> Path:
    """Write a PLINK1-format .bim file. rows = [(chr, id, pos, a1, a2), ...]."""
    lines = [f"{r[0]} {r[1]} 0 {r[2]} {r[3]} {r[4]}" for r in rows]
    p = tmp_path / "test.bim"
    p.write_text("\n".join(lines) + "\n")
    return p


def _lmiss_standard(tmp_path: Path, rows: list[tuple]) -> Path:
    """Write a standard 5-col PLINK1 .lmiss file. rows = [(snp, f_miss), ...]."""
    lines = ["CHR SNP N_MISS N_GENO F_MISS"]
    for snp, f_miss in rows:
        lines.append(f"1 {snp} {int(f_miss * 1000)} 1000 {f_miss}")
    p = tmp_path / "test.lmiss"
    p.write_text("\n".join(lines) + "\n")
    return p


def _lmiss_per_cluster(tmp_path: Path, rows: list[tuple]) -> Path:
    """Write a 6-col per-cluster .lmiss file (produced by --within --missing).
    rows = [(snp, cluster, f_miss), ...].
    """
    lines = ["CHR SNP CLST N_MISS N_GENO F_MISS"]
    for snp, clst, f_miss in rows:
        lines.append(f"1 {snp} {clst} {int(f_miss * 1000)} 1000 {f_miss}")
    p = tmp_path / "test_cluster.lmiss"
    p.write_text("\n".join(lines) + "\n")
    return p


def _frq_plink1(tmp_path: Path, rows: list[tuple], filename="test.frq") -> Path:
    """Write a standard PLINK1 .frq file. rows = [(snp, maf), ...]."""
    lines = [" CHR SNP A1 A2 MAF NCHROBS"]
    for snp, maf in rows:
        lines.append(f" 1 {snp} A T {maf} 1000")
    p = tmp_path / filename
    p.write_text("\n".join(lines) + "\n")
    return p


def _frq_plink2(tmp_path: Path, rows: list[tuple], filename="test.afreq") -> Path:
    """Write a PLINK2 .afreq file. rows = [(snp, alt_freq), ...].
    Use '.' to represent missing frequency (PLINK2 convention).
    """
    lines = ["#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT"]
    for snp, maf in rows:
        lines.append(f"1\t{snp}\tA\tT\t{maf}\t1000")
    p = tmp_path / filename
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# TestReadBim
# ---------------------------------------------------------------------------

class TestReadBim:
    """read_bim() against real PLINK .bim format variants."""

    def test_standard_space_delimited(self, tmp_path):
        """Standard PLINK1 .bim: CHR SNP CM POS A1 A2, space-delimited, no header."""
        bim = _bim(tmp_path, [("1", "rs001", 100, "A", "T"), ("2", "rs002", 200, "C", "G")])
        result = read_bim(bim)
        assert result["rs001"] == ("1", 100, "A", "T")
        assert result["rs002"] == ("2", 200, "C", "G")

    def test_tab_delimited(self, tmp_path):
        """Tab-delimited .bim files are equally valid — PLINK uses both."""
        bim = tmp_path / "tab.bim"
        bim.write_text("1\trs001\t0\t100\tA\tT\n1\trs002\t0\t200\tC\tG\n")
        result = read_bim(bim)
        assert result["rs001"] == ("1", 100, "A", "T")

    def test_chr_prefix_stored_as_string(self, tmp_path):
        """Chromosome names are stored verbatim — 'chr1' is not normalised to '1'."""
        bim = _bim(tmp_path, [("chr1", "rs001", 100, "A", "T"), ("chr2", "rs002", 200, "C", "G")])
        result = read_bim(bim)
        assert result["rs001"][0] == "chr1"
        assert result["rs002"][0] == "chr2"

    def test_multi_character_alleles(self, tmp_path):
        """Indels have multi-char allele strings — parser must not truncate them."""
        bim = tmp_path / "indel.bim"
        bim.write_text("1 rs_ins 0 500 ATG A\n1 rs_del 0 600 CCCCC C\n")
        result = read_bim(bim)
        assert result["rs_ins"] == ("1", 500, "ATG", "A")
        assert result["rs_del"] == ("1", 600, "CCCCC", "C")

    def test_unmapped_snps_chr0(self, tmp_path):
        """Chromosome 0 means unmapped — must be included without error."""
        bim = _bim(tmp_path, [("0", "rs_unmapped", 0, "A", "T"), ("1", "rs001", 100, "A", "T")])
        result = read_bim(bim)
        assert "rs_unmapped" in result
        assert result["rs_unmapped"][0] == "0"

    def test_sex_chromosomes(self, tmp_path):
        """X, Y, XY, MT are valid chromosome designations."""
        bim = tmp_path / "sex.bim"
        bim.write_text(
            "X rs_x 0 100 A T\n"
            "Y rs_y 0 200 C G\n"
            "MT rs_mt 0 300 A C\n"
        )
        result = read_bim(bim)
        assert result["rs_x"][0] == "X"
        assert result["rs_y"][0] == "Y"
        assert result["rs_mt"][0] == "MT"

    def test_duplicate_snp_ids_last_wins(self, tmp_path):
        """When two rows share a SNP ID (e.g., PLINK's '.' convention), last entry wins."""
        bim = tmp_path / "dup_id.bim"
        bim.write_text("1 rs001 0 100 A T\n2 rs001 0 200 C G\n")
        result = read_bim(bim)
        assert "rs001" in result
        # Only one entry survives — the second overwrites the first
        assert len(result) == 1

    def test_nonstandard_cm_position(self, tmp_path):
        """cM position is stored but not used — can be any float, including non-zero."""
        bim = tmp_path / "cm.bim"
        bim.write_text("1 rs001 3.14159 100 A T\n")
        result = read_bim(bim)
        assert "rs001" in result  # cM column is parsed and discarded


# ---------------------------------------------------------------------------
# TestReadLmiss
# ---------------------------------------------------------------------------

class TestReadLmiss:
    """read_lmiss() against PLINK1 format variants.

    PLINK1 --missing produces two variants:
      standard (5-col):     CHR SNP N_MISS N_GENO F_MISS
      per-cluster (6-col):  CHR SNP CLST N_MISS N_GENO F_MISS

    The current implementation hard-codes parts[4] as F_MISS, which is correct
    for 5-col but silently reads N_GENO for 6-col.
    """

    def test_standard_5col_format(self, tmp_path):
        """Standard --missing output: parts[4] correctly maps to F_MISS."""
        lmiss = _lmiss_standard(tmp_path, [("rs001", 0.005), ("rs002", 0.0)])
        result = read_lmiss(lmiss)
        assert result["rs001"] == pytest.approx(0.005)
        assert result["rs002"] == pytest.approx(0.0)

    def test_high_missingness_snp(self, tmp_path):
        """SNP with 100% missingness must not be silently clamped."""
        lmiss = _lmiss_standard(tmp_path, [("rs_all_missing", 1.0)])
        result = read_lmiss(lmiss)
        assert result["rs_all_missing"] == pytest.approx(1.0)

    def test_per_cluster_format_exposes_column_shift_bug(self, tmp_path):
        """BUG: 6-col per-cluster .lmiss causes read_lmiss to read N_GENO as F_MISS.

        --within --missing produces: CHR SNP CLST N_MISS N_GENO F_MISS
        The code does parts[4] = N_GENO (an integer like 1000) instead of
        parts[5] = F_MISS (0.005). This silently corrupts missingness rates.

        Expected behaviour: read_lmiss detects 6-col header and adjusts index.
        Current behaviour: reads N_GENO as the missingness rate.
        """
        lmiss = _lmiss_per_cluster(
            tmp_path,
            [("rs001", "POP1", 0.005), ("rs002", "POP1", 0.050)],
        )
        result = read_lmiss(lmiss)
        assert result["rs001"] == pytest.approx(0.005), (
            "read_lmiss reads N_GENO (1000) as F_MISS for per-cluster .lmiss files. "
            "Fix: inspect header column count and adjust F_MISS index accordingly."
        )
        assert result["rs002"] == pytest.approx(0.050)

    def test_absent_snp_not_in_result(self, tmp_path):
        """SNPs absent from .lmiss are simply not in the dict — caller handles default."""
        lmiss = _lmiss_standard(tmp_path, [("rs001", 0.01)])
        result = read_lmiss(lmiss)
        assert "rs_not_present" not in result

    def test_whitespace_variations(self, tmp_path):
        """PLINK .lmiss output uses variable-width padding — must parse with split()."""
        lmiss = tmp_path / "padded.lmiss"
        lmiss.write_text(
            " CHR          SNP   N_MISS   N_GENO   F_MISS\n"
            "   1    rs001        5     1000   0.005\n"
            "   1    rs002        0     1000   0.000\n"
        )
        result = read_lmiss(lmiss)
        assert result["rs001"] == pytest.approx(0.005)
        assert result["rs002"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestReadFrq
# ---------------------------------------------------------------------------

class TestReadFrq:
    """read_frq() against PLINK1 .frq and PLINK2 .afreq format variants."""

    def test_standard_plink1_frq(self, tmp_path):
        """Standard PLINK1 --freq: CHR SNP A1 A2 MAF NCHROBS."""
        frq = _frq_plink1(tmp_path, [("rs001", 0.234), ("rs002", 0.100)])
        result = read_frq(frq)
        assert result["rs001"] == pytest.approx(0.234)
        assert result["rs002"] == pytest.approx(0.100)

    def test_na_maf_plink1_treated_as_zero(self, tmp_path):
        """PLINK1 uses 'NA' for monomorphic or missing MAF — treated as 0.0."""
        frq = tmp_path / "na.frq"
        frq.write_text(" CHR SNP A1 A2 MAF NCHROBS\n 1 rs001 A T NA 1000\n")
        result = read_frq(frq)
        assert result["rs001"] == pytest.approx(0.0)

    def test_plink2_afreq_format_column_positions(self, tmp_path):
        """PLINK2 --freq (.afreq): #CHROM ID REF ALT ALT_FREQS OBS_CT.

        Column 2 = ID, column 5 = ALT_FREQS — same offsets as PLINK1,
        so the current code accidentally works for PLINK2 when MAF is present.
        """
        frq = _frq_plink2(tmp_path, [("rs001", "0.234"), ("rs002", "0.100")])
        result = read_frq(frq)
        assert result["rs001"] == pytest.approx(0.234)
        assert result["rs002"] == pytest.approx(0.100)

    def test_plink2_dot_missing_crashes(self, tmp_path):
        """BUG: PLINK2 uses '.' for missing MAF, not 'NA'. float('.') raises ValueError.

        Any monomorphic SNP in a PLINK2 .afreq file will crash read_frq.
        Fix: extend the missing-value check to cover both 'NA' and '.'.
        """
        frq = tmp_path / "dot_missing.afreq"
        frq.write_text(
            "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
            "1\trs_mono\tA\tT\t.\t1000\n"
        )
        with pytest.raises(ValueError, match=r"could not convert string to float: '\.'"):
            read_frq(frq)

    def test_plink2_multiallelic_comma_separated_crashes(self, tmp_path):
        """BUG: PLINK2 can emit comma-separated ALT_FREQS for multiallelic sites.

        e.g., '0.234,0.050' — float('0.234,0.050') raises ValueError.
        The current code has no handling for multiallelic sites.
        """
        frq = tmp_path / "multi.afreq"
        frq.write_text(
            "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
            "1\trs_multi\tA\tT,G\t0.234,0.050\t1000\n"
        )
        with pytest.raises(ValueError):
            read_frq(frq)

    def test_absent_snp_not_in_result(self, tmp_path):
        """SNPs absent from .frq are simply not in the dict."""
        frq = _frq_plink1(tmp_path, [("rs001", 0.3)])
        result = read_frq(frq)
        assert "rs_absent" not in result


# ---------------------------------------------------------------------------
# TestFilterSnps
# ---------------------------------------------------------------------------

class TestFilterSnps:
    """filter_snps() core deduplication logic."""

    def test_no_duplicates_all_kept(self):
        snp_info = {
            "rs001": ("1", 100, "A", "T"),
            "rs002": ("1", 200, "C", "G"),
            "rs003": ("2", 100, "A", "T"),
        }
        keep, stats = filter_snps(snp_info, {}, {})
        assert set(keep) == {"rs001", "rs002", "rs003"}
        assert stats["snps_removed"] == 0
        assert stats["duplicate_positions"] == 0

    def test_prefers_lower_missingness(self):
        """Duplicate position: keep SNP with lower F_MISS."""
        snp_info = {
            "rs_bad":  ("1", 100, "A", "T"),
            "rs_good": ("1", 100, "C", "G"),
        }
        missingness = {"rs_bad": 0.5, "rs_good": 0.01}
        maf = {"rs_bad": 0.3, "rs_good": 0.3}
        keep, stats = filter_snps(snp_info, missingness, maf)
        assert keep == ["rs_good"]
        assert stats["snps_removed"] == 1

    def test_prefers_higher_maf_when_missingness_tied(self):
        """Tied missingness: keep SNP with higher MAF."""
        snp_info = {
            "rs_rare":   ("1", 100, "A", "T"),
            "rs_common": ("1", 100, "C", "G"),
        }
        missingness = {"rs_rare": 0.01, "rs_common": 0.01}
        maf = {"rs_rare": 0.05, "rs_common": 0.3}
        keep, stats = filter_snps(snp_info, missingness, maf)
        assert keep == ["rs_common"]

    def test_snp_absent_from_lmiss_gets_max_missingness(self):
        """SNP missing from .lmiss defaults to 1.0 — always loses to a SNP with data."""
        snp_info = {
            "rs_no_miss_data": ("1", 100, "A", "T"),
            "rs_has_data":     ("1", 100, "C", "G"),
        }
        missingness = {"rs_has_data": 0.01}
        maf = {"rs_no_miss_data": 0.9, "rs_has_data": 0.1}  # no_miss_data has better MAF
        keep, stats = filter_snps(snp_info, missingness, maf)
        # rs_has_data wins: its explicit 0.01 beats the implicit 1.0 for rs_no_miss_data
        assert keep == ["rs_has_data"]

    def test_snp_absent_from_frq_gets_zero_maf(self):
        """SNP missing from .frq defaults to 0.0 MAF — loses tiebreaker."""
        snp_info = {
            "rs_no_frq":  ("1", 100, "A", "T"),
            "rs_has_frq": ("1", 100, "C", "G"),
        }
        missingness = {"rs_no_frq": 0.01, "rs_has_frq": 0.01}  # tied
        maf = {"rs_has_frq": 0.3}  # rs_no_frq absent → 0.0
        keep, stats = filter_snps(snp_info, missingness, maf)
        assert keep == ["rs_has_frq"]

    def test_three_snps_same_position_lowest_missingness_wins(self):
        snp_info = {
            "rs_a": ("1", 100, "A", "T"),
            "rs_b": ("1", 100, "C", "G"),
            "rs_c": ("1", 100, "T", "A"),
        }
        missingness = {"rs_a": 0.10, "rs_b": 0.10, "rs_c": 0.01}
        maf = {"rs_a": 0.3, "rs_b": 0.4, "rs_c": 0.2}
        keep, stats = filter_snps(snp_info, missingness, maf)
        assert keep == ["rs_c"]
        assert stats["snps_removed"] == 2
        assert stats["snps_at_duplicate_positions"] == 3

    def test_all_snps_same_position_one_kept(self):
        """Extreme case: 10 SNPs at same position → exactly 1 kept."""
        snp_info = {f"rs{i:03d}": ("1", 100, "A", "T") for i in range(10)}
        keep, stats = filter_snps(snp_info, {}, {})
        assert len(keep) == 1
        assert stats["snps_removed"] == 9
        assert stats["snps_kept"] == 1

    def test_chr_prefix_vs_numeric_not_deduplicated(self):
        """BUG EXPOSURE: 'chr1' and '1' produce different position keys.

        SNPs at chr1:100 and 1:100 are NOT considered duplicates by the
        current implementation, even if they are the same genomic position.
        If your data mixes chr-prefixed and numeric chromosome names, real
        duplicates will be silently missed.
        """
        snp_info = {
            "rs_chr_prefix": ("chr1", 100, "A", "T"),
            "rs_numeric":    ("1",    100, "C", "G"),
        }
        keep, stats = filter_snps(snp_info, {}, {})
        # Both are kept — chr1:100 ≠ 1:100 in current code
        assert set(keep) == {"rs_chr_prefix", "rs_numeric"}, (
            "chr1:100 and 1:100 are treated as different positions. "
            "Mixed chromosome naming conventions will silently miss true duplicates."
        )
        assert stats["duplicate_positions"] == 0

    def test_stats_counts_are_accurate(self):
        snp_info = {
            "rs001": ("1", 100, "A", "T"),
            "rs002": ("1", 100, "C", "G"),
            "rs003": ("2", 200, "T", "A"),
        }
        missingness = {"rs001": 0.1, "rs002": 0.01}
        keep, stats = filter_snps(snp_info, missingness, {})
        assert stats["total_snps"] == 3
        assert stats["unique_positions"] == 2
        assert stats["duplicate_positions"] == 1
        assert stats["snps_at_duplicate_positions"] == 2
        assert stats["snps_removed"] == 1
        assert stats["snps_kept"] == 2
        assert len(keep) == stats["snps_kept"]

    def test_empty_input(self):
        keep, stats = filter_snps({}, {}, {})
        assert keep == []
        assert stats["total_snps"] == 0
        assert stats["snps_removed"] == 0


# ---------------------------------------------------------------------------
# TestFilterSnpsRoundTrip — write real PLINK-format files, then parse + filter
# ---------------------------------------------------------------------------

class TestFilterSnpsRoundTrip:
    """End-to-end: write PLINK text files → read → filter → verify results."""

    def test_basic_round_trip(self, tmp_path):
        bim = _bim(tmp_path, [
            ("1", "rs001", 100, "A", "T"),
            ("1", "rs002", 100, "C", "G"),  # duplicate of rs001 pos
            ("1", "rs003", 200, "A", "T"),  # unique
        ])
        lmiss = _lmiss_standard(tmp_path, [
            ("rs001", 0.10), ("rs002", 0.02), ("rs003", 0.01),
        ])
        frq = _frq_plink1(tmp_path, [
            ("rs001", 0.3), ("rs002", 0.4), ("rs003", 0.2),
        ])
        snp_info = read_bim(bim)
        missingness = read_lmiss(lmiss)
        maf = read_frq(frq)
        keep, stats = filter_snps(snp_info, missingness, maf)

        assert "rs002" in keep   # lower missingness at chr1:100
        assert "rs001" not in keep
        assert "rs003" in keep
        assert stats["snps_removed"] == 1

    def test_plink1_frq_na_round_trip(self, tmp_path):
        """Monomorphic SNP with NA MAF is handled throughout the pipeline."""
        bim = _bim(tmp_path, [("1", "rs_mono", 100, "A", "T"), ("1", "rs_poly", 100, "C", "G")])
        lmiss = _lmiss_standard(tmp_path, [("rs_mono", 0.01), ("rs_poly", 0.01)])
        frq = tmp_path / "mono.frq"
        frq.write_text(" CHR SNP A1 A2 MAF NCHROBS\n 1 rs_mono A T NA 1000\n 1 rs_poly C G 0.3 1000\n")

        keep, _ = filter_snps(read_bim(bim), read_lmiss(lmiss), read_frq(frq))
        # Equal missingness: rs_poly wins (MAF 0.3 > 0.0)
        assert keep == ["rs_poly"]

    def test_plink2_afreq_round_trip(self, tmp_path):
        """PLINK2-format .afreq file works for non-missing values."""
        bim = _bim(tmp_path, [("1", "rs001", 100, "A", "T"), ("1", "rs002", 200, "C", "G")])
        lmiss = _lmiss_standard(tmp_path, [("rs001", 0.01), ("rs002", 0.02)])
        frq = _frq_plink2(tmp_path, [("rs001", "0.3"), ("rs002", "0.4")])

        keep, stats = filter_snps(read_bim(bim), read_lmiss(lmiss), read_frq(frq))
        assert set(keep) == {"rs001", "rs002"}
        assert stats["snps_removed"] == 0
