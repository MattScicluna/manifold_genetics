"""CLI-glue tests for utils/filter_duplicates.main().

read_bim / read_lmiss / read_frq / filter_snps are covered by
test_filter_duplicates.py. These drive main(argv=...) end to end with real
tiny fixture files and cover the argument parsing, input validation, and
output-writing that only main() reaches.
"""

import pytest

from manifold_genetics.utils import filter_duplicates as fd


def _fixtures(tmp_path):
    """A dataset with one clean SNP and a duplicated position (1:200)."""
    bim = tmp_path / "d.bim"
    bim.write_text(
        "1\trs_clean\t0\t100\tA\tG\n"
        "1\trs_dupA\t0\t200\tA\tG\n"  # higher missingness -> dropped
        "1\trs_dupB\t0\t200\tC\tT\n"  # lower missingness -> kept
    )
    lmiss = tmp_path / "d.lmiss"
    lmiss.write_text(
        "CHR SNP N_MISS N_GENO F_MISS\n"
        "1 rs_clean 0 100 0.0\n"
        "1 rs_dupA 20 100 0.20\n"
        "1 rs_dupB 2 100 0.02\n"
    )
    frq = tmp_path / "d.frq"
    frq.write_text(
        "CHR SNP A1 A2 MAF NCHROBS\n"
        "1 rs_clean A G 0.30 200\n"
        "1 rs_dupA A G 0.40 200\n"
        "1 rs_dupB C T 0.10 200\n"
    )
    return bim, lmiss, frq


def test_main_writes_keep_and_stats_files(tmp_path, capsys):
    bim, lmiss, frq = _fixtures(tmp_path)
    out = tmp_path / "result"
    fd.main(["--bim", str(bim), "--lmiss", str(lmiss), "--frq", str(frq), "--out", str(out)])

    kept = (tmp_path / "result.keep_snps.txt").read_text().split()
    assert set(kept) == {"rs_clean", "rs_dupB"}  # lower-missingness dup wins

    stats = dict(
        line.split("\t") for line in (tmp_path / "result.stats.txt").read_text().splitlines()
    )
    assert stats["total_snps"] == "3"
    assert stats["duplicate_positions"] == "1"
    assert stats["snps_removed"] == "1"
    assert stats["snps_kept"] == "2"

    assert "SNPs kept: 2" in capsys.readouterr().out


def test_main_missing_input_file_exits_1(tmp_path, capsys):
    bim, lmiss, _ = _fixtures(tmp_path)
    with pytest.raises(SystemExit) as exc:
        fd.main(
            [
                "--bim",
                str(bim),
                "--lmiss",
                str(lmiss),
                "--frq",
                str(tmp_path / "does_not_exist.frq"),
                "--out",
                str(tmp_path / "r"),
            ]
        )
    assert exc.value.code == 1
    assert "File not found" in capsys.readouterr().err


def test_parse_args_requires_all_four(tmp_path):
    with pytest.raises(SystemExit) as exc:
        fd.parse_args(["--bim", "x.bim"])  # missing --lmiss/--frq/--out
    assert exc.value.code == 2


def test_main_plink2_afreq_and_multiallelic(tmp_path):
    bim = tmp_path / "d.bim"
    bim.write_text("1\trsA\t0\t10\tA\tG\n1\trsB\t0\t10\tC\tT\n")
    lmiss = tmp_path / "d.lmiss"
    lmiss.write_text("CHR SNP N_MISS N_GENO F_MISS\n1 rsA 0 10 0.0\n1 rsB 0 10 0.0\n")
    frq = tmp_path / "d.afreq"
    # PLINK2 header; rsB is multiallelic (comma) with a higher max freq -> kept on the MAF tiebreak
    frq.write_text(
        "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
        "1\trsA\tA\tG\t0.10\t20\n"
        "1\trsB\tC\tT,A\t0.05,0.45\t20\n"
    )
    out = tmp_path / "r"
    fd.main(["--bim", str(bim), "--lmiss", str(lmiss), "--frq", str(frq), "--out", str(out)])
    assert (tmp_path / "r.keep_snps.txt").read_text().strip() == "rsB"
