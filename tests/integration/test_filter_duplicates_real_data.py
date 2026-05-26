"""Integration tests for filter_duplicates.py using real HGDP+1KGP data.

These tests require:
  1. HGDP+1KGP data prepared at examples/hgdp_1kgp/data/
     (run examples/hgdp_1kgp/download_data.sh + prepare_data.sh)
  2. plink2 or plink in PATH (for generating .lmiss and .frq from real PLINK files)

The real dataset (4,151 samples, 172,152 SNPs) is large enough to contain
genuine duplicate positions and multi-allelic sites, making it a strong test
of the actual deduplication logic.

Run with: pytest -m integration tests/integration/test_filter_duplicates_real_data.py
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from manifold_genetics.utils.filter_duplicates import filter_snps, read_bim, read_frq, read_lmiss

# ---------------------------------------------------------------------------
# Fixtures / skip conditions
# ---------------------------------------------------------------------------

HGDP_DATA_DIR = Path(__file__).parent.parent.parent / "examples" / "hgdp_1kgp" / "data"
FIT_PLINK_PREFIX = HGDP_DATA_DIR / "fit_subset"


def hgdp_available() -> bool:
    return (
        FIT_PLINK_PREFIX.with_suffix(".bed").exists()
        and FIT_PLINK_PREFIX.with_suffix(".bim").exists()
        and FIT_PLINK_PREFIX.with_suffix(".fam").exists()
    )


def plink_available() -> bool:
    return bool(shutil.which("plink2") or shutil.which("plink"))


def plink_cmd() -> str:
    return shutil.which("plink2") or shutil.which("plink")


requires_hgdp = pytest.mark.skipif(
    not hgdp_available(),
    reason="HGDP+1KGP data not available — run examples/hgdp_1kgp/download_data.sh",
)
requires_plink = pytest.mark.skipif(
    not plink_available(),
    reason="plink2/plink not in PATH",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_plink_missing(plink_prefix: Path, out_prefix: Path) -> Path:
    """Run plink --missing to produce a .lmiss file. Returns path to .lmiss."""
    cmd = plink_cmd()
    plink_flag = "--pfile" if "plink2" in cmd else "--bfile"
    result = subprocess.run(
        [cmd, plink_flag, str(plink_prefix), "--missing", "--out", str(out_prefix)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"plink --missing failed:\n{result.stderr}")
    lmiss = out_prefix.with_suffix(".lmiss")
    vmiss = out_prefix.with_suffix(".vmiss")  # PLINK2 variant name
    return lmiss if lmiss.exists() else vmiss


def run_plink_freq(plink_prefix: Path, out_prefix: Path) -> Path:
    """Run plink --freq to produce a .frq / .afreq file. Returns the path."""
    cmd = plink_cmd()
    plink_flag = "--pfile" if "plink2" in cmd else "--bfile"
    result = subprocess.run(
        [cmd, plink_flag, str(plink_prefix), "--freq", "--out", str(out_prefix)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"plink --freq failed:\n{result.stderr}")
    frq = out_prefix.with_suffix(".frq")
    afreq = out_prefix.with_suffix(".afreq")  # PLINK2 variant name
    return frq if frq.exists() else afreq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@requires_hgdp
@requires_plink
def test_bim_parses_without_error(tmp_path):
    """Real HGDP .bim file (172k SNPs) must parse cleanly with no exceptions."""
    snp_info = read_bim(FIT_PLINK_PREFIX.with_suffix(".bim"))
    assert len(snp_info) > 100_000, f"Expected >100k SNPs, got {len(snp_info)}"
    # Spot-check: all positions are ints, all chromosomes are non-empty strings
    for snp_id, (chrom, pos, a1, a2) in list(snp_info.items())[:1000]:
        assert isinstance(chrom, str) and chrom, f"Bad chrom for {snp_id}"
        assert isinstance(pos, int) and pos >= 0, f"Bad pos for {snp_id}"


@pytest.mark.integration
@requires_hgdp
@requires_plink
def test_real_data_contains_duplicate_positions(tmp_path):
    """Real genomic data must have at least some duplicate chr:pos entries.

    If filter_snps reports zero duplicates on the full dataset, something is
    wrong with the chromosome normalisation (e.g. chr-prefix mismatch).
    """
    snp_info = read_bim(FIT_PLINK_PREFIX.with_suffix(".bim"))
    keep, stats = filter_snps(snp_info, {}, {})

    assert stats["duplicate_positions"] > 0, (
        "Expected duplicate positions in the HGDP+1KGP dataset. "
        "Zero duplicates suggests chromosome naming inconsistency between "
        ".bim and the grouping key used in filter_snps."
    )
    assert stats["snps_removed"] > 0


@pytest.mark.integration
@requires_hgdp
@requires_plink
def test_filter_snps_is_deterministic(tmp_path):
    """Running filter_snps twice on the same data must produce identical results."""
    snp_info = read_bim(FIT_PLINK_PREFIX.with_suffix(".bim"))
    keep1, stats1 = filter_snps(snp_info, {}, {})
    keep2, stats2 = filter_snps(snp_info, {}, {})
    assert set(keep1) == set(keep2), "filter_snps is not deterministic"
    assert stats1 == stats2


@pytest.mark.integration
@requires_hgdp
@requires_plink
def test_full_pipeline_with_real_lmiss_and_frq(tmp_path):
    """Generate real .lmiss and .frq via PLINK, then filter — end-to-end.

    This is the most realistic test: verifies that the output of an actual
    PLINK run is parseable by read_lmiss/read_frq without crashes or
    silent column-index errors.
    """
    lmiss_path = run_plink_missing(FIT_PLINK_PREFIX, tmp_path / "hgdp")
    frq_path = run_plink_freq(FIT_PLINK_PREFIX, tmp_path / "hgdp")

    snp_info = read_bim(FIT_PLINK_PREFIX.with_suffix(".bim"))
    missingness = read_lmiss(lmiss_path)
    maf = read_frq(frq_path)

    assert len(missingness) > 0, "read_lmiss returned empty dict from real PLINK output"
    assert len(maf) > 0, "read_frq returned empty dict from real PLINK output"

    # All F_MISS values must be in [0, 1]
    bad_miss = {k: v for k, v in missingness.items() if not (0.0 <= v <= 1.0)}
    assert not bad_miss, (
        f"read_lmiss returned out-of-range missingness values: {list(bad_miss.items())[:5]}. "
        "This may indicate a column-shift bug (e.g. reading N_GENO instead of F_MISS)."
    )

    # All MAF values must be in [0, 0.5]  (or [0,1] for PLINK2 ALT_FREQ)
    bad_maf = {k: v for k, v in maf.items() if not (0.0 <= v <= 1.0)}
    assert not bad_maf, (
        f"read_frq returned out-of-range MAF values: {list(bad_maf.items())[:5]}"
    )

    keep, stats = filter_snps(snp_info, missingness, maf)

    assert stats["snps_kept"] == len(keep)
    assert stats["snps_kept"] + stats["snps_removed"] == stats["total_snps"]
    assert stats["snps_kept"] <= stats["total_snps"]


@pytest.mark.integration
@requires_hgdp
@requires_plink
def test_plink_standard_lmiss_not_per_cluster(tmp_path):
    """Standard plink --missing (no --within) produces 5-col output.

    This confirms the real PLINK output that read_lmiss is designed for.
    If PLINK ever changes its default output format to 6-col, this test
    would catch it.
    """
    lmiss_path = run_plink_missing(FIT_PLINK_PREFIX, tmp_path / "hgdp")

    with open(lmiss_path) as f:
        header = f.readline().strip().split()

    # Standard 5-col: CHR SNP N_MISS N_GENO F_MISS
    # PLINK2 vmiss 5-col: #CHROM ID MISSING_CT OBS_CT F_MISS
    assert len(header) == 5, (
        f"Expected 5-column .lmiss header, got {len(header)} cols: {header}. "
        "If this is 6-col per-cluster output, read_lmiss will silently read the wrong column."
    )
    # F_MISS must be the last column
    assert header[-1].upper() in ("F_MISS", "F_MISS_DOSAGE"), (
        f"Expected F_MISS as last column, got '{header[-1]}'"
    )
