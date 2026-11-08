"""Haplotype painting against the real pbwt binary.

Skips when pbwt is not installed (``manifold-genetics setup --pbwt`` builds it).
The unit tests use a fake with the same file layout. This test is what checks
that the layout, the command line and the sample order are really pbwt's.

The data are simulated: two groups whose haplotypes are mosaics of four founder
haplotypes each, drawn from separate founder pools. Painting must separate them.
"""

import random

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.painting.matrix import read_paint_matrix, read_pbwt_samples
from manifold_genetics.pipeline.steps.painting import run_painting
from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pbwt():
    try:
        return ToolResolver().resolve_pbwt()
    except ToolNotFoundError:
        pytest.skip("pbwt is not installed (manifold-genetics setup --pbwt)")


def simulate_vcf(path, samples, n_sites, chrom, seed):
    rng = random.Random(seed)
    pools = {g: [[rng.randint(0, 1) for _ in range(n_sites)] for _ in range(4)] for g in "AB"}
    haps = []
    for name in samples:
        for _ in range(2):
            founder, hap = rng.randrange(4), []
            for s in range(n_sites):
                if rng.random() < 0.05:
                    founder = rng.randrange(4)
                allele = pools[name[0]][founder][s]
                hap.append(1 - allele if rng.random() < 0.01 else allele)
            haps.append(hap)
    cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={chrom},length=100000000>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "\t".join(cols + list(samples)),
    ]
    for s in range(n_sites):
        gts = [f"{haps[2 * i][s]}|{haps[2 * i + 1][s]}" for i in range(len(samples))]
        lines.append(
            "\t".join([chrom, str(1000 + 100 * s), f"rs{s}", "A", "G", ".", ".", ".", "GT"] + gts)
        )
    path.write_text("\n".join(lines) + "\n")


def test_real_pbwt_paints_and_separates_two_groups(tmp_path, pbwt):
    samples = [f"A{i}" for i in range(6)] + [f"B{i}" for i in range(6)]
    for chrom, seed in (("1", 1), ("2", 2)):
        simulate_vcf(tmp_path / f"chr{chrom}.vcf", samples, 400, chrom, seed)

    coords = run_painting(
        tmp_path / "chr{chr}.vcf",
        chromosomes=["1", "2"],
        pbwt_dir=tmp_path / "pbwt",
        project_output=tmp_path / "coords.csv",
        coancestry_output=tmp_path / "coancestry.csv",
        n_pcs=3,
        pbwt=pbwt,
    )

    # pbwt's own files: individual-level, VCF order, zero diagonal.
    labels, chunkcounts = read_paint_matrix(tmp_path / "pbwt" / "chr1.chunkcounts.out")
    assert labels == [f"IND{i}" for i in range(1, 13)]
    assert read_pbwt_samples(tmp_path / "pbwt" / "chr1.samples") == samples
    assert np.all(np.diag(chunkcounts) == 0)
    _, lengths = read_paint_matrix(tmp_path / "pbwt" / "chr1.chunklengths.out")
    np.testing.assert_allclose(lengths.sum(axis=1), 400 * 2, rtol=1e-3)  # n_sites * ploidy

    # Within-group copying dominates, and the first component splits the groups.
    coancestry = pd.read_csv(tmp_path / "coancestry.csv", index_col="sample_id")
    assert coancestry.loc["A0", ["A1", "A2"]].mean() > 3 * coancestry.loc["A0", ["B1", "B2"]].mean()
    first = coords.set_index("sample_id")["dim_1"]
    a, b = first[samples[:6]], first[samples[6:]]
    assert (a.max() < b.min()) or (b.max() < a.min())
