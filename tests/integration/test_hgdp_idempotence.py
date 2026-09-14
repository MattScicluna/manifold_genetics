"""The public HGDP+1KGP panel is already filtered and LD-pruned, so filtering it
again must change nothing. This is the end-to-end check that `acquire`,
`preprocess` and the shell agree on what a cohort directory is -- on real data,
without private access.

Needs the archive: set MG_HGDP_ARCHIVE to a local hgdp_1kgp_full.tar.gz, or
run with network on a login node."""

import os
from pathlib import Path

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess
from manifold_genetics.scaffold import acquire_hgdp
from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _variants(prefix):
    return {tuple(line.split()[i] for i in (0, 3, 4, 5)) for line in open(f"{prefix}.bim")}


def _samples(prefix):
    return [line.split()[1] for line in open(f"{prefix}.fam")]


@pytest.fixture(scope="module")
def hgdp(tmp_path_factory):
    try:
        ToolResolver().resolve_plink2()
        ToolResolver().resolve_plink1()
    except ToolNotFoundError as exc:
        pytest.skip(str(exc))
    archive = os.environ.get("MG_HGDP_ARCHIVE")
    if archive is None:
        pytest.skip("set MG_HGDP_ARCHIVE to the downloaded hgdp_1kgp_full.tar.gz")
    out = tmp_path_factory.mktemp("hgdp")
    return acquire_hgdp(out, archive=Path(archive), download=False)


@pytest.mark.parametrize(
    "preset,extra",
    [
        pytest.param(
            "intersect-only",
            {},
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "the shell's --geno 0.05 filter has no --skip lever, so it is applied even "
                    "under intersect-only. Recomputed on acquire_hgdp's subsets (3,400 fit / "
                    "4,094 project) instead of the full 4,151-sample panel, per-SNP missingness "
                    "shifts enough to newly exceed 5% on 739 reference-side and 36 project-side "
                    "SNPs; the position-overlap step then drops 747 of 172,152 (0.43%) from both "
                    "sides. See spec issue 1."
                ),
            ),
        ),
        pytest.param(
            None,
            {"skip_wrayner": True, "skip_project_maf": True},
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "GIAB+HLA exclusion removes 47,714 of 172,152 reference-side SNPs, the "
                    "unconditional --geno filter another 121, MAF-on-the-reference 3,008 more, "
                    "and LD-pruning at the panel's own r2<0.05/150kb/1 (on the now-shrunk, "
                    "GIAB/HLA/geno/MAF-filtered set) a further 19,952 -- not the 'few' expected "
                    "of an already-pruned panel. 70,795 of 172,152 (41.1%) lost on both sides. "
                    "See spec issue 1."
                ),
            ),
        ),
    ],
    ids=["intersect-only", "ukbb-flags"],
)
def test_filtering_the_filtered_panel_changes_nothing(hgdp, tmp_path, preset, extra):
    before = load_config(hgdp)
    after = load_config(
        preprocess(
            hgdp,
            tmp_path / (preset or "ukbb-flags"),
            options=PreprocessOptions(
                preset=preset,
                threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
                memory=16000,
                **extra,
            ),
        )
    )
    for side in ("fit_plink", "project_plink"):
        assert _samples(after[side]) == _samples(before[side])
        lost = _variants(before[side]) - _variants(after[side])
        assert not lost, f"{len(lost)} of {len(_variants(before[side]))} variants lost on {side}"
        assert _variants(after[side]) == _variants(before[side])
