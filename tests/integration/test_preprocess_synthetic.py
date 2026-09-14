"""The real shell on the simulated cohort: the round trip that proves the
cohort directory survives `preprocess`, and settles whether the intersection
script accepts two subsets of one dataset as its two sides (spec issue 9)."""

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess
from manifold_genetics.scaffold import acquire_synthetic
from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tools():
    resolver = ToolResolver()
    try:
        return resolver.resolve_plink2(), resolver.resolve_plink1()
    except ToolNotFoundError as exc:
        pytest.skip(f"plink binaries unavailable: {exc}")


def _variants(prefix):
    return {tuple(line.split()[i] for i in (0, 3, 4, 5)) for line in open(f"{prefix}.bim")}


def _samples(prefix):
    return [line.split()[1] for line in open(f"{prefix}.fam")]


def test_intersect_only_on_a_whole_cohort_is_lossless(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    config = preprocess(
        tmp_path / "in/config.yaml",
        tmp_path / "out",
        options=PreprocessOptions(
            preset="intersect-only", min_common_snps=100, memory=2000, threads=2
        ),
    )
    before, after = load_config(tmp_path / "in/config.yaml"), load_config(config)
    for side in ("fit_plink", "project_plink"):
        assert _samples(after[side]) == _samples(before[side])
        assert _variants(after[side]) == _variants(before[side])


def test_two_cohorts_intersect_to_the_shared_variants(tmp_path, tools):
    acquire_synthetic(tmp_path / "a", seed=1)
    acquire_synthetic(tmp_path / "b", seed=2)
    config = preprocess(
        tmp_path / "a/config.yaml",
        tmp_path / "out",
        project_config=tmp_path / "b/config.yaml",
        options=PreprocessOptions(
            preset="intersect-only", min_common_snps=100, memory=2000, threads=2
        ),
    )
    loaded = load_config(config)
    assert _variants(loaded["fit_plink"]) == _variants(loaded["project_plink"])


def test_the_output_dry_runs(tmp_path, tools):
    from manifold_genetics.cli import main

    acquire_synthetic(tmp_path / "in")
    config = preprocess(
        tmp_path / "in/config.yaml",
        tmp_path / "out",
        options=PreprocessOptions(
            preset="intersect-only", min_common_snps=100, memory=2000, threads=2
        ),
    )
    assert main(["run", str(config), "--dry-run"]) == 0
