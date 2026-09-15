"""The real shell on the simulated cohort: the round trip that proves the
cohort directory survives `preprocess`, and settles whether the intersection
script accepts two subsets of one dataset as its two sides (spec issue 9)."""

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess
from manifold_genetics.preprocessing.cohort import bed_is_complete
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


def test_a_truncated_intermediate_is_redone_not_reused(tmp_path, tools):
    """An OOM-killed run once left an intermediate .bed truncated, and the next
    run's existence-only checkpoint reused it as finished (issue #133). Resuming
    must detect and redo it instead."""
    acquire_synthetic(tmp_path / "in")
    options = PreprocessOptions(
        preset="intersect-only", min_common_snps=100, memory=2000, threads=2
    )
    config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", options=options)
    loaded = load_config(config)
    baseline = {
        side: (_variants(loaded[side]), _samples(loaded[side]))
        for side in ("fit_plink", "project_plink")
    }

    prefix = tmp_path / "out/data/temp/reference_filtered"
    assert bed_is_complete(prefix), "the first run's intermediate must be complete"
    bed = prefix.with_suffix(".bed")
    data = bed.read_bytes()
    bed.write_bytes(data[: len(data) // 2])
    assert not bed_is_complete(prefix), "the truncation must be detectable"

    config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", options=options, force=True)

    assert bed_is_complete(prefix), "the incomplete intermediate must be redone, not reused"
    loaded = load_config(config)
    for side in ("fit_plink", "project_plink"):
        assert (_variants(loaded[side]), _samples(loaded[side])) == baseline[
            side
        ], f"{side} must still be lossless after the redo"


def test_preprocess_then_subsample_then_dry_run(tmp_path, tools):
    from manifold_genetics.cli import main
    from manifold_genetics.preprocessing import Group, subsample

    acquire_synthetic(tmp_path / "in")
    filtered = preprocess(
        tmp_path / "in/config.yaml",
        tmp_path / "filtered",
        options=PreprocessOptions(
            preset="intersect-only", min_common_snps=100, memory=2000, threads=2
        ),
    )
    config = subsample(filtered, tmp_path / "sub", groups=[Group("branch", ".", 20)])
    assert main(["run", str(config), "--dry-run"]) == 0


def test_rerun_with_different_flags_is_refused_but_a_fresh_out_succeeds(tmp_path, tools):
    """--force after changing flags must not silently reuse intermediates computed
    under the old ones (issue #134): the same --out is refused, a new one works."""
    acquire_synthetic(tmp_path / "in")
    preprocess(
        tmp_path / "in/config.yaml",
        tmp_path / "out",
        options=PreprocessOptions(
            preset="intersect-only", min_common_snps=100, memory=2000, threads=2
        ),
    )

    with pytest.raises(ValueError, match="skip_geno"):
        preprocess(
            tmp_path / "in/config.yaml",
            tmp_path / "out",
            options=PreprocessOptions(
                preset="intersect-only",
                skip_geno=True,
                min_common_snps=100,
                memory=2000,
                threads=2,
            ),
            force=True,
        )

    config = preprocess(
        tmp_path / "in/config.yaml",
        tmp_path / "fresh-out",
        options=PreprocessOptions(
            preset="intersect-only", skip_geno=True, min_common_snps=100, memory=2000, threads=2
        ),
    )
    assert config.exists()
