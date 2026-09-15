"""`acquire synthetic` must produce something that actually runs.

This is the claim the front page makes to someone who has just installed the
package, so it is worth asserting rather than assuming: two commands, no
network, no data of their own, and a figure at the end.
"""

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.pipeline.runner import run_pipeline
from manifold_genetics.scaffold import acquire_synthetic

pytestmark = pytest.mark.integration


def test_a_scaffolded_cohort_runs_end_to_end(tmp_path):
    acquire_synthetic(tmp_path)

    result = run_pipeline(**load_config(tmp_path / "config.yaml"))

    assert result.failed_steps == (), f"stages failed: {result.failed_steps}"
    assert result.figures, "the run produced no figures"
    assert [f for f in result.figures if not f.exists()] == []


def test_the_embedding_recovers_the_tree_it_simulated(tmp_path):
    """A run that completes but shows nothing would be a poor first impression.

    The cohort lies along a branching tree with gaps, so the embedding has a
    known shape to find; this checks it found it, both as labelled branches
    and as the neighbourhoods along them. Measured 252x and 29x on 2026-09-13;
    the floors are far below that, so they trip on a broken pipeline and not
    on a change of PHATE version.
    """
    import pandas as pd

    from manifold_genetics.scaffold import dla_tree
    from tests.science import neighbourhood_preservation_over_chance, separation_over_chance

    acquire_synthetic(tmp_path)
    result = run_pipeline(**load_config(tmp_path / "config.yaml"))

    embedding = pd.read_csv(result.embedding.embedding_file, dtype={"sample_id": str})
    labels = pd.read_csv(tmp_path / "data" / "labels.csv", dtype=str)
    merged = embedding.merge(labels, on="sample_id").sort_values("sample_id")
    coordinates = merged[["dim_1", "dim_2"]].to_numpy()

    separation = separation_over_chance(coordinates, merged["branch"].to_numpy())
    assert separation > 20.0, f"the simulated branches are not separated ({separation:.1f}x chance)"

    # labels.csv is written in dla_tree() order, and SIM ids sort the same way.
    tree, _ = dla_tree()
    preserved = neighbourhood_preservation_over_chance(tree, coordinates, k=30)
    assert (
        preserved > 10.0
    ), f"the tree's neighbourhoods are not preserved ({preserved:.1f}x chance)"


def test_a_workbench_archive_survives_the_real_plink2(tmp_path):
    """The two things plink2 does to the normalised workbench panel unless told not to.

    ``--keep`` matches FID and IID, and the workbench FID is the population; and
    ``--make-bed`` writes ``chr1`` back out as ``1``. Both were found by running
    the layout through the real binary, so both are pinned through it: the unit
    tests only see the argv.
    """
    import tarfile

    import pandas as pd

    from manifold_genetics.scaffold import _detect_hgdp_layout, acquire_hgdp
    from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

    try:
        plink2 = ToolResolver().resolve_plink2()
    except ToolNotFoundError as exc:
        pytest.skip(f"plink2 is not available: {exc}")

    # A workbench-shaped archive around a genuine .bed: acquire synthetic's
    # project set, FIDs rewritten to forReference<branch>, chromosomes numeric,
    # nested under 1KGPHGDP/ as the real tarball is.
    acquire_synthetic(tmp_path / "synthetic")
    synthetic = tmp_path / "synthetic" / "data"
    labels = pd.read_csv(synthetic / "labels.csv", dtype=str).set_index("sample_id")
    fam = pd.read_csv(synthetic / "project_subset.fam", sep=r"\s+", header=None, dtype=str)
    fam[0] = "forReference" + labels.loc[fam[1], "branch"].str.replace(" ", "_").to_numpy()

    src = tmp_path / "src"
    src.mkdir()
    (src / "extractedChrAllUnpruned.bed").write_bytes(
        (synthetic / "project_subset.bed").read_bytes()
    )
    (src / "extractedChrAllUnpruned.bim").write_text((synthetic / "project_subset.bim").read_text())
    fam.to_csv(src / "extractedChrAllUnpruned.fam", sep="\t", header=False, index=False)
    archive = tmp_path / "1KGPHGDP.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for f in src.iterdir():
            tar.add(f, arcname=f"1KGPHGDP/{f.name}")

    out = tmp_path / "out"
    acquire_hgdp(out, archive=archive, plink2=plink2)

    bim = pd.read_csv(out / "data" / "fit_subset.bim", sep=r"\s+", header=None, dtype=str)
    assert set(bim[0]) == {"chr1"}, "the chr prefix must survive --make-bed"
    kept = pd.read_csv(out / "data" / "fit_subset.fam", sep=r"\s+", header=None, dtype=str)
    assert sorted(kept[1]) == sorted(fam[1]), "--keep must select every sample"
    assert set(kept[0]) == set(fam[0].str.replace("^forReference", "", regex=True))
    written = pd.read_csv(out / "data" / "labels.csv", dtype=str)
    assert list(written.columns) == ["sample_id", "Population"]
    assert len(written) == len(fam)

    acquire_hgdp(out, archive=archive, plink2=plink2, force=True)
    assert _detect_hgdp_layout(out / "data" / "raw") == "workbench"
