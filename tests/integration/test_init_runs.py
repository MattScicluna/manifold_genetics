"""`init synthetic` must produce something that actually runs.

This is the claim the front page makes to someone who has just installed the
package, so it is worth asserting rather than assuming: two commands, no
network, no data of their own, and a figure at the end.
"""

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.pipeline.runner import run_pipeline
from manifold_genetics.scaffold import init_synthetic

pytestmark = pytest.mark.integration


def test_a_scaffolded_cohort_runs_end_to_end(tmp_path):
    init_synthetic(tmp_path)

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

    init_synthetic(tmp_path)
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
