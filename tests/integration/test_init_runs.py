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


def test_the_embedding_separates_the_groups_it_simulated(tmp_path):
    """A run that completes but shows nothing would be a poor first impression.

    The cohort is built with three groups at differing allele frequencies, so
    the embedding has real structure to find; this checks it found it.
    """
    import pandas as pd

    from tests.science import separation_over_chance

    init_synthetic(tmp_path)
    result = run_pipeline(**load_config(tmp_path / "config.yaml"))

    embedding = pd.read_csv(result.embedding.embedding_file, dtype={"sample_id": str})
    labels = pd.read_csv(tmp_path / "data" / "labels.csv", dtype=str)
    merged = embedding.merge(labels, on="sample_id")

    separation = separation_over_chance(
        merged[["dim_1", "dim_2"]].to_numpy(), merged["population"].to_numpy()
    )

    assert separation > 3.0, f"the simulated groups are not separated ({separation:.1f}x chance)"
