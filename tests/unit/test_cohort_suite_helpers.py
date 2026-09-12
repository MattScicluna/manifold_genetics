"""The real-cohort suite's own helpers, tested on the cohorts it cannot reach.

``tests/integration/test_cohort_pipeline_real.py`` is skipped everywhere except a
machine holding controlled-access genotypes, so a defect in its helpers surfaces
only during an hour-long run on a cluster -- and a defect that is specific to
those cohorts never surfaces in CI at all.

That is exactly what happened on 2026-09-12. ``_coords`` read the PCA CSV with
inferred dtypes while ``labels`` read with ``dtype=str``. HGDP's sample IDs
(``HG00096``) are non-numeric, so pandas inferred ``object`` and the merge
worked. UK Biobank and All of Us identify samples by all-digit biobank IDs,
which pandas infers as ``int64``, so every merge-based assertion died with
"You are trying to merge on int64 and object columns". The suite had never been
able to run on the three cohorts it was written for.

These tests run in CI, on synthetic data, and hold that boundary.
"""

import pandas as pd

from tests.integration.test_cohort_pipeline_real import _coords


def test_all_digit_sample_ids_are_read_as_strings(tmp_path):
    """Biobank IDs are digits, and must not be inferred as integers.

    The label side of every comparison is read with ``dtype=str``. If this side
    infers ``int64``, the two never meet.
    """
    path = tmp_path / "project_pca_20.csv"
    path.write_text("sample_id,dim_1,dim_2\n1000017,0.1,0.2\n1000024,0.3,0.4\n")

    df, dims = _coords(path)

    assert df["sample_id"].dtype == object
    assert list(df["sample_id"]) == ["1000017", "1000024"]
    assert dims == ["dim_1", "dim_2"]


def test_numeric_ids_merge_against_labels_read_as_strings(tmp_path):
    """The operation the suite actually performs, on the IDs that broke it."""
    pca = tmp_path / "project_pca_20.csv"
    pca.write_text("sample_id,dim_1,dim_2\n1000017,0.1,0.2\n1000024,0.3,0.4\n")
    labels = tmp_path / "project_labels.csv"
    labels.write_text("sample_id,self_described_ancestry\n1000017,British\n1000024,Irish\n")

    df, _ = _coords(pca)
    merged = df.merge(pd.read_csv(labels, dtype=str), on="sample_id")

    assert len(merged) == 2, "numeric sample IDs failed to match their labels"


def test_non_numeric_ids_still_work(tmp_path):
    """HGDP's IDs were never the problem; they must not become one."""
    path = tmp_path / "fit_pca_20.csv"
    path.write_text("sample_id,dim_1\nHG00096,0.1\nHG00097,0.2\n")

    df, _ = _coords(path)

    assert list(df["sample_id"]) == ["HG00096", "HG00097"]
