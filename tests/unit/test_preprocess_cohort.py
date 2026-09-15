"""Reading and writing the cohort directory -- the unit `preprocess`, `subsample`
and `run` exchange."""

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing.cohort import (
    MIN_LABEL_COVERAGE,
    check_label_coverage,
    filter_labels_to_fam,
    read_cohort,
    read_fam_ids,
    write_cohort_config,
)
from manifold_genetics.scaffold import acquire_synthetic


@pytest.fixture
def cohort(tmp_path):
    acquire_synthetic(tmp_path)
    return read_cohort(tmp_path / "config.yaml")


def test_a_whole_cohort_config_uses_one_label_file_for_both_sides(cohort):
    assert cohort.preset == "whole_cohort"
    assert cohort.shared_labels
    assert cohort.fit.labels == cohort.project.labels == cohort.path.parent / "data" / "labels.csv"
    assert cohort.fit.colormap == cohort.project.colormap


def test_a_projection_config_keeps_the_sides_apart(tmp_path):
    (tmp_path / "data").mkdir()
    for name in ("a", "b"):
        for ext in ("bed", "bim", "fam"):
            (tmp_path / "data" / f"{name}.{ext}").write_text("")
        (tmp_path / "data" / f"{name}.csv").write_text("sample_id,x\n")
        (tmp_path / f"{name}.json").write_text("{}")
    (tmp_path / "config.yaml").write_text(
        "preset: projection\ndata:\n  fit_plink: data/a\n  project_plink: data/b\n"
        "  fit_labels: data/a.csv\n  project_labels: data/b.csv\n"
        "  fit_colormap: a.json\n  project_colormap: b.json\n  output_dir: outputs\n"
    )
    cohort = read_cohort(tmp_path / "config.yaml")
    assert not cohort.shared_labels
    assert cohort.fit.labels.name == "a.csv" and cohort.project.labels.name == "b.csv"
    assert cohort.fit.colormap.name == "a.json" and cohort.project.colormap.name == "b.json"


def test_shared_labels_is_false_when_only_one_side_overrides_it(tmp_path):
    (tmp_path / "data").mkdir()
    for name in ("a", "b"):
        for ext in ("bed", "bim", "fam"):
            (tmp_path / "data" / f"{name}.{ext}").write_text("")
        (tmp_path / f"{name}.json").write_text("{}")
    (tmp_path / "shared.csv").write_text("sample_id,x\n")
    (tmp_path / "data" / "b.csv").write_text("sample_id,x\n")
    (tmp_path / "config.yaml").write_text(
        "preset: projection\ndata:\n  fit_plink: data/a\n  project_plink: data/b\n"
        "  labels: shared.csv\n  project_labels: data/b.csv\n"
        "  fit_colormap: a.json\n  project_colormap: b.json\n  output_dir: outputs\n"
    )
    cohort = read_cohort(tmp_path / "config.yaml")
    assert not cohort.shared_labels
    assert cohort.fit.labels.name == "shared.csv"
    assert cohort.project.labels.name == "b.csv"


def test_fam_ids_come_back_in_file_order(cohort):
    ids = read_fam_ids(cohort.fit.plink)
    fam = pd.read_csv(f"{cohort.fit.plink}.fam", sep=r"\s+", header=None, dtype=str)
    assert ids == list(fam[1])


def test_labels_are_filtered_to_the_fam_and_keep_their_columns(cohort, tmp_path):
    out = tmp_path / "filtered.csv"
    n = filter_labels_to_fam(cohort.project.labels, cohort.fit.plink, out)
    written = pd.read_csv(out, dtype=str)
    assert n == len(written) == len(read_fam_ids(cohort.fit.plink))
    assert list(written.columns) == list(pd.read_csv(cohort.project.labels, nrows=0).columns)


def _labels_covering(cohort, fraction, path):
    """The cohort's label file cut to the first ``fraction`` of the .fam."""
    ids = read_fam_ids(cohort.fit.plink)
    keep = set(ids[: int(len(ids) * fraction)])
    labels = pd.read_csv(cohort.project.labels, dtype=str)
    labels[labels["sample_id"].isin(keep)].to_csv(path, index=False)
    return path


def test_full_coverage_writes_every_fam_row_without_a_warning(cohort, tmp_path, caplog):
    out = tmp_path / "filtered.csv"
    with caplog.at_level("WARNING"):
        n = filter_labels_to_fam(cohort.project.labels, cohort.fit.plink, out)
    assert n == len(read_fam_ids(cohort.fit.plink))
    assert not [r for r in caplog.records if "drawn grey" in r.getMessage()]


def test_partial_coverage_above_half_warns_and_writes_only_the_covered_rows(
    cohort, tmp_path, caplog
):
    """`acquire custom` accepts a label file describing half the .fam and `run`
    draws the rest grey; filtering must not turn that into an error hours later."""
    ids = read_fam_ids(cohort.fit.plink)
    partial = _labels_covering(cohort, 0.6, tmp_path / "partial.csv")
    out = tmp_path / "out.csv"
    with caplog.at_level("WARNING"):
        n = filter_labels_to_fam(partial, cohort.fit.plink, out)

    covered = ids[: int(len(ids) * 0.6)]
    written = pd.read_csv(out, dtype=str)
    assert n == len(covered)
    assert list(written["sample_id"]) == covered, "the covered rows, in .fam order"
    warnings = [r.getMessage() for r in caplog.records if "drawn grey" in r.getMessage()]
    assert len(warnings) == 1
    assert f"{len(ids) - len(covered)} of {len(ids)}" in warnings[0]
    assert ids[-1] in warnings[0] or ids[len(covered)] in warnings[0], "names uncovered ids"


def test_labels_that_do_not_cover_the_fam_are_an_error(cohort, tmp_path):
    """Below MIN_LABEL_COVERAGE the file does not describe this cohort."""
    partial = _labels_covering(cohort, 0.3, tmp_path / "partial.csv")
    with pytest.raises(ValueError, match="not in") as excinfo:
        filter_labels_to_fam(partial, cohort.fit.plink, tmp_path / "out.csv")
    assert "30.0%" in str(excinfo.value)
    assert f"{MIN_LABEL_COVERAGE:.0%}" in str(excinfo.value)
    assert not (tmp_path / "out.csv").exists()


def test_check_label_coverage_reports_the_fraction_and_applies_the_same_rule(cohort, tmp_path):
    assert check_label_coverage(cohort.project.labels, cohort.fit.plink) == 1.0
    partial = _labels_covering(cohort, 0.6, tmp_path / "partial.csv")
    assert check_label_coverage(partial, cohort.fit.plink) == pytest.approx(0.6, abs=0.01)
    too_few = _labels_covering(cohort, 0.3, tmp_path / "too_few.csv")
    with pytest.raises(ValueError, match="30.0%"):
        check_label_coverage(too_few, cohort.fit.plink)


def test_the_threshold_is_shared_with_acquire(cohort):
    """One rule: what `acquire custom` and `acquire aou` accept, `preprocess`
    and `subsample` accept too."""
    import inspect

    from manifold_genetics import aou, scaffold

    assert aou._MIN_LABEL_OVERLAP is MIN_LABEL_COVERAGE
    default = inspect.signature(scaffold.acquire_custom).parameters["min_overlap"].default
    assert default is MIN_LABEL_COVERAGE


def test_labels_with_a_duplicate_sample_id_do_not_expand_the_output(cohort, tmp_path):
    labels = pd.read_csv(cohort.project.labels, dtype=str)
    duplicated = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    dup_path = tmp_path / "duplicated.csv"
    duplicated.to_csv(dup_path, index=False)

    out = tmp_path / "filtered.csv"
    n = filter_labels_to_fam(dup_path, cohort.fit.plink, out)
    assert n == len(read_fam_ids(cohort.fit.plink))


def test_the_written_config_is_accepted_by_the_loader_and_carries_settings(cohort, tmp_path):
    out = tmp_path / "next"
    out.mkdir()
    path = write_cohort_config(
        out,
        based_on=cohort,
        preset="projection",
        data={
            "fit_plink": "data/fit_subset",
            "project_plink": "data/project_subset",
            "fit_labels": "data/fit_labels.csv",
            "project_labels": "data/project_labels.csv",
            "fit_colormap": "colormap_fit.json",
            "project_colormap": "colormap_project.json",
            "output_dir": "outputs",
        },
        visualization={
            "projection_plot_fit_column": "branch",
            "projection_plot_project_column": "branch",
        },
        written_by="manifold-genetics preprocess",
    )
    text = path.read_text()
    assert text.startswith("# Written by `manifold-genetics preprocess`")
    loaded = load_config(path)
    assert loaded["fit_plink"] == (out / "data/fit_subset").resolve()
    assert loaded["n_pcs"] == cohort.raw["pca"]["n_pcs"], "pca settings must carry forward"
    assert loaded["projection_plot_fit_column"] == "branch"
    assert "labels" not in loaded, "the old shared-label key must not survive a projection rewrite"
