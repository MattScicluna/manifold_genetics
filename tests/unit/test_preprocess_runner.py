"""What `preprocess` does around the shell, with the shell replaced by a stub
that writes the two outputs the real one writes."""

import dataclasses
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess  # noqa: F401
from manifold_genetics.preprocessing.references import default_tools_dir
from manifold_genetics.preprocessing.runner import SENTINEL_NAME
from manifold_genetics.scaffold import acquire_synthetic


def _fake_shell(argv):
    """Copy reference -> fit_subset and biobank -> project_subset, dropping the last variant.

    The dropped variant's genotype block is also cut from .bed (SNP-major: each
    variant is a contiguous byte-column, so trimming the last one leaves the
    others byte-identical) -- otherwise the stub's own output would be a
    byte-inconsistent triplet that `preprocess`'s completeness check rejects.
    """
    ref = Path(argv[argv.index("--reference-plink") + 1])
    bio = Path(argv[argv.index("--biobank-plink") + 1])
    out = Path(argv[argv.index("--output-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    for src, name in ((ref, "fit_subset"), (bio, "project_subset")):
        shutil.copy(f"{src}.fam", out / f"{name}.fam")
        lines = Path(f"{src}.bim").read_text().splitlines()[:-1]
        (out / f"{name}.bim").write_text("\n".join(lines) + "\n")

        n_variants = len(lines)
        n_samples = len(Path(f"{src}.fam").read_text().splitlines())
        bytes_per_variant = (n_samples + 3) // 4
        expected_size = 3 + n_variants * bytes_per_variant
        data = Path(f"{src}.bed").read_bytes()
        (out / f"{name}.bed").write_bytes(data[:expected_size])


@pytest.fixture
def tools(monkeypatch):
    from manifold_genetics.utils import tools as t

    monkeypatch.setattr(t.ToolResolver, "resolve_plink2", lambda self: "/stub/plink2")
    monkeypatch.setattr(t.ToolResolver, "resolve_plink1", lambda self: "/stub/plink")


def test_one_config_keeps_the_cohort_shape(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)

    loaded = load_config(config)
    assert loaded["fit_plink"] == (tmp_path / "out/data/fit_subset").resolve()
    assert loaded["labels"] == (tmp_path / "out/data/labels.csv").resolve()
    assert (tmp_path / "out/colormap.json").exists()
    labels = pd.read_csv(tmp_path / "out/data/labels.csv", dtype=str)
    fam = pd.read_csv(tmp_path / "out/data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
    assert set(labels["sample_id"]) >= set(fam[1])


def test_two_configs_make_a_projection(tmp_path, tools):
    acquire_synthetic(tmp_path / "a", seed=1)
    acquire_synthetic(tmp_path / "b", seed=2)
    config = preprocess(
        tmp_path / "a/config.yaml",
        tmp_path / "out",
        project_config=tmp_path / "b/config.yaml",
        runner=_fake_shell,
    )
    loaded = load_config(config)
    assert loaded["embedding_input"] == "both", "projection preset expected"
    assert loaded["fit_labels"].name == "fit_labels.csv"
    assert loaded["project_labels"].name == "project_labels.csv"
    assert loaded["fit_colormap"].name == "colormap_fit.json"
    assert loaded["projection_plot_fit_column"] == "branch"


def test_the_shell_gets_the_resolved_tools(tmp_path, tools):
    seen = {}

    def spy(argv):
        seen["argv"] = argv
        _fake_shell(argv)

    acquire_synthetic(tmp_path / "in")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=spy)
    argv = seen["argv"]
    assert argv[argv.index("--plink2") + 1] == "/stub/plink2"
    assert argv[argv.index("--plink") + 1] == "/stub/plink"
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "out/data")


def test_tools_dir_defaults_when_not_given(tmp_path, tools):
    seen = {}

    def spy(argv):
        seen["argv"] = argv
        _fake_shell(argv)

    acquire_synthetic(tmp_path / "in")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=spy)
    argv = seen["argv"]
    assert argv[argv.index("--tools-dir") + 1] == str(default_tools_dir())


def test_tools_dir_is_kept_when_given(tmp_path, tools):
    seen = {}

    def spy(argv):
        seen["argv"] = argv
        _fake_shell(argv)

    given = tmp_path / "custom-tools"
    acquire_synthetic(tmp_path / "in")
    preprocess(
        tmp_path / "in/config.yaml",
        tmp_path / "out",
        options=dataclasses.replace(PreprocessOptions(), tools_dir=given),
        runner=spy,
    )
    argv = seen["argv"]
    assert argv[argv.index("--tools-dir") + 1] == str(given)


def test_refuses_to_overwrite_a_config_without_force(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    with pytest.raises(FileExistsError):
        preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell, force=True)


def test_labels_are_rewritten_not_reused(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    (tmp_path / "out/data").mkdir(parents=True)
    (tmp_path / "out/data/labels.csv").write_text("sample_id,branch\nSTALE,0\n")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    assert "STALE" not in (tmp_path / "out/data/labels.csv").read_text()


def test_a_duplicated_shared_label_row_is_not_duplicated_in_the_output(tmp_path, tools):
    """A shared labels.csv with a repeated sample_id must not silently expand the
    output: the union of both .fam files, not len(labels) * duplicates, decides
    the row count."""
    acquire_synthetic(tmp_path / "in")
    labels_path = tmp_path / "in/data/labels.csv"
    lines = labels_path.read_text().splitlines()
    labels_path.write_text("\n".join(lines[:2] + [lines[1]] + lines[2:]) + "\n")

    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)

    fit_ids = set(
        pd.read_csv(tmp_path / "out/data/fit_subset.fam", sep=r"\s+", header=None, dtype=str)[1]
    )
    project_ids = set(
        pd.read_csv(tmp_path / "out/data/project_subset.fam", sep=r"\s+", header=None, dtype=str)[1]
    )
    labels = pd.read_csv(tmp_path / "out/data/labels.csv", dtype=str)
    assert len(labels) == len(fit_ids | project_ids)


def test_missing_shell_output_raises_runtime_error(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    with pytest.raises(RuntimeError, match="fit_subset"):
        preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=lambda argv: None)


def test_truncated_shell_output_raises_runtime_error_naming_the_prefix(tmp_path, tools):
    """An OOM-killed shell can leave a truncated fit_subset.bed behind; that must
    not be mistaken for a finished output."""

    def truncating_shell(argv):
        _fake_shell(argv)
        out = Path(argv[argv.index("--output-dir") + 1])
        bed = out / "fit_subset.bed"
        bed.write_bytes(bed.read_bytes()[:-1])

    acquire_synthetic(tmp_path / "in")
    with pytest.raises(RuntimeError, match="incomplete") as excinfo:
        preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=truncating_shell)
    assert str(tmp_path / "out/data/fit_subset") in str(excinfo.value)


def _cut_labels_to(fraction, config_dir):
    """Rewrite the cohort's labels.csv to cover only ``fraction`` of the .fam."""
    fam = pd.read_csv(config_dir / "data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
    keep = set(fam[1].iloc[: int(len(fam) * fraction)])
    labels_path = config_dir / "data/labels.csv"
    labels = pd.read_csv(labels_path, dtype=str)
    labels[labels["sample_id"].isin(keep)].to_csv(labels_path, index=False)


def test_labels_below_half_coverage_fail_before_the_shell_runs(tmp_path, tools):
    """The shell takes hours and removes no samples, so the input's coverage is
    the output's: check it first."""
    acquire_synthetic(tmp_path / "in")
    _cut_labels_to(0.3, tmp_path / "in")
    calls = []

    def shell(argv):
        calls.append(argv)
        _fake_shell(argv)

    with pytest.raises(ValueError, match="30.0%"):
        preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=shell)
    assert calls == [], "the shell must not have been started"
    assert not (tmp_path / "out/config.yaml").exists()


def test_labels_above_half_coverage_run_and_keep_the_covered_rows(tmp_path, tools, caplog):
    acquire_synthetic(tmp_path / "in")
    _cut_labels_to(0.6, tmp_path / "in")
    with caplog.at_level("WARNING"):
        config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    fam = pd.read_csv(tmp_path / "out/data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
    labels = pd.read_csv(load_config(config)["labels"], dtype=str)
    assert len(labels) == int(len(fam) * 0.6)
    assert set(labels["sample_id"]) <= set(fam[1])
    assert any("drawn grey" in r.getMessage() for r in caplog.records)


def _add_geographic_coords(cohort_dir, covered_ids):
    """Append a geographic_coords entry to a synthetic cohort's data section,
    and write a CSV covering only ``covered_ids``."""
    geo = pd.DataFrame(
        {
            "sample_id": covered_ids,
            "latitude": range(len(covered_ids)),
            "longitude": range(len(covered_ids)),
        }
    )
    geo.to_csv(cohort_dir / "data" / "geographic.csv", index=False)
    config_path = cohort_dir / "config.yaml"
    text = config_path.read_text()
    assert "output_dir: outputs\n" in text
    config_path.write_text(
        text.replace(
            "output_dir: outputs\n",
            "output_dir: outputs\n  geographic_coords: data/geographic.csv\n",
        )
    )


def test_geographic_coords_are_carried_and_filtered_to_the_output_project_fam(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    fam = pd.read_csv(tmp_path / "in/data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
    ids = list(fam[1])
    covered = ids[::2]
    _add_geographic_coords(tmp_path / "in", covered)

    config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    loaded = load_config(config)
    assert loaded["geographic_coords"] == (tmp_path / "out/data/geographic.csv").resolve()

    out_fam = pd.read_csv(
        tmp_path / "out/data/project_subset.fam", sep=r"\s+", header=None, dtype=str
    )
    geo = pd.read_csv(loaded["geographic_coords"], dtype=str)
    assert set(geo["sample_id"]) <= set(out_fam[1])
    assert list(geo["sample_id"]) == [i for i in covered if i in set(out_fam[1])]


def test_no_geographic_coords_in_the_input_means_none_in_the_output(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    assert "geographic_coords" not in load_config(config)


def _sentinel(out_dir):
    return json.loads((out_dir / "data/temp" / SENTINEL_NAME).read_text())


def test_rerun_with_the_same_flags_and_force_proceeds_with_the_sentinel_unchanged(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    options = PreprocessOptions(preset="intersect-only")
    calls = []

    def shell(argv):
        calls.append(argv)
        _fake_shell(argv)

    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", options=options, runner=shell)
    first = _sentinel(tmp_path / "out")

    preprocess(
        tmp_path / "in/config.yaml", tmp_path / "out", options=options, force=True, runner=shell
    )
    second = _sentinel(tmp_path / "out")

    assert len(calls) == 2, "both runs must have invoked the shell"
    assert first == second


def test_rerun_with_different_flags_and_force_is_refused(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    calls = []

    def shell(argv):
        calls.append(argv)
        _fake_shell(argv)

    preprocess(
        tmp_path / "in/config.yaml",
        tmp_path / "out",
        options=PreprocessOptions(preset="intersect-only"),
        runner=shell,
    )
    assert len(calls) == 1

    with pytest.raises(ValueError, match="skip_geno") as excinfo:
        preprocess(
            tmp_path / "in/config.yaml",
            tmp_path / "out",
            options=PreprocessOptions(preset="intersect-only", skip_geno=True),
            force=True,
            runner=shell,
        )
    assert str(tmp_path / "out/data/temp") in str(excinfo.value)
    assert len(calls) == 1, "the shell must not have been invoked a second time"


def test_rerun_pointing_a_different_cohort_at_the_same_out_is_refused(tmp_path, tools):
    acquire_synthetic(tmp_path / "in")
    acquire_synthetic(tmp_path / "in2", seed=2)
    calls = []

    def shell(argv):
        calls.append(argv)
        _fake_shell(argv)

    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=shell)
    assert len(calls) == 1

    with pytest.raises(ValueError, match="fit_plink"):
        preprocess(tmp_path / "in2/config.yaml", tmp_path / "out", force=True, runner=shell)
    assert len(calls) == 1, "the shell must not have been invoked a second time"


def test_a_temp_dir_with_intermediates_but_no_sentinel_warns_and_proceeds(tmp_path, tools, caplog):
    acquire_synthetic(tmp_path / "in")
    temp_dir = tmp_path / "out/data/temp"
    temp_dir.mkdir(parents=True)
    (temp_dir / "reference_filtered.bed").write_bytes(b"\x00")

    with caplog.at_level("WARNING"):
        config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)

    assert config.exists()
    assert any("no " + SENTINEL_NAME in r.getMessage() for r in caplog.records)
    assert (temp_dir / SENTINEL_NAME).exists(), "a sentinel is written for the next run"


def test_a_sentinel_missing_a_key_whose_current_value_is_none_proceeds(tmp_path, tools):
    """A key absent from an older sentinel (e.g. one written before a new
    signature field existed) must read the same as an explicit None, not as a
    mismatch -- otherwise every such sentinel would be refused with an empty
    diff."""
    acquire_synthetic(tmp_path / "in")
    options = PreprocessOptions(preset="intersect-only")
    calls = []

    def shell(argv):
        calls.append(argv)
        _fake_shell(argv)

    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", options=options, runner=shell)
    sentinel_path = tmp_path / "out/data/temp" / SENTINEL_NAME
    sentinel = json.loads(sentinel_path.read_text())
    assert sentinel["maf"] is None
    del sentinel["maf"]
    sentinel_path.write_text(json.dumps(sentinel, indent=2, sort_keys=True) + "\n")

    preprocess(
        tmp_path / "in/config.yaml", tmp_path / "out", options=options, force=True, runner=shell
    )

    assert len(calls) == 2, "the second run must have proceeded, not been refused"
    assert json.loads(sentinel_path.read_text())["maf"] is None
