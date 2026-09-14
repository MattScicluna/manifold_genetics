"""What `preprocess` does around the shell, with the shell replaced by a stub
that writes the two outputs the real one writes."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess  # noqa: F401
from manifold_genetics.scaffold import init_synthetic


def _fake_shell(argv):
    """Copy reference -> fit_subset and biobank -> project_subset, dropping the last variant."""
    ref = Path(argv[argv.index("--reference-plink") + 1])
    bio = Path(argv[argv.index("--biobank-plink") + 1])
    out = Path(argv[argv.index("--output-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    for src, name in ((ref, "fit_subset"), (bio, "project_subset")):
        for ext in ("bed", "fam"):
            shutil.copy(f"{src}.{ext}", out / f"{name}.{ext}")
        lines = Path(f"{src}.bim").read_text().splitlines()[:-1]
        (out / f"{name}.bim").write_text("\n".join(lines) + "\n")


@pytest.fixture
def tools(monkeypatch):
    from manifold_genetics.utils import tools as t

    monkeypatch.setattr(t.ToolResolver, "resolve_plink2", lambda self: "/stub/plink2")
    monkeypatch.setattr(t.ToolResolver, "resolve_plink1", lambda self: "/stub/plink")


def test_one_config_keeps_the_cohort_shape(tmp_path, tools):
    init_synthetic(tmp_path / "in")
    config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)

    loaded = load_config(config)
    assert loaded["fit_plink"] == (tmp_path / "out/data/fit_subset").resolve()
    assert loaded["labels"] == (tmp_path / "out/data/labels.csv").resolve()
    assert (tmp_path / "out/colormap.json").exists()
    labels = pd.read_csv(tmp_path / "out/data/labels.csv", dtype=str)
    fam = pd.read_csv(tmp_path / "out/data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
    assert set(labels["sample_id"]) >= set(fam[1])


def test_two_configs_make_a_projection(tmp_path, tools):
    init_synthetic(tmp_path / "a", seed=1)
    init_synthetic(tmp_path / "b", seed=2)
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

    init_synthetic(tmp_path / "in")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=spy)
    argv = seen["argv"]
    assert argv[argv.index("--plink2") + 1] == "/stub/plink2"
    assert argv[argv.index("--plink") + 1] == "/stub/plink"
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "out/data")


def test_refuses_to_overwrite_a_config_without_force(tmp_path, tools):
    init_synthetic(tmp_path / "in")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    with pytest.raises(FileExistsError):
        preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell, force=True)


def test_labels_are_rewritten_not_reused(tmp_path, tools):
    init_synthetic(tmp_path / "in")
    (tmp_path / "out/data").mkdir(parents=True)
    (tmp_path / "out/data/labels.csv").write_text("sample_id,branch\nSTALE,0\n")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    assert "STALE" not in (tmp_path / "out/data/labels.csv").read_text()


def test_a_duplicated_shared_label_row_is_not_duplicated_in_the_output(tmp_path, tools):
    """A shared labels.csv with a repeated sample_id must not silently expand the
    output: the union of both .fam files, not len(labels) * duplicates, decides
    the row count."""
    init_synthetic(tmp_path / "in")
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
    init_synthetic(tmp_path / "in")
    with pytest.raises(RuntimeError, match="fit_subset"):
        preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=lambda argv: None)
