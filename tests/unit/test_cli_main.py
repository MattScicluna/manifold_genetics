"""Tests for cli.main() - argument-parser construction and command dispatch.

These exercise the ~700-line argparse tree in main() (which no other test
touches) plus the thin cmd_* dispatch wrappers. Everything heavy - PCA,
NeuralAdmixture, embeddings, plotting, metrics, tool download - is stubbed,
so the whole module runs in milliseconds with no torch, binaries, or real data.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics import cli as mg_cli

# Every subcommand registered in main()'s parser.
SUBCOMMANDS = [
    "pca",
    "admixture",
    "plot-admixture",
    "plot-admixture-embedding",
    "plot-knn-composition",
    "embed",
    "plot",
    "plot-pca",
    "plot-projection",
    "setup",
    "metrics-geographic",
    "metrics-admixture",
    "pipeline",
]

_VALIDATORS = [
    "validate_admixture_csv",
    "validate_colormap_json",
    "validate_column_in_csv",
    "validate_embedding_csv",
    "validate_geographic_csv",
    "validate_label_column",
    "validate_labels_colormap_match",
    "validate_labels_csv",
    "validate_sample_id_overlap",
]


@pytest.fixture
def stub_validation(monkeypatch):
    """Turn every validate_* helper imported into cli into a no-op."""
    for name in _VALIDATORS:
        monkeypatch.setattr(mg_cli, name, lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Parser construction: `--help` builds and walks the entire argparse tree
# ---------------------------------------------------------------------------


def test_top_level_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        mg_cli.main(["--help"])
    assert exc.value.code == 0


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        mg_cli.main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help_exits_zero(subcommand):
    """Each subparser's add_argument block runs when its --help is built."""
    with pytest.raises(SystemExit) as exc:
        mg_cli.main([subcommand, "--help"])
    assert exc.value.code == 0


def test_no_command_prints_help_and_returns_1(capsys):
    assert mg_cli.main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_unknown_command_is_argparse_error():
    with pytest.raises(SystemExit) as exc:
        mg_cli.main(["definitely-not-a-command"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Dispatch wrapper: main() routes to args.func and wraps exceptions
# ---------------------------------------------------------------------------


def test_dispatch_calls_selected_command(monkeypatch):
    seen = {}

    def fake_cmd(args):
        seen["called"] = True
        return 0

    # main() re-reads the module global when it runs set_defaults(func=cmd_setup),
    # so patching here (before the call) is enough.
    monkeypatch.setattr(mg_cli, "cmd_setup", fake_cmd)
    assert mg_cli.main(["setup"]) == 0
    assert seen.get("called") is True


def test_command_exception_is_caught_and_returns_1(monkeypatch, capsys):
    def boom(args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mg_cli, "cmd_setup", boom)
    assert mg_cli.main(["setup"]) == 1
    assert "kaboom" in capsys.readouterr().err


def test_command_exception_with_verbose_prints_traceback(monkeypatch, capsys):
    def boom(args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mg_cli, "cmd_setup", boom)
    assert mg_cli.main(["setup", "--verbose"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err


# ---------------------------------------------------------------------------
# setup_logging / _resolve_k_values helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verbose", [True, False])
def test_setup_logging_runs(verbose):
    mg_cli.setup_logging(verbose)


def test_resolve_k_values_explicit_list():
    assert mg_cli._resolve_k_values(None, None, [3, 2, 2, 5]) == [2, 3, 5]


def test_resolve_k_values_range():
    assert mg_cli._resolve_k_values(2, 5, None) == [2, 3, 4, 5]


def test_resolve_k_values_autodetect_from_files(tmp_path):
    for k in (2, 4):
        (tmp_path / f"q.{k}.csv").write_text("sample_id\n")
    assert mg_cli._resolve_k_values(None, None, None, q_prefix=tmp_path / "q") == [2, 4]


def test_resolve_k_values_no_input_raises():
    with pytest.raises(ValueError):
        mg_cli._resolve_k_values(None, None, None)


# ---------------------------------------------------------------------------
# Individual command dispatch (via main(), heavy deps stubbed)
# ---------------------------------------------------------------------------


def test_cmd_setup_reports_installed_tools(monkeypatch, capsys):
    class FakeResolver:
        def install_tools(self, include_plink1):
            assert include_plink1 is True
            return {"plink2": "/bin/plink2", "flashpca": "/bin/flashpca"}

    monkeypatch.setattr(mg_cli, "ToolResolver", FakeResolver)
    assert mg_cli.main(["setup"]) == 0
    out = capsys.readouterr().out
    assert "plink2: /bin/plink2" in out


def test_cmd_setup_skip_plink1(monkeypatch):
    captured = {}

    class FakeResolver:
        def install_tools(self, include_plink1):
            captured["include_plink1"] = include_plink1
            return {}

    monkeypatch.setattr(mg_cli, "ToolResolver", FakeResolver)
    assert mg_cli.main(["setup", "--skip-plink1"]) == 0
    assert captured["include_plink1"] is False


def test_cmd_pca_fit_project(monkeypatch, tmp_path):
    calls = []

    class FakePCA:
        def __init__(self, n_components, force):
            calls.append(("init", n_components, force))

        def fit(self, prefix, output_dir=None):
            calls.append(("fit", prefix))

        def project(self, prefix, output_path=None):
            calls.append(("project", prefix))
            df = pd.DataFrame({"sample_id": ["s1", "s2"], "dim_1": [0.1, 0.2]})
            if output_path:
                df.to_csv(output_path, index=False)
            return df

    monkeypatch.setattr(mg_cli, "PCA", FakePCA)
    out = tmp_path / "proj.csv"
    rc = mg_cli.main(
        [
            "pca",
            "--fit-plink",
            "fit",
            "--project-plink",
            "proj",
            "--project-output",
            str(out),
            "--n-pcs",
            "2",
        ]
    )
    assert rc == 0
    assert out.exists()
    assert ("fit", "fit") in calls


def test_cmd_admixture_fit_project(monkeypatch, tmp_path):
    calls = []

    class FakeAdmix:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def fit(self, prefix, output_dir=None, model_name=None):
            calls.append(("fit", prefix, model_name))

        def transform(self, prefix, output_prefix=None):
            calls.append(("transform", prefix))
            out = Path(output_prefix)
            out.parent.mkdir(parents=True, exist_ok=True)
            paths = {}
            for k in (2, 3):
                csv = Path(f"{out}.{k}.csv")
                pd.DataFrame({"sample_id": ["s1"], f"component_{k}": [1.0]}).to_csv(
                    csv, index=False
                )
                paths[k] = csv
            return paths

    monkeypatch.setattr(mg_cli, "NeuralAdmixture", FakeAdmix)
    out_dir = tmp_path / "admix"
    rc = mg_cli.main(
        [
            "admixture",
            "--fit-plink",
            "fit",
            "--project-plink",
            "proj",
            "--neuraladmixture-output-dir",
            str(out_dir / "ckpt"),
            "--fit-output",
            str(out_dir / "fit"),
            "--project-output",
            str(out_dir / "transform"),
            "--k-min",
            "2",
            "--k-max",
            "3",
        ]
    )
    assert rc == 0
    assert (out_dir / "transform.3.csv").exists()
    assert calls[0][0] == "init"
    assert ("fit", "fit", "fit") in calls


def test_cmd_admixture_missing_outputs_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mg_cli, "NeuralAdmixture", lambda **k: None)
    rc = mg_cli.main(["admixture", "--fit-plink", "fit", "--output", str(tmp_path)])
    assert rc == 1
    assert "fit-output" in capsys.readouterr().err


def test_cmd_embed_fit_project(monkeypatch, tmp_path, stub_validation):
    calls = []

    class FakeEmbed:
        def __init__(self, n_components=2, **kwargs):
            calls.append(("init", kwargs))

        def fit(self, X):
            calls.append(("fit", X))
            return self

        def transform(self, X):
            calls.append(("transform", X))
            return pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1], "dim_2": [0.2]})

    for name in ("PHATE", "UMAP", "TSNE", "DiffusionMap"):
        monkeypatch.setattr(mg_cli, name, FakeEmbed)

    out = tmp_path / "emb.csv"
    rc = mg_cli.main(
        [
            "embed",
            "--method",
            "umap",
            "--fit-input",
            str(tmp_path / "fit.csv"),
            "--project-input",
            str(tmp_path / "proj.csv"),
            "--project-output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert ("fit", str(tmp_path / "fit.csv")) in calls
    assert ("transform", str(tmp_path / "proj.csv")) in calls


def test_cmd_embed_unknown_method_returns_1(monkeypatch, tmp_path, stub_validation, capsys):
    # argparse restricts --method, so reach cmd_embed's fallback branch directly.
    import argparse

    args = argparse.Namespace(
        input=None,
        fit_input=str(tmp_path / "f.csv"),
        project_input=None,
        fit_output=None,
        project_output=str(tmp_path / "o.csv"),
        output=None,
        method="bogus",
        knn=5,
        t="auto",
        n_landmark=None,
        random_landmarking=False,
        n_neighbors=15,
        min_dist=0.1,
        perplexity=30.0,
        verbose=False,
    )
    assert mg_cli.cmd_embed(args) == 1
    assert "Unknown method" in capsys.readouterr().out


def test_cmd_metrics_geographic_writes_json(monkeypatch, tmp_path, stub_validation):
    monkeypatch.setattr(
        mg_cli,
        "compute_geographic_preservation",
        lambda **k: {"correlation": 0.9, "p_value": 1e-3},
    )
    out = tmp_path / "geo.json"
    rc = mg_cli.main(
        [
            "metrics-geographic",
            "--embedding",
            "emb.csv",
            "--geographic",
            "geo.csv",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text())["correlation"] == 0.9


def test_cmd_metrics_admixture_writes_json(monkeypatch, tmp_path, stub_validation):
    monkeypatch.setattr(
        mg_cli,
        "compute_admixture_preservation",
        lambda **k: {"2": {"correlation": 0.5}},
    )
    out = tmp_path / "adm.json"
    rc = mg_cli.main(
        [
            "metrics-admixture",
            "--embedding",
            "emb.csv",
            "--admixture-output",
            str(tmp_path / "q"),
            "--k-min",
            "2",
            "--k-max",
            "3",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text())["2"]["correlation"] == 0.5


def test_cmd_plot_dispatch(monkeypatch, tmp_path, stub_validation):
    monkeypatch.setattr(mg_cli, "visualize", lambda **k: [tmp_path / "a.png"])
    rc = mg_cli.main(
        [
            "plot",
            "--input",
            "emb.csv",
            "--labels",
            "labels.csv",
            "--colormap",
            "cmap.json",
            "--output",
            str(tmp_path / "out.png"),
        ]
    )
    assert rc == 0


def test_cmd_plot_pca_dispatch(monkeypatch, tmp_path, stub_validation):
    cmap = tmp_path / "cmap.json"
    cmap.write_text(json.dumps({"Population": {"A": "#000000"}}))
    monkeypatch.setattr(mg_cli, "plot_pca_pairs", lambda **k: tmp_path / "pca.png")
    rc = mg_cli.main(
        [
            "plot-pca",
            "--input",
            "pca.csv",
            "--labels",
            "labels.csv",
            "--colormap",
            str(cmap),
            "--output",
            str(tmp_path / "figs"),
            "--n-pcs",
            "4",
        ]
    )
    assert rc == 0


def test_cmd_plot_admixture_dispatch(monkeypatch, tmp_path, stub_validation):
    monkeypatch.setattr(mg_cli, "read_colormap", lambda p: {})
    monkeypatch.setattr(mg_cli, "plot_admixture_bar_grid", lambda **k: None)
    rc = mg_cli.main(
        [
            "plot-admixture",
            "--q-prefix",
            str(tmp_path / "q"),
            "--labels",
            "labels.csv",
            "--group-column",
            "Population",
            "--colormap",
            "cmap.json",
            "--ks",
            "2",
            "3",
            "--output",
            str(tmp_path / "bars.png"),
        ]
    )
    assert rc == 0


def test_cmd_plot_admixture_embedding_dispatch(monkeypatch, tmp_path, stub_validation):
    monkeypatch.setattr(mg_cli, "plot_admixture_embedding_grid", lambda **k: None)
    rc = mg_cli.main(
        [
            "plot-admixture-embedding",
            "--q-prefix",
            str(tmp_path / "q"),
            "--embedding",
            "emb.csv",
            "--ks",
            "2",
            "3",
            "--output",
            str(tmp_path / "ae.png"),
        ]
    )
    assert rc == 0


def test_cmd_plot_knn_composition_dispatch(monkeypatch, tmp_path, stub_validation):
    monkeypatch.setattr(mg_cli, "plot_knn_composition", lambda **k: tmp_path / "knn.png")
    rc = mg_cli.main(
        [
            "plot-knn-composition",
            "--fit-embedding",
            "fe.csv",
            "--project-embedding",
            "pe.csv",
            "--fit-labels",
            "fl.csv",
            "--project-labels",
            "pl.csv",
            "--fit-colormap",
            "fc.json",
            "--fit-label-column",
            "Population",
            "--project-label-column",
            "Region",
            "--output",
            str(tmp_path / "knn.png"),
        ]
    )
    assert rc == 0


def test_cmd_plot_projection_dispatch(monkeypatch, tmp_path, stub_validation):
    monkeypatch.setattr(mg_cli, "plot_projection", lambda **k: tmp_path / "proj.png")
    rc = mg_cli.main(
        [
            "plot-projection",
            "--fit-embedding",
            "fe.csv",
            "--project-embedding",
            "pe.csv",
            "--fit-labels",
            "fl.csv",
            "--project-labels",
            "pl.csv",
            "--fit-colormap",
            "fc.json",
            "--project-colormap",
            "pc.json",
            "--fit-column",
            "Population",
            "--project-column",
            "Population",
            "--output",
            str(tmp_path / "proj.png"),
        ]
    )
    assert rc == 0


def test_cmd_pipeline_forwards_to_run_pipeline(monkeypatch, tmp_path, stub_validation):
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(mg_cli, "run_pipeline", fake_run_pipeline)
    rc = mg_cli.main(
        [
            "pipeline",
            "--fit-plink",
            "fit",
            "--project-plink",
            "proj",
            "--output",
            str(tmp_path / "out"),
            "--labels",
            "labels.csv",
            "--colormap",
            "cmap.json",
            "--n-pcs",
            "10",
            "--k-min",
            "2",
            "--k-max",
            "3",
            "--embedding",
            "phate",
            "--knn",
            "50",
        ]
    )
    assert rc == 0
    assert captured["fit_plink"] == "fit"
    assert captured["embedding"] == "phate"
    assert captured["embedding_params"]["knn"] == 50


def test_cmd_pipeline_prints_metrics(monkeypatch, tmp_path, stub_validation, capsys):
    monkeypatch.setattr(
        mg_cli,
        "run_pipeline",
        lambda **k: {
            "metrics": {
                "geographic": {"correlation": 0.8, "p_value": 1e-4},
                "admixture": {"2": {"correlation": 0.7}},
            }
        },
    )
    rc = mg_cli.main(
        [
            "pipeline",
            "--fit-plink",
            "fit",
            "--project-plink",
            "proj",
            "--output",
            str(tmp_path / "out"),
            "--embedding",
            "umap",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Geographic preservation" in out
    assert "K=2" in out
