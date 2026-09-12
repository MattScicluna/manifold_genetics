"""`manifold-genetics run config.yaml`.

The subcommand that lets examples stop being shell scripts. It does three things
and no more: load the config, apply a few command-line overrides, call
run_pipeline.

``--dry-run`` is not a convenience. These runs take tens of minutes on real
cohorts, and the failure this repo keeps hitting is a setting that quietly was
not what the file said -- so being able to see the resolved arguments before
committing to the run is the point.
"""

from unittest.mock import patch

import pytest
import yaml

from manifold_genetics.cli import main

MINIMAL = {
    "data": {
        "fit_plink": "data/fit",
        "project_plink": "data/project",
        "labels": "data/labels.csv",
        "colormap": "colors.json",
        "output_dir": "outputs",
    },
    "preset": "transform",
}


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(MINIMAL))
    return path


class TestRun:
    def test_calls_run_pipeline_with_the_config_contents(self, config, tmp_path):
        with patch("manifold_genetics.cli.run_pipeline") as run:
            rc = main(["run", str(config)])

        assert rc == 0
        assert run.call_args.kwargs["fit_plink"] == tmp_path / "data/fit"
        assert run.call_args.kwargs["embedding_input"] == "project"

    def test_output_can_be_overridden_on_the_command_line(self, config, tmp_path):
        # So one config can drive a verification run without being edited.
        with patch("manifold_genetics.cli.run_pipeline") as run:
            main(["run", str(config), "--output", str(tmp_path / "elsewhere")])

        assert run.call_args.kwargs["output_dir"] == tmp_path / "elsewhere"

    def test_skip_flags_can_be_added_on_the_command_line(self, config):
        with patch("manifold_genetics.cli.run_pipeline") as run:
            main(["run", str(config), "--skip-admixture"])

        assert run.call_args.kwargs["skip_admixture"] is True

    def test_a_command_line_skip_does_not_clear_the_config_s_own(self, config, tmp_path):
        path = tmp_path / "skips.yaml"
        path.write_text(yaml.safe_dump({**MINIMAL, "skip": {"metrics": True}}))

        with patch("manifold_genetics.cli.run_pipeline") as run:
            main(["run", str(path), "--skip-admixture"])

        assert run.call_args.kwargs["skip_admixture"] is True
        assert run.call_args.kwargs["skip_metrics"] is True


class TestDryRun:
    def test_dry_run_does_not_execute_the_pipeline(self, config):
        with patch("manifold_genetics.cli.run_pipeline") as run:
            rc = main(["run", str(config), "--dry-run"])

        assert rc == 0
        run.assert_not_called()

    def test_dry_run_prints_the_resolved_paths(self, config, capsys, tmp_path):
        main(["run", str(config), "--dry-run"])

        out = capsys.readouterr().out
        assert str(tmp_path / "data/fit") in out

    def test_dry_run_prints_the_preset_derived_embedding_settings(self, config, capsys):
        # The preset is exactly the part a reader cannot see in the file itself.
        main(["run", str(config), "--dry-run"])

        out = capsys.readouterr().out
        assert "project" in out  # embedding_input from the transform preset
        assert "knn" in out

    def test_dry_run_prints_settings_that_come_from_package_defaults(self, config, capsys):
        """The admixture batch size is the case that matters.

        It is a workaround for an upstream bug, supplied by the package rather
        than by any config, so a reader of the file cannot see it at all -- and
        it is the first thing to check before a long run on a large cohort.
        """
        main(["run", str(config), "--dry-run"])

        out = capsys.readouterr().out
        assert "admix_batch_size" in out
        assert "400" in out

    def test_it_marks_which_values_came_from_a_default(self, config, capsys):
        main(["run", str(config), "--dry-run"])

        line = next(
            line for line in capsys.readouterr().out.splitlines() if "admix_batch_size" in line
        )

        assert "default" in line

    def test_it_does_not_mark_the_config_s_own_values(self, config, capsys):
        main(["run", str(config), "--dry-run"])

        line = next(
            line
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("fit_plink")
        )

        assert "default" not in line

    def test_it_omits_the_test_only_backend_injection_point(self, config, capsys):
        # admixture_backend exists so tests can substitute a fake; printing it
        # would invite somebody to try setting it in a config file.
        main(["run", str(config), "--dry-run"])

        assert "admixture_backend" not in capsys.readouterr().out


class TestErrors:
    def test_a_missing_config_reports_cleanly(self, tmp_path, capsys):
        rc = main(["run", str(tmp_path / "nope.yaml")])

        assert rc != 0
        assert "not found" in capsys.readouterr().err

    def test_an_unknown_key_reports_cleanly(self, tmp_path, capsys):
        path = tmp_path / "typo.yaml"
        path.write_text(yaml.safe_dump({**MINIMAL, "pca": {"n_pc": 20}}))

        rc = main(["run", str(path)])

        assert rc != 0
        assert "n_pcs" in capsys.readouterr().err
