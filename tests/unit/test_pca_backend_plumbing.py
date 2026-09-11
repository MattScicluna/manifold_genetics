"""The PCA backend choice must reach the PCA object from config and CLI.

A backend nobody can select is dead code. These pin the two routes a user has:
``run_pipeline(pca_backend=...)`` and ``--pca-backend`` on the CLI.
"""

from unittest.mock import patch

import pytest

from manifold_genetics.cli import main
from manifold_genetics.pipeline.config import PCAConfig
from manifold_genetics.pipeline.steps.pca import run_pca


class TestPCAConfig:
    def test_defaults_to_python(self):
        # The package must work after a plain pip install, which means the
        # default cannot require a Linux-x86-64 binary.
        assert PCAConfig().backend == "python"

    def test_carries_an_explicit_backend(self):
        assert PCAConfig(backend="python").backend == "python"


class TestRunPCAPassesBackendThrough:
    def test_backend_reaches_the_pca_constructor(self, tmp_path):
        with patch("manifold_genetics.pipeline.steps.pca.PCA") as MockPCA:
            run_pca(
                "fit_prefix",
                project_output=tmp_path / "out.csv",
                n_pcs=5,
                backend="python",
            )

        assert MockPCA.call_args.kwargs["backend"] == "python"

    def test_default_backend_needs_no_external_binary(self, tmp_path):
        with patch("manifold_genetics.pipeline.steps.pca.PCA") as MockPCA:
            run_pca("fit_prefix", project_output=tmp_path / "out.csv", n_pcs=5)

        assert MockPCA.call_args.kwargs["backend"] == "python"


class TestCLIFlag:
    @pytest.mark.parametrize("subcommand", ["pca", "pipeline"])
    def test_pca_backend_flag_is_accepted(self, subcommand, capsys):
        # --help exits 0 and lists the flag; a missing flag would be a parser error.
        with pytest.raises(SystemExit) as exc:
            main([subcommand, "--help"])

        assert exc.value.code == 0
        assert "--pca-backend" in capsys.readouterr().out

    def test_invalid_backend_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit) as exc:
            main(["pca", "--input", "x", "--pca-backend", "magic"])

        assert exc.value.code != 0
