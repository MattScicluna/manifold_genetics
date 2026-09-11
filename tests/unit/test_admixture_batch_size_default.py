"""Neural admixture must always be given a batch size.

Left unset, neural-admixture batches the entire dataset at once. That is a bug
in the tool, and 400 is the workaround this project has used for it -- so it is
not a tuning knob and must not be something a caller can forget.

It used to be a shell default on `subsample` mode only, which left
`aou/hgdp_1kgp_proj` with no batch size while projecting onto a large cohort.
Making it a package default rather than a preset value closes that, and covers
the Python API too.
"""

from unittest.mock import patch

from manifold_genetics.admixture.backends.neural import NeuralAdmixtureBackend
from manifold_genetics.pipeline.config import AdmixtureConfig, build_configs
from manifold_genetics.pipeline.configfile import PRESETS

DEFAULT_BATCH_SIZE = 400


class TestConfigDefault:
    def test_admixture_config_defaults_to_the_workaround_value(self):
        assert AdmixtureConfig().batch_size == DEFAULT_BATCH_SIZE

    def test_build_configs_defaults_to_it(self, tmp_path):
        configs = build_configs(
            fit_plink="fit",
            project_plink="project",
            output_dir=tmp_path,
            labels="labels.csv",
            colormap="colors.json",
        )

        assert configs.admixture.batch_size == DEFAULT_BATCH_SIZE

    def test_an_explicit_value_still_wins(self, tmp_path):
        configs = build_configs(
            fit_plink="fit",
            project_plink="project",
            output_dir=tmp_path,
            labels="labels.csv",
            colormap="colors.json",
            admix_batch_size=64,
        )

        assert configs.admixture.batch_size == 64


class TestPresetsNoLongerCarryIt:
    def test_no_preset_sets_a_batch_size(self):
        """A package default makes the preset copies redundant.

        Leaving them would reintroduce the duplication the config file exists to
        remove -- and a value in two places drifts.
        """
        for name, preset in PRESETS.items():
            assert "admix_batch_size" not in preset, f"{name} still sets it"


class TestBackendAlwaysPassesIt:
    def _commands(self, backend, method, *args):
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            try:
                getattr(backend, method)(*args)
            except Exception:
                pass  # only the command line matters here
            return [c.args[0] for c in run.call_args_list if c.args]

    def test_backend_defaults_to_the_workaround_value(self):
        assert NeuralAdmixtureBackend().batch_size == DEFAULT_BATCH_SIZE

    def test_none_is_still_honoured_as_an_explicit_opt_out(self):
        # Someone debugging the upstream bug needs to be able to turn it off.
        assert NeuralAdmixtureBackend(batch_size=None).batch_size is None
