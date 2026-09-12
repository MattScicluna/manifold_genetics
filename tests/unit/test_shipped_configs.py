"""Every config.yaml shipped in examples/ must be loadable and well formed.

These files are documentation people copy. A typo in one is only discovered when
somebody starts a run that takes tens of minutes, so it is worth catching here
instead -- the same reasoning as rejecting unknown keys in the loader.

Deliberately does not check that the referenced data exists: most of it is
controlled-access and absent from any clean checkout. What is checked is that the
config parses, that every key is a real ``run_pipeline`` argument, and that the
preset each example claims is the one its pipeline mode needs.
"""

import inspect
from pathlib import Path

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.pipeline.runner import run_pipeline

REPO = Path(__file__).resolve().parents[2]
CONFIGS = sorted(REPO.glob("examples/**/config.yaml"))

# Which preset each example is expected to use, and the embedding_input it
# implies. Written out rather than derived so that changing an example's mode is
# a deliberate edit here too.
EXPECTED = {
    "examples/hgdp_1kgp/config.yaml": ("whole_cohort", "project"),
    "examples/ukbb/hgdp_1kgp_proj/config.yaml": ("projection", "both"),
    "examples/ukbb/10k_WB_5K_Irish/config.yaml": ("subsample", "fit"),
    "examples/ukbb/geosketch_phate/config.yaml": ("subsample", "fit"),
    "examples/aou/hgdp_1kgp_proj/config.yaml": ("projection", "both"),
    "examples/aou/10k_WBH/config.yaml": ("subsample", "fit"),
    "examples/aou/geosketch_phate/config.yaml": ("subsample", "fit"),
    "examples/generic/subset/config.yaml": ("subsample", "fit"),
    "examples/generic/hgdp_1kgp_proj/config.yaml": ("projection", "both"),
}


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def test_at_least_one_config_ships():
    # Guards against the parametrised tests below silently becoming no-ops.
    assert CONFIGS, "no example configs found"


@pytest.mark.parametrize("config", CONFIGS, ids=rel)
class TestShippedConfig:
    def test_loads(self, config):
        load_config(config)

    def test_every_key_is_a_real_run_pipeline_argument(self, config):
        # A key the loader accepts but run_pipeline does not would fail only at
        # the end of a long run.
        inspect.signature(run_pipeline).bind(**load_config(config))

    def test_names_its_inputs_and_output(self, config):
        kwargs = load_config(config)

        for required in ("fit_plink", "project_plink", "output_dir"):
            assert required in kwargs, f"{rel(config)} does not set {required}"

    def test_labels_and_colormap_are_resolvable(self, config):
        kwargs = load_config(config)

        assert "labels" in kwargs or {"fit_labels", "project_labels"} <= set(kwargs)
        assert "colormap" in kwargs or {"fit_colormap", "project_colormap"} <= set(kwargs)

    def test_paths_are_absolute_after_loading(self, config):
        # Relative paths resolve against the config's own directory; anything
        # still relative would depend on the working directory.
        kwargs = load_config(config)

        assert Path(kwargs["fit_plink"]).is_absolute()
        assert Path(kwargs["output_dir"]).is_absolute()

    def test_uses_the_preset_its_mode_requires(self, config):
        expected = EXPECTED.get(rel(config))
        assert expected is not None, f"{rel(config)} is new; add it to EXPECTED"
        _, embedding_input = expected

        assert load_config(config)["embedding_input"] == embedding_input


def test_colormaps_referenced_by_configs_exist():
    """Colormaps are tracked in the repo, so unlike the data these must resolve."""
    missing = []
    for config in CONFIGS:
        kwargs = load_config(config)
        for key in ("colormap", "fit_colormap", "project_colormap"):
            path = kwargs.get(key)
            if path is None or "colormaps" not in str(path):
                continue
            # Templates deliberately point at a placeholder the user supplies.
            if Path(path).name in {"your_colormap.json", "custom.json"}:
                continue
            if not Path(path).exists():
                missing.append(f"{rel(config)} -> {key}: {path}")
    assert not missing, "configs reference colormaps that do not exist:\n" + "\n".join(missing)
