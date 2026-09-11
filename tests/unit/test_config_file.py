"""Loading a pipeline run from a YAML config file.

`examples/_shared/run_pipeline.sh` is ~400 lines that re-declare an argument
surface `cli.py` already owns. That duplication is where this repo's drift keeps
coming from: a subsample default every caller had to correct, one script
silently taking the most expensive landmarking path, another skipping label
regeneration. A config file is a serialisation of `build_configs`' keyword
arguments, so the shell layer stops existing rather than being rewritten.

Two properties matter most and are pinned hardest:

* **paths resolve relative to the config file**, not the working directory, so an
  example is runnable from anywhere;
* **presets carry the mode policy** that used to live in bash, including the
  landmarking rules settled in #82 -- in Python, under test, rather than in
  shell defaults every caller had to remember to correct.
"""

from pathlib import Path

import pytest
import yaml

from manifold_genetics.pipeline.configfile import ConfigFileError, load_config


def write_config(tmp_path, config, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return path


MINIMAL = {
    "data": {
        "fit_plink": "data/fit",
        "project_plink": "data/project",
        "labels": "data/labels.csv",
        "colormap": "colors.json",
        "output_dir": "outputs",
    }
}


class TestBasicLoading:
    def test_returns_run_pipeline_keyword_arguments(self, tmp_path):
        path = write_config(tmp_path, MINIMAL)

        kwargs = load_config(path)

        assert kwargs["fit_plink"] == tmp_path / "data/fit"
        assert kwargs["project_plink"] == tmp_path / "data/project"
        assert kwargs["output_dir"] == tmp_path / "outputs"

    def test_scalar_sections_map_onto_their_flat_argument_names(self, tmp_path):
        path = write_config(
            tmp_path,
            {**MINIMAL, "pca": {"n_pcs": 20}, "admixture": {"k_min": 3, "k_max": 7}},
        )

        kwargs = load_config(path)

        assert kwargs["n_pcs"] == 20
        assert kwargs["k_min"] == 3
        assert kwargs["k_max"] == 7

    def test_skip_section_becomes_skip_prefixed_flags(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "skip": {"admixture": True}})

        kwargs = load_config(path)

        assert kwargs["skip_admixture"] is True
        assert kwargs["skip_pca"] is False


class TestPathResolution:
    def test_relative_paths_resolve_against_the_config_file_not_the_cwd(self, tmp_path):
        nested = tmp_path / "examples" / "demo"
        nested.mkdir(parents=True)
        path = write_config(nested, MINIMAL)

        kwargs = load_config(path)

        assert kwargs["fit_plink"] == nested / "data/fit"

    def test_parent_relative_paths_work(self, tmp_path):
        nested = tmp_path / "examples" / "demo"
        nested.mkdir(parents=True)
        cfg = {"data": {**MINIMAL["data"], "colormap": "../shared/colors.json"}}
        path = write_config(nested, cfg)

        kwargs = load_config(path)

        assert kwargs["colormap"] == (nested / "../shared/colors.json").resolve()

    def test_absolute_paths_are_left_alone(self, tmp_path):
        cfg = {"data": {**MINIMAL["data"], "fit_plink": "/somewhere/else/fit"}}
        path = write_config(tmp_path, cfg)

        kwargs = load_config(path)

        assert kwargs["fit_plink"] == Path("/somewhere/else/fit")

    def test_non_path_values_are_not_turned_into_paths(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "embedding": {"method": "umap"}})

        kwargs = load_config(path)

        assert kwargs["embedding"] == "umap"


class TestPresets:
    """The mode policy from examples/_shared/run_pipeline.sh, now in Python."""

    def test_transform_preset_embeds_the_project_cohort_without_landmarking(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "preset": "transform"})

        kwargs = load_config(path)

        assert kwargs["embedding_input"] == "project"
        assert kwargs["embedding_params"]["knn"] == 100
        assert kwargs["embedding_params"]["t"] == 3
        assert kwargs["embedding_params"]["n_landmark"] is None

    def test_projection_preset_embeds_both_cohorts_without_landmarking(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "preset": "projection"})

        kwargs = load_config(path)

        assert kwargs["embedding_input"] == "both"
        assert kwargs["embedding_params"]["n_landmark"] is None

    def test_subsample_preset_pairs_landmarks_with_random_selection(self, tmp_path):
        # The #82 policy: n_landmark without random_landmarking silently selects
        # the far more expensive spectral path, so the two are set together.
        path = write_config(tmp_path, {**MINIMAL, "preset": "subsample"})

        kwargs = load_config(path)

        assert kwargs["embedding_input"] == "fit"
        assert kwargs["embedding_params"]["knn"] == 500
        assert kwargs["embedding_params"]["t"] == 50
        assert kwargs["embedding_params"]["n_landmark"] == 10000
        assert kwargs["embedding_params"]["random_landmarking"] is True

    def test_no_preset_sets_an_admixture_batch_size(self, tmp_path):
        """It is a package default now, not a preset value.

        It started life as a subsample-only shell default, which left
        aou/hgdp_1kgp_proj with none at all while projecting onto a large
        cohort. Since it exists to stop neural-admixture batching the entire
        dataset -- a bug in that tool -- it cannot be something a mode selection
        turns on. See tests/unit/test_admixture_batch_size_default.py.
        """
        for preset in ("subsample", "projection", "transform"):
            path = write_config(tmp_path, {**MINIMAL, "preset": preset}, f"{preset}.yaml")

            assert "admix_batch_size" not in load_config(path)

    def test_an_explicit_batch_size_is_still_carried_through(self, tmp_path):
        path = write_config(
            tmp_path,
            {**MINIMAL, "preset": "subsample", "admixture": {"batch_size": 64}},
        )

        assert load_config(path)["admix_batch_size"] == 64

    def test_explicit_values_override_the_preset(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "preset": "subsample", "embedding": {"t": 100}})

        kwargs = load_config(path)

        assert kwargs["embedding_params"]["t"] == 100
        assert kwargs["embedding_params"]["knn"] == 500  # still from the preset

    def test_no_preset_is_allowed(self, tmp_path):
        path = write_config(tmp_path, MINIMAL)

        load_config(path)

    def test_an_unknown_preset_names_the_valid_ones(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "preset": "wishful"})

        with pytest.raises(ConfigFileError, match="subsample"):
            load_config(path)


class TestErrors:
    """A typo must fail loudly. Silently ignoring a key is how settings drift."""

    def test_an_unknown_section_is_rejected(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "embeddings": {"method": "umap"}})

        with pytest.raises(ConfigFileError, match="embeddings"):
            load_config(path)

    def test_an_unknown_key_within_a_section_is_rejected(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "pca": {"n_pcs": 20, "npcs": 30}})

        with pytest.raises(ConfigFileError, match="npcs"):
            load_config(path)

    def test_a_near_miss_key_suggests_the_real_one(self, tmp_path):
        path = write_config(tmp_path, {**MINIMAL, "pca": {"n_pc": 20}})

        with pytest.raises(ConfigFileError, match="n_pcs"):
            load_config(path)

    def test_a_missing_data_section_is_rejected(self, tmp_path):
        path = write_config(tmp_path, {"pca": {"n_pcs": 20}})

        with pytest.raises(ConfigFileError, match="data"):
            load_config(path)

    def test_a_missing_config_file_is_rejected(self, tmp_path):
        with pytest.raises(ConfigFileError, match="not found"):
            load_config(tmp_path / "absent.yaml")

    def test_malformed_yaml_is_reported_as_such(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("data: {unclosed\n")

        with pytest.raises(ConfigFileError, match="YAML"):
            load_config(path)

    def test_an_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")

        with pytest.raises(ConfigFileError):
            load_config(path)


class TestLandmarkNone:
    def test_the_string_none_becomes_a_real_none(self, tmp_path):
        # YAML users write `n_landmark: none`; the CLI accepts the same spelling.
        path = write_config(tmp_path, {**MINIMAL, "embedding": {"n_landmark": "none"}})

        kwargs = load_config(path)

        assert kwargs["embedding_params"]["n_landmark"] is None

    def test_random_landmarking_without_landmarks_is_rejected(self, tmp_path):
        path = write_config(
            tmp_path,
            {**MINIMAL, "embedding": {"n_landmark": "none", "random_landmarking": True}},
        )

        with pytest.raises(ConfigFileError, match="random_landmarking"):
            load_config(path)
