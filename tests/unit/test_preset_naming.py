"""Preset names, and the one word this project had used for two things.

``transform`` was both the second cohort's dataset role and the sklearn-style
method verb. The role was renamed to ``project`` (PR #67), which left exactly one
holdout: a preset still called ``transform``, denoting a *mode* while every other
use of the word denotes an *operation*.

So the rule is now: **``transform`` is the method verb, ``project`` is the
dataset role, and a preset is named for the shape of the run.** These tests hold
the line, because the cost of getting it wrong rises the moment a version is
published with the old name in it.

Renaming a preset breaks every config that names it, so ``transform`` stays as a
deprecated alias rather than disappearing.
"""

import warnings

import pytest
import yaml

from manifold_genetics.pipeline.configfile import PRESETS, ConfigFileError, load_config

MINIMAL = {
    "data": {
        "fit_plink": "data/fit",
        "project_plink": "data/project",
        "labels": "data/labels.csv",
        "colormap": "colors.json",
        "output_dir": "outputs",
    }
}


def write(tmp_path, preset):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**MINIMAL, "preset": preset}))
    return path


class TestTheNames:
    def test_the_three_presets_are_named_for_the_shape_of_the_run(self):
        assert set(PRESETS) == {"projection", "subsample", "whole_cohort"}

    def test_no_preset_is_named_after_a_method(self):
        """``transform`` and ``fit`` are operations, not modes.

        A preset called ``transform`` reads as "call transform", which is not
        what it selects -- it selects which cohort gets embedded.
        """
        assert not {"transform", "fit", "fit_transform"} & set(PRESETS)

    def test_whole_cohort_embeds_the_project_set(self):
        # Fit on a subset, embed everything: the project set contains the fit
        # set, so there is nothing to cross-project.
        assert PRESETS["whole_cohort"]["embedding_input"] == "project"


class TestTheDeprecatedAlias:
    def test_the_old_name_still_loads(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            kwargs = load_config(write(tmp_path, "transform"))

        assert kwargs["embedding_input"] == "project"

    def test_it_resolves_to_exactly_the_new_preset(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            old = load_config(write(tmp_path, "transform"))
        new = load_config(write(tmp_path, "whole_cohort"))

        assert old == new

    def test_using_it_warns_and_names_the_replacement(self, tmp_path):
        with pytest.warns(DeprecationWarning, match="whole_cohort"):
            load_config(write(tmp_path, "transform"))

    def test_the_new_name_does_not_warn(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            load_config(write(tmp_path, "whole_cohort"))

    def test_the_alias_is_not_offered_as_a_choice(self, tmp_path):
        """It is accepted, not advertised.

        An alias that appears in the error message is a name people will keep
        choosing, which defeats renaming it.
        """
        with pytest.raises(ConfigFileError) as error:
            load_config(write(tmp_path, "nonsense"))

        assert "whole_cohort" in str(error.value)
        assert "transform" not in str(error.value)
