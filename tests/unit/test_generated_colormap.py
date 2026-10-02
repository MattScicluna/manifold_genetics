"""The colormap `acquire` generates: one distinct colour per value, and the
published colours kept where they exist.

The palette used to wrap at ten, so an eleventh value reused the first value's
colour; and colours followed alphabetical position, so a value new in a release
recoloured every value after it. On All of Us v9, "American Indian or Alaska
Native" sorted first and moved every race group's colour.
"""

import json

import pandas as pd
import pytest

from manifold_genetics.aou import AOU_COLOURS
from manifold_genetics.scaffold import _distinct_colours, _write_generated_colormap


def _generate(tmp_path, column, values, known=None):
    labels = pd.DataFrame({"sample_id": [str(i) for i in range(len(values))], column: values})
    _write_generated_colormap(labels, tmp_path / "colormap.json", known=known)
    return json.loads((tmp_path / "colormap.json").read_text())[column]


@pytest.mark.parametrize("n", [1, 10, 11, 30, 31, 100])
def test_no_two_values_share_a_colour(tmp_path, n):
    colours = _generate(tmp_path, "g", [f"v{i:03d}" for i in range(n)])

    assert len(colours) == n
    assert len({c.upper() for c in colours.values()}) == n


def test_distinct_colours_avoid_the_excluded_ones():
    excluded = ["#0072B2", "#d55e00"]

    colours = _distinct_colours(40, exclude=excluded)

    assert not {c.upper() for c in colours} & {c.upper() for c in excluded}
    assert len(set(colours)) == 40


def test_known_colours_are_kept_and_come_first(tmp_path):
    known = {"g": {"White": "#9B59B6", "Asian": "#4C6FFF"}}

    colours = _generate(tmp_path, "g", ["Asian", "White", "Zeta", "Alpha"], known=known)

    assert colours["White"] == "#9B59B6" and colours["Asian"] == "#4C6FFF"
    assert list(colours)[:2] == ["White", "Asian"], "the legend follows known's order"
    assert not {colours["Zeta"], colours["Alpha"]} & {"#9B59B6", "#4C6FFF"}


def test_a_new_value_does_not_recolour_the_known_ones(tmp_path):
    known = {"g": {"Asian": "#4C6FFF", "Black": "#3FA34D", "White": "#9B59B6"}}

    before = _generate(tmp_path, "g", ["Asian", "Black", "White"], known=known)
    after = _generate(tmp_path, "g", ["Asian", "Black", "White", "Another"], known=known)

    assert {k: after[k] for k in before} == before


def test_known_values_absent_from_the_data_are_left_out(tmp_path):
    colours = _generate(tmp_path, "g", ["White"], known={"g": {"White": "#9B59B6", "Gone": "#000"}})

    assert colours == {"White": "#9B59B6"}


class TestAllOfUsColours:
    # examples/colormaps/aou.json as last committed (1aca7ad), removed in #132.
    PUBLISHED = {
        "Black or African American": "#3FA34D",
        "Middle Eastern or North African": "#9E9E9E",
        "White": "#9B59B6",
        "Hispanic or Latino": "#FF5A5F",
        "Asian": "#4C6FFF",
        "Native Hawaiian or Other Pacific Islander": "#FFD84D",
        "More than one population": "#3BA99C",
        "No information": "#E5E5E5",
    }
    # The values seen in v9 (C2025Q4R6).
    V9 = {
        "race_ethnicity": [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanic or Latino",
            "Middle Eastern or North African",
            "More than one population",
            "Native Hawaiian or Other Pacific Islander",
            "No information",
            "White",
        ],
        "ethnicity": [
            "Hispanic or Latino",
            "No information",
            "No matching concept",
            "Not Hispanic or Latino",
        ],
    }

    def test_the_published_colours_are_unchanged(self):
        assert {k: AOU_COLOURS["race_ethnicity"][k] for k in self.PUBLISHED} == self.PUBLISHED

    def test_the_published_legend_order_is_kept(self):
        order = [k for k in AOU_COLOURS["race_ethnicity"] if k in self.PUBLISHED]

        assert order == list(self.PUBLISHED)

    @pytest.mark.parametrize("column", ["race_ethnicity", "ethnicity"])
    def test_every_v9_value_has_a_fixed_colour(self, column):
        assert set(self.V9[column]) <= set(AOU_COLOURS[column])

    @pytest.mark.parametrize("column", ["race_ethnicity", "race", "ethnicity"])
    def test_no_two_groups_share_a_colour(self, column):
        colours = [c.upper() for c in AOU_COLOURS[column].values()]

        assert len(set(colours)) == len(colours)
