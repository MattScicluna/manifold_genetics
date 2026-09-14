"""A label value with no colour must be impossible to miss.

`acquire hgdp` once wrote "Unknown" as every sample's label. The colormap had one
entry, the run completed, the figure was drawn, and every point in it was the
same grey. Nothing in the logs said anything was wrong -- the only detector was
a person looking at the picture and knowing what it should look like.

The same shape of failure is what makes a stale label file expensive: values
that no longer match the colormap are drawn as background and silently dropped
from the legend.
"""

import logging

import pandas as pd
import pytest

from manifold_genetics.visualization.plotting import warn_about_unmatched_labels


def test_it_warns_and_names_the_values_with_no_colour(caplog):
    labels = pd.Series(["Africa", "Europe", "Atlantis", "Mu"])
    colours = {"Africa": "#008000", "Europe": "#800080"}

    with caplog.at_level(logging.WARNING):
        warn_about_unmatched_labels(labels, colours, "genetic_region")

    message = caplog.text
    assert "Atlantis" in message and "Mu" in message
    assert "genetic_region" in message


def test_the_warning_says_how_many_points_it_affects(caplog):
    """Two stray samples and half the cohort are different problems."""
    labels = pd.Series(["Africa"] * 10 + ["Atlantis"] * 90)

    with caplog.at_level(logging.WARNING):
        warn_about_unmatched_labels(labels, {"Africa": "#008000"}, "region")

    assert "90" in caplog.text, "the warning must quantify the damage"


def test_silent_when_every_value_has_a_colour(caplog):
    labels = pd.Series(["Africa", "Europe"])

    with caplog.at_level(logging.WARNING):
        warn_about_unmatched_labels(labels, {"Africa": "#008000", "Europe": "#800080"}, "region")

    assert caplog.text == "", "a correct colormap must not produce noise"


def test_it_is_louder_when_nothing_matches_at_all(caplog):
    """Every value unmatched is a wrong file, not a missing entry."""
    labels = pd.Series(["Unknown"] * 100)

    with caplog.at_level(logging.WARNING):
        warn_about_unmatched_labels(labels, {"Africa": "#008000"}, "region")

    assert "no colour" in caplog.text.lower() or "none of" in caplog.text.lower()
    assert "100%" in caplog.text or "every" in caplog.text.lower()


@pytest.mark.parametrize("column", ["region", "self_described_ancestry"])
def test_the_column_is_always_named(caplog, column):
    with caplog.at_level(logging.WARNING):
        warn_about_unmatched_labels(pd.Series(["x"]), {}, column)

    assert column in caplog.text
