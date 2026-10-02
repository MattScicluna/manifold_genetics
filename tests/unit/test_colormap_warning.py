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


class TestUnlistedValuesAreDrawn:
    """A value the colormap does not list is drawn grey, as Unknown.

    Only empty labels used to reach the grey layer, so an unlisted value was
    drawn by no layer at all: its points vanished while the warning said they
    were grey. On All of Us v9 that was every "American Indian or Alaska
    Native" participant.
    """

    @pytest.fixture
    def drawn(self, monkeypatch):
        import matplotlib.axes

        calls = {"scatter": [], "legend": []}
        real_scatter, real_legend = matplotlib.axes.Axes.scatter, matplotlib.axes.Axes.legend

        def scatter(self, x, y, *a, **kw):
            calls["scatter"].append((kw.get("color"), len(x)))
            return real_scatter(self, x, y, *a, **kw)

        def legend(self, *a, **kw):
            handles = kw.get("handles") or (a[0] if a else [])
            calls["legend"].append([h.get_label() for h in handles])
            return real_legend(self, *a, **kw)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", scatter)
        monkeypatch.setattr(matplotlib.axes.Axes, "legend", legend)
        return calls

    @staticmethod
    def _frames(values):
        ids = [f"s{i}" for i in range(len(values))]
        embedding = pd.DataFrame(
            {"sample_id": ids, "dim_1": range(len(ids)), "dim_2": range(len(ids))}
        )
        return embedding, pd.DataFrame({"sample_id": ids, "group": values})

    def test_every_point_is_drawn(self, tmp_path, drawn):
        from manifold_genetics.visualization.plotting import plot_embedding

        embedding, labels = self._frames(["White", "Asian", "New", "New", None])
        colormap = {"group": {"White": "#9B59B6", "Asian": "#4C6FFF"}}

        plot_embedding(embedding, labels, colormap, tmp_path / "e.png")

        assert sum(n for _, n in drawn["scatter"]) == 5
        assert ("lightgray", 3) in drawn["scatter"], "two unlisted + one empty, in grey"
        assert drawn["legend"][0] == ["White", "Asian", "Unknown"]

    def test_an_unlisted_value_alone_still_gets_the_unknown_entry(self, tmp_path, drawn):
        from manifold_genetics.visualization.plotting import plot_embedding

        embedding, labels = self._frames(["White", "New"])

        plot_embedding(embedding, labels, {"group": {"White": "#9B59B6"}}, tmp_path / "e.png")

        assert ("lightgray", 1) in drawn["scatter"]
        assert "Unknown" in drawn["legend"][0]

    def test_listed_values_with_no_samples_are_harmless(self, tmp_path, drawn):
        from manifold_genetics.visualization.plotting import plot_embedding

        embedding, labels = self._frames(["White", "Asian"])
        colormap = {"group": {"White": "#9B59B6", "Asian": "#4C6FFF", "Gone": "#000000"}}

        plot_embedding(embedding, labels, colormap, tmp_path / "e.png")

        assert drawn["legend"][0] == ["White", "Asian"]
        assert sum(n for _, n in drawn["scatter"]) == 2
