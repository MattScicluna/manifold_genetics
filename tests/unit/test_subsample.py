"""Choosing the fit samples of a cohort by label counts or a list."""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing.subsample import (
    Group,
    parse_group,
    select_by_groups,
    subsample,
)
from manifold_genetics.scaffold import acquire_synthetic


class TestParseGroup:
    def test_column_pattern_count(self):
        assert parse_group("race_ethnicity=White|European:10000") == Group(
            "race_ethnicity", "White|European", 10000
        )

    def test_pattern_may_contain_colons_and_equals(self):
        assert parse_group("col=a=b:c:5") == Group("col", "a=b:c", 5)

    @pytest.mark.parametrize("bad", ["nocolumn:5", "col=:5", "col=x", "col=x:0", "col=x:five"])
    def test_malformed_specs_are_rejected(self, bad):
        with pytest.raises(ValueError, match="COLUMN=PATTERN:COUNT"):
            parse_group(bad)


@pytest.fixture
def labels():
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(10)],
            "group": ["British"] * 4 + ["Irish"] * 3 + ["Other"] * 3,
        }
    )


class TestSelectByGroups:
    def test_takes_count_from_each_group_and_all_when_fewer(self, labels):
        chosen = select_by_groups(
            labels,
            [Group("group", "British", 2), Group("group", "Irish", 5)],
            include_rest=False,
            seed=0,
        )
        kinds = labels.set_index("sample_id").loc[chosen, "group"]
        assert (kinds == "British").sum() == 2 and (kinds == "Irish").sum() == 3
        assert "Other" not in set(kinds)

    def test_include_rest_appends_the_unmatched(self, labels):
        # "unmatched" means matched no group's pattern: with only a "British"
        # group, both Irish and Other are unmatched and both land in the rest
        # (mirrors examples/aou/shared/select_samples.py's `matched_ids`).
        chosen = select_by_groups(labels, [Group("group", "British", 1)], include_rest=True, seed=0)
        kinds = labels.set_index("sample_id").loc[chosen, "group"]
        assert (kinds == "Other").sum() == 3 and (kinds == "Irish").sum() == 3

    def test_matching_is_case_insensitive_regex(self, labels):
        chosen = select_by_groups(
            labels, [Group("group", "brit|iri", 10)], include_rest=False, seed=0
        )
        assert len(chosen) == 7

    def test_later_groups_do_not_reuse_samples(self, labels):
        chosen = select_by_groups(
            labels,
            [Group("group", "British", 4), Group("group", "B", 10)],
            include_rest=False,
            seed=0,
        )
        assert len(chosen) == len(set(chosen)) == 4

    def test_seed_makes_it_reproducible(self, labels):
        a = select_by_groups(labels, [Group("group", "British", 2)], include_rest=False, seed=7)
        b = select_by_groups(labels, [Group("group", "British", 2)], include_rest=False, seed=7)
        assert a == b

    def test_an_unknown_column_is_named(self, labels):
        with pytest.raises(ValueError, match="nope.*group"):
            select_by_groups(labels, [Group("nope", "x", 1)], include_rest=False, seed=0)


def _fake_keep(bfile, keep, out, plink2):
    ids = [line.split()[1] for line in Path(keep).read_text().splitlines()]
    fam = {line.split()[1]: line for line in Path(f"{bfile}.fam").read_text().splitlines()}
    Path(f"{out}.fam").write_text("".join(fam[i] + "\n" for i in ids))
    for ext in ("bed", "bim"):
        Path(f"{out}.{ext}").write_bytes(Path(f"{bfile}.{ext}").read_bytes())


class TestSubsample:
    @pytest.fixture
    def cohort(self, tmp_path, monkeypatch):
        from manifold_genetics.utils import tools

        monkeypatch.setattr(tools.ToolResolver, "resolve_plink2", lambda self: "/stub/plink2")
        acquire_synthetic(tmp_path / "in")
        return tmp_path / "in/config.yaml"

    def test_writes_a_subsample_cohort(self, cohort, tmp_path):
        # The synthetic cohort's `branch` column is "Branch 1".."Branch 8" (never
        # "0"), so the pattern picks out one whole branch (300 candidates) rather
        # than the empty match "^0$" would give.
        config = subsample(
            cohort,
            tmp_path / "out",
            groups=[Group("branch", "^Branch 1$", 3)],
            include_rest=False,
            keep_runner=_fake_keep,
        )
        loaded = load_config(config)
        assert loaded["embedding_input"] == "fit", "subsample preset expected"
        fit = [line.split()[1] for line in open(f"{loaded['fit_plink']}.fam")]
        assert len(fit) == 3
        project = [line.split()[1] for line in open(f"{loaded['project_plink']}.fam")]
        assert len(project) > 3, "the project side is the whole cohort"
        fit_labels = pd.read_csv(loaded["fit_labels"], dtype=str)
        assert list(fit_labels["sample_id"]) == fit

    def test_a_sample_list_is_honoured_verbatim(self, cohort, tmp_path):
        fam = [line.split() for line in open(tmp_path / "in/data/project_subset.fam")]
        keep = tmp_path / "keep.txt"
        keep.write_text("".join(f"{f[0]} {f[1]}\n" for f in fam[:5]))
        config = subsample(cohort, tmp_path / "out", fit_samples=keep, keep_runner=_fake_keep)
        fit = [line.split()[1] for line in open(f"{load_config(config)['fit_plink']}.fam")]
        assert fit == [f[1] for f in fam[:5]]

    def test_needs_exactly_one_way_of_choosing(self, cohort, tmp_path):
        with pytest.raises(ValueError, match="one of"):
            subsample(cohort, tmp_path / "out", keep_runner=_fake_keep)
        with pytest.raises(ValueError, match="one of"):
            subsample(
                cohort,
                tmp_path / "out",
                groups=[Group("branch", "0", 1)],
                fit_samples=tmp_path / "x",
                keep_runner=_fake_keep,
            )
