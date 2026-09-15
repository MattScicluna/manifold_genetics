"""Choosing the fit samples of a cohort by label counts, a list, or geosketch."""

import builtins
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing.subsample import (
    Group,
    parse_group,
    select_by_geosketch,
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


class TestSelectByGeosketch:
    def test_chooses_n_ids_from_the_pca_table(self):
        pca = pd.DataFrame(
            {"sample_id": [f"S{i}" for i in range(50)], "dim_1": range(50), "dim_2": range(50)}
        )
        chosen = select_by_geosketch(pca, 10, seed=0, sketch=lambda X, n, **kw: list(range(n)))
        assert chosen == [f"S{i}" for i in range(10)]

    def test_uses_every_column_but_sample_id_as_a_dimension(self):
        # examples/_shared/select_samples_geosketch.py's rule is "everything
        # except sample_id", not "columns named dim_*".
        captured = {}

        def sketch(X, n, **kw):
            captured["X"] = X
            captured["kwargs"] = kw
            return list(range(n))

        pca = pd.DataFrame(
            {
                "sample_id": ["A", "B", "C"],
                "pc1": [1, 2, 3],
                "pc2": [4, 5, 6],
                "extra": [7, 8, 9],
            }
        )
        select_by_geosketch(pca, 2, seed=3, sketch=sketch)
        assert captured["X"].shape == (3, 3)
        assert captured["X"].dtype == np.float32
        assert captured["kwargs"] == {"seed": 3, "replace": False}

    def test_n_pcs_truncates_the_dimensions_used(self):
        captured = {}

        def sketch(X, n, **kw):
            captured["shape"] = X.shape
            return list(range(n))

        pca = pd.DataFrame({"sample_id": ["A", "B", "C"], "dim_1": [1, 2, 3], "dim_2": [4, 5, 6]})
        select_by_geosketch(pca, 2, seed=0, sketch=sketch, n_pcs=1)
        assert captured["shape"] == (3, 1)

    def test_missing_geosketch_raises_a_named_import_error(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "geosketch":
                raise ImportError("no module named geosketch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        pca = pd.DataFrame({"sample_id": ["A"], "dim_1": [1]})
        with pytest.raises(ImportError, match="geosketch"):
            select_by_geosketch(pca, 1, seed=0)

    def test_n_larger_than_available_rows_is_clamped_with_a_warning(self, caplog):
        pca = pd.DataFrame({"sample_id": ["A", "B", "C"], "dim_1": [1, 2, 3]})
        with caplog.at_level(logging.WARNING):
            chosen = select_by_geosketch(pca, 10, seed=0, sketch=lambda X, n, **kw: list(range(n)))
        assert chosen == ["A", "B", "C"]
        assert any(
            "3" in record.getMessage() and "10" in record.getMessage() for record in caplog.records
        )

    def test_no_rows_raises_a_named_value_error(self):
        pca = pd.DataFrame({"sample_id": [], "dim_1": []})
        with pytest.raises(ValueError, match="no PCA rows"):
            select_by_geosketch(pca, 5, seed=0, sketch=lambda X, n, **kw: list(range(n)))

    def test_sketch_indices_are_sorted_before_selecting(self):
        pca = pd.DataFrame({"sample_id": ["A", "B", "C", "D"], "dim_1": [1, 2, 3, 4]})
        chosen = select_by_geosketch(pca, 3, seed=0, sketch=lambda X, n, **kw: [3, 0, 1])
        assert chosen == ["A", "B", "D"]


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

    def test_an_empty_fit_samples_file_is_a_named_error_before_plink_runs(self, cohort, tmp_path):
        keep = tmp_path / "keep.txt"
        keep.write_text("\n\n")
        calls = []

        def keep_runner(*args):
            calls.append(args)
            _fake_keep(*args)

        with pytest.raises(ValueError, match="empty"):
            subsample(cohort, tmp_path / "out", fit_samples=keep, keep_runner=keep_runner)
        assert calls == [], "plink must not have been started"

    def test_geosketch_selects_from_the_pca_table_restricted_to_the_fam(
        self, cohort, tmp_path, monkeypatch
    ):
        import sys

        subsample_module = sys.modules["manifold_genetics.preprocessing.subsample"]

        fam = [line.split() for line in open(tmp_path / "in/data/project_subset.fam")]
        iids = [f[1] for f in fam]
        pca_path = tmp_path / "pca.csv"
        # One extra row outside the .fam must be dropped before sketching.
        pd.DataFrame({"sample_id": iids + ["not_in_fam"], "dim_1": range(len(iids) + 1)}).to_csv(
            pca_path, index=False
        )

        captured = {}

        def fake_select(pca_df, n, seed, sketch=None, n_pcs=None):
            captured["ids"] = set(pca_df["sample_id"])
            captured["n_pcs"] = n_pcs
            return list(pca_df["sample_id"])[:n]

        monkeypatch.setattr(subsample_module, "select_by_geosketch", fake_select)
        config = subsample(
            cohort,
            tmp_path / "out",
            geosketch=3,
            pca=pca_path,
            n_pcs=1,
            keep_runner=_fake_keep,
        )
        assert captured["ids"] == set(iids)
        assert captured["n_pcs"] == 1
        fit = [line.split()[1] for line in open(f"{load_config(config)['fit_plink']}.fam")]
        assert len(fit) == 3

    def test_geosketch_selecting_nothing_is_a_named_error_before_plink_runs(
        self, cohort, tmp_path, monkeypatch
    ):
        """A fake ``sketch`` that returns no indices must fail before plink runs,
        the same way an empty --group selection does."""
        import sys

        subsample_module = sys.modules["manifold_genetics.preprocessing.subsample"]

        fam = [line.split() for line in open(tmp_path / "in/data/project_subset.fam")]
        iids = [f[1] for f in fam]
        pca_path = tmp_path / "pca.csv"
        pd.DataFrame({"sample_id": iids, "dim_1": range(len(iids))}).to_csv(pca_path, index=False)

        def fake_select(pca_df, n, seed, sketch=None, n_pcs=None):
            return []

        monkeypatch.setattr(subsample_module, "select_by_geosketch", fake_select)
        calls = []

        def keep_runner(*args):
            calls.append(args)
            _fake_keep(*args)

        with pytest.raises(ValueError, match="no samples"):
            subsample(
                cohort,
                tmp_path / "out",
                geosketch=3,
                pca=pca_path,
                keep_runner=keep_runner,
            )
        assert calls == [], "plink must not have been started"

    def test_geosketch_requires_pca(self, cohort, tmp_path):
        with pytest.raises(ValueError, match="--pca"):
            subsample(cohort, tmp_path / "out", geosketch=3, keep_runner=_fake_keep)

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
        with pytest.raises(ValueError, match="one of"):
            subsample(
                cohort,
                tmp_path / "out",
                groups=[Group("branch", "0", 1)],
                geosketch=5,
                pca=tmp_path / "pca.csv",
                keep_runner=_fake_keep,
            )

    def test_a_group_matching_nothing_is_a_named_error_before_plink_runs(self, cohort, tmp_path):
        calls = []

        def keep_runner(*args):
            calls.append(args)
            _fake_keep(*args)

        with pytest.raises(ValueError, match="no samples"):
            subsample(
                cohort,
                tmp_path / "out",
                groups=[Group("branch", "^NoSuchBranch$", 3)],
                keep_runner=keep_runner,
            )
        assert calls == [], "plink must not have been started"

    def test_labels_below_half_coverage_fail_before_plink_runs(self, cohort, tmp_path):
        """The fit set is drawn from the project .fam, so its label coverage can
        only be the input's: check that before plink2 --keep is started."""
        fam = [line.split() for line in open(tmp_path / "in/data/project_subset.fam")]
        keep_ids = {f[1] for f in fam[: int(len(fam) * 0.3)]}
        labels_path = tmp_path / "in/data/labels.csv"
        labels = pd.read_csv(labels_path, dtype=str)
        labels[labels["sample_id"].isin(keep_ids)].to_csv(labels_path, index=False)
        calls = []

        def keep_runner(*args):
            calls.append(args)
            _fake_keep(*args)

        with pytest.raises(ValueError, match="30.0%"):
            subsample(
                cohort,
                tmp_path / "out",
                fit_samples=tmp_path / "in/data/project_subset.fam",
                keep_runner=keep_runner,
            )
        assert calls == [], "plink must not have been started"

    def _add_geographic_coords(self, tmp_path, covered_ids):
        """Append a geographic_coords entry to the fixture cohort's data
        section, and write a CSV covering only ``covered_ids``."""
        geo = pd.DataFrame(
            {
                "sample_id": covered_ids,
                "latitude": range(len(covered_ids)),
                "longitude": range(len(covered_ids)),
            }
        )
        geo.to_csv(tmp_path / "in/data/geographic.csv", index=False)
        config_path = tmp_path / "in/config.yaml"
        text = config_path.read_text()
        assert "output_dir: outputs\n" in text
        config_path.write_text(
            text.replace(
                "output_dir: outputs\n",
                "output_dir: outputs\n  geographic_coords: data/geographic.csv\n",
            )
        )

    def test_geographic_coords_are_carried_and_filtered_to_the_project_fam(self, cohort, tmp_path):
        fam = pd.read_csv(
            tmp_path / "in/data/project_subset.fam", sep=r"\s+", header=None, dtype=str
        )
        ids = list(fam[1])
        covered = ids[::2]
        self._add_geographic_coords(tmp_path, covered)

        config = subsample(
            cohort,
            tmp_path / "out",
            groups=[Group("branch", "^Branch 1$", 3)],
            include_rest=False,
            keep_runner=_fake_keep,
        )
        loaded = load_config(config)
        assert loaded["geographic_coords"] == (tmp_path / "out/data/geographic.csv").resolve()

        out_project_fam = pd.read_csv(
            tmp_path / "out/data/project_subset.fam", sep=r"\s+", header=None, dtype=str
        )
        geo = pd.read_csv(loaded["geographic_coords"], dtype=str)
        assert set(geo["sample_id"]) <= set(out_project_fam[1])
        assert list(geo["sample_id"]) == [i for i in covered if i in set(out_project_fam[1])]

    def test_no_geographic_coords_in_the_input_means_none_in_the_output(self, cohort, tmp_path):
        config = subsample(
            cohort,
            tmp_path / "out",
            groups=[Group("branch", "^Branch 1$", 3)],
            include_rest=False,
            keep_runner=_fake_keep,
        )
        assert "geographic_coords" not in load_config(config)
