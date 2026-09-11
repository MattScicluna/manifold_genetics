"""The cohort registry: which real datasets the pipeline is tested against.

Most of these need controlled-access data that no clean checkout has, so the
tests that use them must skip cleanly rather than fail. The resolution logic is
the part worth testing quickly -- it decides whether an expensive test runs at
all, and a bug here would silently skip everything, which looks exactly like
success.

Data roots come from one environment variable per cohort (``MG_HGDP_DATA``,
``MG_AOU_PROJECTION_DATA``, ...) falling back to the example directories, so no
test hardcodes a cluster path and none needs configuring where the examples are
already prepared.
"""

from pathlib import Path

from tests.integration.cohorts import COHORTS, Cohort, rebase, resolve_data_root

REPO = Path(__file__).resolve().parents[2]


class TestRegistry:
    def test_every_shipped_config_has_a_cohort_entry(self):
        """A new example must come with a statement of how it is tested.

        Otherwise an example can be added, look supported, and never be exercised
        against real data.
        """
        shipped = {str(p.relative_to(REPO)) for p in REPO.glob("examples/**/config.yaml")}
        registered = {c.config for c in COHORTS.values()}
        # Templates are placeholders, not runnable cohorts.
        shipped -= {
            "examples/generic/subset/config.yaml",
            "examples/generic/hgdp_1kgp_proj/config.yaml",
        }

        assert shipped == registered, (
            f"configs without a cohort entry: {sorted(shipped - registered)}\n"
            f"cohort entries without a config: {sorted(registered - shipped)}"
        )

    def test_public_cohort_is_marked_public(self):
        assert COHORTS["hgdp"].public is True

    def test_controlled_cohorts_are_marked_private(self):
        for name in ("ukbb_projection", "ukbb_subsample", "aou_projection"):
            assert COHORTS[name].public is False, f"{name} should be private"

    def test_every_cohort_names_an_environment_variable(self):
        for name, cohort in COHORTS.items():
            assert cohort.env_var, f"{name} has no env var for its data root"


class TestResolveDataRoot:
    def test_uses_the_environment_variable_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MG_TEST_ROOT", str(tmp_path))
        cohort = Cohort(
            name="x",
            config="examples/hgdp_1kgp/config.yaml",
            env_var="MG_TEST_ROOT",
            public=True,
            requires=(),
        )

        assert resolve_data_root(cohort) == tmp_path

    def test_falls_back_to_the_config_s_own_directory(self, monkeypatch):
        monkeypatch.delenv("MG_TEST_ROOT", raising=False)
        cohort = Cohort(
            name="x",
            config="examples/hgdp_1kgp/config.yaml",
            env_var="MG_TEST_ROOT",
            public=True,
            requires=(),
        )

        assert resolve_data_root(cohort) == REPO / "examples/hgdp_1kgp"

    def test_returns_none_when_the_environment_variable_points_nowhere(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MG_TEST_ROOT", str(tmp_path / "absent"))
        cohort = Cohort(
            name="x",
            config="examples/hgdp_1kgp/config.yaml",
            env_var="MG_TEST_ROOT",
            public=True,
            requires=(),
        )

        assert resolve_data_root(cohort) is None


class TestAvailability:
    def test_a_cohort_missing_its_required_files_is_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MG_TEST_ROOT", str(tmp_path))
        cohort = Cohort(
            name="x",
            config="examples/hgdp_1kgp/config.yaml",
            env_var="MG_TEST_ROOT",
            public=True,
            requires=("data/fit_subset.bed",),
        )

        assert cohort.missing(tmp_path) == ["data/fit_subset.bed"]

    def test_a_cohort_with_all_its_files_is_available(self, tmp_path, monkeypatch):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "fit_subset.bed").write_bytes(b"")
        cohort = Cohort(
            name="x",
            config="examples/hgdp_1kgp/config.yaml",
            env_var="MG_TEST_ROOT",
            public=True,
            requires=("data/fit_subset.bed",),
        )

        assert cohort.missing(tmp_path) == []

    def test_the_public_cohort_lists_the_files_it_needs(self):
        # A cohort that requires nothing would appear available always, and its
        # test would fail confusingly instead of skipping.
        assert COHORTS["hgdp"].requires


def test_marker_is_registered():
    """An unregistered marker is silently ignored under --strict-markers."""
    config = (REPO / "pyproject.toml").read_text()
    assert "requires_private_data" in config


class TestRebaseOntoDataRoot:
    """Config paths resolve against the config file; the data root may be elsewhere.

    Without this, setting the environment variable would look like it worked and
    the test would quietly read the in-repo example data instead.
    """

    cohort = Cohort(
        name="x",
        config="examples/hgdp_1kgp/config.yaml",
        env_var="MG_TEST_ROOT",
        public=True,
        requires=(),
    )

    def test_a_path_inside_the_example_moves_to_the_root(self, tmp_path):
        inside = REPO / "examples/hgdp_1kgp/data/fit_subset.bed"

        assert rebase(inside, self.cohort, tmp_path) == tmp_path / "data/fit_subset.bed"

    def test_a_path_outside_the_example_is_left_alone(self, tmp_path):
        # Colormaps live in examples/colormaps/ and are tracked in the repo, so
        # they are not part of anybody's private data root.
        outside = REPO / "examples/colormaps/hgdp_1kgp.json"

        assert rebase(outside, self.cohort, tmp_path) == outside

    def test_rebasing_onto_the_example_itself_is_a_no_op(self):
        inside = REPO / "examples/hgdp_1kgp/data/fit_subset.bed"
        root = REPO / "examples/hgdp_1kgp"

        assert rebase(inside, self.cohort, root) == inside
