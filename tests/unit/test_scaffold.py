"""`manifold-genetics init` — the scaffolding a fresh install has nothing without.

`pip install manifold-genetics` ships no example config and no data: `examples/`
is in neither the wheel nor the sdist. Until this command existed the
documentation's opening move, `manifold-genetics run config.yaml`, was
unreachable for everyone who installed the package the documented way.
"""

import json

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.scaffold import (
    DLA_TREE_EDGES,
    DLA_TREE_GAPS,
    dla_tree,
    genotypes_from_coordinates,
    hgdp_subsets,
    init_synthetic,
)


class TestDlaTree:
    """The cohort lies along the tree in manylatents' `dla_tree_from_graph.yaml`.

    The generator is a port, not an import, and was checked against the
    original array for array (seed 42, 2026-09-13). These guard the properties
    that check relied on, so a later edit cannot quietly break the match.
    """

    def test_one_sample_per_position_on_each_data_edge(self):
        coordinates, branch = dla_tree()

        data_edges = [edge for edge in DLA_TREE_EDGES if edge[2] not in DLA_TREE_GAPS]
        assert coordinates.shape == (sum(edge[3] for edge in data_edges), 100)
        counts = dict(zip(*np.unique(branch, return_counts=True)))
        assert counts == {edge[2]: edge[3] for edge in data_edges}

    def test_is_reproducible(self):
        first, _ = dla_tree()
        second, _ = dla_tree()

        assert np.array_equal(first, second)

    def test_a_gap_edge_leaves_a_gap(self):
        """Edge 2 continues edge 1 at node 2, so it starts where edge 1 ends.
        Edge 3 starts at node 8, on the far side of a gap edge from node 2, so
        it does not -- that gap is the point of having gap edges."""
        coordinates, branch = dla_tree(sigma=0.0)

        end_of_1 = coordinates[branch == 1][-1]
        step = np.linalg.norm(np.diff(coordinates[branch == 1], axis=0), axis=1).max()

        assert np.allclose(coordinates[branch == 2][0], end_of_1)
        assert np.linalg.norm(coordinates[branch == 3][0] - end_of_1) > 10 * step

    def test_genotypes_are_plink_dosages(self):
        coordinates, _ = dla_tree()

        dosages = genotypes_from_coordinates(coordinates, n_variants=50)

        assert dosages.shape == (len(coordinates), 50)
        assert set(np.unique(dosages)) <= {0, 1, 2}


class TestSyntheticScaffold:
    def test_writes_everything_a_run_needs(self, tmp_path):
        written = init_synthetic(tmp_path)

        for name in (
            "config.yaml",
            "data/fit_subset.bed",
            "data/fit_subset.bim",
            "data/fit_subset.fam",
            "data/project_subset.bed",
            "data/project_subset.bim",
            "data/project_subset.fam",
            "data/labels.csv",
            "colormap.json",
        ):
            assert (tmp_path / name).exists(), f"init did not write {name}"
        assert written == tmp_path / "config.yaml"

    def test_the_config_it_writes_is_accepted_by_the_loader(self, tmp_path):
        """The point of the command: what it writes must actually load."""
        from manifold_genetics.pipeline.configfile import load_config

        init_synthetic(tmp_path)
        kwargs = load_config(tmp_path / "config.yaml")

        assert kwargs["fit_plink"] == tmp_path / "data/fit_subset"
        assert kwargs["output_dir"] == tmp_path / "outputs"

    def test_labels_cover_every_genotyped_sample(self, tmp_path):
        """A label file that does not match its genotypes is this project's
        most expensive failure mode, so the one it ships cannot have it."""
        init_synthetic(tmp_path)

        fam = pd.read_csv(tmp_path / "data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
        labels = pd.read_csv(tmp_path / "data/labels.csv", dtype=str)

        assert set(labels["sample_id"]) >= set(fam[1])

    def test_the_colormap_names_a_real_label_column(self, tmp_path):
        init_synthetic(tmp_path)

        colormap = json.loads((tmp_path / "colormap.json").read_text())
        labels = pd.read_csv(tmp_path / "data/labels.csv")

        for column, values in colormap.items():
            assert column in labels.columns, f"colormap names {column!r}, which is not a column"
            assert set(labels[column].astype(str)) <= set(values), "a label value has no colour"

    def test_refuses_to_overwrite_without_force(self, tmp_path):
        init_synthetic(tmp_path)

        with pytest.raises(FileExistsError, match="--force"):
            init_synthetic(tmp_path)

        init_synthetic(tmp_path, force=True)  # explicit is fine

    def test_the_bed_it_writes_is_readable(self, tmp_path):
        """It hand-encodes PLINK's 2-bit format, so round-tripping is the check."""
        from manifold_genetics.pca.plink import count_lines, read_bed_dosages

        init_synthetic(tmp_path)
        prefix = tmp_path / "data/fit_subset"
        n_samples = count_lines(f"{prefix}.fam")
        n_variants = count_lines(f"{prefix}.bim")

        dosages = read_bed_dosages(prefix, n_samples=n_samples, n_variants=n_variants)

        assert dosages.shape == (n_samples, n_variants)
        assert set(pd.unique(dosages.ravel())) <= {0.0, 1.0, 2.0}


class TestHgdpSubsets:
    """The two subsets are boolean filters on the cohort's metadata.

    Ported from examples/hgdp_1kgp/prepare_data.sh, which produced the subsets
    every published figure was made from, so the rule has to match it exactly:
    the fit set additionally excludes related samples, the project set does not.
    """

    @pytest.fixture
    def metadata(self):
        return pd.DataFrame(
            {
                "project_meta.sample_id": ["clean", "related", "outlier", "hard", "dirty"],
                "filter_pca_outlier": [False, False, True, False, False],
                "hard_filtered": [False, False, False, True, False],
                "filter_king_related": [False, True, False, False, False],
                "filter_contaminated": [False, False, False, False, True],
            }
        )

    def test_fit_excludes_related_samples(self, metadata):
        fit, _ = hgdp_subsets(metadata)

        assert list(fit) == ["clean"]

    def test_project_keeps_related_samples(self, metadata):
        _, project = hgdp_subsets(metadata)

        assert list(project) == ["clean", "related"]

    def test_both_drop_outliers_hard_filtered_and_contaminated(self, metadata):
        fit, project = hgdp_subsets(metadata)

        for excluded in ("outlier", "hard", "dirty"):
            assert excluded not in list(fit)
            assert excluded not in list(project)
