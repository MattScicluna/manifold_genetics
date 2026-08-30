"""Tests for pipeline sub-config dataclasses and build_configs()."""

import dataclasses
from pathlib import Path

import pytest

from manifold_genetics.pipeline.config import (
    AdmixtureConfig,
    EmbeddingConfig,
    IOConfig,
    PCAConfig,
    PipelineConfigs,
    SkipConfig,
    VizConfig,
    build_configs,
)


def _io(**overrides):
    base = dict(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=Path("out"),
        fit_labels=Path("fit_labels.csv"),
        project_labels=Path("project_labels.csv"),
        fit_colormap=Path("fit_cmap.json"),
        project_colormap=Path("project_cmap.json"),
    )
    base.update(overrides)
    return IOConfig(**base)


class TestSubConfigs:
    def test_ioconfig_holds_paths_and_defaults_geographic_to_none(self):
        io = _io()
        assert io.fit_plink == Path("data/fit")
        assert io.geographic_coords is None

    def test_all_configs_are_frozen(self):
        for cfg in (
            _io(),
            PCAConfig(),
            AdmixtureConfig(),
            EmbeddingConfig(),
            VizConfig(),
            SkipConfig(),
        ):
            first_field = next(iter(dataclasses.fields(cfg))).name
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(cfg, first_field, "x")

    def test_defaults_match_current_pipeline(self):
        assert PCAConfig().n_pcs == 50
        assert PCAConfig().force is False
        assert (AdmixtureConfig().k_min, AdmixtureConfig().k_max) == (2, 10)
        assert AdmixtureConfig().threads is None
        assert EmbeddingConfig().method == "phate"
        assert EmbeddingConfig().input_mode == "both"
        assert EmbeddingConfig().params == {}
        assert VizConfig().admix_within_group_order == "chron"
        assert SkipConfig().skip_pca is False

    def test_embedding_params_default_is_not_shared_between_instances(self):
        a = EmbeddingConfig()
        b = EmbeddingConfig()
        assert a.params is not b.params

    def test_pipeline_configs_container_groups_the_six(self):
        pc = PipelineConfigs(
            io=_io(),
            pca=PCAConfig(),
            admixture=AdmixtureConfig(),
            embedding=EmbeddingConfig(),
            viz=VizConfig(),
            skips=SkipConfig(),
        )
        assert isinstance(pc.io, IOConfig)
        assert isinstance(pc.skips, SkipConfig)


def _required(**overrides):
    base = dict(
        fit_plink="data/fit",
        project_plink="data/project",
        output_dir="out",
        labels="labels.csv",
        colormap="cmap.json",
    )
    base.update(overrides)
    return base


class TestBuildConfigs:
    def test_shared_labels_and_colormap_fan_out_to_both_cohorts(self):
        pc = build_configs(**_required())
        assert pc.io.fit_labels == Path("labels.csv")
        assert pc.io.project_labels == Path("labels.csv")
        assert pc.io.fit_colormap == Path("cmap.json")
        assert pc.io.project_colormap == Path("cmap.json")

    def test_separate_labels_and_colormaps_pass_through(self):
        pc = build_configs(
            fit_plink="f",
            project_plink="p",
            output_dir="out",
            fit_labels="fl.csv",
            project_labels="pl.csv",
            fit_colormap="fc.json",
            project_colormap="pc.json",
        )
        assert pc.io.fit_labels == Path("fl.csv")
        assert pc.io.project_labels == Path("pl.csv")
        assert pc.io.fit_colormap == Path("fc.json")
        assert pc.io.project_colormap == Path("pc.json")

    def test_str_paths_become_path_objects(self):
        pc = build_configs(**_required())
        assert isinstance(pc.io.fit_plink, Path)
        assert isinstance(pc.io.output_dir, Path)

    def test_geographic_coords_optional(self):
        assert build_configs(**_required()).io.geographic_coords is None
        pc = build_configs(**_required(geographic_coords="geo.csv"))
        assert pc.io.geographic_coords == Path("geo.csv")

    def test_scalar_params_land_in_the_right_sub_config(self):
        pc = build_configs(
            **_required(
                n_pcs=30,
                force_pca=True,
                k_min=3,
                k_max=6,
                admix_threads=8,
                admix_gpus=1,
                admix_batch_size=400,
                embedding="umap",
                embedding_input="fit",
                embedding_params={"n_neighbors": 20},
                admix_group_column="region",
                admix_within_group_order="tree",
                projection_plot_fit_column="Population",
                projection_plot_project_column="ancestry",
                skip_metrics=True,
            )
        )
        assert (pc.pca.n_pcs, pc.pca.force) == (30, True)
        assert (pc.admixture.k_min, pc.admixture.k_max) == (3, 6)
        assert pc.admixture.threads == 8
        assert pc.admixture.num_gpus == 1
        assert pc.admixture.batch_size == 400
        assert pc.embedding.method == "umap"
        assert pc.embedding.input_mode == "fit"
        assert pc.embedding.params == {"n_neighbors": 20}
        assert pc.viz.admix_group_column == "region"
        assert pc.viz.projection_plot_fit_column == "Population"
        assert pc.viz.admix_within_group_order == "tree"
        assert pc.viz.projection_plot_project_column == "ancestry"
        assert pc.skips.skip_metrics is True

    def test_missing_labels_raises_with_the_current_message(self):
        with pytest.raises(ValueError, match="Must provide either 'labels'"):
            build_configs(fit_plink="f", project_plink="p", output_dir="out", colormap="c.json")

    def test_only_one_of_fit_project_labels_raises(self):
        with pytest.raises(ValueError, match="only one of 'fit_labels'"):
            build_configs(
                fit_plink="f",
                project_plink="p",
                output_dir="out",
                fit_labels="fl.csv",
                colormap="c.json",
            )

    def test_missing_colormap_raises(self):
        with pytest.raises(ValueError, match="Must provide either 'colormap'"):
            build_configs(fit_plink="f", project_plink="p", output_dir="out", labels="l.csv")

    def test_unknown_embedding_method_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding method: 'wavelet'"):
            build_configs(**_required(embedding="wavelet"))

    def test_unknown_embedding_input_mode_raises(self):
        with pytest.raises(ValueError, match="embedding_input"):
            build_configs(**_required(embedding_input="sideways"))

    def test_k_min_greater_than_k_max_raises(self):
        with pytest.raises(ValueError, match="k_min .* k_max"):
            build_configs(**_required(k_min=8, k_max=3))

    def test_non_positive_n_pcs_raises(self):
        with pytest.raises(ValueError, match="n_pcs"):
            build_configs(**_required(n_pcs=0))

    def test_unknown_within_group_order_raises(self):
        with pytest.raises(ValueError, match="admix_within_group_order"):
            build_configs(**_required(admix_within_group_order="sideways"))

    def test_empty_string_labels_raises(self):
        with pytest.raises(ValueError, match="Must provide either 'labels'"):
            build_configs(
                fit_plink="f",
                project_plink="p",
                output_dir="out",
                labels="",
                colormap="c.json",
            )

    def test_empty_string_one_of_fit_project_labels_raises(self):
        with pytest.raises(ValueError, match="only one of 'fit_labels'"):
            build_configs(
                fit_plink="f",
                project_plink="p",
                output_dir="out",
                fit_labels="",
                project_labels="pl.csv",
                colormap="c.json",
            )

    def test_empty_string_colormap_raises(self):
        with pytest.raises(ValueError, match="Must provide either 'colormap'"):
            build_configs(
                fit_plink="f",
                project_plink="p",
                output_dir="out",
                labels="l.csv",
                colormap="",
            )
