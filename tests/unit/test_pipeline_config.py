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
