"""painting.genotype_weight: the genotype PCA and the painting side by side.

The weighting is tested on synthetic coordinate CSVs, and the orchestrator
with the painting, PCA and embedding steps replaced by fakes.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from manifold_genetics.pipeline.config import PAINTING_DEFAULT_T, build_painting_config
from manifold_genetics.pipeline.configfile import ConfigFileError, load_config
from manifold_genetics.pipeline.steps.painting import weight_with_genotype
from manifold_genetics.pipeline.steps.pca import PCAStepResult


def blobs(seed, n_per=40, dims=12, groups=4, spread=5.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(groups, dims)) * spread
    x = np.vstack([c + rng.normal(size=(n_per, dims)) for c in centers])
    df = pd.DataFrame(x, columns=[f"dim_{i + 1}" for i in range(dims)])
    df.insert(0, "sample_id", [f"s{i}" for i in range(len(x))])
    return df


@pytest.fixture
def cohorts(tmp_path):
    """Genotype and painting coordinates: a fit cohort of 120 and a project cohort of 160."""
    files = {}
    for name, df in (("geno", blobs(5)), ("paint", blobs(0))):
        files[f"{name}_fit"] = tmp_path / f"{name}_fit.csv"
        files[f"{name}_project"] = tmp_path / f"{name}_project.csv"
        df.iloc[:120].to_csv(files[f"{name}_fit"], index=False)
        df.to_csv(files[f"{name}_project"], index=False)
    geno = PCAStepResult(fit_pca=files["geno_fit"], project_pca=files["geno_project"])
    paint = PCAStepResult(fit_pca=files["paint_fit"], project_pca=files["paint_project"])
    return geno, paint


def _weigh(cohorts, tmp_path, weight, paint=True):
    geno, painting = cohorts
    return weight_with_genotype(
        genotype_fit=geno.fit_pca,
        genotype_project=geno.project_pca,
        painting_fit=painting.fit_pca if paint else None,
        painting_project=painting.project_pca if paint else None,
        weight=weight,
        fit_output=tmp_path / f"w{weight}_fit.csv",
        project_output=tmp_path / f"w{weight}_project.csv",
    )


def _squared_distance_sum(x):
    return float((((x[:, None, :] - x[None, :, :]) ** 2).sum()))


# --- config -----------------------------------------------------------------------


@pytest.mark.parametrize("weight", [0, 0.25, 1, 1.0])
def test_genotype_weight_accepts_zero_to_one(weight):
    assert build_painting_config(genotype_weight=weight).genotype_weight == float(weight)


@pytest.mark.parametrize("weight", [-0.1, 1.5, "half", True, None])
def test_genotype_weight_refuses_anything_else(weight):
    with pytest.raises(ValueError, match="genotype_weight must be a number from 0"):
        build_painting_config(genotype_weight=weight)


def test_genotype_weight_one_skips_the_painting():
    assert build_painting_config(genotype_weight=0.99).paints is False  # not enabled
    on = dict(enabled=True, phased_vcf="chr{chr}.vcf.gz")
    assert build_painting_config(genotype_weight=0.99, **on).paints is True
    assert build_painting_config(genotype_weight=1, **on).paints is False


def _config(tmp_path, painting, embedding=None):
    cfg = {
        "data": {
            "fit_plink": "fit",
            "project_plink": "project",
            "labels": "labels.csv",
            "colormap": "cmap.json",
            "output_dir": "out",
            "phased_vcf": "chr{chr}.vcf.gz",
        },
        "painting": painting,
    }
    if embedding is not None:
        cfg["embedding"] = embedding
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_config_file_carries_genotype_weight(tmp_path):
    kwargs = load_config(_config(tmp_path, {"enabled": True, "genotype_weight": 0.25}))
    assert kwargs["painting_genotype_weight"] == 0.25


@pytest.mark.parametrize(
    "key", ["combine_genotype", "combine_pcs", "combine_weight", "combine_norm"]
)
def test_config_file_refuses_the_removed_combine_keys(tmp_path, key):
    with pytest.raises(ConfigFileError, match=key):
        load_config(_config(tmp_path, {"enabled": True, key: 1}))


@pytest.mark.parametrize("key", ["init", "init_blend"])
def test_config_file_refuses_the_removed_init_keys(tmp_path, key):
    with pytest.raises(ConfigFileError, match=key):
        load_config(_config(tmp_path, {"enabled": True}, {"method": "phate", key: 0.5}))


# --- weight_with_genotype ------------------------------------------------------


@pytest.mark.parametrize("weight", [0.1, 0.5, 0.8])
def test_weight_is_the_genotype_share_of_squared_distance(cohorts, tmp_path, weight):
    fit_out, _ = _weigh(cohorts, tmp_path, weight)
    x = pd.read_csv(fit_out).iloc[:, 1:].to_numpy()
    assert x.shape == (120, 20)
    geno, paint = x[:, :10], x[:, 10:]
    for block, share in ((geno, weight), (paint, 1 - weight)):
        np.testing.assert_allclose(block.mean(axis=0), 0, atol=1e-12)
        assert np.linalg.norm(block) ** 2 == pytest.approx(share)
    total = _squared_distance_sum(x)
    assert _squared_distance_sum(geno) / total == pytest.approx(weight)


def test_blocks_keep_their_internal_shape(cohorts, tmp_path):
    """Each block is scaled as a whole, never column by column."""
    geno, _ = cohorts
    fit_out, _ = _weigh(cohorts, tmp_path, 0.5)
    x = pd.read_csv(fit_out).iloc[:, 1:11].to_numpy()
    raw = pd.read_csv(geno.fit_pca).iloc[:, 1:11].to_numpy()
    ratio = x / (raw - raw.mean(axis=0))
    np.testing.assert_allclose(ratio, ratio[0, 0])


def test_project_cohort_uses_the_fit_cohort_statistics(cohorts, tmp_path):
    fit_out, project_out = _weigh(cohorts, tmp_path, 0.5)
    fit, project = pd.read_csv(fit_out), pd.read_csv(project_out)
    assert len(project) == 160
    pd.testing.assert_frame_equal(project.iloc[:120], fit)


def test_painting_block_is_unchanged_by_its_scale(cohorts, tmp_path):
    geno, paint = cohorts
    big = pd.read_csv(paint.project_pca)
    big.iloc[:, 1:] *= 100
    big.to_csv(tmp_path / "big_project.csv", index=False)
    big.iloc[:120].to_csv(tmp_path / "big_fit.csv", index=False)
    scaled, _ = weight_with_genotype(
        genotype_fit=geno.fit_pca,
        genotype_project=None,
        painting_fit=tmp_path / "big_fit.csv",
        painting_project=None,
        weight=0.5,
        fit_output=tmp_path / "big_out.csv",
        project_output=None,
    )
    plain, _ = _weigh(cohorts, tmp_path, 0.5)
    pd.testing.assert_frame_equal(pd.read_csv(scaled), pd.read_csv(plain))


def test_weight_one_is_the_genotype_pca_alone(cohorts, tmp_path):
    geno, _ = cohorts
    fit_out, project_out = _weigh(cohorts, tmp_path, 1, paint=False)
    x = pd.read_csv(project_out).iloc[:, 1:].to_numpy()
    raw = pd.read_csv(geno.project_pca).iloc[:, 1:11].to_numpy()
    assert x.shape == (160, 10)
    # The top 10 genotype PCs up to one shift and one scale: same distance ratios.
    ratio = (x[1:] - x[0]) / (raw[1:] - raw[0])
    np.testing.assert_allclose(ratio, ratio[0, 0])


def test_weight_below_one_needs_the_painting(cohorts, tmp_path):
    with pytest.raises(ValueError, match="needs the painting"):
        _weigh(cohorts, tmp_path, 0.5, paint=False)


@pytest.mark.parametrize("weight", [0, 1.5])
def test_weight_outside_the_mixing_range_is_refused(cohorts, tmp_path, weight):
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        _weigh(cohorts, tmp_path, weight)


def test_painted_samples_without_genotype_rows_are_refused(cohorts, tmp_path):
    geno, paint = cohorts
    with pytest.raises(ValueError, match="no genotype PCA row"):
        weight_with_genotype(
            genotype_fit=geno.fit_pca,
            genotype_project=geno.fit_pca,  # lacks 40 project samples
            painting_fit=paint.fit_pca,
            painting_project=paint.project_pca,
            weight=0.5,
            fit_output=tmp_path / "a.csv",
            project_output=tmp_path / "b.csv",
        )


def test_genotype_rows_follow_the_painting_order(cohorts, tmp_path):
    geno, paint = cohorts
    shuffled = pd.read_csv(geno.project_pca).sample(frac=1, random_state=0)
    shuffled.to_csv(tmp_path / "shuffled.csv", index=False)
    _, a = _weigh(cohorts, tmp_path, 0.5)
    _, b = weight_with_genotype(
        genotype_fit=geno.fit_pca,
        genotype_project=tmp_path / "shuffled.csv",
        painting_fit=paint.fit_pca,
        painting_project=paint.project_pca,
        weight=0.5,
        fit_output=tmp_path / "s_fit.csv",
        project_output=tmp_path / "s_project.csv",
    )
    pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b))


# --- the orchestrator -------------------------------------------------------------


def _run(tmp_path, cohorts, monkeypatch, **kwargs):
    import manifold_genetics.pipeline.orchestrator as orch
    from manifold_genetics.pipeline.runner import run_pipeline

    geno, paint = cohorts
    seen = {"pca_runs": 0, "paint_runs": 0}

    def fake_pca(io, cfg):
        seen["pca_runs"] += 1
        return geno

    def fake_paint(io, pca_cfg, painting_cfg):
        seen["paint_runs"] += 1
        return SimpleNamespace(fit_pca=paint.fit_pca, project_pca=paint.project_pca, coords_df=None)

    def fake_embedding(io, emb, *, pca):
        seen["pca"], seen["emb"] = pca, emb

    monkeypatch.setattr(orch, "run_pca_step", fake_pca)
    monkeypatch.setattr(orch, "run_painting_step", fake_paint)
    monkeypatch.setattr(orch, "run_embedding_step", fake_embedding)
    run_pipeline(
        fit_plink=tmp_path / "fit",
        project_plink=tmp_path / "project",
        output_dir=tmp_path / "out",
        labels=tmp_path / "labels.csv",
        colormap=tmp_path / "cmap.json",
        phased_vcf=tmp_path / "chr{chr}.vcf",
        painting=True,
        painting_chromosomes=["1"],
        n_pcs=12,
        skip_admixture=True,
        skip_pca_visualization=True,
        skip_visualization=True,
        skip_metrics=True,
        **kwargs,
    )
    return seen


def test_weight_zero_embeds_the_painting_and_runs_no_genotype_pca(tmp_path, cohorts, monkeypatch):
    _, paint = cohorts
    seen = _run(tmp_path, cohorts, monkeypatch)
    assert (seen["pca_runs"], seen["paint_runs"]) == (0, 1)
    assert seen["pca"].fit_pca == paint.fit_pca
    assert seen["pca"].project_pca == paint.project_pca


@pytest.mark.parametrize("method", ["phate", "umap"])
def test_weight_between_embeds_both_views(tmp_path, cohorts, monkeypatch, method):
    seen = _run(tmp_path, cohorts, monkeypatch, embedding=method, painting_genotype_weight=0.25)
    d = tmp_path / "out" / "painting"
    assert (seen["pca_runs"], seen["paint_runs"]) == (1, 1)
    assert seen["pca"].fit_pca == d / "fit_painting_hc_genotype_0.25.csv"
    assert seen["pca"].project_pca == d / "project_painting_hc_genotype_0.25.csv"
    assert pd.read_csv(seen["pca"].project_pca).shape == (160, 21)


def test_weight_one_skips_the_painting(tmp_path, cohorts, monkeypatch):
    seen = _run(tmp_path, cohorts, monkeypatch, painting_genotype_weight=1)
    assert (seen["pca_runs"], seen["paint_runs"]) == (1, 0)
    assert pd.read_csv(seen["pca"].project_pca).shape == (160, 11)
    # Painting provides none of the input, so PHATE keeps its own default t.
    assert "t" not in seen["emb"].params


def test_skip_pca_reads_both_views_from_disk(tmp_path, cohorts, monkeypatch):
    """With skip: pca the genotype PCA is read from its usual paths."""
    import shutil

    geno, paint = cohorts
    pca_dir = tmp_path / "out" / "pca"
    pca_dir.mkdir(parents=True)
    shutil.copy(geno.fit_pca, pca_dir / "fit_pca_12.csv")
    shutil.copy(geno.project_pca, pca_dir / "project_pca_12.csv")
    paint_dir = tmp_path / "out" / "painting"
    paint_dir.mkdir(parents=True)
    shutil.copy(paint.fit_pca, paint_dir / "fit_painting_hc_pca_12.csv")
    shutil.copy(paint.project_pca, paint_dir / "project_painting_hc_pca_12.csv")
    seen = _run(tmp_path, cohorts, monkeypatch, skip_pca=True, painting_genotype_weight=0.5)
    assert (seen["pca_runs"], seen["paint_runs"]) == (0, 0)
    assert pd.read_csv(seen["pca"].fit_pca).shape == (120, 21)


@pytest.mark.parametrize(
    "params, expected",
    [({}, PAINTING_DEFAULT_T), ({"t": 3}, 3), ({"t": "auto"}, "auto")],
)
def test_painting_input_defaults_phate_t_to_ten(tmp_path, cohorts, monkeypatch, params, expected):
    seen = _run(tmp_path, cohorts, monkeypatch, embedding_params=params)
    assert seen["emb"].params["t"] == expected


def test_preset_t_wins_over_the_painting_default(tmp_path, cohorts, monkeypatch):
    kwargs = load_config(_preset_config(tmp_path))
    assert kwargs["embedding_params"]["t"] == 3
    seen = _run(tmp_path, cohorts, monkeypatch, embedding_params=kwargs["embedding_params"])
    assert seen["emb"].params["t"] == 3


def _preset_config(tmp_path):
    path = _config(tmp_path, {"enabled": True})
    cfg = yaml.safe_load(path.read_text())
    cfg["preset"] = "whole_cohort"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_t_default_is_only_for_phate(tmp_path, cohorts, monkeypatch):
    seen = _run(tmp_path, cohorts, monkeypatch, embedding="umap")
    assert "t" not in seen["emb"].params
