"""painting.genotype_weight: PHATE on a weighted sum of the genotype and painting kernels.

The kernels and the PHATE model are tested on synthetic coordinates, and the
orchestrator with the painting and PCA steps replaced by fakes (the embedding
step real where the output is compared).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial import procrustes

from manifold_genetics.embeddings import PHATE
from manifold_genetics.embeddings.phate import GenotypePaintingPHATE, mix_kernels
from manifold_genetics.pipeline.config import (
    GENOTYPE_KERNEL_PARAM,
    PAINTING_DEFAULT_KNN,
    PAINTING_DEFAULT_T,
    GenotypeKernel,
    build_painting_config,
    check_genotype_weight,
)
from manifold_genetics.pipeline.configfile import ConfigFileError, load_config
from manifold_genetics.pipeline.steps.embedding import build_embedding_model
from manifold_genetics.pipeline.steps.pca import PCAStepResult

PHATE_PARAMS = dict(knn=10, t=5, n_landmark=None)


def blobs(seed, n_per=40, dims=12, groups=4, spread=5.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(groups, dims)) * spread
    x = np.vstack([c + rng.normal(size=(n_per, dims)) for c in centers])
    df = pd.DataFrame(x, columns=[f"dim_{i + 1}" for i in range(dims)])
    df.insert(0, "sample_id", [f"s{i}" for i in range(len(x))])
    return df


def xy(df):
    return df[["dim_1", "dim_2"]].to_numpy()


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


# --- config -----------------------------------------------------------------------


@pytest.mark.parametrize("weight", [0, 0.25, 1, 1.0])
def test_genotype_weight_accepts_zero_to_one(weight):
    assert build_painting_config(genotype_weight=weight).genotype_weight == float(weight)


@pytest.mark.parametrize("weight", [-0.1, 1.5, "half", True, None])
def test_genotype_weight_refuses_anything_else(weight):
    with pytest.raises(ValueError, match="genotype_weight must be a number from 0"):
        build_painting_config(genotype_weight=weight)


def test_painting_knn_defaults_to_30_and_must_be_a_positive_integer():
    assert build_painting_config().knn == PAINTING_DEFAULT_KNN == 30
    assert build_painting_config(knn=10).knn == 10
    for bad in (0, -3, 2.5, "ten", True):
        with pytest.raises(ValueError, match="painting knn must be a positive integer"):
            build_painting_config(knn=bad)


def test_paints_and_mixes():
    on = dict(enabled=True, phased_vcf="chr{chr}.vcf.gz")
    assert build_painting_config(genotype_weight=0.5).paints is False  # not enabled
    assert build_painting_config(genotype_weight=0.5).mixes is False
    assert build_painting_config(genotype_weight=0.5, **on).mixes is True
    assert build_painting_config(genotype_weight=0, **on).mixes is False
    assert build_painting_config(genotype_weight=1, **on).paints is False
    assert build_painting_config(genotype_weight=1, **on).mixes is False


@pytest.mark.parametrize("weight", [0, 1])
@pytest.mark.parametrize("method", ["phate", "umap", "tsne", "diffusion_map"])
def test_weights_zero_and_one_run_anything(weight, method):
    check_genotype_weight(
        weight, method, "both", {"n_landmark": 2000}, {"umap": None}, {"n_landmark": [None, 50]}
    )


@pytest.mark.parametrize(
    "args, match",
    [
        (("umap", "project"), "needs embedding method phate, not 'umap'"),
        (("tsne", "fit"), "needs embedding method phate, not 'tsne'"),
        (("diffusion_map", "fit"), "needs embedding method phate, not 'diffusion_map'"),
        (("phate", "both"), "input_mode both"),
        (("phate", "fit", {"n_landmark": 2000}), "n_landmark to none"),
        (("phate", "fit", {}, {"phate": None, "umap": None}), "method_grid can hold only phate"),
        (("phate", "fit", {}, {"phate": {"n_landmark": 50}}), "method_grid's phate panel"),
        (("phate", "fit", {}, None, {"n_landmark": [None, 50]}), "phate_grid cannot set"),
    ],
)
def test_weights_between_refuse_what_the_mixed_kernel_cannot_run(args, match):
    with pytest.raises(ValueError, match=match):
        check_genotype_weight(0.5, *args)


def test_weights_between_accept_exact_phate_on_one_cohort_and_its_grids():
    check_genotype_weight(
        0.5,
        "phate",
        "project",
        {"knn": 100, "t": 10, "n_landmark": None},
        {"phate": {"knn": 30}},
        {"knn": [30, 100], "t": [3, 10], "n_landmark": None},
    )


def _config(tmp_path, painting, embedding=None, preset=None, visualization=None):
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
    if preset is not None:
        cfg["preset"] = preset
    if visualization is not None:
        cfg["visualization"] = visualization
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_config_file_carries_genotype_weight_and_painting_knn(tmp_path):
    kwargs = load_config(
        _config(
            tmp_path,
            {"enabled": True, "genotype_weight": 0.25, "knn": 15},
            {"input_mode": "project"},
        )
    )
    assert kwargs["painting_genotype_weight"] == 0.25
    assert kwargs["painting_knn"] == 15


@pytest.mark.parametrize(
    "embedding, preset, visualization, match",
    [
        ({"method": "umap", "input_mode": "fit"}, None, None, "not 'umap'"),
        (None, None, None, "input_mode both"),  # the run's default
        (None, "projection", None, "projection preset"),
        (None, "subsample", None, "n_landmark to none"),
        (
            {"input_mode": "fit"},
            None,
            {"method_grid": {"phate": None, "tsne": None}},
            "only phate, not 'tsne'",
        ),
    ],
)
def test_config_file_refuses_mixed_kernels_where_they_cannot_run(
    tmp_path, embedding, preset, visualization, match
):
    path = _config(
        tmp_path, {"enabled": True, "genotype_weight": 0.5}, embedding, preset, visualization
    )
    with pytest.raises(ConfigFileError, match=match):
        load_config(path)


def test_config_file_accepts_the_whole_cohort_preset(tmp_path):
    path = _config(tmp_path, {"enabled": True, "genotype_weight": 0.5}, preset="whole_cohort")
    assert load_config(path)["painting_genotype_weight"] == 0.5


def test_config_file_checks_the_weight_only_with_painting_on(tmp_path):
    path = _config(tmp_path, {"enabled": False, "genotype_weight": 0.5}, {"method": "umap"})
    assert load_config(path)["painting_genotype_weight"] == 0.5


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


# --- the kernels ------------------------------------------------------------------


def _kernels():
    model = GenotypePaintingPHATE(
        genotype=blobs(5), genotype_weight=0.5, painting_knn=10, **PHATE_PARAMS
    )
    geno = blobs(5).drop(columns="sample_id").to_numpy()[:, :10]
    paint = blobs(0).drop(columns="sample_id").to_numpy()
    return model.view_kernel(geno, 10), model.view_kernel(paint, 10)


def test_mixed_kernel_is_symmetric_and_nonnegative():
    K = mix_kernels(*_kernels(), 0.3).toarray()
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert K.min() >= 0 and K.sum(axis=1).min() > 0


def test_each_kernel_is_divided_by_its_mean_degree():
    K_g, K_p = _kernels()
    # Scaling one view changes nothing: the weight alone sets its share.
    np.testing.assert_allclose(
        mix_kernels(100 * K_g, K_p, 0.3).toarray(), mix_kernels(K_g, K_p, 0.3).toarray()
    )
    only_genotype = mix_kernels(K_g, K_p, 1.0)
    assert only_genotype.sum(axis=1).mean() == pytest.approx(1.0)
    mixed = mix_kernels(K_g, K_p, 0.3)
    assert mixed.sum(axis=1).mean() == pytest.approx(1.0)


def test_mix_kernels_validates():
    eye = np.eye(3)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        mix_kernels(eye, eye, -0.1)
    with pytest.raises(ValueError, match="differ in shape"):
        mix_kernels(eye, np.eye(4), 0.5)


@pytest.mark.parametrize("weight, view", [(1.0, "genotype"), (0.0, "painting")])
def test_weight_one_or_zero_reproduces_single_view_phate(weight, view):
    geno, paint = blobs(5), blobs(0)
    if view == "genotype":
        single_input, knn = geno.iloc[:, :11], 10  # sample_id and the top 10 PCs
    else:
        single_input, knn = paint, 7
    single = PHATE(**dict(PHATE_PARAMS, knn=knn))
    single.fit(single_input)
    expected = single.transform(single_input)

    model = GenotypePaintingPHATE(
        genotype=geno, genotype_weight=weight, painting_knn=7, **PHATE_PARAMS
    )
    got = model.fit_transform(paint)
    assert list(got["sample_id"]) == list(paint["sample_id"])
    np.testing.assert_allclose(xy(got), xy(expected), atol=1e-3)
    assert procrustes(xy(expected), xy(got))[2] < 1e-6


def test_genotype_rows_follow_the_painting_sample_ids():
    geno, paint = blobs(5), blobs(0)
    kwargs = dict(genotype_weight=0.5, painting_knn=10, **PHATE_PARAMS)
    in_order = GenotypePaintingPHATE(genotype=geno, **kwargs).fit_transform(paint)
    shuffled = geno.sample(frac=1.0, random_state=0)
    got = GenotypePaintingPHATE(genotype=shuffled, **kwargs).fit_transform(paint)
    np.testing.assert_allclose(xy(got), xy(in_order), atol=1e-6)


def test_painted_samples_without_genotype_rows_are_refused():
    geno, paint = blobs(5), blobs(0)
    model = GenotypePaintingPHATE(
        genotype=geno.iloc[:150], genotype_weight=0.5, painting_knn=10, **PHATE_PARAMS
    )
    with pytest.raises(ValueError, match="10 of the 160 painted samples have no genotype"):
        model.fit(paint)


def test_mixed_phate_refuses_landmarks_and_new_samples():
    geno, paint = blobs(5), blobs(0)
    with pytest.raises(ValueError, match="exact only"):
        GenotypePaintingPHATE(
            genotype=geno, genotype_weight=0.5, painting_knn=10, **dict(PHATE_PARAMS, n_landmark=50)
        )
    model = GenotypePaintingPHATE(
        genotype=geno, genotype_weight=0.5, painting_knn=10, **PHATE_PARAMS
    )
    model.fit(paint)
    with pytest.raises(ValueError, match="cannot place new samples"):
        model.transform(paint.iloc[:10])


def test_build_embedding_model_reads_the_genotype_kernel(tmp_path):
    kernel = GenotypeKernel(genotype=tmp_path / "geno.csv", weight=0.25, painting_knn=12)
    model = build_embedding_model("phate", {"knn": 80, "t": 10, GENOTYPE_KERNEL_PARAM: kernel})
    assert isinstance(model, GenotypePaintingPHATE)
    assert (model.knn, model.painting_knn, model.genotype_weight, model.t) == (80, 12, 0.25, 10)
    assert model.genotype_components == 10
    with pytest.raises(ValueError, match="needs phate, not 'umap'"):
        build_embedding_model("umap", {GENOTYPE_KERNEL_PARAM: kernel})
    assert type(build_embedding_model("phate", {"knn": 80})) is PHATE


# --- the orchestrator -------------------------------------------------------------


def _run(tmp_path, cohorts, monkeypatch, real_embedding=False, painting=True, **kwargs):
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

    real = orch.run_embedding_step

    def fake_embedding(io, emb, *, pca):
        seen["pca"], seen["emb"] = pca, emb
        if real_embedding:
            seen["result"] = real(io, emb, pca=pca)
        return seen.get("result")

    monkeypatch.setattr(orch, "run_pca_step", fake_pca)
    monkeypatch.setattr(orch, "run_painting_step", fake_paint)
    monkeypatch.setattr(orch, "run_embedding_step", fake_embedding)
    kwargs.setdefault("embedding_input", "project")
    kwargs.setdefault("skip_visualization", True)
    run_pipeline(
        fit_plink=tmp_path / "fit",
        project_plink=tmp_path / "project",
        output_dir=tmp_path / "out",
        labels=tmp_path / "labels.csv",
        colormap=tmp_path / "cmap.json",
        phased_vcf=tmp_path / "chr{chr}.vcf",
        painting=painting,
        painting_chromosomes=["1"],
        n_pcs=12,
        skip_admixture=True,
        skip_pca_visualization=True,
        skip_metrics=True,
        **kwargs,
    )
    return seen


def test_weight_zero_embeds_the_painting_with_painting_knn(tmp_path, cohorts, monkeypatch):
    _, paint = cohorts
    seen = _run(tmp_path, cohorts, monkeypatch, embedding_params={"knn": 100})
    assert (seen["pca_runs"], seen["paint_runs"]) == (0, 1)
    assert seen["pca"].project_pca == paint.project_pca
    assert seen["emb"].params == {"knn": PAINTING_DEFAULT_KNN, "t": PAINTING_DEFAULT_T}
    seen = _run(tmp_path, cohorts, monkeypatch, painting_knn=12)
    assert seen["emb"].params["knn"] == 12


def test_weight_zero_leaves_other_methods_alone(tmp_path, cohorts, monkeypatch):
    seen = _run(tmp_path, cohorts, monkeypatch, embedding="umap", embedding_params={"knn": 9})
    assert seen["emb"].params == {"knn": 9}


def test_weight_between_hands_phate_the_genotype_kernel(tmp_path, cohorts, monkeypatch):
    geno, paint = cohorts
    seen = _run(
        tmp_path,
        cohorts,
        monkeypatch,
        painting_genotype_weight=0.25,
        painting_knn=15,
        embedding_params={"knn": 100},
    )
    assert (seen["pca_runs"], seen["paint_runs"]) == (1, 1)
    assert seen["pca"].project_pca == paint.project_pca  # PHATE's input stays the painting
    params = dict(seen["emb"].params)
    kernel = params.pop(GENOTYPE_KERNEL_PARAM)
    assert params == {"knn": 100, "t": PAINTING_DEFAULT_T}
    assert kernel == GenotypeKernel(genotype=geno.project_pca, weight=0.25, painting_knn=15)


def test_weight_between_takes_the_fit_cohorts_genotype_in_fit_mode(tmp_path, cohorts, monkeypatch):
    geno, _ = cohorts
    seen = _run(tmp_path, cohorts, monkeypatch, painting_genotype_weight=0.5, embedding_input="fit")
    assert seen["emb"].params[GENOTYPE_KERNEL_PARAM].genotype == geno.fit_pca


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(embedding="umap"), "not 'umap'"),
        (dict(embedding_input="both"), "input_mode both"),
        (dict(embedding_params={"n_landmark": 100}), "n_landmark"),
    ],
)
def test_pipeline_refuses_before_any_step(tmp_path, cohorts, monkeypatch, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, cohorts, monkeypatch, painting_genotype_weight=0.5, **kwargs)


def test_weight_one_embeds_the_top_genotype_pcs(tmp_path, cohorts, monkeypatch):
    geno, _ = cohorts
    seen = _run(tmp_path, cohorts, monkeypatch, painting_genotype_weight=1, embedding_input="both")
    assert (seen["pca_runs"], seen["paint_runs"]) == (1, 0)
    top = pd.read_csv(seen["pca"].project_pca)
    expected = pd.read_csv(geno.project_pca).iloc[:, :11]
    pd.testing.assert_frame_equal(top, expected)
    assert pd.read_csv(seen["pca"].fit_pca).shape == (120, 11)
    # Painting provides none of the input, so PHATE keeps its own t and knn.
    assert seen["emb"].params == {}


def test_skip_pca_reads_the_genotype_pca_from_disk(tmp_path, cohorts, monkeypatch):
    import shutil

    geno, paint = cohorts
    pca_dir = tmp_path / "out" / "pca"
    pca_dir.mkdir(parents=True)
    shutil.copy(geno.project_pca, pca_dir / "project_pca_12.csv")
    paint_dir = tmp_path / "out" / "painting"
    paint_dir.mkdir(parents=True)
    shutil.copy(paint.project_pca, paint_dir / "project_painting_hc_pca_12.csv")
    seen = _run(tmp_path, cohorts, monkeypatch, skip_pca=True, painting_genotype_weight=0.5)
    assert (seen["pca_runs"], seen["paint_runs"]) == (0, 0)
    kernel = seen["emb"].params[GENOTYPE_KERNEL_PARAM]
    assert kernel.genotype == pca_dir / "project_pca_12.csv"


def test_skip_pca_without_the_genotype_pca_is_an_error(tmp_path, cohorts, monkeypatch):
    with pytest.raises(RuntimeError, match="needs the project cohort's genotype PCA"):
        _run(tmp_path, cohorts, monkeypatch, skip_pca=True, painting_genotype_weight=0.5)


@pytest.mark.parametrize(
    "params, expected",
    [({}, PAINTING_DEFAULT_T), ({"t": 3}, 3), ({"t": "auto"}, "auto")],
)
def test_painting_input_defaults_phate_t_to_ten(tmp_path, cohorts, monkeypatch, params, expected):
    seen = _run(tmp_path, cohorts, monkeypatch, embedding_params=params)
    assert seen["emb"].params["t"] == expected


def test_preset_t_wins_over_the_painting_default(tmp_path, cohorts, monkeypatch):
    path = _config(tmp_path, {"enabled": True}, preset="whole_cohort")
    kwargs = load_config(path)
    assert kwargs["embedding_params"]["t"] == 3
    seen = _run(tmp_path, cohorts, monkeypatch, embedding_params=kwargs["embedding_params"])
    assert seen["emb"].params["t"] == 3


def test_t_default_is_only_for_phate(tmp_path, cohorts, monkeypatch):
    seen = _run(tmp_path, cohorts, monkeypatch, embedding="umap")
    assert "t" not in seen["emb"].params


# --- regression: the embeddings themselves ------------------------------------------


def _embedded(seen):
    return pd.read_csv(seen["result"].embedding_file, dtype={"sample_id": str})


def test_weight_zero_is_plain_painting_phate(tmp_path, cohorts, monkeypatch):
    _, paint = cohorts
    seen = _run(tmp_path, cohorts, monkeypatch, real_embedding=True)
    expected = PHATE(knn=PAINTING_DEFAULT_KNN, t=PAINTING_DEFAULT_T, n_landmark=None)
    expected.fit(paint.project_pca)
    pd.testing.assert_frame_equal(_embedded(seen), expected.transform(paint.project_pca))


def test_runs_without_painting_are_unaffected(tmp_path, cohorts, monkeypatch):
    geno, _ = cohorts
    params = {"knn": 20, "t": "auto"}
    seen = _run(
        tmp_path,
        cohorts,
        monkeypatch,
        real_embedding=True,
        painting=False,
        painting_genotype_weight=0.5,
        painting_knn=5,
        embedding_params=params,
    )
    assert (seen["pca_runs"], seen["paint_runs"]) == (1, 0)
    assert seen["emb"].params == params
    expected = PHATE(knn=20, t="auto", n_landmark=None)
    expected.fit(geno.project_pca)
    pd.testing.assert_frame_equal(_embedded(seen), expected.transform(geno.project_pca))


def test_weight_between_embeds_the_painted_cohort(tmp_path, cohorts, monkeypatch):
    geno, paint = cohorts
    seen = _run(
        tmp_path,
        cohorts,
        monkeypatch,
        real_embedding=True,
        painting_genotype_weight=0.5,
        painting_knn=10,
        embedding_params={"knn": 10, "t": 5},
    )
    got = _embedded(seen)
    expected = GenotypePaintingPHATE(
        genotype=geno.project_pca, genotype_weight=0.5, painting_knn=10, **PHATE_PARAMS
    ).fit_transform(paint.project_pca)
    pd.testing.assert_frame_equal(got, expected)
    assert seen["result"].fit_embedding_file is None


# --- grids -------------------------------------------------------------------------


def test_grid_phate_panels_mix_the_same_kernels(tmp_path, cohorts, monkeypatch):
    import manifold_genetics.pipeline.steps.viz as viz

    panels = []
    monkeypatch.setattr(
        viz,
        "run_embedding",
        lambda fit, project, *, project_output, method, params: panels.append((method, params)),
    )
    monkeypatch.setattr(viz, "plot_embedding_grid", lambda **kw: kw["output_path"])
    monkeypatch.setattr(viz, "read_colormap", lambda p: {"Population": {}})
    _run(
        tmp_path,
        cohorts,
        monkeypatch,
        painting_genotype_weight=0.5,
        painting_knn=15,
        skip_embedding=True,
        skip_visualization=False,
        method_grid={"phate": {"knn": 50}},
        phate_grid={"knn": [30, 100]},
    )
    assert [p["knn"] for _, p in panels] == [50, 30, 100]
    kernels = {p[GENOTYPE_KERNEL_PARAM] for _, p in panels}
    assert len(kernels) == 1
    (kernel,) = kernels
    assert (kernel.weight, kernel.painting_knn) == (0.5, 15)
