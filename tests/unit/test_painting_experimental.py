"""EXPERIMENTAL painting options: genotype-initialized PHATE and combined inputs.

* ``embedding: init: genotype`` (+ ``init_blend``) starts painting PHATE's
  final MDS from the genotype PHATE layout.
* ``painting: combine_genotype: concat`` embeds genotype and painting PCs side
  by side, and ``kernel_sum`` / ``kernel_product`` combine PHATE's kernels.

The embedding layer is tested on synthetic coordinate CSVs, and the
orchestrator with the painting and PCA steps replaced by fakes.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import euclidean_distances

from manifold_genetics.embeddings import PHATE, InitializedPHATE
from manifold_genetics.embeddings.phate import (
    CombinedKernelPHATE,
    combine_kernels,
    metric_mds_from,
)
from manifold_genetics.pipeline.config import (
    EmbeddingConfig,
    IOConfig,
    build_painting_config,
    check_embedding_init,
)
from manifold_genetics.pipeline.configfile import ConfigFileError, load_config
from manifold_genetics.pipeline.steps.embedding import KernelCombine, run_embedding_step
from manifold_genetics.pipeline.steps.painting import combine_with_genotype
from manifold_genetics.pipeline.steps.pca import PCAStepResult

PHATE_PARAMS = dict(knn=10, t=5, n_landmark=None)


def blobs(seed, n_per=40, dims=10, groups=4, spread=5.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(groups, dims)) * spread
    x = np.vstack([c + rng.normal(size=(n_per, dims)) for c in centers])
    df = pd.DataFrame(x, columns=[f"dim_{i + 1}" for i in range(dims)])
    df.insert(0, "sample_id", [f"s{i}" for i in range(len(x))])
    return df


def xy(df):
    return df[["dim_1", "dim_2"]].to_numpy()


# --- config validation ------------------------------------------------------


def test_check_embedding_init_accepts_the_valid_settings():
    check_embedding_init("phate", {}, painting_enabled=False)
    check_embedding_init("umap", {"init": "none"}, painting_enabled=False)
    check_embedding_init("phate", {"init": "genotype"}, painting_enabled=True)
    check_embedding_init("phate", {"init": "genotype", "init_blend": 0.5}, painting_enabled=True)


@pytest.mark.parametrize(
    "method, params, painting, match",
    [
        ("phate", {"init": "pca"}, True, "Unknown embedding init"),
        ("umap", {"init": "genotype"}, True, "only with method phate"),
        ("phate", {"init": "genotype"}, False, "needs painting enabled"),
        ("phate", {"init_blend": 0.5}, True, "needs init: genotype"),
        ("phate", {"init": "genotype", "init_blend": 1.5}, True, r"in \[0, 1\]"),
        ("phate", {"init": "genotype", "init_blend": "half"}, True, r"in \[0, 1\]"),
    ],
)
def test_check_embedding_init_refuses(method, params, painting, match):
    with pytest.raises(ValueError, match=match):
        check_embedding_init(method, params, painting_enabled=painting)


def write_config(tmp_path, embedding, painting):
    cfg = {
        "data": {
            "fit_plink": "fit",
            "project_plink": "project",
            "labels": "labels.csv",
            "colormap": "cmap.json",
            "output_dir": "out",
            "phased_vcf": "chr{chr}.vcf.gz",
        },
        "embedding": embedding,
    }
    if painting is not None:
        cfg["painting"] = painting
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_config_file_carries_init_and_combine(tmp_path):
    kwargs = load_config(
        write_config(
            tmp_path,
            {"method": "phate", "init": "genotype", "init_blend": 0.3},
            {"enabled": True, "combine_genotype": "concat", "combine_pcs": 5},
        )
    )
    assert kwargs["embedding_params"]["init"] == "genotype"
    assert kwargs["embedding_params"]["init_blend"] == 0.3
    assert kwargs["painting_combine_genotype"] == "concat"
    assert kwargs["painting_combine_pcs"] == 5


def test_config_file_carries_the_kernel_settings(tmp_path):
    path = write_config(
        tmp_path,
        {"method": "phate"},
        {"enabled": True, "combine_genotype": "kernel_sum", "combine_weight": 0.25},
    )
    cfg = yaml.safe_load(path.read_text())
    cfg["preset"] = "whole_cohort"
    cfg["painting"]["combine_norm"] = "markov"
    path.write_text(yaml.safe_dump(cfg))
    kwargs = load_config(path)
    assert kwargs["painting_combine_genotype"] == "kernel_sum"
    assert kwargs["painting_combine_weight"] == 0.25
    assert kwargs["painting_combine_norm"] == "markov"


@pytest.mark.parametrize(
    "embedding, preset, match",
    [
        ({"method": "umap"}, "whole_cohort", "only with embedding method phate"),
        ({"method": "phate"}, "projection", "input_mode fit or project"),
        ({"method": "phate"}, "subsample", "n_landmark: none"),
        ({"method": "phate", "init": "genotype"}, "whole_cohort", "cannot be used together"),
    ],
)
def test_config_file_refuses_kernel_modes_where_they_cannot_run(tmp_path, embedding, preset, match):
    path = write_config(tmp_path, embedding, {"enabled": True, "combine_genotype": "kernel_sum"})
    cfg = yaml.safe_load(path.read_text())
    cfg["preset"] = preset
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigFileError, match=match):
        load_config(path)


def test_config_file_refuses_init_without_painting(tmp_path):
    with pytest.raises(ConfigFileError, match="needs painting enabled"):
        load_config(write_config(tmp_path, {"method": "phate", "init": "genotype"}, None))


def test_config_file_refuses_init_with_another_method(tmp_path):
    with pytest.raises(ConfigFileError, match="only with method phate"):
        load_config(
            write_config(tmp_path, {"method": "umap", "init": "genotype"}, {"enabled": True})
        )


def test_combine_genotype_is_validated():
    vcf = "chr{chr}.vcf.gz"
    on = dict(phased_vcf=vcf, enabled=True)
    with pytest.raises(ValueError, match="needs painting enabled"):
        build_painting_config(phased_vcf=vcf, combine_genotype="concat")
    with pytest.raises(ValueError, match="Unknown painting combine_genotype"):
        build_painting_config(**on, combine_genotype="average")
    with pytest.raises(ValueError, match="positive integer"):
        build_painting_config(**on, combine_genotype="concat", combine_pcs=0)
    with pytest.raises(ValueError, match="more than pca n_pcs"):
        build_painting_config(**on, combine_genotype="concat", combine_pcs=10, n_pcs=5)
    with pytest.raises(ValueError, match=r"combine_weight must be a number in \[0, 1\]"):
        build_painting_config(**on, combine_genotype="kernel_sum", combine_weight=1.5)
    with pytest.raises(ValueError, match="Unknown painting combine_norm"):
        build_painting_config(**on, combine_genotype="kernel_sum", combine_norm="l2")
    cfg = build_painting_config(**on, combine_genotype="concat")
    assert cfg.combine_genotype == "concat" and cfg.combine_pcs == 10
    assert build_painting_config(**on, combine_genotype=False).combine_genotype is None


# --- the custom MDS ---------------------------------------------------------


@pytest.mark.parametrize("solver", ["sgd", "smacof"])
def test_mds_started_at_the_optimum_barely_moves(solver):
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(80, 2))
    D = euclidean_distances(truth)
    # Rotated, shifted and shrunk: the same configuration, not at D's scale.
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    init = 0.1 * truth @ rot + 3.0
    Y = metric_mds_from(D, init, solver=solver, random_state=0)
    assert procrustes(truth, Y)[2] < 1e-3


def test_initialized_phate_started_at_its_own_layout_barely_moves():
    df = blobs(0)
    # SMACOF for both: it converges, so PHATE's own layout is an optimum of it.
    plain = PHATE(mds_solver="smacof", **PHATE_PARAMS)
    plain.fit(df)
    layout = plain.transform(df)
    model = InitializedPHATE(init_layout=layout, mds_solver="smacof", **PHATE_PARAMS)
    model.fit(df)
    assert model.init_disparity < 0.02


def test_initialized_phate_starts_from_the_given_layout():
    # A random start: the result is a PHATE layout, not the start.
    df = blobs(0)
    rng = np.random.default_rng(2)
    noise = df[["sample_id"]].assign(dim_1=rng.normal(size=len(df)), dim_2=rng.normal(size=len(df)))
    model = InitializedPHATE(init_layout=noise, **PHATE_PARAMS)
    model.fit(df)
    assert model.init_disparity > 0.3


@pytest.mark.parametrize("landmarks", [None, 40])
def test_blend_alpha_one_reproduces_the_init_geometry(landmarks):
    df = blobs(0)
    other = blobs(5)  # an unrelated "genotype" layout for the same samples
    plain = PHATE(**PHATE_PARAMS)
    plain.fit(other)
    geno = plain.transform(other)

    params = dict(PHATE_PARAMS, n_landmark=landmarks, random_landmarking=landmarks is not None)
    model = InitializedPHATE(init_layout=geno, init_blend=1.0, **params)
    model.fit(df)
    # Exact: the MDS target is the layout's own distances. With landmarks the
    # samples are placed through their transitions, so only close.
    assert model.init_disparity < (1e-3 if landmarks is None else 0.1)


def test_init_layout_rows_follow_sample_ids_not_order():
    df = blobs(0)
    plain = PHATE(**PHATE_PARAMS)
    plain.fit(df)
    layout = plain.transform(df)
    shuffled = layout.sample(frac=1.0, random_state=0)
    model = InitializedPHATE(init_layout=shuffled, init_blend=1.0, **PHATE_PARAMS)
    model.fit(df)
    assert model.init_disparity < 1e-3


def test_init_layout_missing_a_fit_sample_is_refused():
    df = blobs(0)
    layout = df[["sample_id"]].assign(dim_1=0.0, dim_2=0.0).iloc[1:]
    with pytest.raises(ValueError, match="no row for 1"):
        InitializedPHATE(init_layout=layout, **PHATE_PARAMS).fit(df)


# --- the embedding step, end to end on coordinate CSVs -------------------------


@pytest.fixture
def cohorts(tmp_path):
    """Genotype and painting coordinates for a fit cohort and a project cohort."""
    geno, paint = blobs(5), blobs(0)
    files = {}
    for name, df in (("geno", geno), ("paint", paint)):
        files[f"{name}_fit"] = tmp_path / f"{name}_fit.csv"
        files[f"{name}_project"] = tmp_path / f"{name}_project.csv"
        df.iloc[:120].to_csv(files[f"{name}_fit"], index=False)
        df.to_csv(files[f"{name}_project"], index=False)
    io = IOConfig(
        fit_plink=tmp_path / "fit",
        project_plink=tmp_path / "project",
        output_dir=tmp_path / "out",
        fit_labels=tmp_path / "l.csv",
        project_labels=tmp_path / "l.csv",
        fit_colormap=tmp_path / "c.json",
        project_colormap=tmp_path / "c.json",
    )
    geno_pca = PCAStepResult(fit_pca=files["geno_fit"], project_pca=files["geno_project"])
    paint_pca = PCAStepResult(fit_pca=files["paint_fit"], project_pca=files["paint_project"])
    return io, geno_pca, paint_pca


@pytest.mark.parametrize("mode", ["both", "project"])
def test_init_step_writes_both_layouts_and_a_report(cohorts, mode):
    io, geno_pca, paint_pca = cohorts
    emb = EmbeddingConfig(
        method="phate", input_mode=mode, params=dict(PHATE_PARAMS, init="genotype")
    )
    result = run_embedding_step(io, emb, pca=paint_pca, init_pca=geno_pca)

    d = io.output_dir / "embeddings"
    assert result.embedding_file == d / "phate_2d.csv"
    painted = pd.read_csv(d / "phate_2d.csv")
    genotype = pd.read_csv(d / "phate_init_genotype_2d.csv")
    assert len(painted) == len(genotype) == 160
    if mode == "both":
        assert result.fit_embedding_file == d / "phate_fit_2d.csv"
        assert len(pd.read_csv(d / "phate_init_genotype_fit_2d.csv")) == 120
    report = json.loads((d / "phate_init_report_2d.json").read_text())
    assert report["init"] == "genotype" and 0 <= report["procrustes_disparity_fit_samples"] <= 1


def test_init_step_blend_one_gives_the_genotype_layout(cohorts):
    io, geno_pca, paint_pca = cohorts
    emb = EmbeddingConfig(
        method="phate",
        input_mode="project",
        params=dict(PHATE_PARAMS, init="genotype", init_blend=1.0),
    )
    run_embedding_step(io, emb, pca=paint_pca, init_pca=geno_pca)
    d = io.output_dir / "embeddings"
    painted = pd.read_csv(d / "phate_2d.csv")
    genotype = pd.read_csv(d / "phate_init_genotype_2d.csv")
    assert procrustes(xy(genotype), xy(painted))[2] < 1e-3


def test_init_step_without_genotype_pca_is_an_error(cohorts):
    io, _, paint_pca = cohorts
    emb = EmbeddingConfig(method="phate", input_mode="project", params={"init": "genotype"})
    with pytest.raises(RuntimeError, match="needs the genotype PCA"):
        run_embedding_step(io, emb, pca=paint_pca)


# --- combine_genotype ---------------------------------------------------------


def test_combine_scales_each_block_to_equal_weight_on_the_fit_cohort(cohorts, tmp_path):
    _, geno_pca, paint_pca = cohorts
    # Make the painting block 100 times larger: it must not dominate.
    big = pd.read_csv(paint_pca.project_pca)
    big.iloc[:, 1:] *= 100
    big.to_csv(tmp_path / "big_project.csv", index=False)
    big.iloc[:120].to_csv(tmp_path / "big_fit.csv", index=False)

    fit_out, project_out = combine_with_genotype(
        genotype_fit=geno_pca.fit_pca,
        genotype_project=geno_pca.project_pca,
        painting_fit=tmp_path / "big_fit.csv",
        painting_project=tmp_path / "big_project.csv",
        n=3,
        fit_output=tmp_path / "c_fit.csv",
        project_output=tmp_path / "c_project.csv",
    )
    fit = pd.read_csv(fit_out)
    assert list(fit.columns) == ["sample_id"] + [f"dim_{i}" for i in range(1, 7)]
    x = fit.iloc[:, 1:].to_numpy()
    for block in (x[:, :3], x[:, 3:]):
        np.testing.assert_allclose(block.mean(axis=0), 0, atol=1e-9)
        assert np.sqrt((block**2).sum(axis=1).mean()) == pytest.approx(1.0)
    # Within a block, columns keep their relative scale (no per-column z-scoring).
    geno = pd.read_csv(geno_pca.fit_pca).iloc[:, 1:4].to_numpy()
    np.testing.assert_allclose(
        x[:, :3].std(axis=0) / x[:, 0].std(), geno.std(axis=0) / geno[:, 0].std()
    )
    # The project cohort goes through the fit cohort's centering and scale.
    project = pd.read_csv(project_out)
    pd.testing.assert_frame_equal(project.iloc[:120], fit)
    assert len(project) == 160


def test_combine_refuses_painted_samples_without_genotype(cohorts, tmp_path):
    _, geno_pca, paint_pca = cohorts
    with pytest.raises(ValueError, match="no genotype PCA row"):
        combine_with_genotype(
            genotype_fit=geno_pca.fit_pca,
            genotype_project=geno_pca.fit_pca,  # lacks 40 project samples
            painting_fit=paint_pca.fit_pca,
            painting_project=paint_pca.project_pca,
            n=2,
            fit_output=tmp_path / "a.csv",
            project_output=tmp_path / "b.csv",
        )


def test_combine_refuses_too_few_components(cohorts, tmp_path):
    _, geno_pca, paint_pca = cohorts
    with pytest.raises(ValueError, match="only 10 components"):
        combine_with_genotype(
            genotype_fit=geno_pca.fit_pca,
            genotype_project=None,
            painting_fit=paint_pca.fit_pca,
            painting_project=None,
            n=11,
            fit_output=tmp_path / "a.csv",
            project_output=None,
        )


# --- the orchestrator -------------------------------------------------------------


def _run(tmp_path, cohorts, monkeypatch, **kwargs):
    import manifold_genetics.pipeline.orchestrator as orch
    from manifold_genetics.pipeline.runner import run_pipeline

    _, geno_pca, paint_pca = cohorts
    seen = {"pca_runs": 0}

    def fake_pca(io, cfg):
        seen["pca_runs"] += 1
        return geno_pca

    def fake_paint(io, pca_cfg, painting_cfg):
        return SimpleNamespace(
            fit_pca=paint_pca.fit_pca, project_pca=paint_pca.project_pca, coords_df=None
        )

    def fake_embedding(io, emb, *, pca, **extra):
        seen["pca"], seen["extra"], seen["emb"] = pca, extra, emb

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
        n_pcs=10,
        skip_admixture=True,
        skip_pca_visualization=True,
        skip_visualization=True,
        skip_metrics=True,
        **kwargs,
    )
    return seen


def test_pipeline_init_runs_the_genotype_pca_and_passes_it_on(tmp_path, cohorts, monkeypatch):
    seen = _run(tmp_path, cohorts, monkeypatch, embedding_params={"init": "genotype"})
    _, geno_pca, paint_pca = cohorts
    assert seen["pca_runs"] == 1
    assert seen["pca"].project_pca == paint_pca.project_pca
    assert seen["extra"] == {"init_pca": geno_pca}


def test_pipeline_without_the_options_runs_no_genotype_pca(tmp_path, cohorts, monkeypatch):
    seen = _run(tmp_path, cohorts, monkeypatch)
    assert seen["pca_runs"] == 0 and seen["extra"] == {}


def test_pipeline_combine_embeds_the_combined_coordinates(tmp_path, cohorts, monkeypatch):
    seen = _run(
        tmp_path,
        cohorts,
        monkeypatch,
        embedding="umap",
        painting_combine_genotype="concat",
        painting_combine_pcs=4,
    )
    d = tmp_path / "out" / "painting"
    assert seen["pca_runs"] == 1
    assert seen["pca"].fit_pca == d / "fit_combined_genotype_hc_4x2.csv"
    assert seen["pca"].project_pca == d / "project_combined_genotype_hc_4x2.csv"
    assert pd.read_csv(seen["pca"].project_pca).shape == (160, 9)


def test_pipeline_kernel_mode_hands_the_genotype_view_to_the_embedding(
    tmp_path, cohorts, monkeypatch
):
    seen = _run(
        tmp_path,
        cohorts,
        monkeypatch,
        embedding_input="project",
        painting_combine_genotype="kernel_sum",
        painting_combine_weight=0.25,
        painting_combine_norm="markov",
        painting_combine_pcs=4,
    )
    _, geno_pca, paint_pca = cohorts
    assert seen["pca"].project_pca == paint_pca.project_pca
    assert seen["extra"] == {
        "kernel": KernelCombine(
            genotype=geno_pca, mode="kernel_sum", weight=0.25, norm="markov", n_pcs=4
        )
    }


def test_pipeline_refuses_a_kernel_mode_in_both_mode(tmp_path, cohorts, monkeypatch):
    with pytest.raises(ValueError, match="input_mode fit or project"):
        _run(tmp_path, cohorts, monkeypatch, painting_combine_genotype="kernel_product")


# --- kernel combination --------------------------------------------------------


def _kernels():
    a = CombinedKernelPHATE(second_view=np.zeros((1, 1)), **PHATE_PARAMS)
    return a.view_kernel(xy10(blobs(5))), a.view_kernel(xy10(blobs(0)))


def xy10(df):
    return df.drop(columns="sample_id").to_numpy()


@pytest.mark.parametrize(
    "mode, norm", [("kernel_sum", "degree"), ("kernel_sum", "markov"), ("kernel_product", "degree")]
)
def test_combined_kernels_are_symmetric_and_nonnegative(mode, norm):
    K_a, K_b = _kernels()
    K = combine_kernels(K_a, K_b, mode=mode, weight=0.3, norm=norm).toarray()
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert K.min() >= 0 and K.sum(axis=1).min() > 0


def test_markov_norm_gives_both_views_equal_total_weight():
    K_a, K_b = _kernels()
    # Each view is a symmetrized Markov matrix: total weight n, whatever its scale.
    big = combine_kernels(100 * K_a, K_b, weight=1.0, norm="markov")
    small = combine_kernels(K_a, K_b, weight=1.0, norm="markov")
    np.testing.assert_allclose(big.toarray(), small.toarray())
    assert big.sum() == pytest.approx(K_a.shape[0])


def test_combine_kernels_validates():
    eye = np.eye(3)
    with pytest.raises(ValueError, match="Unknown kernel mode"):
        combine_kernels(eye, eye, mode="kernel_max")
    with pytest.raises(ValueError, match="Unknown kernel norm"):
        combine_kernels(eye, eye, norm="l1")
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        combine_kernels(eye, eye, weight=-0.1)
    with pytest.raises(ValueError, match="differ in shape"):
        combine_kernels(eye, np.eye(4))


@pytest.mark.parametrize("weight, view", [(1.0, "genotype"), (0.0, "painting")])
def test_weight_one_or_zero_reproduces_single_view_phate(weight, view):
    geno, paint = blobs(5), blobs(0)
    single = PHATE(**PHATE_PARAMS)
    single.fit(geno if view == "genotype" else paint)
    expected = single.transform(geno if view == "genotype" else paint)

    model = CombinedKernelPHATE(second_view=geno, weight=weight, norm="degree", **PHATE_PARAMS)
    got = model.fit_transform(paint)
    assert list(got["sample_id"]) == list(paint["sample_id"])
    # Equal up to the last digits of the MDS (layouts span about 0.2).
    np.testing.assert_allclose(xy(got), xy(expected), atol=1e-3)
    assert procrustes(xy(expected), xy(got))[2] < 1e-6


def test_kernel_phate_aligns_the_second_view_by_sample_id():
    geno, paint = blobs(5), blobs(0)
    in_order = CombinedKernelPHATE(second_view=geno, **PHATE_PARAMS).fit_transform(paint)
    shuffled = geno.sample(frac=1.0, random_state=0)
    got = CombinedKernelPHATE(second_view=shuffled, **PHATE_PARAMS).fit_transform(paint)
    np.testing.assert_allclose(xy(got), xy(in_order), atol=1e-6)


def test_kernel_phate_refuses_landmarks_and_new_samples():
    geno, paint = blobs(5), blobs(0)
    with pytest.raises(ValueError, match="exact PHATE"):
        CombinedKernelPHATE(second_view=geno, **dict(PHATE_PARAMS, n_landmark=50))
    model = CombinedKernelPHATE(second_view=geno, mode="kernel_product", **PHATE_PARAMS)
    model.fit(paint)
    with pytest.raises(ValueError, match="cannot place new samples"):
        model.transform(paint.iloc[:10])


def test_kernel_step_embeds_the_painted_cohort(cohorts):
    io, geno_pca, paint_pca = cohorts
    emb = EmbeddingConfig(method="phate", input_mode="project", params=dict(PHATE_PARAMS))
    result = run_embedding_step(
        io, emb, pca=paint_pca, kernel=KernelCombine(genotype=geno_pca, n_pcs=5)
    )
    out = pd.read_csv(result.embedding_file)
    assert result.embedding_file == io.output_dir / "embeddings" / "phate_2d.csv"
    assert len(out) == 160 and result.fit_embedding_file is None
