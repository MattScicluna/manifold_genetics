"""The painting step and how it stands in for PCA.

pbwt is replaced by a fake that writes its real file layout (see fake_pbwt.py),
so these run anywhere. tests/integration/test_painting_pbwt_real.py runs the
real binary when it is installed.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from manifold_genetics.pipeline.config import (
    IOConfig,
    PaintingConfig,
    PCAConfig,
    build_configs,
    build_painting_config,
)
from manifold_genetics.pipeline.configfile import ConfigFileError, load_config
from manifold_genetics.pipeline.steps.painting import run_painting, run_painting_step
from manifold_genetics.pipeline.steps.paths import painting_output_paths
from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

from .fake_pbwt import calls, install_fake_pbwt, write_fam, write_vcf

SAMPLES = ["A1", "A2", "A3", "B1", "B2", "B3"]


@pytest.fixture
def painted(tmp_path):
    """Two chromosomes of six samples, in two groups, and a fake pbwt."""
    pbwt = install_fake_pbwt(tmp_path / "bin")
    for chrom in ("1", "2"):
        write_vcf(tmp_path / "data" / f"chr{chrom}.vcf", SAMPLES, chrom)
    return tmp_path, str(pbwt)


def paint(tmp_path, pbwt, **kwargs):
    defaults = dict(
        chromosomes=["1", "2"],
        pbwt_dir=tmp_path / "pbwt",
        project_output=tmp_path / "project.csv",
        n_pcs=3,
        pbwt=pbwt,
    )
    defaults.update(kwargs)
    return run_painting(tmp_path / "data" / "chr{chr}.vcf", **defaults)


@pytest.mark.parametrize("method", ["finestructure", "hc"])
def test_both_methods_separate_the_groups(tmp_path, method):
    # Twenty per group: hc keeps the zero self-copying diagonal, which in
    # groups of three would outweigh the group signal.
    pbwt = str(install_fake_pbwt(tmp_path / "bin"))
    names = [f"{g}{i}" for g in "AB" for i in range(20)]
    for chrom in ("1", "2"):
        write_vcf(tmp_path / "data" / f"chr{chrom}.vcf", names, chrom)
    coords = paint(tmp_path, pbwt, method=method, n_pcs=2).set_index("sample_id")

    x = coords.to_numpy()
    dist = np.linalg.norm(x[:, None] - x[None], axis=-1)
    group = np.array([s[0] for s in coords.index])
    same = group[:, None] == group[None]
    off = ~np.eye(len(group), dtype=bool)
    assert dist[same & off].max() < dist[~same].min()


def test_hc_coancestry_is_in_centimorgans(painted):
    tmp_path, pbwt = painted
    paint(tmp_path, pbwt, method="hc", coancestry_output=tmp_path / "coancestry.csv")

    # The fake writes chunk length rows of 2 * 100 + 3 * 10 = 230 per
    # chromosome, i.e. 115 sites, so each is scaled by g_c / 115.
    coancestry = pd.read_csv(tmp_path / "coancestry.csv", index_col="sample_id")
    np.testing.assert_allclose(coancestry.loc["A1", "A2"], 100 * (286.28 + 268.83) / 115)
    np.testing.assert_allclose(coancestry.sum(axis=1), 2 * (286.28 + 268.83))


def test_paints_every_sample_and_writes_coordinates(painted):
    tmp_path, pbwt = painted
    coords = paint(
        tmp_path, pbwt, method="finestructure", coancestry_output=tmp_path / "coancestry.csv"
    )

    assert list(coords.columns) == ["sample_id", "dim_1", "dim_2", "dim_3"]
    assert list(coords["sample_id"]) == SAMPLES
    assert pd.read_csv(tmp_path / "project.csv").shape == (6, 4)
    assert len(calls(tmp_path / "bin")) == 2  # one run per chromosome

    # The two groups separate on the first component.
    first = coords.set_index("sample_id")["dim_1"]
    assert np.sign(first[["A1", "A2", "A3"]]).nunique() == 1
    assert np.sign(first["A1"]) != np.sign(first["B1"])

    # Summed over both chromosomes: 10 within a group, 1 across, per chromosome.
    coancestry = pd.read_csv(tmp_path / "coancestry.csv", index_col="sample_id")
    assert coancestry.loc["A1", "A2"] == 20.0
    assert coancestry.loc["A1", "B1"] == 2.0
    assert coancestry.loc["A1", "A1"] == 0.0


def test_fit_and_project_cohorts_come_from_the_fam_files(painted):
    tmp_path, pbwt = painted
    write_fam(tmp_path / "fit", ["A1", "A2", "B1", "B2"])
    write_fam(tmp_path / "project", ["A3", "B3"])

    project = paint(
        tmp_path,
        pbwt,
        fit_plink=tmp_path / "fit",
        project_plink=tmp_path / "project",
        fit_output=tmp_path / "fit.csv",
        method="finestructure",
    )
    fit = pd.read_csv(tmp_path / "fit.csv")

    assert list(fit["sample_id"]) == ["A1", "A2", "B1", "B2"]
    assert list(project["sample_id"]) == ["A3", "B3"]
    # Projected rows land beside their group's fitted rows.
    a_fit = fit.set_index("sample_id").loc["A1", "dim_1"]
    assert np.sign(project.set_index("sample_id").loc["A3", "dim_1"]) == np.sign(a_fit)


def test_vcf_names_of_the_form_fid_iid_match_the_fam(tmp_path):
    pbwt = install_fake_pbwt(tmp_path / "bin")
    names = [f"{s}fam_{s}" for s in SAMPLES]  # first letter keeps the groups
    write_vcf(tmp_path / "data" / "chr1.vcf", names)
    write_fam(tmp_path / "fit", SAMPLES, fids=[f"{s}fam" for s in SAMPLES])

    coords = paint(tmp_path, str(pbwt), chromosomes=["1"], fit_plink=tmp_path / "fit")

    assert list(coords["sample_id"]) == SAMPLES  # written under the IID


def test_a_cohort_sample_missing_from_the_vcf_fails_before_painting(painted):
    tmp_path, pbwt = painted
    write_fam(tmp_path / "fit", ["A1", "A2", "ZZ9"])
    with pytest.raises(ValueError, match="1 of 3 fit samples.*ZZ9") as err:
        paint(tmp_path, pbwt, fit_plink=tmp_path / "fit")
    assert "1 is absent from the VCF" in str(err.value)
    assert "ID format mismatch" not in str(err.value)
    assert "missing: drop" in str(err.value)  # says how to proceed
    assert calls(tmp_path / "bin") == []  # pbwt never ran


def test_a_project_sample_missing_from_the_vcf_fails_before_painting(painted):
    tmp_path, pbwt = painted
    write_fam(tmp_path / "fit", ["A1", "B1"])
    write_fam(tmp_path / "project", SAMPLES + ["ZZ9"])
    with pytest.raises(ValueError, match="1 of 7 project samples"):
        paint(tmp_path, pbwt, fit_plink=tmp_path / "fit", project_plink=tmp_path / "project")
    assert calls(tmp_path / "bin") == []


def test_an_id_format_mismatch_is_told_apart_from_absent_samples(tmp_path):
    pbwt = install_fake_pbwt(tmp_path / "bin")
    # The VCF prefixes each name with a batch, not the FID, and one sample is gone.
    write_vcf(tmp_path / "data" / "chr1.vcf", [f"batch1_{s}" for s in SAMPLES[:-1]])
    write_fam(tmp_path / "fit", SAMPLES)
    with pytest.raises(ValueError) as err:
        paint(tmp_path, str(pbwt), chromosomes=["1"], fit_plink=tmp_path / "fit")
    message = str(err.value)
    assert "6 of 6 fit samples" in message
    assert "5 of them look like an ID format mismatch" in message
    assert "A1 vs batch1_A1" in message
    assert "1 is absent from the VCF under any naming, for example B3" in message


def test_an_iid_that_is_itself_fid_iid_is_a_format_mismatch(tmp_path):
    pbwt = install_fake_pbwt(tmp_path / "bin")
    write_vcf(tmp_path / "data" / "chr1.vcf", SAMPLES)
    write_fam(tmp_path / "fit", [f"fam_{s}" for s in SAMPLES], fids=["x"] * 6)
    with pytest.raises(ValueError, match="fam_A1 vs A1"):
        paint(tmp_path, str(pbwt), chromosomes=["1"], fit_plink=tmp_path / "fit")


def test_missing_drop_leaves_the_sample_out_with_a_warning(painted, caplog):
    tmp_path, pbwt = painted
    write_fam(tmp_path / "fit", ["A1", "A2", "B1", "B2", "ZZ9"])
    write_fam(tmp_path / "project", SAMPLES + ["ZZ8"])
    with caplog.at_level("WARNING"):
        project = paint(
            tmp_path,
            pbwt,
            fit_plink=tmp_path / "fit",
            project_plink=tmp_path / "project",
            fit_output=tmp_path / "fit.csv",
            missing="drop",
            n_pcs=2,
        )
    assert list(project["sample_id"]) == SAMPLES
    assert list(pd.read_csv(tmp_path / "fit.csv")["sample_id"]) == ["A1", "A2", "B1", "B2"]
    assert "left out of the fit cohort" in caplog.text
    assert "left out of the project cohort" in caplog.text


def test_missing_drop_still_refuses_an_empty_cohort(painted):
    tmp_path, pbwt = painted
    write_fam(tmp_path / "fit", ["ZZ8", "ZZ9"])
    with pytest.raises(ValueError, match="No fit sample is left"):
        paint(tmp_path, pbwt, fit_plink=tmp_path / "fit", missing="drop")


def test_chromosomes_with_different_samples_are_refused_before_painting(painted):
    tmp_path, pbwt = painted
    write_vcf(tmp_path / "data" / "chr2.vcf", list(reversed(SAMPLES)), "2")
    with pytest.raises(ValueError, match="chromosome 2 .* does not list the same samples"):
        paint(tmp_path, pbwt)
    assert calls(tmp_path / "bin") == []

    write_vcf(tmp_path / "data" / "chr2.vcf", SAMPLES[:-1], "2")
    with pytest.raises(ValueError, match="1 missing from it, for example B3"):
        paint(tmp_path, pbwt)


def test_a_missing_chromosome_vcf_fails_before_painting(painted):
    tmp_path, pbwt = painted
    with pytest.raises(FileNotFoundError, match="chr3.vcf"):
        paint(tmp_path, pbwt, chromosomes=["1", "2", "3"])
    assert calls(tmp_path / "bin") == []


def test_n_pcs_is_capped_by_the_fit_cohort(painted, caplog):
    tmp_path, pbwt = painted
    coords = paint(tmp_path, pbwt, n_pcs=50)
    assert coords.shape[1] - 1 == 5
    assert "supports 5" in caplog.text


def test_step_writes_where_painting_output_paths_says(painted):
    tmp_path, pbwt = painted
    write_fam(tmp_path / "fit", ["A1", "A2", "B1", "B2"])
    write_fam(tmp_path / "project", SAMPLES)
    io = IOConfig(
        fit_plink=tmp_path / "fit",
        project_plink=tmp_path / "project",
        output_dir=tmp_path / "out",
        fit_labels=tmp_path / "l.csv",
        project_labels=tmp_path / "l.csv",
        fit_colormap=tmp_path / "c.json",
        project_colormap=tmp_path / "c.json",
        phased_vcf=tmp_path / "data" / "chr{chr}.vcf",
    )
    pca = PCAConfig(n_pcs=2)
    painting = PaintingConfig(enabled=True, chromosomes=("1", "2"), method="hc")

    result = run_painting_step(io, pca, painting, pbwt=pbwt)
    paths = painting_output_paths(io, pca, painting)

    assert result.fit_pca == paths["fit_pca"] == tmp_path / "out/painting/fit_painting_hc_pca_2.csv"
    assert result.project_pca == paths["project_pca"]
    assert result.coancestry == tmp_path / "out/painting/coancestry_hc.csv"
    assert all(p.exists() for p in (result.fit_pca, result.project_pca, result.coancestry))
    assert (paths["pbwt_roots"]["1"].parent / "chr1.chunkcounts.out").exists()


# --- configuration ---------------------------------------------------------


def test_painting_config_validation():
    assert build_painting_config().enabled is False
    with pytest.raises(ValueError, match="no phased_vcf"):
        build_painting_config(enabled=True)
    with pytest.raises(ValueError, match="must contain"):
        build_painting_config(enabled=True, phased_vcf="all.vcf.gz")
    with pytest.raises(ValueError, match="Unknown painting method"):
        build_painting_config(method="chunkcounts")
    with pytest.raises(ValueError, match="autosomes 1-22, not X"):
        build_painting_config(method="hc", chromosomes="21,22,X")
    cfg = build_painting_config(method="finestructure", chromosomes="X")
    assert cfg.chromosomes == ("X",)

    cfg = build_painting_config(enabled=True, phased_vcf="c{chr}.vcf", chromosomes="21-22")
    assert cfg.chromosomes == ("21", "22")


def test_build_configs_carries_painting():
    configs = build_configs(
        fit_plink="f",
        project_plink="p",
        output_dir="o",
        labels="l.csv",
        colormap="c.json",
        phased_vcf="data/chr{chr}.vcf.gz",
        painting=True,
        painting_method="hc",
    )
    assert str(configs.io.phased_vcf) == "data/chr{chr}.vcf.gz"
    assert configs.painting.enabled and configs.painting.method == "hc"


def test_config_file_painting_section(tmp_path):
    config = {
        "data": {
            "fit_plink": "data/fit",
            "project_plink": "data/project",
            "labels": "labels.csv",
            "colormap": "cmap.json",
            "output_dir": "out",
            "phased_vcf": "data/chr{chr}.vcf.gz",
        },
        "painting": {"enabled": True, "method": "hc", "chromosomes": "1-2"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    kwargs = load_config(path)

    assert kwargs["phased_vcf"] == (tmp_path / "data" / "chr{chr}.vcf.gz").resolve()
    assert kwargs["painting"] is True
    assert kwargs["painting_method"] == "hc"
    assert kwargs["painting_chromosomes"] == "1-2"


def test_config_file_rejects_unknown_painting_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"data": {"output_dir": "o"}, "painting": {"enable": True}}))
    with pytest.raises(ConfigFileError, match="did you mean 'enabled'"):
        load_config(path)


# --- the orchestrator ------------------------------------------------------


def test_painting_replaces_pca_as_the_embedding_input(painted, monkeypatch):
    import manifold_genetics.pipeline.orchestrator as orch
    from manifold_genetics.pipeline.runner import run_pipeline

    tmp_path, pbwt = painted
    monkeypatch.setenv("MANIFOLD_GENETICS_TOOL_DIR", str(tmp_path / "bin"))
    write_fam(tmp_path / "fit", ["A1", "A2", "B1", "B2"])
    write_fam(tmp_path / "project", SAMPLES)

    def no_pca(*args, **kwargs):
        raise AssertionError("genotype PCA ran although painting is enabled")

    seen = {}

    def fake_embedding(io, emb, *, pca):
        seen["pca"] = pca
        return None

    monkeypatch.setattr(orch, "run_pca_step", no_pca)
    monkeypatch.setattr(orch, "run_embedding_step", fake_embedding)

    result = run_pipeline(
        fit_plink=tmp_path / "fit",
        project_plink=tmp_path / "project",
        output_dir=tmp_path / "out",
        labels=tmp_path / "labels.csv",
        colormap=tmp_path / "cmap.json",
        phased_vcf=tmp_path / "data" / "chr{chr}.vcf",
        painting=True,
        painting_chromosomes=["1", "2"],
        n_pcs=2,
        skip_admixture=True,
        skip_pca_visualization=True,
        skip_visualization=True,
        skip_metrics=True,
    )

    out = tmp_path / "out" / "painting"
    assert seen["pca"].fit_pca == out / "fit_painting_hc_pca_2.csv"
    assert seen["pca"].project_pca == out / "project_painting_hc_pca_2.csv"
    assert result.painting is not None and result.pca.project_pca == result.painting.project_pca

    # skip_pca reuses the cached painting coordinates rather than repainting.
    monkeypatch.setattr(orch, "run_painting_step", no_pca)
    run_pipeline(
        fit_plink=tmp_path / "fit",
        project_plink=tmp_path / "project",
        output_dir=tmp_path / "out",
        labels=tmp_path / "labels.csv",
        colormap=tmp_path / "cmap.json",
        phased_vcf=tmp_path / "data" / "chr{chr}.vcf",
        painting=True,
        painting_chromosomes=["1", "2"],
        n_pcs=2,
        skip_pca=True,
        skip_admixture=True,
        skip_pca_visualization=True,
        skip_visualization=True,
        skip_metrics=True,
    )
    assert seen["pca"].skipped and seen["pca"].project_pca == out / "project_painting_hc_pca_2.csv"


def _run_painting_pipeline(tmp_path, **kwargs):
    from manifold_genetics.pipeline.runner import run_pipeline

    defaults = dict(
        fit_plink=tmp_path / "fit",
        project_plink=tmp_path / "project",
        output_dir=tmp_path / "out",
        labels=tmp_path / "labels.csv",
        colormap=tmp_path / "cmap.json",
        phased_vcf=tmp_path / "data" / "chr{chr}.vcf",
        painting=True,
        painting_chromosomes=["1", "2"],
        n_pcs=2,
        skip_admixture=True,
        skip_pca_visualization=True,
        skip_visualization=True,
        skip_metrics=True,
    )
    defaults.update(kwargs)
    return run_pipeline(**defaults)


def test_switching_method_keeps_both_outputs_and_skip_reads_only_its_own(painted, monkeypatch):
    import manifold_genetics.pipeline.orchestrator as orch

    tmp_path, _ = painted
    monkeypatch.setenv("MANIFOLD_GENETICS_TOOL_DIR", str(tmp_path / "bin"))
    write_fam(tmp_path / "fit", ["A1", "A2", "B1", "B2"])
    write_fam(tmp_path / "project", SAMPLES)
    seen = {}
    monkeypatch.setattr(
        orch, "run_embedding_step", lambda io, emb, *, pca: seen.__setitem__("pca", pca)
    )

    _run_painting_pipeline(tmp_path, painting_method="hc")
    out = tmp_path / "out" / "painting"
    hc = pd.read_csv(out / "project_painting_hc_pca_2.csv")

    # skip: pca with a method that has never run finds nothing, and says so,
    # rather than embedding the hc coordinates.
    with pytest.raises(RuntimeError, match="project_painting_finestructure_pca_2.csv"):
        _run_painting_pipeline(tmp_path, painting_method="finestructure", skip_pca=True)

    _run_painting_pipeline(tmp_path, painting_method="finestructure")
    assert seen["pca"].project_pca == out / "project_painting_finestructure_pca_2.csv"
    # The second method reused pbwt's output and left the first one's alone.
    assert len(calls(tmp_path / "bin")) == 2
    pd.testing.assert_frame_equal(pd.read_csv(out / "project_painting_hc_pca_2.csv"), hc)

    _run_painting_pipeline(tmp_path, painting_method="hc", skip_pca=True)
    assert seen["pca"].project_pca == out / "project_painting_hc_pca_2.csv"


def test_pbwt_dir_is_shared_between_output_dirs(painted, monkeypatch):
    import manifold_genetics.pipeline.orchestrator as orch

    tmp_path, _ = painted
    monkeypatch.setenv("MANIFOLD_GENETICS_TOOL_DIR", str(tmp_path / "bin"))
    write_fam(tmp_path / "fit", ["A1", "A2", "B1", "B2"])
    write_fam(tmp_path / "project", SAMPLES)
    monkeypatch.setattr(orch, "run_embedding_step", lambda io, emb, *, pca: None)
    shared = tmp_path / "shared" / "pbwt"

    _run_painting_pipeline(tmp_path, output_dir=tmp_path / "hc", painting_pbwt_dir=shared)
    _run_painting_pipeline(
        tmp_path,
        output_dir=tmp_path / "fs",
        painting_method="finestructure",
        painting_pbwt_dir=shared,
    )

    assert (shared / "chr1.chunkcounts.out").exists()
    assert not (tmp_path / "hc" / "painting" / "pbwt").exists()
    assert len(calls(tmp_path / "bin")) == 2  # painted once, for the first run
    assert (tmp_path / "fs/painting/project_painting_finestructure_pca_2.csv").exists()


def test_painting_output_paths_name_the_method(tmp_path):
    io = IOConfig(
        fit_plink=tmp_path / "f",
        project_plink=tmp_path / "p",
        output_dir=tmp_path / "out",
        fit_labels=tmp_path / "l.csv",
        project_labels=tmp_path / "l.csv",
        fit_colormap=tmp_path / "c.json",
        project_colormap=tmp_path / "c.json",
    )
    pca = PCAConfig(n_pcs=4)
    hc = painting_output_paths(io, pca, PaintingConfig(method="hc"))
    fs = painting_output_paths(io, pca, PaintingConfig(method="finestructure"))
    for key in ("fit_pca", "project_pca", "coancestry"):
        assert hc[key] != fs[key]
    assert hc["pbwt_dir"] == fs["pbwt_dir"] == tmp_path / "out/painting/pbwt"
    shared = painting_output_paths(io, pca, PaintingConfig(pbwt_dir=tmp_path / "s"))
    assert shared["pbwt_roots"]["1"] == tmp_path / "s" / "chr1"


def test_config_file_painting_missing_and_pbwt_dir(tmp_path):
    config = {
        "data": {"output_dir": "o", "phased_vcf": "chr{chr}.vcf"},
        "painting": {"missing": "drop", "pbwt_dir": "shared/pbwt"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    kwargs = load_config(path)
    assert kwargs["painting_missing"] == "drop"
    assert kwargs["painting_pbwt_dir"] == (tmp_path / "shared" / "pbwt").resolve()

    cfg = build_painting_config(missing="drop", pbwt_dir="x/pbwt")
    assert cfg.missing == "drop" and str(cfg.pbwt_dir) == "x/pbwt"
    with pytest.raises(ValueError, match="Unknown painting missing policy"):
        build_painting_config(missing="warn")


# --- tool resolution -------------------------------------------------------


def test_resolve_pbwt_order(tmp_path, monkeypatch):
    monkeypatch.delenv("PBWT_PATH", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    resolver = ToolResolver(download_dir=tmp_path / "tools")

    with pytest.raises(ToolNotFoundError, match="setup --pbwt"):
        resolver.resolve_pbwt()

    on_path = install_fake_pbwt(tmp_path / "onpath")
    monkeypatch.setenv("PATH", str(tmp_path / "onpath"))
    assert resolver.resolve_pbwt() == str(on_path)

    monkeypatch.setenv("PBWT_PATH", str(tmp_path / "missing"))
    with pytest.raises(ToolNotFoundError, match="PBWT_PATH"):
        resolver.resolve_pbwt()

    local = install_fake_pbwt(tmp_path / "tools")
    assert resolver.resolve_pbwt() == str(local)  # setup's build wins


def test_install_pbwt_without_a_compiler_says_how(tmp_path, monkeypatch):
    import manifold_genetics.utils.tools as tools

    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    with pytest.raises(ToolNotFoundError, match="C compiler"):
        ToolResolver(download_dir=tmp_path).install_pbwt()


def test_paint_subcommand(painted, monkeypatch, capsys):
    from manifold_genetics.cli import main

    tmp_path, _ = painted
    monkeypatch.setenv("MANIFOLD_GENETICS_TOOL_DIR", str(tmp_path / "bin"))
    out = tmp_path / "coords.csv"

    assert (
        main(
            [
                "paint",
                "--phased-vcf",
                str(tmp_path / "data" / "chr{chr}.vcf"),
                "--chromosomes",
                "1-2",
                "--pbwt-dir",
                str(tmp_path / "pbwt"),
                "--output",
                str(out),
                "--n-pcs",
                "2",
                "--method",
                "hc",
            ]
        )
        == 0
    )

    assert list(pd.read_csv(out).columns) == ["sample_id", "dim_1", "dim_2"]
    assert "Shape: (6, 2)" in capsys.readouterr().out
