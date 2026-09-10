"""Output-tree contract test for the pipeline orchestrator.

This is PR5b's mitigation for spec tension T2: every pipeline stage now runs
in-process (Tasks 1-3 of this PR), and ``test_pipeline_output_layout`` here is
the independent, from-the-outside check on that migration. It exercises a
real ``Pipeline(...).run()`` — real FlashPCA, a precomputed admixture backend,
real embedding — for each of the three shapes the example scripts drive
(``examples/_shared/run_pipeline.sh``: projection / subsample / transform) and
asserts the *complete* output tree for that shape, including which files must
NOT exist. A test that only checked presence would pass against an
orchestrator that wrote every file in every mode — exactly the branch bug T2
exists to catch.

Plotting is stubbed (see ``stub_plotting`` below): the real step functions in
``pipeline/steps/viz.py`` run unmodified and compute the real output paths,
but the underlying matplotlib-rendering calls are replaced with a fast
"touch the expected file" stand-in. This keeps the test fast while still
exercising the real path logic — a plotting call made at the wrong path, or
not made at all, is still caught.
"""

from pathlib import Path

import pytest

import manifold_genetics.pipeline.steps.viz as viz_mod
from manifold_genetics.admixture.backends import PrecomputedAdmixtureBackend
from manifold_genetics.pipeline.orchestrator import Pipeline

N_PCS = 5
K_MIN, K_MAX = 2, 3
METHOD = "phate"
EMBEDDING_INPUT_BY_MODE = {
    "projection": "both",
    "subsample": "fit",
    "transform": "project",
}
FIT_PLOT_COLUMN = "Population"
PROJECT_PLOT_COLUMN = "self_described_ancestry"


def _touch_plot(**kwargs) -> Path:
    """Stand-in for the four plotting functions that take an ``output_path``
    keyword and return it. Writes an empty file at the real path the step
    computed, so presence/absence assertions still exercise real path logic.
    """
    output_path = Path(kwargs["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch()
    return output_path


def _fake_visualize(
    embedding, labels, colormap, output_dir=None, output_prefix="embedding", dataset_prefix=""
):
    """Stand-in for ``visualize()``: same output-path formula
    (``{dataset_prefix}{output_prefix}_by_{label_col}.png``, one per colormap
    key), without rendering any figure.
    """
    output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    colormap_dict = viz_mod.read_colormap(colormap)
    paths = []
    for label_col in colormap_dict.keys():
        p = output_dir / f"{dataset_prefix}{output_prefix}_by_{label_col}.png"
        p.touch()
        paths.append(p)
    return paths


@pytest.fixture
def stub_plotting(monkeypatch):
    """Replace the matplotlib-rendering functions imported into
    ``pipeline/steps/viz.py`` with fast stand-ins (see module docstring).
    """
    monkeypatch.setattr(viz_mod, "plot_pca_pairs", _touch_plot)
    monkeypatch.setattr(viz_mod, "plot_admixture_bar_grid", _touch_plot)
    monkeypatch.setattr(viz_mod, "plot_admixture_embedding_grid", _touch_plot)
    monkeypatch.setattr(viz_mod, "plot_projection", _touch_plot)
    monkeypatch.setattr(viz_mod, "visualize", _fake_visualize)


def _assert_exists(path: Path, why: str):
    assert path.exists(), f"expected to exist ({why}): {path}"


def _assert_absent(path: Path, why: str):
    assert not path.exists(), f"expected to be ABSENT ({why}): {path}"


def _assert_nonempty_dir(path: Path, why: str):
    assert path.is_dir(), f"expected directory to exist ({why}): {path}"
    assert any(path.iterdir()), f"expected directory to be non-empty ({why}): {path}"


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["projection", "subsample", "transform"])
def test_pipeline_output_layout(
    mode,
    fit_plink_files,
    project_plink_files,
    cross_cohort_fixtures,
    temp_dir,
    stub_plotting,
):
    """Run a full pipeline for one mode and assert the complete output tree.

    Modes (from ``examples/_shared/run_pipeline.sh``):
      - projection: embedding_input="both" — cross-cohort, fit AND project
        embeddings, projection plot.
      - subsample:  embedding_input="fit" — within-cohort, fit subset only.
      - transform:  embedding_input="project" — within-cohort, project set
        only.

    Only "projection" mode produces a fit embedding and a projection plot;
    the other two modes must NOT produce them.
    """
    output_dir = temp_dir / "pipeline_output"
    embedding_input = EMBEDDING_INPUT_BY_MODE[mode]

    admixture_fixtures_dir = Path(__file__).parent.parent / "fixtures" / "admixture"
    admix_backend = PrecomputedAdmixtureBackend(
        k_min=K_MIN, k_max=K_MAX, fixtures_dir=str(admixture_fixtures_dir)
    )

    pipeline = Pipeline(
        fit_plink_prefix=fit_plink_files,
        project_plink_prefix=project_plink_files,
        fit_labels=cross_cohort_fixtures["fit_labels"],
        project_labels=cross_cohort_fixtures["project_labels"],
        fit_colormap=cross_cohort_fixtures["fit_colormap"],
        project_colormap=cross_cohort_fixtures["project_colormap"],
        geographic_coords=cross_cohort_fixtures["geographic"],
        output_dir=output_dir,
        admixture_backend=admix_backend,
        projection_plot_fit_column=FIT_PLOT_COLUMN,
        projection_plot_project_column=PROJECT_PLOT_COLUMN,
    )

    result = pipeline.run(
        n_pcs=N_PCS,
        k_min=K_MIN,
        k_max=K_MAX,
        embedding=METHOD,
        embedding_params={"knn": 5, "n_components": 2},
        embedding_input=embedding_input,
    )

    # --- PipelineResult shape -------------------------------------------------
    assert result.failed_steps == (), f"unexpected failed steps: {result.failed_steps}"
    assert result.pca is not None
    assert result.admixture is not None
    assert result.embedding is not None

    is_projection = mode == "projection"

    if is_projection:
        assert result.embedding.fit_embedding_file is not None
        assert result.projection_plot is not None
        assert result.fit_embedding_figures, "expected fit embedding figures in projection mode"
    else:
        assert (
            result.embedding.fit_embedding_file is None
        ), f"fit_embedding_file must be None in {mode} mode"
        assert result.projection_plot is None, f"projection_plot must be None in {mode} mode"
        assert (
            result.fit_embedding_figures == ()
        ), f"fit_embedding_figures must be empty in {mode} mode"

    assert result.pca_figures, "expected PCA figures"
    assert result.embedding_figures, "expected project embedding figures"
    assert "bars" in result.admixture_figures
    assert "admixture_colored_embedding" in result.admixture_figures

    # --- On-disk tree ----------------------------------------------------------
    pca_dir = output_dir / "pca"
    admixture_dir = output_dir / "admixture"
    embeddings_dir = output_dir / "embeddings"
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"

    # PCA — same in every mode: PCA does not depend on embedding_input.
    _assert_exists(pca_dir / f"fit_pca_{N_PCS}.csv", "fit PCA is always produced")
    _assert_exists(pca_dir / f"project_pca_{N_PCS}.csv", "project PCA is always produced")
    _assert_nonempty_dir(pca_dir / "flashpca_outputs", "flashpca raw outputs always written")

    # Admixture — always runs against both cohorts, independent of embedding_input.
    _assert_exists(admixture_dir / "checkpoints", "checkpoints dir always created")
    for k in range(K_MIN, K_MAX + 1):
        _assert_exists(admixture_dir / f"fit.{k}.csv", f"fit Q file for K={k}")
        _assert_exists(admixture_dir / f"project.{k}.csv", f"project Q file for K={k}")

    # Embeddings — the project-direction CSV always exists; the fit-direction
    # CSV is the T2 branch: it must exist ONLY in projection mode.
    _assert_exists(embeddings_dir / f"{METHOD}_2d.csv", "project embedding always produced")
    fit_embedding_csv = embeddings_dir / f"{METHOD}_fit_2d.csv"
    if is_projection:
        _assert_exists(fit_embedding_csv, "fit embedding only in projection mode")
    else:
        _assert_absent(fit_embedding_csv, f"fit embedding must not exist in {mode} mode")

    # Figures.
    _assert_nonempty_dir(figures_dir / "pca", "PCA-pairs figure per colormap column")
    _assert_nonempty_dir(figures_dir / "embeddings", "project embedding figure(s) always")
    _assert_exists(
        figures_dir / "admixture" / "project_bars.png", "admixture bar plot always produced"
    )
    _assert_exists(
        figures_dir / "admixture" / "project_admixture_colored_embedding.png",
        "admixture-colored embedding plot always produced",
    )

    fit_embedding_figure = figures_dir / "embeddings" / f"fit_{METHOD}_by_{FIT_PLOT_COLUMN}.png"
    projection_plot_figure = (
        figures_dir
        / "embeddings"
        / f"{METHOD}_projection_fit_{FIT_PLOT_COLUMN}_project_{PROJECT_PLOT_COLUMN}.png"
    )
    if is_projection:
        _assert_exists(fit_embedding_figure, "fit embedding figure only in projection mode")
        _assert_exists(projection_plot_figure, "projection plot only in projection mode")
    else:
        _assert_absent(fit_embedding_figure, f"fit embedding figure must not exist in {mode} mode")
        _assert_absent(projection_plot_figure, f"projection plot must not exist in {mode} mode")

    # Metrics — geographic_coords is configured and admixture always runs, so
    # both metrics always run regardless of embedding_input.
    _assert_exists(metrics_dir / "geographic.json", "geographic metrics always run here")
    _assert_exists(metrics_dir / "admixture.json", "admixture metrics always run here")
