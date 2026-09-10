"""Tests for PipelineResult (pipeline/result.py).

PipelineResult replaces the loose eighteen-key ``results`` dict that
``Pipeline.run()`` used to build. Most of those keys are already reachable
through the step results this object holds (e.g. ``result.pca.project_pca``),
so this type deliberately does not re-expose them — these tests only cover the
new surface: defaults, the ``figures`` aggregate, the ``metrics`` mapping, and
equality in the presence of DataFrame-bearing step results.
"""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.result import PipelineResult
from manifold_genetics.pipeline.steps.admixture import AdmixtureStepResult
from manifold_genetics.pipeline.steps.embedding import EmbeddingStepResult
from manifold_genetics.pipeline.steps.metrics import MetricsStepResult
from manifold_genetics.pipeline.steps.pca import PCAStepResult

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_are_all_empty_or_none():
    result = PipelineResult()

    assert result.pca is None
    assert result.admixture is None
    assert result.embedding is None
    assert result.geographic_metrics is None
    assert result.admixture_metrics is None

    assert result.pca_figures == ()
    assert result.fit_embedding_figures == ()
    assert result.embedding_figures == ()
    assert result.projection_plot is None
    assert result.admixture_figures == {}

    assert result.failed_steps == ()


def test_default_figures_and_metrics_are_empty():
    result = PipelineResult()

    assert result.figures == ()
    assert result.metrics == {}


# ---------------------------------------------------------------------------
# figures — stage order
# ---------------------------------------------------------------------------


def test_figures_aggregates_in_stage_order():
    pca_fig = Path("figures/pca/pca_pairs_by_Population.png")
    fit_emb_fig = Path("figures/embeddings/phate_fit_2d.png")
    project_emb_fig = Path("figures/embeddings/phate_project_2d.png")
    projection_fig = Path("figures/embeddings/phate_projection.png")
    admix_bars = Path("figures/admixture/project_bars.png")
    admix_emb = Path("figures/admixture/project_admixture_colored_embedding.png")

    result = PipelineResult(
        pca_figures=(pca_fig,),
        fit_embedding_figures=(fit_emb_fig,),
        embedding_figures=(project_emb_fig,),
        projection_plot=projection_fig,
        admixture_figures={"bars": admix_bars, "admixture_colored_embedding": admix_emb},
    )

    assert result.figures == (
        pca_fig,
        fit_emb_fig,
        project_emb_fig,
        projection_fig,
        admix_bars,
        admix_emb,
    )


def test_figures_skips_projection_plot_when_none():
    pca_fig = Path("pca.png")
    admix_bars = Path("bars.png")

    result = PipelineResult(
        pca_figures=(pca_fig,),
        admixture_figures={"bars": admix_bars},
    )

    assert result.figures == (pca_fig, admix_bars)


# ---------------------------------------------------------------------------
# metrics — the shape the CLI summary prints
# ---------------------------------------------------------------------------


def test_metrics_empty_when_neither_ran():
    result = PipelineResult()
    assert result.metrics == {}


def test_metrics_geographic_only():
    geo = MetricsStepResult(path=Path("metrics/geographic.json"), values={"spearman": 0.9})
    result = PipelineResult(geographic_metrics=geo)

    assert result.metrics == {"geographic": {"spearman": 0.9}}


def test_metrics_admixture_only():
    admix = MetricsStepResult(path=Path("metrics/admixture.json"), values={"2": {"spearman": 0.5}})
    result = PipelineResult(admixture_metrics=admix)

    assert result.metrics == {"admixture": {"2": {"spearman": 0.5}}}


def test_metrics_both():
    geo = MetricsStepResult(path=Path("metrics/geographic.json"), values={"spearman": 0.9})
    admix = MetricsStepResult(path=Path("metrics/admixture.json"), values={"2": {"spearman": 0.5}})
    result = PipelineResult(geographic_metrics=geo, admixture_metrics=admix)

    assert result.metrics == {
        "geographic": {"spearman": 0.9},
        "admixture": {"2": {"spearman": 0.5}},
    }


# ---------------------------------------------------------------------------
# Equality with DataFrame-bearing step results
# ---------------------------------------------------------------------------


def test_equal_results_compare_equal_even_with_dataframe_step_results():
    """PCAStepResult.coords_df and EmbeddingStepResult.coords_df are both
    field(compare=False), so a frozen PipelineResult carrying them must not
    raise on __eq__ — and two results built from equal-valued step results
    must compare equal.
    """
    pca_df = pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1]})
    emb_df = pd.DataFrame({"sample_id": ["s1"], "dim_1": [1.0], "dim_2": [2.0]})

    def make():
        return PipelineResult(
            pca=PCAStepResult(
                fit_pca=Path("fit_pca_10.csv"),
                project_pca=Path("project_pca_10.csv"),
                coords_df=pca_df,
            ),
            embedding=EmbeddingStepResult(
                embedding_file=Path("phate_2d.csv"),
                coords_df=emb_df,
            ),
        )

    a, b = make(), make()

    # Must not raise, even though the held DataFrames are elementwise-`==`.
    assert a == b


def test_results_with_different_dataframes_but_equal_other_fields_still_compare_equal():
    """coords_df is excluded from comparison, so differing DataFrame contents
    alone must not make two otherwise-identical results unequal."""
    a = PipelineResult(
        pca=PCAStepResult(
            fit_pca=Path("fit.csv"),
            project_pca=Path("project.csv"),
            coords_df=pd.DataFrame({"sample_id": ["s1"]}),
        )
    )
    b = PipelineResult(
        pca=PCAStepResult(
            fit_pca=Path("fit.csv"),
            project_pca=Path("project.csv"),
            coords_df=pd.DataFrame({"sample_id": ["s2", "s3"]}),
        )
    )

    assert a == b


def test_results_with_different_paths_compare_unequal():
    a = PipelineResult(pca=PCAStepResult(fit_pca=Path("fit.csv"), project_pca=Path("project.csv")))
    b = PipelineResult(pca=PCAStepResult(fit_pca=Path("fit.csv"), project_pca=Path("other.csv")))

    assert a != b


def test_admixture_step_result_is_reachable_but_not_duplicated():
    """Spec: the compute step results carry what the old dict's keys duplicated.
    Nothing on PipelineResult re-exposes them as top-level fields — only through
    the step result object itself.
    """
    admix = AdmixtureStepResult(
        q_prefix=Path("admixture/project"),
        k_values=(2, 3),
        dir=Path("admixture"),
        checkpoints_dir=Path("admixture/checkpoints"),
        fit_prefix=Path("admixture/fit"),
    )
    result = PipelineResult(admixture=admix)

    assert result.admixture.q_prefix == Path("admixture/project")
    assert not hasattr(result, "admixture_dir")
    assert not hasattr(result, "q_files")


@pytest.mark.parametrize(
    "field_name",
    [
        "pca_file",
        "project_pca_file",
        "fit_pca_file",
        "pca_coords",
        "admixture_dir",
        "admixture_checkpoints_dir",
        "fit_q_files",
        "q_files",
        "project_q_files",
        "embedding_file",
        "fit_embedding_file",
        "embedding_coords",
    ],
)
def test_old_dict_keys_are_not_reexposed_as_top_level_fields(field_name):
    """The twelve old results-dict keys reachable through step results must not
    be duplicated as PipelineResult attributes."""
    assert not hasattr(PipelineResult(), field_name)
