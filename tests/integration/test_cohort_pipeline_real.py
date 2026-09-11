"""One end-to-end pipeline run per real cohort, with assertions about the science.

Every shipped example is driven here through exactly the config file it ships,
so what is tested is what a user runs -- not a reduced stand-in that shares no
settings with it.

Selection is explicit. Public cohorts run whenever their data is present; a
controlled-access cohort runs only when named, because these are hour-scale runs
on hundreds of thousands of samples:

    pytest tests/integration/test_cohort_pipeline_real.py                      # HGDP
    pytest tests/integration/test_cohort_pipeline_real.py --cohort ukbb_projection
    pytest tests/integration/test_cohort_pipeline_real.py --cohort all --cohort-admixture

Admixture is off unless asked for: it wants a GPU and hours, and it is not what
these assertions are about.

The assertions are chance-corrected (see ``tests/science.py``). The cohorts span
3,400 to 486,748 samples and 7 to 300-odd label groups, so a fixed threshold on
a raw statistic would be either vacuous on one end or impossible on the other.
Run ``test_cohort_preflight.py`` first -- it catches data/config disagreements in
seconds that would otherwise surface at the end of one of these runs.
"""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.pipeline.runner import run_pipeline
from tests.integration.cohorts import COHORTS, rebase, resolve_data_root
from tests.integration.test_cohort_preflight import COHORT_PARAMS, fam_ids
from tests.science import neighbourhood_preservation, separation_over_chance

pytestmark = [pytest.mark.slow, pytest.mark.integration]

# Config keys naming a file on disk, which move with the data root.
PATH_KEYS = (
    "fit_plink",
    "project_plink",
    "labels",
    "fit_labels",
    "project_labels",
    "colormap",
    "fit_colormap",
    "project_colormap",
    "geographic_coords",
)

# Floors, not targets. Calibrated on HGDP+1KGP (3,400 fit / 4,094 project, 172k
# variants, 2m35s), the only cohort small enough to measure cheaply; a healthy
# run clears each of them by more than an order of magnitude, so what they catch
# is a pipeline that broke, not one that drifted. Each test prints what it
# actually scored -- see submit_cohort_tests.sh, which runs with -s.
MIN_PC_SEPARATION = 3.0  # HGDP: 54.3 on Population over the first 10 PCs
MIN_EMBEDDING_SEPARATION = 3.0  # HGDP: 57.5
MIN_NEIGHBOURHOOD_PRESERVATION = 0.05  # HGDP: 0.333 at k=30; chance here is 0.007


class CohortRun:
    """One completed pipeline run, with the inputs needed to check it."""

    def __init__(self, cohort, root, config, result):
        self.cohort = cohort
        self.root = root
        self.config = config
        self.result = result

    @property
    def embedded_side(self):
        """Which set the embedding covers. ``both`` embeds them jointly."""
        return {"fit": "fit", "project": "project"}.get(
            self.config.get("embedding_input", "both"), "project"
        )

    def labels(self, side):
        path = self.config.get(f"{side}_labels") or self.config["labels"]
        return pd.read_csv(path, dtype=str)

    def group_column(self, side):
        """The label column this cohort is coloured by.

        Taken from the colormap rather than declared separately: the colormap is
        already the statement of which vocabulary this cohort is described in,
        and a second copy would drift from it.
        """
        import json
        from pathlib import Path

        path = self.config.get(f"{side}_colormap") or self.config["colormap"]
        colormap = json.loads(Path(path).read_text())
        columns = set(self.labels(side).columns)
        for column in colormap:
            if column in columns:
                return column
        raise AssertionError(f"no colormap key of {sorted(colormap)} is a label column")


def _coords(path):
    df = pd.read_csv(path)
    dims = [c for c in df.columns if c.startswith("dim_")]
    return df, dims


def _report(cohort, name, value):
    """Print a measurement so a batch log records what the run actually scored.

    The thresholds below are floors; knowing how far above them a cohort sits is
    what tells you whether a later change degraded something. Visible with
    ``pytest -s``, which is how submit_cohort_tests.sh runs.
    """
    print(f"[{cohort}] {name} = {value:.4g}")


@pytest.fixture(scope="module", params=COHORT_PARAMS)
def run(request, tmp_path_factory):
    cohort = COHORTS[request.param]
    selected = request.config.getoption("--cohort")

    if not (cohort.public or "all" in selected or cohort.name in selected):
        pytest.skip(f"{cohort.name}: not selected (pass --cohort {cohort.name})")

    root = resolve_data_root(cohort)
    if root is None:
        pytest.skip(f"{cohort.name}: no data root (set {cohort.env_var})")

    missing = cohort.missing(root)
    if missing:
        pytest.skip(f"{cohort.name}: data not prepared, missing {missing[0]}")

    config = load_config(cohort.config_path)
    config.pop("output_dir")
    config = {k: (rebase(v, cohort, root) if k in PATH_KEYS else v) for k, v in config.items()}

    # load_config always emits every skip flag, so set rather than pass again.
    if not request.config.getoption("--cohort-admixture"):
        config["skip_admixture"] = True

    result = run_pipeline(output_dir=tmp_path_factory.mktemp(cohort.name), **config)
    return CohortRun(cohort, root, config, result)


# ---------------------------------------------------------------------------
# The run completed
# ---------------------------------------------------------------------------


def test_no_stage_failed(run):
    """The orchestrator tolerates some failures; a cohort run must have none."""
    assert run.result.failed_steps == ()


def test_every_figure_it_reported_was_actually_written(run):
    figures = run.result.figures

    assert figures, "the run produced no figures"
    assert [f for f in figures if not f.exists()] == []


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["fit", "project"])
def test_pca_covers_exactly_the_genotyped_samples(run, side):
    path = getattr(run.result.pca, f"{side}_pca")
    df, _ = _coords(path)

    assert set(df["sample_id"].astype(str)) == fam_ids(run.config[f"{side}_plink"])


@pytest.mark.parametrize("side", ["fit", "project"])
def test_pca_has_the_requested_number_of_finite_components(run, side):
    df, dims = _coords(getattr(run.result.pca, f"{side}_pca"))

    assert len(dims) == run.config.get("n_pcs", 50)
    assert np.isfinite(df[dims].to_numpy()).all()


def test_components_are_ordered_by_the_variance_they_carry(run):
    """A backend that returned singular vectors unsorted would still look fine
    in a scatter plot of PC1 vs PC2, and be wrong everywhere downstream."""
    df, dims = _coords(run.result.pca.fit_pca)
    variances = df[dims].var().to_numpy()

    assert np.all(np.diff(variances) <= 1e-9), variances


def test_the_components_separate_the_cohort_s_label_groups(run):
    """The headline claim of the PCA step, stated as a number.

    Chance-corrected, so it means the same thing for HGDP's seven regions and
    UK Biobank's heavily imbalanced ancestry labels.
    """
    side = run.embedded_side
    df, dims = _coords(getattr(run.result.pca, f"{side}_pca"))
    merged = df.merge(run.labels(side), on="sample_id")
    assert len(merged) == len(df), "labels did not cover the PCA output"

    X = merged[dims[:10]].to_numpy()
    y = merged[run.group_column(side)].to_numpy()

    separation = separation_over_chance(X, y)
    _report(run.cohort.name, f"pc_separation({run.group_column(side)})", separation)

    assert separation > MIN_PC_SEPARATION, (
        f"{run.cohort.name}: the first 10 PCs explain only {separation:.1f}x more "
        f"of the variance in {run.group_column(side)} than shuffled labels do"
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_the_embedding_covers_the_set_it_was_asked_to_embed(run):
    df, dims = _coords(run.result.embedding.embedding_file)

    assert dims == ["dim_1", "dim_2"]
    assert np.isfinite(df[dims].to_numpy()).all()
    assert set(df["sample_id"].astype(str)) == fam_ids(run.config[f"{run.embedded_side}_plink"])


def test_the_embedding_keeps_the_neighbourhoods_pca_found(run):
    """An embedding that lost them would be a picture of nothing in particular."""
    side = run.embedded_side
    pcs, pc_dims = _coords(getattr(run.result.pca, f"{side}_pca"))
    emb, emb_dims = _coords(run.result.embedding.embedding_file)

    # Aligned by index rather than merged: the two frames share dim_1 and dim_2
    # but not dim_3..dim_50, so pandas' merge suffixes would rename only the
    # overlap and leave the rest ambiguous.
    pcs = pcs.set_index("sample_id")
    emb = emb.set_index("sample_id").reindex(pcs.index)
    assert emb[emb_dims].notna().all().all(), "the embedding does not cover the PCA output"

    preserved = neighbourhood_preservation(pcs[pc_dims].to_numpy(), emb[emb_dims].to_numpy(), k=30)
    _report(run.cohort.name, "neighbourhood_preservation@30", preserved)

    assert preserved > MIN_NEIGHBOURHOOD_PRESERVATION, (
        f"{run.cohort.name}: only {preserved:.1%} of each sample's 30 nearest "
        f"neighbours in PC space survive into the embedding"
    )


def test_the_embedding_separates_the_cohort_s_label_groups(run):
    side = run.embedded_side
    df, dims = _coords(run.result.embedding.embedding_file)
    merged = df.merge(run.labels(side), on="sample_id")

    separation = separation_over_chance(
        merged[dims].to_numpy(), merged[run.group_column(side)].to_numpy()
    )
    _report(run.cohort.name, f"embedding_separation({run.group_column(side)})", separation)

    assert separation > MIN_EMBEDDING_SEPARATION, (
        f"{run.cohort.name}: the embedding separates {run.group_column(side)} only "
        f"{separation:.1f}x better than shuffled labels"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_geographic_preservation_is_positive_and_significant(run):
    if run.config.get("geographic_coords") is None:
        pytest.skip(f"{run.cohort.name} computes no geographic metric")

    values = run.result.geographic_metrics.values
    _report(run.cohort.name, "geographic_correlation", values["correlation"])

    # HGDP: 0.589, which is what test_hgdp_pipeline_real.py has long recorded for
    # this pipeline -- a useful check that this harness reproduces the old one.
    assert values["correlation"] > 0.4, values
    assert values["p_value"] < 0.01, values
