"""Cheap checks that a cohort's real data agrees with the config that describes it.

Every bug this repo has found in a real run was of this shape: something in the
data did not match what the config claimed, and it surfaced forty minutes in, or
not at all. Labels belonging to a superseded sample selection coloured 40% of a
published figure. A colormap keyed on the wrong vocabulary would have done the
same. A plot column that does not exist fails after PCA and admixture are done.

None of that needs the pipeline to run. These tests read the ``.fam`` files, the
labels and the colormap and cross-check them, in seconds -- so run them first,
on any machine that has the data, before committing to a real run:

    pytest tests/integration/test_cohort_preflight.py -v

Cohorts whose data is absent skip; controlled-access ones also carry the
``requires_private_data`` marker so they can be deselected outright.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from tests.integration.cohorts import COHORTS, Cohort, rebase, resolve_data_root

pytestmark = pytest.mark.integration

COHORT_PARAMS = [
    pytest.param(
        name,
        id=name,
        marks=() if cohort.public else (pytest.mark.requires_private_data,),
    )
    for name, cohort in sorted(COHORTS.items())
]


@dataclass
class Prepared:
    """A cohort whose data is present, with its config already resolved."""

    cohort: Cohort
    root: Path
    config: dict

    def path(self, key):
        value = self.config.get(key)
        return None if value is None else rebase(value, self.cohort, self.root)

    def side(self, side, key):
        """``fit_labels`` if the config sets it, else the shared ``labels``."""
        return self.path(f"{side}_{key}") or self.path(key)


@pytest.fixture(params=COHORT_PARAMS)
def prepared(request):
    cohort = COHORTS[request.param]

    root = resolve_data_root(cohort)
    if root is None:
        pytest.skip(f"{cohort.name}: no data root (set {cohort.env_var})")

    missing = cohort.missing(root)
    if missing:
        pytest.skip(f"{cohort.name}: data not prepared, missing {missing[0]}")

    return Prepared(cohort, root, load_config(cohort.config_path))


def fam_ids(prefix: Path) -> set:
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, dtype=str)
    return set(fam[1])


def labels_of(prepared: Prepared, side: str) -> pd.DataFrame:
    return pd.read_csv(prepared.side(side, "labels"), dtype=str)


# ---------------------------------------------------------------------------
# The genotypes themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["fit", "project"])
def test_bed_size_matches_its_bim_and_fam(prepared, side):
    """A truncated or mismatched .bed is otherwise found deep inside flashpca.

    PLINK 1 SNP-major: a 3-byte magic header, then ceil(n_samples/4) bytes for
    each variant.
    """
    prefix = prepared.path(f"{side}_plink")
    n_samples = sum(1 for _ in open(f"{prefix}.fam"))
    n_variants = sum(1 for _ in open(f"{prefix}.bim"))

    expected = 3 + n_variants * -(-n_samples // 4)

    assert (
        Path(f"{prefix}.bed").stat().st_size == expected
    ), f"{prefix}.bed is not {n_variants} variants x {n_samples} samples"


@pytest.mark.parametrize("side", ["fit", "project"])
def test_sample_ids_are_unique(prepared, side):
    prefix = prepared.path(f"{side}_plink")
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, dtype=str)

    duplicated = fam[1][fam[1].duplicated()].tolist()

    assert not duplicated, f"{side} set repeats sample ids: {duplicated[:5]}"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["fit", "project"])
def test_labels_have_a_sample_id_column(prepared, side):
    assert "sample_id" in labels_of(prepared, side).columns


@pytest.mark.parametrize("side", ["fit", "project"])
def test_labels_cover_every_genotyped_sample(prepared, side):
    """The stale-labels defect of 2026-09-10, as a test.

    A label file left over from a superseded sample selection still loads, still
    merges, and silently drops every sample it does not know -- which is how a
    published figure came to colour 40% of its points from the wrong selection.
    """
    genotyped = fam_ids(prepared.path(f"{side}_plink"))
    labelled = set(labels_of(prepared, side)["sample_id"])

    uncovered = genotyped - labelled

    assert not uncovered, (
        f"{prepared.cohort.name} {side}: {len(uncovered)} of {len(genotyped)} genotyped "
        f"samples have no label ({len(genotyped & labelled) / len(genotyped):.1%} covered) "
        f"-- regenerate with prepare_data.sh"
    )


@pytest.mark.parametrize("side", ["fit", "project"])
def test_the_columns_the_plots_name_exist(prepared, side):
    """A missing plot column fails after PCA and admixture have already run."""
    columns = set(labels_of(prepared, side).columns)

    named = [
        prepared.config.get("admix_group_column"),
        prepared.config.get(f"projection_plot_{side}_column"),
    ]
    for column in filter(None, named):
        # admix_group_column describes the cohort being plotted, which for a
        # projection run is the project side only.
        if column in columns:
            continue
        other = set(labels_of(prepared, "fit" if side == "project" else "project").columns)
        assert column in other, f"{column!r} is in neither side's labels"


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["fit", "project"])
def test_the_colormap_covers_the_label_values_it_claims(prepared, side):
    """A colormap keyed on a different vocabulary leaves points uncoloured.

    ``examples/aou/hgdp_1kgp_proj`` needs ``hgdp_1kgp_aou_aligned.json`` rather
    than the plain one; picking the wrong file produces a plot, not an error.
    """
    colormap = json.loads(Path(prepared.side(side, "colormap")).read_text())
    labels = labels_of(prepared, side)

    checked = []
    for column, colours in colormap.items():
        if column not in labels.columns:
            continue
        checked.append(column)
        values = set(labels[column].dropna())

        uncoloured = values - set(colours)

        assert not uncoloured, (
            f"{prepared.cohort.name} {side}: {column} has values with no colour: "
            f"{sorted(uncoloured)[:5]}"
        )

    assert checked, (
        f"{prepared.cohort.name} {side}: the colormap keys "
        f"{sorted(colormap)} match none of the label columns {sorted(labels.columns)}"
    )


# ---------------------------------------------------------------------------
# Metrics inputs
# ---------------------------------------------------------------------------


def test_geographic_coords_cover_the_embedded_samples(prepared):
    coords_path = prepared.path("geographic_coords")
    if coords_path is None:
        pytest.skip(f"{prepared.cohort.name} computes no geographic metric")

    coords = pd.read_csv(coords_path, dtype=str)
    embedded = "fit" if prepared.config.get("embedding_input") == "fit" else "project"
    genotyped = fam_ids(prepared.path(f"{embedded}_plink"))

    covered = genotyped & set(coords["sample_id"])

    assert len(covered) / len(genotyped) > 0.5, (
        f"geographic coords cover only {len(covered)}/{len(genotyped)} of the "
        f"{embedded} set; the metric would be computed on a biased remnant"
    )
