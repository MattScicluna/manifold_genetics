"""examples/ukbb/hgdp_1kgp_proj/data was produced by prepare_data.sh, which
calls preprocess_cross_projection.sh with --skip-wrayner --skip-biobank-maf.
Task 9 found that re-running lossy filters (--geno, LD-prune) on an already-
filtered *subset* is not exactly idempotent. Feeding the intersected UKBB
output back through the UKBB flag set would just repeat that finding at 200x
the cost, so this file tests two different, cheaper things instead:

Test A -- lossless round trip (strict). `intersect-only` with `skip_geno=True`
on the published, already-intersected cohort must reproduce it exactly: no
lossy step runs at all, so nothing should change. Same lossless case Task 9
proved on HGDP.

Test B -- reproduction from raw inputs. Acquiring the raw HGDP/UKBB prefixes
named in mappings_private.json and running `preprocess` with exactly the flags
prepare_data.sh used must reproduce the published fit_subset/project_subset.
This is the actual equivalence check the spec asks for.

`test_subsample_reproduces_the_wb_irish_selection` only needs Test A's `bio`
cohort, so it is independent of Test B.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess, subsample
from manifold_genetics.scaffold import acquire_custom
from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.requires_private_data]

REPO_ROOT = Path(__file__).resolve().parents[2]
UKBB = REPO_ROOT / "examples/ukbb/hgdp_1kgp_proj"
MAPPINGS = UKBB / "data/mappings_private.json"
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))


def _variants(prefix):
    return {tuple(line.split()[i] for i in (0, 3, 4, 5)) for line in open(f"{prefix}.bim")}


def _samples(prefix):
    return [line.split()[1] for line in open(f"{prefix}.fam")]


def _require_tools():
    try:
        ToolResolver().resolve_plink2()
        ToolResolver().resolve_plink1()
    except ToolNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Named, resumable output root: MG_UKBB_SCRATCH persists across runs so a
    long Test B can be stopped and resumed via the shell's own checkpointing."""
    scratch = os.environ.get("MG_UKBB_SCRATCH")
    if scratch:
        out = Path(scratch)
        out.mkdir(parents=True, exist_ok=True)
        return out
    return tmp_path_factory.mktemp("ukbb_repro")


@pytest.fixture(scope="module")
def bio(root):
    """acquire_custom on the published, already-intersected UKBB cohort."""
    _require_tools()
    if not (UKBB / "data/project_subset.bed").exists():
        pytest.skip("UKBB intersected data is not on this machine")
    ref = acquire_custom(
        root / "a_ref",
        fit_plink=UKBB / "data/fit_subset",
        labels=UKBB / "data/fit_labels.csv",
        force=True,
    )
    bio_config = acquire_custom(
        root / "a_bio",
        fit_plink=UKBB / "data/project_subset",
        labels=UKBB / "data/project_labels.csv",
        force=True,
    )
    return ref, bio_config


def test_a_lossless_round_trip_on_the_intersected_cohort(bio, root):
    ref, bio_config = bio
    config = preprocess(
        ref,
        root / "a_out",
        project_config=bio_config,
        options=PreprocessOptions(
            preset="intersect-only",
            skip_geno=True,
            threads=THREADS,
            memory=28000,
            temp_dir=root / "temp",
        ),
        force=True,
    )
    after = load_config(config)
    assert _samples(after["fit_plink"]) == _samples(UKBB / "data/fit_subset")
    assert _samples(after["project_plink"]) == _samples(UKBB / "data/project_subset")
    assert _variants(after["fit_plink"]) == _variants(UKBB / "data/fit_subset")
    assert _variants(after["project_plink"]) == _variants(UKBB / "data/project_subset")


def test_subsample_reproduces_the_wb_irish_selection(bio, root):
    _, bio_config = bio
    keep = UKBB.parent / "10k_WB_5K_Irish/data/fit_samples.txt"
    if not keep.exists():
        pytest.skip("10k_WB_5K_Irish selection is not on this machine")
    config = subsample(bio_config, root / "sub", fit_samples=keep, force=True)
    fit = _samples(load_config(config)["fit_plink"])
    expected = [line.split()[1] for line in open(keep)]
    assert sorted(fit) == sorted(expected)


def _resolve_mapping_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / value)


def _hgdp_labels_as_sample_id(labels_path: Path, root: Path) -> Path:
    """metadata.csv (the raw HGDP label source) uses 'project_meta.sample_id',
    not 'sample_id' -- prepare_data.sh's own embedded Python renames it before
    use. `_checked_labels` requires the latter, so do the same rename here on
    a copy; the content is untouched, only the column name."""
    frame = pd.read_csv(labels_path, dtype=str)
    if "sample_id" in frame.columns:
        return labels_path
    if "project_meta.sample_id" not in frame.columns:
        pytest.skip(
            f"{labels_path} has neither 'sample_id' nor 'project_meta.sample_id'; "
            f"found {list(frame.columns)}"
        )
    frame = frame.rename(columns={"project_meta.sample_id": "sample_id"})
    renamed = root / "fit_labels_sample_id.csv"
    frame.to_csv(renamed, index=False)
    return renamed


@pytest.fixture(scope="module")
def raw_cohorts(root):
    """acquire_custom on the raw HGDP/UKBB prefixes named in mappings_private.json."""
    _require_tools()
    if not MAPPINGS.exists():
        pytest.skip("mappings_private.json is not on this machine")
    mappings = json.loads(MAPPINGS.read_text())
    for key in ("ukbb_plink", "hgdp_plink", "fit_labels", "project_labels"):
        if key not in mappings:
            pytest.skip(f"mappings_private.json is missing {key!r}")
    resolved = {key: _resolve_mapping_path(value) for key, value in mappings.items()}
    for plink_key in ("ukbb_plink", "hgdp_plink"):
        if not Path(f"{resolved[plink_key]}.bed").exists():
            pytest.skip(f"{resolved[plink_key]}.bed is not on this machine")

    fit_labels = _hgdp_labels_as_sample_id(resolved["fit_labels"], root)
    ref = acquire_custom(
        root / "b_ref",
        fit_plink=resolved["hgdp_plink"],
        labels=fit_labels,
        force=True,
    )
    bio_config = acquire_custom(
        root / "b_bio",
        fit_plink=resolved["ukbb_plink"],
        labels=resolved["project_labels"],
        force=True,
    )
    return ref, bio_config


def test_b_reproduction_from_raw_inputs(raw_cohorts, root):
    ref, bio_config = raw_cohorts
    config = preprocess(
        ref,
        root / "b_out",
        project_config=bio_config,
        options=PreprocessOptions(
            skip_wrayner=True,
            skip_project_maf=True,
            threads=THREADS,
            # 28000 OOM-killed the biobank ID-standardisation --make-bed on
            # 486,748 samples (33.4 GB RSS observed); 100000 matches what the
            # original prepare_data.sh wrappers passed.
            memory=100000,
            temp_dir=root / "temp_b",
        ),
        force=True,
    )
    after = load_config(config)
    assert _samples(after["fit_plink"]) == _samples(UKBB / "data/fit_subset")
    assert _samples(after["project_plink"]) == _samples(UKBB / "data/project_subset")
    assert _variants(after["fit_plink"]) == _variants(UKBB / "data/fit_subset")
    assert _variants(after["project_plink"]) == _variants(UKBB / "data/project_subset")
