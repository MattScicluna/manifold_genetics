"""End-to-end integration test on the real HGDP+1KGP example data.

Runs the actual `manifold-genetics pca` -> `embed` -> `metrics-geographic`
CLI against the ~7.5K-sample / 172K-SNP subset in examples/hgdp_1kgp/data/,
using the real flashpca binary. This is the only test that exercises the
external-tool contracts and checks that the pipeline recovers real
population structure.

It is marked `slow` and `integration`, so CI (which runs
`-m "not slow and not integration and not network"`) skips it. Run manually:

    uv run pytest tests/integration/test_hgdp_pipeline_real.py -m "slow and integration" -v

The module skips itself when the example data or the flashpca binary are not
present (e.g. a fresh checkout without `manifold-genetics setup` +
`examples/hgdp_1kgp/download_data.sh`).

Neural Admixture is deliberately not run here (GPU, hours); the precomputed
backend covers the admixture code path in tests/integration/.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "examples" / "hgdp_1kgp" / "data"
FLASHPCA = REPO / "bin" / "flashpca"

pytestmark = [pytest.mark.slow, pytest.mark.integration]

if not (DATA / "fit_subset.bed").exists() or not FLASHPCA.exists():
    pytest.skip(
        "HGDP+1KGP example data or bin/flashpca not available",
        allow_module_level=True,
    )

N_PCS = 20
FIT_N = 3400  # examples/hgdp_1kgp/data/fit_subset.fam
PROJECT_N = 4094  # examples/hgdp_1kgp/data/project_subset.fam


def _cli(*args, cwd=REPO):
    subprocess.run(
        [sys.executable, "-m", "manifold_genetics.cli", *map(str, args)],
        check=True,
        cwd=cwd,
        env={**__import__("os").environ, "FLASHPCA_PATH": str(FLASHPCA)},
    )


@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("hgdp")
    _cli(
        "pca",
        "--fit-plink",
        DATA / "fit_subset",
        "--project-plink",
        DATA / "project_subset",
        "--fit-output",
        out / "fit_pca.csv",
        "--project-output",
        out / "project_pca.csv",
        "--flashpca-output-dir",
        out / "flashpca",
        "--n-pcs",
        N_PCS,
    )
    _cli(
        "embed",
        "--method",
        "phate",
        "--fit-input",
        out / "project_pca.csv",
        "--project-output",
        out / "phate_2d.csv",
        "--knn",
        30,
    )
    _cli(
        "metrics-geographic",
        "--embedding",
        out / "phate_2d.csv",
        "--geographic",
        DATA / "hgdp_project_geographic.csv",
        "--output",
        out / "geographic.json",
    )
    return out


def _fam_ids(prefix):
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None)
    return set(fam[1].astype(str))


# ---------------------------------------------------------------------------
# PCA output shape + identity
# ---------------------------------------------------------------------------


def test_pca_fit_output_shape(pipeline_outputs):
    df = pd.read_csv(pipeline_outputs / "fit_pca.csv")
    assert len(df) == FIT_N
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    assert len(dim_cols) == N_PCS
    assert np.isfinite(df[dim_cols].to_numpy()).all()
    assert set(df["sample_id"].astype(str)) == _fam_ids(DATA / "fit_subset")


def test_pca_project_output_shape(pipeline_outputs):
    df = pd.read_csv(pipeline_outputs / "project_pca.csv")
    assert len(df) == PROJECT_N
    assert set(df["sample_id"].astype(str)) == _fam_ids(DATA / "project_subset")


# ---------------------------------------------------------------------------
# PCA correctness — it must recover real population structure
# ---------------------------------------------------------------------------


def test_pca_recovers_continental_structure(pipeline_outputs):
    pca = pd.read_csv(pipeline_outputs / "project_pca.csv")
    labels = pd.read_csv(DATA / "hgdp_project_labels.csv")
    m = pca.merge(labels, on="sample_id")
    assert len(m) == PROJECT_N

    dims = [f"dim_{i}" for i in range(1, 11)]
    X = m[dims].to_numpy()
    X = (X - X.mean(0)) / X.std(0)
    y = m["Genetic_region_merged"].to_numpy()

    # Nearest-centroid classification of genetic region from the first 10 PCs.
    # On this data a correct projection gives ~0.96; a broken one collapses.
    regions = np.unique(y)
    centroids = np.array([X[y == r].mean(0) for r in regions])
    dists = ((X[:, None, :] - centroids[None]) ** 2).sum(-1)
    pred = regions[np.argmin(dists, axis=1)]
    accuracy = (pred == y).mean()
    assert accuracy > 0.85, f"PC-based region classification only {accuracy:.2f}"

    # PC1 should separate African from non-African samples (the largest axis
    # of human genetic variation).
    pc1 = m["dim_1"].to_numpy()
    sep = abs(pc1[y == "Africa"].mean() - pc1[y != "Africa"].mean()) / pc1.std()
    assert sep > 1.5, f"PC1 Africa/non-Africa separation only {sep:.2f} SD"


# ---------------------------------------------------------------------------
# PHATE + geographic metric
# ---------------------------------------------------------------------------


def test_phate_embedding_shape(pipeline_outputs):
    df = pd.read_csv(pipeline_outputs / "phate_2d.csv")
    assert len(df) == PROJECT_N
    assert [c for c in df.columns if c.startswith("dim_")] == ["dim_1", "dim_2"]
    assert np.isfinite(df[["dim_1", "dim_2"]].to_numpy()).all()


def test_geographic_preservation_is_strong(pipeline_outputs):
    result = json.loads((pipeline_outputs / "geographic.json").read_text())
    # reference run of this pipeline gives ~0.59
    assert result["correlation"] > 0.4, result
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_samples"] > 3000
