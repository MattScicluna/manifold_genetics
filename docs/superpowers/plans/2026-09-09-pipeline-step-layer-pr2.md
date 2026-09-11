# Pipeline Step Layer — PR 2: PCA step + metrics steps

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the PCA and the two metrics stages out of `subprocess.run(["manifold-genetics", ...])` and into in-process step functions that both `Pipeline.run()` and the `cmd_*` CLI handlers call.

**Architecture:** Add `src/manifold_genetics/pipeline/steps/pca.py` and `steps/metrics.py` (layer L2). `steps/pca.py` exposes `run_pca()` — the shared seam the CLI and the step both call — plus `run_pca_step(io, pca) -> PCAStepResult`, which adds path resolution and the stale-component-count guard on top. `steps/metrics.py` exposes `run_geographic_metrics_step()` / `run_admixture_metrics_step()`, which take plain `Path` arguments (not step results) so they stay decoupled from whichever layer produced the embedding CSV and the Q files. The orchestrator's three `subprocess.run` calls for these stages are then replaced with direct calls. Admixture, embedding and visualization keep shelling out — those land in PRs 3–5.

**Tech Stack:** Python 3.10–3.12, pandas, pytest, `uv` for env management. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-30-pipeline-step-layer-design.md` (see the migration table row #2). PR 1 landed as GitHub PR #68 (`config.py`, `steps/paths.py`, cross-cohort fixtures).

## Global Constraints

Copied from the spec's "Hard constraints" table. Every task's requirements implicitly include these.

- **A — The CLI surface is frozen.** No flag renames, removals, or additions to any `subparsers.add_parser(...)` block in `src/manifold_genetics/cli.py`. This PR must not touch `main()` at all.
- **B — Output paths are a contract.** PCA writes `output_dir/pca/fit_pca_{n}.csv`, `output_dir/pca/project_pca_{n}.csv`, `output_dir/pca/flashpca_outputs/`. Metrics write `output_dir/metrics/geographic.json` and `output_dir/metrics/admixture.json`. These paths come from `pca_output_paths()` / `metrics_output_paths()` in `src/manifold_genetics/pipeline/steps/paths.py` and nowhere else.
- **C — With a step skipped, `run()` still supplies its output paths downstream.** `skip_pca=True` must keep populating `results["fit_pca_file"]` / `results["project_pca_file"]` from disk when those files exist. (The "raise a clear error" half of constraint C is PR 5's job; PR 2 preserves today's behaviour exactly.)
- **D — Visualization steps are non-fatal, compute steps are fatal.** PCA and metrics are compute steps: they must raise on failure. Do not add try/except around them.
- **F — `project` naming everywhere; no dead params.**
- **Behaviour-preserving.** `run_pipeline()` still returns the same loose `results` dict with the same keys and the same value types. `PipelineResult` is PR 5.
- **Style.** `black` (line length 100), `isort` (profile: black), `flake8`. Run all three before every commit.
- **Tests.** `uv run pytest -m "not slow and not network"` must be green at the end of every task. The login node is slow and shared; run the suite on a compute node: `srun --jobid=<JOBID> --overlap --ntasks=1 --time=10:00 bash -c 'cd $REPO_ROOT && source .venv/bin/activate && pytest -m "not slow and not network" -q'`.

## Deviation from the spec (deliberate, carry forward)

The spec's §"L2 — step layer" sketch has each step module exporting its own `*_output_paths`. PR 1 instead put **all** path helpers in one module, `steps/paths.py`. Keep it that way: `steps/pca.py` and `steps/metrics.py` **import** their path helper from `steps/paths.py` and re-export it. The spec's actual requirement — "path-layout knowledge lives in exactly one function per step" — is satisfied, and `tests/unit/test_step_paths.py` stays intact.

## File Structure

**Create:**
- `src/manifold_genetics/pipeline/steps/pca.py` — `PCAStepResult`, `run_pca()` (shared CLI/step seam), `_existing_pcs_mismatch()`, `run_pca_step()`.
- `src/manifold_genetics/pipeline/steps/metrics.py` — `MetricsStepResult`, `_write_and_reload()`, `run_geographic_metrics_step()`, `run_admixture_metrics_step()`.
- `tests/unit/test_step_pca.py`
- `tests/unit/test_step_metrics.py`

**Modify:**
- `src/manifold_genetics/pipeline/steps/__init__.py` — re-export the new public names.
- `src/manifold_genetics/cli.py` — `cmd_pca` (lines 77–120), `cmd_metrics_geographic` (406–432), `cmd_metrics_admixture` (435–470), and the `from .pca import PCA` import (line 17). **No changes below line 732 (`main()`).**
- `src/manifold_genetics/pipeline/orchestrator.py` — imports (lines 1–24), new `_io_config()` method, the PCA block (lines ~175–250), the PCA-viz path lookup (~253–256), the metrics block (~580–630).
- `tests/unit/test_orchestrator.py` — delete `TestPipelineRunPCAForce`; rework `TestPipelineRunFullFlow`.
- `tests/unit/test_cli_main.py` — `test_cmd_pca_fit_project`, `test_cmd_metrics_geographic_writes_json`, `test_cmd_metrics_admixture_writes_json`.

---

### Task 1: `steps/pca.py` — the PCA step and its shared seam

**Files:**
- Create: `src/manifold_genetics/pipeline/steps/pca.py`
- Create: `tests/unit/test_step_pca.py`
- Modify: `src/manifold_genetics/pipeline/steps/__init__.py`

**Interfaces:**
- Consumes: `IOConfig`, `PCAConfig` from `manifold_genetics.pipeline.config`; `pca_output_paths(io, pca) -> Dict[str, Path]` from `manifold_genetics.pipeline.steps.paths` (keys: `"fit_pca"`, `"project_pca"`, `"flashpca_dir"`); `PCA(n_components: int, force: bool)` from `manifold_genetics.pca` with methods `fit(plink_prefix, output_dir=None)`, `project(plink_prefix, output_path=None) -> pd.DataFrame`, `fit_transform(plink_prefix, output_path=None) -> pd.DataFrame`.
- Produces:
  - `PCAStepResult(fit_pca: Path, project_pca: Path, coords_df: pd.DataFrame | None = None, skipped: bool = False)` — frozen dataclass; `coords_df` is declared `compare=False`.
  - `run_pca(fit_plink, project_plink=None, *, project_output, fit_output=None, flashpca_dir=None, n_pcs=50, force=False) -> pd.DataFrame`
  - `run_pca_step(io: IOConfig, pca: PCAConfig) -> PCAStepResult`

Nothing is wired up in this task — it is a pure addition.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_step_pca.py`:

```python
"""Tests for the in-process PCA step (pipeline/steps/pca.py).

The step replaces a `subprocess.run(["manifold-genetics", "pca", ...])` call, so
these tests assert *behaviour* — which PLINK prefix is fitted, which files are
written, when a stale cache forces recomputation — rather than an argv list.

The real `PCA` class shells out to the flashpca binary; it is faked here.
"""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.config import IOConfig, PCAConfig
from manifold_genetics.pipeline.steps.paths import pca_output_paths
from manifold_genetics.pipeline.steps.pca import PCAStepResult, run_pca, run_pca_step


def coords_frame(n_dims: int, n_rows: int = 3) -> pd.DataFrame:
    cols = {"sample_id": [f"s{i}" for i in range(n_rows)]}
    cols.update({f"dim_{i}": [float(i)] * n_rows for i in range(1, n_dims + 1)})
    return pd.DataFrame(cols)


def write_pca_csv(path: Path, n_dims: int, n_rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coords_frame(n_dims, n_rows).to_csv(path, index=False)


class FakePCA:
    """Stand-in for manifold_genetics.pca.PCA. Records calls, writes real CSVs."""

    instances = []

    def __init__(self, n_components, force=False):
        self.n_components = n_components
        self.force = force
        self.calls = []
        FakePCA.instances.append(self)

    def fit(self, plink_prefix, output_dir=None):
        self.calls.append(("fit", str(plink_prefix), str(output_dir)))

    def project(self, plink_prefix, output_path=None):
        self.calls.append(("project", str(plink_prefix), str(output_path)))
        df = coords_frame(self.n_components)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        return df

    def fit_transform(self, plink_prefix, output_path=None):
        self.calls.append(("fit_transform", str(plink_prefix), str(output_path)))
        df = coords_frame(self.n_components)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        return df


@pytest.fixture
def fake_pca(monkeypatch):
    FakePCA.instances = []
    monkeypatch.setattr("manifold_genetics.pipeline.steps.pca.PCA", FakePCA)
    return FakePCA


def make_io(tmp_path) -> IOConfig:
    return IOConfig(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=tmp_path / "out",
        fit_labels=Path("fl.csv"),
        project_labels=Path("pl.csv"),
        fit_colormap=Path("fc.json"),
        project_colormap=Path("pc.json"),
    )


# ---------------------------------------------------------------------------
# run_pca — the seam shared with `manifold-genetics pca`
# ---------------------------------------------------------------------------


class TestRunPca:
    """run_pca is called by both cmd_pca and run_pca_step. If the two ever
    disagree about which dataset is fitted vs. projected, cross-cohort results
    are silently wrong — that is the drift this seam exists to prevent."""

    def test_fits_on_fit_prefix_and_projects_project_prefix(self, tmp_path, fake_pca):
        out = tmp_path / "project.csv"
        run_pca("fitset", "projectset", project_output=out, n_pcs=4)

        (inst,) = fake_pca.instances
        assert inst.calls[0] == ("fit", "fitset", "None")
        # No --fit-output requested, so the only projection is the project set.
        assert [c[:2] for c in inst.calls] == [("fit", "fitset"), ("project", "projectset")]

    def test_writes_fit_output_when_requested(self, tmp_path, fake_pca):
        fit_out = tmp_path / "fit.csv"
        proj_out = tmp_path / "project.csv"
        run_pca("fitset", "projectset", fit_output=fit_out, project_output=proj_out, n_pcs=4)

        assert fit_out.exists(), "--fit-output equivalent must be written"
        assert proj_out.exists()
        (inst,) = fake_pca.instances
        assert [c[:2] for c in inst.calls] == [
            ("fit", "fitset"),
            ("project", "fitset"),
            ("project", "projectset"),
        ]

    def test_single_dataset_uses_fit_transform(self, tmp_path, fake_pca):
        """project_plink=None is the `manifold-genetics pca --input X` shape."""
        out = tmp_path / "all.csv"
        run_pca("allsamples", None, project_output=out, n_pcs=4)

        (inst,) = fake_pca.instances
        assert [c[0] for c in inst.calls] == ["fit_transform"]
        assert out.exists()

    def test_flashpca_dir_forwarded_to_fit(self, tmp_path, fake_pca):
        d = tmp_path / "flashpca_outputs"
        run_pca("fitset", "projectset", project_output=tmp_path / "p.csv", flashpca_dir=d)

        (inst,) = fake_pca.instances
        assert inst.calls[0] == ("fit", "fitset", str(d))

    def test_force_forwarded_to_pca_constructor(self, tmp_path, fake_pca):
        run_pca("fitset", "projectset", project_output=tmp_path / "p.csv", force=True)
        assert fake_pca.instances[0].force is True

    def test_returns_project_coordinates(self, tmp_path, fake_pca):
        df = run_pca("fitset", "projectset", project_output=tmp_path / "p.csv", n_pcs=3)
        assert list(df.columns) == ["sample_id", "dim_1", "dim_2", "dim_3"]


# ---------------------------------------------------------------------------
# run_pca_step — config in, typed result out
# ---------------------------------------------------------------------------


class TestRunPcaStep:
    def test_writes_the_documented_output_layout(self, tmp_path, fake_pca):
        """Spec constraint B: downstream example scripts read these exact paths."""
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)

        run_pca_step(io, cfg)

        paths = pca_output_paths(io, cfg)
        assert paths["fit_pca"].exists()
        assert paths["project_pca"].exists()
        assert paths["flashpca_dir"].is_dir()

    def test_result_carries_paths_and_coordinates(self, tmp_path, fake_pca):
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)

        result = run_pca_step(io, cfg)

        paths = pca_output_paths(io, cfg)
        assert isinstance(result, PCAStepResult)
        assert result.fit_pca == paths["fit_pca"]
        assert result.project_pca == paths["project_pca"]
        assert result.skipped is False
        assert isinstance(result.coords_df, pd.DataFrame)
        assert len(result.coords_df.columns) == 6  # sample_id + 5 dims

    def test_fits_on_fit_plink_and_projects_project_plink(self, tmp_path, fake_pca):
        io = make_io(tmp_path)
        run_pca_step(io, PCAConfig(n_pcs=5))

        (inst,) = fake_pca.instances
        assert [c[:2] for c in inst.calls] == [
            ("fit", "data/fit"),
            ("project", "data/fit"),
            ("project", "data/project"),
        ]

    def test_fresh_run_does_not_force(self, tmp_path, fake_pca):
        """Forcing on every run would recompute expensive flashpca needlessly."""
        run_pca_step(make_io(tmp_path), PCAConfig(n_pcs=5))
        assert fake_pca.instances[0].force is False

    def test_matching_cached_dim_count_does_not_force(self, tmp_path, fake_pca):
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)
        write_pca_csv(pca_output_paths(io, cfg)["project_pca"], n_dims=5)

        run_pca_step(io, cfg)
        assert fake_pca.instances[0].force is False

    def test_mismatched_cached_dim_count_forces_recompute(self, tmp_path, fake_pca):
        """A cached 10-PC file with n_pcs=50 requested must not be silently reused —
        stale dimensionality would propagate into every downstream embedding."""
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=50)
        write_pca_csv(pca_output_paths(io, cfg)["project_pca"], n_dims=10)

        run_pca_step(io, cfg)
        assert fake_pca.instances[0].force is True

    def test_unreadable_cached_file_forces_recompute(self, tmp_path, fake_pca):
        """An empty/corrupt cache file must trigger recomputation, not a crash."""
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)
        project_pca = pca_output_paths(io, cfg)["project_pca"]
        project_pca.parent.mkdir(parents=True, exist_ok=True)
        project_pca.write_text("")  # pandas raises EmptyDataError

        run_pca_step(io, cfg)
        assert fake_pca.instances[0].force is True

    def test_config_force_wins_over_matching_cache(self, tmp_path, fake_pca):
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5, force=True)
        write_pca_csv(pca_output_paths(io, cfg)["project_pca"], n_dims=5)

        run_pca_step(io, cfg)
        assert fake_pca.instances[0].force is True

    def test_second_call_is_idempotent(self, tmp_path, fake_pca):
        """Checkpointing: re-running the step on valid outputs re-uses them and
        does not force. PCA(force=False) skips the flashpca work internally."""
        io = make_io(tmp_path)
        cfg = PCAConfig(n_pcs=5)

        run_pca_step(io, cfg)
        second = run_pca_step(io, cfg)

        assert fake_pca.instances[1].force is False
        assert second.project_pca.exists()
        assert second.project_pca == pca_output_paths(io, cfg)["project_pca"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_step_pca.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'manifold_genetics.pipeline.steps.pca'`.

- [ ] **Step 3: Write `src/manifold_genetics/pipeline/steps/pca.py`**

```python
"""In-process PCA step.

``run_pca()`` is the seam shared by the ``manifold-genetics pca`` subcommand and
``run_pca_step()``, so the CLI and the orchestrator cannot drift in how they fit
and project (spec goal 2). ``run_pca_step()`` adds the pipeline's config, path
and checkpoint layer on top.

The real work still happens in a separate process: ``PCA`` shells out to the
flashpca binary. That boundary is L1 and is unchanged here.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ...pca import PCA
from ..config import IOConfig, PCAConfig
from .paths import pca_output_paths

logger = logging.getLogger(__name__)

__all__ = ["PCAStepResult", "pca_output_paths", "run_pca", "run_pca_step"]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PCAStepResult:
    """Typed outputs of the PCA step.

    ``coords_df`` is excluded from comparison: DataFrame ``==`` is elementwise
    and would make a generated ``__eq__`` raise.
    """

    fit_pca: Path
    project_pca: Path
    coords_df: Optional[pd.DataFrame] = field(default=None, compare=False)
    skipped: bool = False


def run_pca(
    fit_plink: PathLike,
    project_plink: Optional[PathLike] = None,
    *,
    project_output: PathLike,
    fit_output: Optional[PathLike] = None,
    flashpca_dir: Optional[PathLike] = None,
    n_pcs: int = 50,
    force: bool = False,
) -> pd.DataFrame:
    """Run FlashPCA and write coordinate CSVs.

    Args:
        fit_plink: PLINK prefix the PCA model is fitted on.
        project_plink: PLINK prefix to project into that space. ``None`` fits and
            projects the same dataset (the ``pca --input X`` shape).
        project_output: CSV path for the projected coordinates.
        fit_output: Optional CSV path for the fit set's own coordinates.
        flashpca_dir: Directory for flashpca's raw intermediate outputs.
        n_pcs: Number of principal components.
        force: Recompute even when cached flashpca outputs exist.

    Returns:
        DataFrame written to ``project_output`` (sample_id, dim_1, ..., dim_N).
    """
    pca = PCA(n_components=n_pcs, force=force)

    if project_plink is None:
        return pca.fit_transform(fit_plink, output_path=project_output)

    pca.fit(fit_plink, output_dir=flashpca_dir)
    if fit_output:
        pca.project(fit_plink, output_path=fit_output)
    return pca.project(project_plink, output_path=project_output)


def _existing_pcs_mismatch(project_pca: Path, n_pcs: int) -> bool:
    """True when a cached project PCA CSV cannot be trusted for ``n_pcs``.

    Reusing a cached file with the wrong number of ``dim_`` columns would feed
    wrong-dimensionality coordinates to every downstream step.
    """
    if not project_pca.exists():
        return False
    try:
        header = pd.read_csv(project_pca, nrows=1)
    except Exception as e:  # unreadable/corrupt cache — recompute rather than crash
        logger.warning(f"Could not check existing PCA file: {e}")
        return True

    existing_n_pcs = len([col for col in header.columns if col.startswith("dim_")])
    if existing_n_pcs != n_pcs:
        logger.info(f"PCA component mismatch: existing={existing_n_pcs}, requested={n_pcs}")
        logger.info("Forcing PCA recomputation...")
        return True
    return False


def run_pca_step(io: IOConfig, pca: PCAConfig) -> PCAStepResult:
    """Fit PCA on the fit cohort and project the project cohort.

    Idempotent: ``PCA(force=False)`` reuses cached flashpca outputs, and a cached
    CSV whose component count disagrees with ``pca.n_pcs`` forces a recompute.
    """
    paths = pca_output_paths(io, pca)
    fit_pca, project_pca = paths["fit_pca"], paths["project_pca"]
    flashpca_dir = paths["flashpca_dir"]

    project_pca.parent.mkdir(parents=True, exist_ok=True)
    flashpca_dir.mkdir(parents=True, exist_ok=True)

    force = pca.force or _existing_pcs_mismatch(project_pca, pca.n_pcs)

    logger.info(f"Running PCA (fit: {io.fit_plink}, project: {io.project_plink})")
    coords = run_pca(
        io.fit_plink,
        io.project_plink,
        fit_output=fit_pca,
        project_output=project_pca,
        flashpca_dir=flashpca_dir,
        n_pcs=pca.n_pcs,
        force=force,
    )

    return PCAStepResult(fit_pca=fit_pca, project_pca=project_pca, coords_df=coords)
```

- [ ] **Step 4: Re-export from `steps/__init__.py`**

Replace the whole file with:

```python
"""In-process pipeline steps.

Each compute step lives in its own module and exposes a pure
``<step>_output_paths()`` helper (defined once in ``steps.paths``) plus a
``run_<step>_step()`` function. Both the ``manifold-genetics`` CLI subcommands
and the pipeline orchestrator call these directly, in-process.
"""

from .metrics import (
    MetricsStepResult,
    metrics_output_paths,
    run_admixture_metrics_step,
    run_geographic_metrics_step,
)
from .paths import admixture_output_paths, embedding_output_paths
from .pca import PCAStepResult, pca_output_paths, run_pca, run_pca_step

__all__ = [
    "MetricsStepResult",
    "PCAStepResult",
    "admixture_output_paths",
    "embedding_output_paths",
    "metrics_output_paths",
    "pca_output_paths",
    "run_admixture_metrics_step",
    "run_geographic_metrics_step",
    "run_pca",
    "run_pca_step",
]
```

This imports `steps.metrics`, which does not exist yet. To keep this task independently green, **omit the `.metrics` import and the four metrics names for now** — write only the `.paths` and `.pca` lines plus their `__all__` entries. Task 3 adds the metrics half.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_step_pca.py tests/unit/test_step_paths.py -q
```

Expected: PASS, 0 failures.

- [ ] **Step 6: Lint and commit**

```bash
uv run black src/ tests/ && uv run isort src/ tests/ && uv run flake8 src/ tests/
git add src/manifold_genetics/pipeline/steps/pca.py \
        src/manifold_genetics/pipeline/steps/__init__.py \
        tests/unit/test_step_pca.py
git commit -m "feat(pipeline): add in-process PCA step

run_pca() is the seam shared by cmd_pca and run_pca_step so the CLI and the
orchestrator cannot drift. run_pca_step() adds path resolution and moves the
stale-component-count guard out of Pipeline.run().

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 2: Point `cmd_pca` at the shared seam

**Files:**
- Modify: `src/manifold_genetics/cli.py:17` (import), `src/manifold_genetics/cli.py:77-120` (`cmd_pca`)
- Test: `tests/unit/test_cli_main.py:188-224` (`test_cmd_pca_fit_project`)

**Interfaces:**
- Consumes: `run_pca(...)` from Task 1.
- Produces: nothing new. `cmd_pca` keeps returning `0` and printing the same lines.

Argument resolution (`args.fit_plink or args.input`, `args.project_output or args.output`, `--flashpca-output-dir` overriding `--model-dir`) stays in `cmd_pca` per the spec. **Do not touch the `pca` subparser** (constraint A).

- [ ] **Step 1: Update the CLI test to the new fake target**

The old test patched `mg_cli.PCA`; `PCA` is now instantiated inside `steps.pca`. Replace `test_cmd_pca_fit_project` in `tests/unit/test_cli_main.py` with:

```python
def test_cmd_pca_fit_project(monkeypatch, tmp_path):
    """--fit-plink + --project-plink fits one dataset and projects the other."""
    calls = []

    class FakePCA:
        def __init__(self, n_components, force=False):
            calls.append(("init", n_components, force))

        def fit(self, prefix, output_dir=None):
            calls.append(("fit", prefix))

        def project(self, prefix, output_path=None):
            calls.append(("project", prefix))
            df = pd.DataFrame({"sample_id": ["s1", "s2"], "dim_1": [0.1, 0.2]})
            if output_path:
                df.to_csv(output_path, index=False)
            return df

    monkeypatch.setattr("manifold_genetics.pipeline.steps.pca.PCA", FakePCA)
    out = tmp_path / "proj.csv"
    rc = mg_cli.main(
        [
            "pca",
            "--fit-plink",
            "fit",
            "--project-plink",
            "proj",
            "--project-output",
            str(out),
            "--n-pcs",
            "2",
        ]
    )
    assert rc == 0
    assert out.exists()
    assert ("fit", "fit") in calls
    assert ("project", "proj") in calls


def test_cmd_pca_single_input_uses_fit_transform(monkeypatch, tmp_path):
    """`pca --input X --output Y` fits and projects the same dataset."""
    calls = []

    class FakePCA:
        def __init__(self, n_components, force=False):
            calls.append(("init", n_components, force))

        def fit_transform(self, prefix, output_path=None):
            calls.append(("fit_transform", prefix))
            df = pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1]})
            if output_path:
                df.to_csv(output_path, index=False)
            return df

    monkeypatch.setattr("manifold_genetics.pipeline.steps.pca.PCA", FakePCA)
    out = tmp_path / "all.csv"
    rc = mg_cli.main(["pca", "--input", "all", "--output", str(out), "--n-pcs", "1"])
    assert rc == 0
    assert out.exists()
    assert ("fit_transform", "all") in calls


def test_cmd_pca_flashpca_output_dir_overrides_model_dir(monkeypatch, tmp_path):
    """--flashpca-output-dir wins over --model-dir (both name the same thing)."""
    seen = {}

    class FakePCA:
        def __init__(self, n_components, force=False):
            pass

        def fit(self, prefix, output_dir=None):
            seen["output_dir"] = output_dir

        def project(self, prefix, output_path=None):
            df = pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1]})
            if output_path:
                df.to_csv(output_path, index=False)
            return df

    monkeypatch.setattr("manifold_genetics.pipeline.steps.pca.PCA", FakePCA)
    rc = mg_cli.main(
        [
            "pca",
            "--fit-plink",
            "fit",
            "--project-plink",
            "proj",
            "--project-output",
            str(tmp_path / "p.csv"),
            "--model-dir",
            str(tmp_path / "ignored"),
            "--flashpca-output-dir",
            str(tmp_path / "flash"),
        ]
    )
    assert rc == 0
    assert seen["output_dir"] == tmp_path / "flash"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_cli_main.py -k pca -q
```

Expected: `test_cmd_pca_fit_project` FAILS — the real `PCA` is still constructed in `cli.py`, so `ToolResolver` looks for a flashpca binary (or `calls` stays empty).

- [ ] **Step 3: Rewrite `cmd_pca`**

Replace lines 77–120 of `src/manifold_genetics/cli.py` with:

```python
def cmd_pca(args):
    """Run PCA command."""
    setup_logging(args.verbose)

    # Resolve fit prefix and outputs (CLI-shape concerns stay here)
    fit_prefix = args.fit_plink or args.input
    if fit_prefix is None:
        raise ValueError("Please provide --input or --fit-plink for PCA fitting.")

    project_output = args.project_output or args.output
    if project_output is None:
        raise ValueError("Please provide --output or --project-output for PCA projection.")

    model_dir = Path(args.model_dir) if args.model_dir else None
    if args.flashpca_output_dir:
        model_dir = Path(args.flashpca_output_dir)

    pca_coords = run_pca(
        fit_prefix,
        args.project_plink,
        fit_output=args.fit_output,
        project_output=project_output,
        flashpca_dir=model_dir,
        n_pcs=args.n_pcs,
        force=args.force,
    )

    if args.project_plink:
        print(f"PCA fit on {fit_prefix} and projected {args.project_plink}")
        if args.fit_output:
            print(f"Fit PCA coords: {args.fit_output}")
        print(f"Projected PCA coords: {project_output}")
    else:
        print(f"PCA complete: {project_output}")

    # Report shape excluding sample_id column
    n_samples = pca_coords.shape[0]
    n_pcs = pca_coords.shape[1] - 1  # Exclude sample_id column
    print(f"Shape: ({n_samples}, {n_pcs}) [excluding sample_id column]")
    return 0
```

Then swap the import on line 17. `PCA` is used nowhere else in `cli.py` (verify with `grep -n 'PCA(' src/manifold_genetics/cli.py`), so replace

```python
from .pca import PCA
```

with

```python
from .pipeline.steps import run_pca
```

placed in isort order (`isort` will sort it — run the formatter and accept the result).

> Note: `from .pipeline...` at cli import time already happens via `from .pipeline import run_pipeline` on line 18, so this introduces no new import cost or cycle.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_cli_main.py -q
uv run pytest -m "not slow and not network" -q
```

Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run black src/ tests/ && uv run isort src/ tests/ && uv run flake8 src/ tests/
git add src/manifold_genetics/cli.py tests/unit/test_cli_main.py
git commit -m "refactor(cli): route cmd_pca through the shared run_pca seam

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 3: `steps/metrics.py` — the two metrics steps

**Files:**
- Create: `src/manifold_genetics/pipeline/steps/metrics.py`
- Create: `tests/unit/test_step_metrics.py`
- Modify: `src/manifold_genetics/pipeline/steps/__init__.py` (add the metrics half deferred in Task 1)

**Interfaces:**
- Consumes: `compute_geographic_preservation(embedding, geographic_coords, longitude_col, latitude_col, num_samples, ignore_missing) -> dict` and `compute_admixture_preservation(embedding, q_files: Dict[int, Path], k_value, num_samples, subsample) -> dict` from `manifold_genetics.metrics`; `validate_embedding_csv`, `validate_geographic_csv`, `validate_admixture_csv`, `validate_sample_id_overlap` from `manifold_genetics.utils.validation`; `metrics_output_paths(io) -> {"geographic": Path, "admixture": Path}` from `steps.paths`.
- Produces:
  - `MetricsStepResult(path: Path, values: dict = {}, skipped: bool = False)` — frozen dataclass.
  - `run_geographic_metrics_step(embedding_csv, geo_csv, out, *, longitude_col="longitude", latitude_col="latitude", num_samples=50000, ignore_missing=True) -> MetricsStepResult`
  - `run_admixture_metrics_step(embedding_csv, q_prefix, k_range, out, *, k_value=None, num_samples=50000, subsample=None) -> MetricsStepResult`

**Two behaviours to preserve exactly**, because the orchestrator currently reaches these through `cmd_metrics_*`:

1. **Validation runs inside the step.** Today the pipeline gets `validate_embedding_csv` etc. for free by shelling out to the CLI. If validation stayed in `cmd_*`, the pipeline would silently lose it.
2. **`values` is the parsed JSON, not the raw return.** `compute_admixture_preservation` returns int keys; the JSON on disk has string keys, and `run_pipeline()`'s `results["metrics"]["admixture"]` has string keys today. The step writes the file and reads it back, so the round-trip normalisation is preserved. This is what the spec's `values: dict  # parsed JSON` means.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_step_metrics.py`:

```python
"""Tests for the in-process metrics steps (pipeline/steps/metrics.py).

These steps replace `subprocess.run(["manifold-genetics", "metrics-*", ...])`.
The two behaviours that used to come free from going through the CLI — input
validation, and JSON round-tripping of the result dict — must survive the move,
so both are asserted here.
"""

import json
from pathlib import Path

import pytest

from manifold_genetics.pipeline.steps.metrics import (
    MetricsStepResult,
    run_admixture_metrics_step,
    run_geographic_metrics_step,
)

MODULE = "manifold_genetics.pipeline.steps.metrics"
_VALIDATORS = (
    "validate_embedding_csv",
    "validate_geographic_csv",
    "validate_admixture_csv",
    "validate_sample_id_overlap",
)


@pytest.fixture
def recorded_validators(monkeypatch):
    """Replace every validator with a recorder so the step runs on fake paths."""
    seen = {name: [] for name in _VALIDATORS}
    for name in _VALIDATORS:
        monkeypatch.setattr(
            f"{MODULE}.{name}",
            lambda *a, _n=name, **k: seen[_n].append((a, k)),
        )
    return seen


class TestGeographicMetricsStep:
    def test_writes_json_and_returns_parsed_values(self, tmp_path, monkeypatch, recorded_validators):
        monkeypatch.setattr(
            f"{MODULE}.compute_geographic_preservation",
            lambda **k: {"correlation": 0.9, "p_value": 1e-3, "n_pairs": 12},
        )
        out = tmp_path / "metrics" / "geographic.json"

        result = run_geographic_metrics_step("emb.csv", "geo.csv", out)

        assert isinstance(result, MetricsStepResult)
        assert result.path == out
        assert result.skipped is False
        assert out.exists(), "parent directory must be created for the caller"
        assert json.loads(out.read_text())["correlation"] == 0.9
        assert result.values == json.loads(out.read_text())

    def test_forwards_options_to_the_compute_function(self, tmp_path, monkeypatch, recorded_validators):
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_geographic_preservation",
            lambda **k: seen.update(k) or {"correlation": 0.1},
        )

        run_geographic_metrics_step(
            "emb.csv",
            "geo.csv",
            tmp_path / "g.json",
            longitude_col="lon",
            latitude_col="lat",
            num_samples=17,
            ignore_missing=False,
        )

        assert seen["longitude_col"] == "lon"
        assert seen["latitude_col"] == "lat"
        assert seen["num_samples"] == 17
        assert seen["ignore_missing"] is False

    def test_defaults_match_the_cli_defaults(self, tmp_path, monkeypatch, recorded_validators):
        """Drift here would silently change pipeline metrics vs. the CLI."""
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_geographic_preservation",
            lambda **k: seen.update(k) or {"correlation": 0.1},
        )

        run_geographic_metrics_step("emb.csv", "geo.csv", tmp_path / "g.json")

        assert seen["longitude_col"] == "longitude"
        assert seen["latitude_col"] == "latitude"
        assert seen["num_samples"] == 50000
        assert seen["ignore_missing"] is True

    def test_validates_inputs(self, tmp_path, monkeypatch, recorded_validators):
        monkeypatch.setattr(
            f"{MODULE}.compute_geographic_preservation", lambda **k: {"correlation": 0.1}
        )
        run_geographic_metrics_step("emb.csv", "geo.csv", tmp_path / "g.json")

        assert recorded_validators["validate_embedding_csv"], "embedding CSV must be validated"
        assert recorded_validators["validate_geographic_csv"], "geographic CSV must be validated"
        assert recorded_validators["validate_sample_id_overlap"], "sample overlap must be checked"


class TestAdmixtureMetricsStep:
    def test_builds_q_file_map_from_prefix_and_range(self, tmp_path, monkeypatch, recorded_validators):
        """<prefix>.K.csv is the documented Q-file layout (spec constraint B)."""
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation",
            lambda **k: seen.update(k) or {},
        )
        prefix = tmp_path / "admixture" / "project"

        run_admixture_metrics_step("emb.csv", prefix, range(2, 5), tmp_path / "a.json")

        assert seen["q_files"] == {
            2: Path(f"{prefix}.2.csv"),
            3: Path(f"{prefix}.3.csv"),
            4: Path(f"{prefix}.4.csv"),
        }

    def test_json_round_trip_stringifies_k_keys(self, tmp_path, monkeypatch, recorded_validators):
        """compute_* returns int keys; the JSON artifact and run_pipeline()'s
        results dict have always had string keys. Preserve that."""
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation",
            lambda **k: {2: {"correlation": 0.5}, 3: {"correlation": 0.6}},
        )
        out = tmp_path / "a.json"

        result = run_admixture_metrics_step("emb.csv", tmp_path / "q", range(2, 4), out)

        assert set(result.values) == {"2", "3"}
        assert result.values["2"]["correlation"] == 0.5
        assert json.loads(out.read_text()) == result.values

    def test_forwards_options_to_the_compute_function(self, tmp_path, monkeypatch, recorded_validators):
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation", lambda **k: seen.update(k) or {}
        )

        run_admixture_metrics_step(
            "emb.csv",
            tmp_path / "q",
            range(2, 4),
            tmp_path / "a.json",
            k_value=3,
            num_samples=11,
            subsample=100,
        )

        assert seen["k_value"] == 3
        assert seen["num_samples"] == 11
        assert seen["subsample"] == 100

    def test_defaults_match_the_cli_defaults(self, tmp_path, monkeypatch, recorded_validators):
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation", lambda **k: seen.update(k) or {}
        )

        run_admixture_metrics_step("emb.csv", tmp_path / "q", range(2, 4), tmp_path / "a.json")

        assert seen["k_value"] is None
        assert seen["num_samples"] == 50000
        assert seen["subsample"] is None

    def test_empty_k_range_raises(self, tmp_path, monkeypatch, recorded_validators):
        """An empty range would otherwise write an empty JSON and look successful."""
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation", lambda **k: {}
        )
        with pytest.raises(ValueError, match="[Nn]o admixture files"):
            run_admixture_metrics_step("emb.csv", tmp_path / "q", range(5, 5), tmp_path / "a.json")

    def test_validates_inputs_against_the_first_k(self, tmp_path, monkeypatch, recorded_validators):
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation", lambda **k: {}
        )
        prefix = tmp_path / "q"

        run_admixture_metrics_step("emb.csv", prefix, range(3, 6), tmp_path / "a.json")

        assert recorded_validators["validate_admixture_csv"]
        overlap_args = recorded_validators["validate_sample_id_overlap"][0][0]
        assert overlap_args[1] == f"{prefix}.3.csv", "overlap is checked against the lowest K"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_step_metrics.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'manifold_genetics.pipeline.steps.metrics'`.

- [ ] **Step 3: Write `src/manifold_genetics/pipeline/steps/metrics.py`**

```python
"""In-process metrics steps.

Both steps take plain paths rather than upstream step-result objects, so they
stay decoupled from whichever layer produced the embedding CSV and the Q files
(spec §"L2 — step layer"). ``cmd_metrics_*`` and the orchestrator both call
these, so validation lives here rather than in the CLI handlers — otherwise the
pipeline would lose the validation it used to get by shelling out.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from ...metrics import compute_admixture_preservation, compute_geographic_preservation
from ...utils.validation import (
    validate_admixture_csv,
    validate_embedding_csv,
    validate_geographic_csv,
    validate_sample_id_overlap,
)
from .paths import metrics_output_paths

logger = logging.getLogger(__name__)

__all__ = [
    "MetricsStepResult",
    "metrics_output_paths",
    "run_admixture_metrics_step",
    "run_geographic_metrics_step",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class MetricsStepResult:
    """Typed outputs of a metrics step. ``values`` is the parsed JSON artifact."""

    path: Path
    values: dict = field(default_factory=dict)
    skipped: bool = False


def _write_and_reload(values: dict, out: PathLike) -> MetricsStepResult:
    """Write ``values`` as JSON and return what the file actually contains.

    Reading back is deliberate: ``compute_admixture_preservation`` returns int K
    keys, while the JSON artifact — and therefore every existing consumer of
    ``run_pipeline()``'s metrics dict — has string keys.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(values, f, indent=2)
    with open(out) as f:
        return MetricsStepResult(path=out, values=json.load(f))


def run_geographic_metrics_step(
    embedding_csv: PathLike,
    geo_csv: PathLike,
    out: PathLike,
    *,
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
    num_samples: int = 50000,
    ignore_missing: bool = True,
) -> MetricsStepResult:
    """Correlate pairwise geographic distance with pairwise embedding distance."""
    validate_embedding_csv(embedding_csv)
    validate_geographic_csv(geo_csv, longitude_col, latitude_col)
    validate_sample_id_overlap(embedding_csv, geo_csv, "embedding", "geographic coordinates")

    logger.info("Computing geographic preservation...")
    values = compute_geographic_preservation(
        embedding=embedding_csv,
        geographic_coords=geo_csv,
        longitude_col=longitude_col,
        latitude_col=latitude_col,
        num_samples=num_samples,
        ignore_missing=ignore_missing,
    )
    return _write_and_reload(values, out)


def run_admixture_metrics_step(
    embedding_csv: PathLike,
    q_prefix: PathLike,
    k_range: Iterable[int],
    out: PathLike,
    *,
    k_value: Optional[int] = None,
    num_samples: int = 50000,
    subsample: Optional[int] = None,
) -> MetricsStepResult:
    """Correlate pairwise admixture distance with pairwise embedding distance.

    ``q_prefix`` names the Q files as ``<prefix>.{K}.csv`` (spec constraint B).
    """
    q_prefix = Path(q_prefix)
    k_values = list(k_range)
    if not k_values:
        raise ValueError(f"No admixture files found for K range: {k_range!r}")

    validate_embedding_csv(embedding_csv)
    validate_admixture_csv(str(q_prefix), k_values)
    first_q = f"{q_prefix}.{k_values[0]}.csv"
    validate_sample_id_overlap(embedding_csv, first_q, "embedding", "admixture")

    q_files = {k: Path(f"{q_prefix}.{k}.csv") for k in k_values}

    logger.info("Computing admixture preservation...")
    values = compute_admixture_preservation(
        embedding=embedding_csv,
        q_files=q_files,
        k_value=k_value,
        num_samples=num_samples,
        subsample=subsample,
    )
    return _write_and_reload(values, out)
```

- [ ] **Step 4: Complete `steps/__init__.py`**

Add the metrics import block and the four `__all__` entries that Task 1 deferred, so the file matches the full listing shown in Task 1 Step 4.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_step_metrics.py tests/unit/test_step_pca.py -q
```

Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run black src/ tests/ && uv run isort src/ tests/ && uv run flake8 src/ tests/
git add src/manifold_genetics/pipeline/steps/metrics.py \
        src/manifold_genetics/pipeline/steps/__init__.py \
        tests/unit/test_step_metrics.py
git commit -m "feat(pipeline): add in-process geographic and admixture metrics steps

Validation moves into the steps: the pipeline used to get it for free by
shelling out to the metrics CLI subcommands.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 4: Point `cmd_metrics_geographic` / `cmd_metrics_admixture` at the steps

**Files:**
- Modify: `src/manifold_genetics/cli.py:406-470`
- Test: `tests/unit/test_cli_main.py:345-390`

**Interfaces:**
- Consumes: `run_geographic_metrics_step`, `run_admixture_metrics_step` from Task 3.
- Produces: nothing new.

Keep `import json` in `cli.py` — it is still used by `cmd_plot_pca` (line ~517). Keep every `validate_*` import — all four are still used by other `cmd_*` handlers.

- [ ] **Step 1: Update the two CLI tests**

The `stub_validation` fixture patches validators on the `cli` module; validation now happens in `steps.metrics`. Replace `test_cmd_metrics_geographic_writes_json` and `test_cmd_metrics_admixture_writes_json` in `tests/unit/test_cli_main.py` with:

```python
_STEP_METRICS = "manifold_genetics.pipeline.steps.metrics"


@pytest.fixture
def stub_step_metrics_validation(monkeypatch):
    """Turn every validator used by the metrics steps into a no-op."""
    for name in (
        "validate_embedding_csv",
        "validate_geographic_csv",
        "validate_admixture_csv",
        "validate_sample_id_overlap",
    ):
        monkeypatch.setattr(f"{_STEP_METRICS}.{name}", lambda *a, **k: None)


def test_cmd_metrics_geographic_writes_json(monkeypatch, tmp_path, stub_step_metrics_validation):
    monkeypatch.setattr(
        f"{_STEP_METRICS}.compute_geographic_preservation",
        lambda **k: {"correlation": 0.9, "p_value": 1e-3},
    )
    out = tmp_path / "geo.json"
    rc = mg_cli.main(
        [
            "metrics-geographic",
            "--embedding",
            "emb.csv",
            "--geographic",
            "geo.csv",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text())["correlation"] == 0.9


def test_cmd_metrics_admixture_writes_json(monkeypatch, tmp_path, stub_step_metrics_validation):
    monkeypatch.setattr(
        f"{_STEP_METRICS}.compute_admixture_preservation",
        lambda **k: {"2": {"correlation": 0.5}},
    )
    out = tmp_path / "adm.json"
    rc = mg_cli.main(
        [
            "metrics-admixture",
            "--embedding",
            "emb.csv",
            "--admixture-output",
            str(tmp_path / "q"),
            "--k-min",
            "2",
            "--k-max",
            "3",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text())["2"]["correlation"] == 0.5


def test_cmd_metrics_admixture_forwards_subsample(
    monkeypatch, tmp_path, stub_step_metrics_validation
):
    """--subsample must reach compute_admixture_preservation; dropping it would
    silently run full pairwise distances on a large cohort."""
    seen = {}
    monkeypatch.setattr(
        f"{_STEP_METRICS}.compute_admixture_preservation",
        lambda **k: seen.update(k) or {"2": {"correlation": 0.5}},
    )
    rc = mg_cli.main(
        [
            "metrics-admixture",
            "--embedding",
            "emb.csv",
            "--admixture-output",
            str(tmp_path / "q"),
            "--k-min",
            "2",
            "--k-max",
            "2",
            "--output",
            str(tmp_path / "a.json"),
            "--subsample",
            "500",
        ]
    )
    assert rc == 0
    assert seen["subsample"] == 500
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_cli_main.py -k metrics -q
```

Expected: FAIL — the real validators still run inside `cmd_metrics_*` on non-existent `emb.csv` / `geo.csv`.

- [ ] **Step 3: Rewrite the two handlers**

Replace `cmd_metrics_geographic` and `cmd_metrics_admixture` in `src/manifold_genetics/cli.py` with:

```python
def cmd_metrics_geographic(args):
    """Compute geographic preservation metrics."""
    setup_logging(args.verbose)

    result = run_geographic_metrics_step(
        args.embedding,
        args.geographic,
        Path(args.output),
        longitude_col=args.longitude_col,
        latitude_col=args.latitude_col,
        num_samples=args.num_dists_sampled,
        ignore_missing=not args.keep_missing,
    )

    print(f"Geographic metrics saved to: {result.path}")
    return 0


def cmd_metrics_admixture(args):
    """Compute admixture preservation metrics."""
    setup_logging(args.verbose)

    result = run_admixture_metrics_step(
        args.embedding,
        Path(args.admixture_output),
        range(args.k_min, args.k_max + 1),
        Path(args.output),
        k_value=args.k_value,
        num_samples=args.num_dists_sampled,
        subsample=args.subsample,
    )

    print(f"Admixture metrics saved to: {result.path}")
    return 0
```

Extend the `pipeline.steps` import added in Task 2:

```python
from .pipeline.steps import run_admixture_metrics_step, run_geographic_metrics_step, run_pca
```

`compute_admixture_preservation` / `compute_geographic_preservation` are no longer referenced in `cli.py` — verify with `grep -n 'compute_.*_preservation' src/manifold_genetics/cli.py` and, if there are no remaining uses, delete line 16 (`from .metrics import ...`) so flake8 stays clean.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_cli_main.py -q
uv run pytest -m "not slow and not network" -q
```

Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run black src/ tests/ && uv run isort src/ tests/ && uv run flake8 src/ tests/
git add src/manifold_genetics/cli.py tests/unit/test_cli_main.py
git commit -m "refactor(cli): route the metrics subcommands through the step layer

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 5: Orchestrator calls `run_pca_step` instead of shelling out

**Files:**
- Modify: `src/manifold_genetics/pipeline/orchestrator.py` — imports, new `_io_config()`, the PCA block and the PCA-viz path lookup
- Test: `tests/unit/test_orchestrator.py` — delete `TestPipelineRunPCAForce`, rework `TestPipelineRunFullFlow`

**Interfaces:**
- Consumes: `run_pca_step(io, pca) -> PCAStepResult`, `pca_output_paths(io, pca)`, `IOConfig`, `PCAConfig`.
- Produces: `Pipeline._io_config() -> IOConfig` — a temporary adapter. `Pipeline` still stores loose attributes; PR 5 replaces this with `build_configs()` wiring.

The `results` dict keys must be byte-for-byte what they were: `fit_pca_file`, `project_pca_file`, `pca_file`, `pca_coords`. The one visible change is that `pca_coords` is now the DataFrame `PCA.project()` returned rather than a `pd.read_csv()` of the file it just wrote — same rows, same columns, no re-read of a potentially large file.

- [ ] **Step 1: Delete the argv-assertion tests and add the new orchestrator tests**

In `tests/unit/test_orchestrator.py`:

1. Delete the entire `class TestPipelineRunPCAForce` block (lines ~157–255, including its section comment banner). The force logic it covered now lives in `tests/unit/test_step_pca.py::TestRunPcaStep`.

2. Add this helper next to `capture_subprocess`:

```python
def stub_pca_step(monkeypatch, n_pcs):
    """Replace the in-process PCA step with a fake that writes both CSVs.

    Returns the list of (io, pca_config) it was called with, so tests can assert
    what the orchestrator handed the step.
    """
    import manifold_genetics.pipeline.orchestrator as orch
    from manifold_genetics.pipeline.steps.pca import PCAStepResult

    calls = []

    def fake_run_pca_step(io, pca):
        calls.append((io, pca))
        paths = orch.pca_output_paths(io, pca)
        write_pca_csv(paths["fit_pca"], pca.n_pcs)
        write_pca_csv(paths["project_pca"], pca.n_pcs)
        return PCAStepResult(
            fit_pca=paths["fit_pca"],
            project_pca=paths["project_pca"],
            coords_df=pd.read_csv(paths["project_pca"]),
        )

    monkeypatch.setattr(orch, "run_pca_step", fake_run_pca_step)
    return calls
```

3. In `class TestPipelineRunFullFlow`, drop the `"pca" in cmd` branch from `_writer` (the PCA stage no longer goes through subprocess), leaving:

```python
    def _writer(self, pipeline, n_pcs, method="phate"):
        emb_dir = pipeline.output_dir / "embeddings"
        metrics_dir = pipeline.output_dir / "metrics"

        def on_call(cmd):
            if "embed" in cmd:
                write_pca_csv(emb_dir / f"{method}_2d.csv", 2)
                write_pca_csv(emb_dir / f"{method}_fit_2d.csv", 2)
            elif "metrics-geographic" in cmd:
                metrics_dir.mkdir(parents=True, exist_ok=True)
                (metrics_dir / "geographic.json").write_text('{"correlation": 0.9}')
            elif "metrics-admixture" in cmd:
                metrics_dir.mkdir(parents=True, exist_ok=True)
                (metrics_dir / "admixture.json").write_text('{"2": {"correlation": 0.5}}')

        return on_call
```

4. Add `pca_step_calls = stub_pca_step(monkeypatch, n_pcs)` immediately before each `pipeline.run(...)` in `test_full_run_with_backend_and_metrics`, `test_full_run_via_cli_admixture`, and `test_phate_landmark_and_batch_flags_forwarded`.

5. Add a new class at the end of the file:

```python
# ---------------------------------------------------------------------------
# TestPipelineRunPCAInProcess — PR 2 cutover: no subprocess for PCA
# ---------------------------------------------------------------------------


class TestPipelineRunPCAInProcess:
    """run() calls run_pca_step() directly. Re-introducing a subprocess hop
    would restore the API → CLI → API inversion this refactor removed, along
    with its cold-start cost and buried tracebacks."""

    def test_pca_no_longer_shells_out(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        stub_pca_step(monkeypatch, 3)
        calls = capture_subprocess(monkeypatch)

        pipeline.run(
            n_pcs=3,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        assert pca_calls(calls) == [], f"PCA must run in-process, got: {calls}"

    def test_step_receives_pipeline_io_and_n_pcs(self, tmp_path, monkeypatch):
        """The step's IOConfig must carry the pipeline's own prefixes; passing
        the wrong cohort would fit PCA on the projection set."""
        pipeline = make_pipeline(
            tmp_path, fit_plink_prefix="fitset", project_plink_prefix="projectset"
        )
        step_calls = stub_pca_step(monkeypatch, 7)
        capture_subprocess(monkeypatch)

        pipeline.run(
            n_pcs=7,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        assert len(step_calls) == 1
        io, pca_cfg = step_calls[0]
        assert io.fit_plink == Path("fitset")
        assert io.project_plink == Path("projectset")
        assert io.output_dir == pipeline.output_dir
        assert pca_cfg.n_pcs == 7

    def test_results_expose_pca_paths_and_coords(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        stub_pca_step(monkeypatch, 4)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=4,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        pca_dir = pipeline.output_dir / "pca"
        assert results["fit_pca_file"] == pca_dir / "fit_pca_4.csv"
        assert results["project_pca_file"] == pca_dir / "project_pca_4.csv"
        assert results["pca_file"] == results["project_pca_file"]
        assert isinstance(results["pca_coords"], pd.DataFrame)

    def test_skip_pca_still_resolves_existing_outputs(self, tmp_path, monkeypatch):
        """Spec constraint C: a skipped step still supplies its paths downstream."""
        pipeline = make_pipeline(tmp_path)
        pca_dir = pipeline.output_dir / "pca"
        write_pca_csv(pca_dir / "fit_pca_4.csv", 4)
        write_pca_csv(pca_dir / "project_pca_4.csv", 4)

        step_calls = stub_pca_step(monkeypatch, 4)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=4,
            skip_pca=True,
            skip_admixture=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        assert step_calls == [], "skip_pca must not run the step"
        assert results["fit_pca_file"] == pca_dir / "fit_pca_4.csv"
        assert results["project_pca_file"] == pca_dir / "project_pca_4.csv"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_orchestrator.py -q
```

Expected: FAIL — `AttributeError: module 'manifold_genetics.pipeline.orchestrator' has no attribute 'run_pca_step'`.

- [ ] **Step 3: Wire the orchestrator**

In `src/manifold_genetics/pipeline/orchestrator.py`, add to the imports (isort will place them):

```python
from .config import IOConfig, PCAConfig
from .steps.paths import pca_output_paths
from .steps.pca import run_pca_step
```

Add this method to `Pipeline`, directly above `run()`:

```python
    def _io_config(self) -> IOConfig:
        """Adapt the pipeline's loose attributes to the step layer's IOConfig.

        Transitional: PR 5 wires ``build_configs()`` into ``__init__`` and drops
        this. ``__init__`` has already guaranteed every field is set.
        """
        return IOConfig(
            fit_plink=self.fit_plink_prefix,
            project_plink=self.project_plink_prefix,
            output_dir=self.output_dir,
            fit_labels=self.fit_labels,
            project_labels=self.project_labels,
            fit_colormap=self.fit_colormap,
            project_colormap=self.project_colormap,
            geographic_coords=self.geographic_coords,
        )
```

Replace the whole PCA block (from `results = {}` through the end of the `else:` skip branch) with:

```python
        results = {}

        io = self._io_config()
        pca_cfg = PCAConfig(n_pcs=n_pcs)
        pca_paths = pca_output_paths(io, pca_cfg)

        # Step 1: PCA
        if not skip_pca:
            logger.info("=" * 70)
            logger.info("STEP 1: PCA")
            logger.info("=" * 70)

            pca_result = run_pca_step(io, pca_cfg)

            results["fit_pca_file"] = pca_result.fit_pca
            results["project_pca_file"] = pca_result.project_pca
            results["pca_file"] = pca_result.project_pca
            results["pca_coords"] = pca_result.coords_df

        else:
            # PCA skipped — resolve expected paths so embedding can still run
            if pca_paths["fit_pca"].exists():
                results["fit_pca_file"] = pca_paths["fit_pca"]
            if pca_paths["project_pca"].exists():
                results["project_pca_file"] = pca_paths["project_pca"]
                results["pca_file"] = pca_paths["project_pca"]
```

In the "Step 1.5: PCA VISUALIZATION" block, replace the two re-derived locals

```python
            pca_dir = self.output_dir / "pca"
            pca_file = pca_dir / f"project_pca_{n_pcs}.csv"
```

with

```python
            pca_file = pca_paths["project_pca"]
```

and change the figures directory line to keep using `self.output_dir` (it already does: `pca_figures_dir = self.output_dir / "figures" / "pca"` — leave it).

Leave `import subprocess` in place: the admixture and embedding stages still use it.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_orchestrator.py -q
uv run pytest -m "not slow and not network" -q
```

Expected: PASS. If `test_full_run_*` fails on a missing `pca_dir` local, confirm you removed every reference to the deleted `pca_dir` variable inside the viz block.

- [ ] **Step 5: Lint and commit**

```bash
uv run black src/ tests/ && uv run isort src/ tests/ && uv run flake8 src/ tests/
git add src/manifold_genetics/pipeline/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "refactor(pipeline): run PCA in-process instead of via the CLI

Pipeline.run() now calls run_pca_step() directly. Deletes the argv-assertion
tests for --force in favour of behaviour tests on the step itself.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 6: Orchestrator calls the metrics steps instead of shelling out

**Files:**
- Modify: `src/manifold_genetics/pipeline/orchestrator.py` — imports, the "Step 5: Metrics" block (~lines 580–630)
- Test: `tests/unit/test_orchestrator.py` — `TestPipelineRunFullFlow`

**Interfaces:**
- Consumes: `run_geographic_metrics_step`, `run_admixture_metrics_step`, `metrics_output_paths` from Task 3.
- Produces: nothing new. `results["metrics"]` keeps the same shape: `{"geographic": {...}, "admixture": {"2": {...}, ...}}` with string K keys.

- [ ] **Step 1: Update `TestPipelineRunFullFlow` to exercise the real step code**

Rather than stubbing the step functions, stub the *compute* functions and validators inside `steps.metrics`, so the orchestrator → step → JSON wiring is actually tested.

In `tests/unit/test_orchestrator.py`, add next to `stub_pca_step`:

```python
def stub_metrics_steps(monkeypatch, geo=None, admix=None):
    """Stub the metric computations and validators inside steps.metrics so the
    real step code (JSON write + reload) runs against fake inputs."""
    import manifold_genetics.pipeline.steps.metrics as m

    for name in (
        "validate_embedding_csv",
        "validate_geographic_csv",
        "validate_admixture_csv",
        "validate_sample_id_overlap",
    ):
        monkeypatch.setattr(m, name, lambda *a, **k: None)

    monkeypatch.setattr(
        m, "compute_geographic_preservation", lambda **k: geo or {"correlation": 0.9}
    )
    # int keys on purpose: the JSON round-trip inside the step is what turns
    # them into the string keys run_pipeline() has always returned.
    monkeypatch.setattr(
        m, "compute_admixture_preservation", lambda **k: admix or {2: {"correlation": 0.5}}
    )
```

Drop the two `metrics-*` branches from `TestPipelineRunFullFlow._writer` (the metrics stages no longer go through subprocess), leaving only the `"embed"` branch.

Rewrite `test_full_run_with_backend_and_metrics` to call `stub_metrics_steps(monkeypatch)` before `pipeline.run(...)` and to add these assertions after the existing ones:

```python
        # Metrics ran in-process and landed at the documented paths (constraint B)
        assert (pipeline.output_dir / "metrics" / "geographic.json").exists()
        assert (pipeline.output_dir / "metrics" / "admixture.json").exists()
        assert [c for c in calls if "metrics-geographic" in c] == []
        assert [c for c in calls if "metrics-admixture" in c] == []
```

The two existing assertions

```python
        assert results["metrics"]["geographic"]["correlation"] == 0.9
        assert results["metrics"]["admixture"]["2"]["correlation"] == 0.5
```

stay unchanged — they now prove the int→string K-key round trip is preserved.

Add one more test to `TestPipelineRunFullFlow`:

```python
    def test_metrics_skipped_when_no_geographic_coords(self, tmp_path, monkeypatch):
        """Without geographic_coords the geographic metric must not run, but the
        admixture metric still must."""
        self._stub_plots(monkeypatch)
        stub_metrics_steps(monkeypatch)
        pipeline = make_pipeline(tmp_path, admixture_backend=_FakeBackend())
        n_pcs = 3
        stub_pca_step(monkeypatch, n_pcs)
        capture_subprocess(monkeypatch, on_call=self._writer(pipeline, n_pcs))

        results = pipeline.run(
            n_pcs=n_pcs,
            k_min=2,
            k_max=3,
            embedding="phate",
            embedding_params={"knn": 5},
        )

        assert "geographic" not in results["metrics"]
        assert not (pipeline.output_dir / "metrics" / "geographic.json").exists()
        assert results["metrics"]["admixture"]["2"]["correlation"] == 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_orchestrator.py -k FullFlow -q
```

Expected: FAIL — the orchestrator still shells out, so `metrics/geographic.json` is written by the fake subprocess (which no longer writes it), and `results["metrics"]` is empty or a `FileNotFoundError` surfaces.

- [ ] **Step 3: Wire the orchestrator's metrics block**

Add to the imports:

```python
from .steps.metrics import run_admixture_metrics_step, run_geographic_metrics_step
from .steps.paths import metrics_output_paths, pca_output_paths
```

(merge with the `steps.paths` import added in Task 5).

Replace the body of the "Step 5: Metrics" block with:

```python
        # Step 5: Metrics
        if not skip_metrics and not skip_embedding:
            logger.info("=" * 70)
            logger.info("STEP 5: METRICS")
            logger.info("=" * 70)

            metrics = {}

            metrics_dir = self.output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_paths = metrics_output_paths(io)

            # Geographic preservation
            if self.geographic_coords:
                geo_result = run_geographic_metrics_step(
                    embedding_file,
                    self.geographic_coords,
                    metrics_paths["geographic"],
                )
                metrics["geographic"] = geo_result.values

            # Admixture preservation
            if not skip_admixture and "admixture_dir" in results:
                admix_result = run_admixture_metrics_step(
                    embedding_file,
                    results["admixture_dir"] / "project",
                    range(k_min, k_max + 1),
                    metrics_paths["admixture"],
                )
                metrics["admixture"] = admix_result.values

            results["metrics"] = metrics
```

Keep the unconditional `metrics_dir.mkdir(...)`: today the directory is created even when neither metric runs, and the output tree is a contract.

Then delete `import json` from the top of `orchestrator.py` — it has no remaining uses. Verify: `grep -n 'json\.' src/manifold_genetics/pipeline/orchestrator.py` must print nothing.

- [ ] **Step 4: Run the full suite**

```bash
uv run pytest -m "not slow and not network" -q
uv run pytest tests/integration/test_generic_pipeline.py -q
```

Expected: PASS. The `test_generic_pipeline.py` cases that need a real flashpca binary behave exactly as they did before this PR (if they were failing/erroring for lack of the binary on this machine, they still do — compare against `git stash` output rather than assuming).

- [ ] **Step 5: Lint and commit**

```bash
uv run black src/ tests/ && uv run isort src/ tests/ && uv run flake8 src/ tests/
git add src/manifold_genetics/pipeline/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "refactor(pipeline): compute metrics in-process instead of via the CLI

Removes the last two subprocess hops to our own CLI outside of admixture and
embedding, and with them the JSON re-read the orchestrator did by hand.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 7: Verify the PR against the spec's constraints and open it

**Files:** none modified (unless a check fails).

- [ ] **Step 1: Confirm the CLI surface is untouched (constraint A)**

```bash
git diff origin/main -- src/manifold_genetics/cli.py | grep -E '^[-+].*add_(parser|argument)'
```

Expected: **no output**. Any hit is a constraint-A violation — revert it.

- [ ] **Step 2: Confirm the remaining subprocess calls are the expected three**

```bash
grep -n "subprocess.run" src/manifold_genetics/pipeline/orchestrator.py
```

Expected: exactly two remaining call sites — the `admixture` and `embed` CLI commands (PRs 3 and 4). No `pca`, `metrics-geographic`, or `metrics-admixture` argv lists anywhere in the file:

```bash
grep -n '"metrics-geographic"\|"metrics-admixture"\|"pca",' src/manifold_genetics/pipeline/orchestrator.py
```

Expected: no output.

- [ ] **Step 3: Confirm no path literals were reintroduced (constraint B)**

```bash
grep -rn 'fit_pca_\|project_pca_\|geographic.json\|admixture.json' src/manifold_genetics/ --include=*.py | grep -v steps/paths.py
```

Expected: no output outside `steps/paths.py` (docstrings and the `pipeline` subparser's epilog description are fine — inspect any hits rather than blindly deleting).

- [ ] **Step 4: Full check**

```bash
uv run black --check src/ tests/ && uv run isort --check src/ tests/ && uv run flake8 src/ tests/
uv run pytest -m "not slow and not network" -q
```

Expected: all green.

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin HEAD
~/bin/gh pr create --base main --title "feat(pipeline): run PCA and metrics in-process (step layer PR 2)" --body "$(cat <<'EOF'
PR 2 of the in-process pipeline step layer (`docs/superpowers/specs/2026-08-30-pipeline-step-layer-design.md`, migration table row 2). Follows #68.

## What changed

- **New `pipeline/steps/pca.py`** — `run_pca()` is the seam shared by `cmd_pca` and `run_pca_step()`, so the CLI and the orchestrator can no longer drift on which cohort is fitted vs. projected. `run_pca_step(io, pca) -> PCAStepResult` adds path resolution and takes over the stale-component-count guard that used to live inline in `Pipeline.run()`.
- **New `pipeline/steps/metrics.py`** — `run_geographic_metrics_step()` / `run_admixture_metrics_step()`, taking plain paths so they stay decoupled from whichever layer produced the embedding CSV and the Q files. Input validation moved into the steps: the pipeline used to get it for free by shelling out to the CLI.
- **`Pipeline.run()`** calls all three directly. Three `subprocess.run(["manifold-genetics", ...])` hops are gone; `admixture` and `embed` still shell out (PRs 3 and 4).
- **Tests**: argv-assertion tests for `--force` deleted, replaced by behaviour tests on `run_pca_step`. New `tests/unit/test_step_pca.py` and `tests/unit/test_step_metrics.py`.

## Deviation from the spec

The spec sketched each step module owning its own `*_output_paths`. PR 1 put them all in `steps/paths.py`; this PR keeps that and imports from it, so path-layout knowledge still lives in exactly one function per step.

## Behaviour notes

- `results["pca_coords"]` is now the DataFrame `PCA.project()` returned rather than a re-read of the CSV it just wrote — same rows and columns, one less full-file read.
- `results["metrics"]` keeps string K keys: the metrics steps write the JSON and read it back, preserving the round-trip normalisation the orchestrator used to do by hand.
- The `manifold-genetics` CLI surface is unchanged (constraint A) and every output path is unchanged (constraint B).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF
EOF
)"
```

---

## Self-Review

**Spec coverage (migration table row 2 — "PCA step + metrics steps"):**

| Spec requirement | Task |
|---|---|
| `steps/pca.py` with `run_pca_step` | 1 |
| `steps/metrics.py` with both metrics steps taking plain `Path` args | 3 |
| Replace those `subprocess.run` calls in `run()` | 5, 6 |
| Point `cmd_pca` / `cmd_metrics_*` at the step functions | 2, 4 |
| Rewrite their tests | 1, 2, 3, 4, 5, 6 |
| `PCAStepResult` / `MetricsStepResult` shapes | 1, 3 |
| `*_output_paths` is the only place path layout lives | 1, 3, 7 (verified) |
| Dim-count guard moves *into* `run_pca_step` | 1 |
| Constraint A (frozen CLI surface) | 7 Step 1 |
| Constraint B (output paths) | 1 Step 1, 6 Step 1, 7 Step 3 |
| Constraint C (skipped step still supplies paths) | 5 Step 1 |
| Constraint D (compute steps stay fatal) | no try/except added anywhere |

**Deferred to later PRs, by design:** `PipelineResult` (PR 5), `build_configs()` wiring into `Pipeline.__init__` (PR 5), `_skipped_pca` raising a clear error (PR 5), the admixture and embedding steps (PRs 3–4), `test_pipeline_output_layout` parametrized over the three modes (PR 5).

**Type consistency check:** `PCAStepResult(fit_pca, project_pca, coords_df, skipped)` and `MetricsStepResult(path, values, skipped)` are used with exactly those field names in Tasks 1, 3, 5, 6. `pca_output_paths` keys `"fit_pca"` / `"project_pca"` / `"flashpca_dir"` and `metrics_output_paths` keys `"geographic"` / `"admixture"` match `steps/paths.py` as it exists on `main`. `run_pca`'s keyword-only `project_output` is required in every call site shown.
