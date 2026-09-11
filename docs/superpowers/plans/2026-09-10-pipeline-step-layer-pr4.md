# Pipeline Step Layer — PR 4: admixture step

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the admixture stage out of `subprocess.run(["manifold-genetics", "admixture", ...])` into an in-process step function, and **delete the special-case backend branch** so one code path serves the real backend and injected test backends alike.

**Architecture:** Add `src/manifold_genetics/pipeline/steps/admixture.py` exposing `run_admixture(...)` — the seam shared by `cmd_admixture` and the step — plus `run_admixture_step(io, admix, *, backend=None) -> AdmixtureStepResult`. The unification point already exists: `NeuralAdmixture(..., backend=None)` constructs a real `NeuralAdmixtureBackend`, and `NeuralAdmixture` is otherwise a thin delegator. So passing the caller's optional backend straight into `NeuralAdmixture` collapses today's two branches into one.

**Tech Stack:** Python 3.10–3.12, pandas, pytest, `uv`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-30-pipeline-step-layer-design.md` (migration table row 4). PR 1 = #68, PR 2 = #70, PR 3 = #74. Base this branch on `main` **after #74 merges**.

## Global Constraints

- **A — The CLI surface is frozen.** No flag renames, removals, or additions in `src/manifold_genetics/cli.py`; nothing below `main()` changes.
- **B — Output paths are a contract.** The stage writes `output_dir/admixture/fit.{k}.csv`, `output_dir/admixture/project.{k}.csv`, and `output_dir/admixture/checkpoints/`. These come from `admixture_output_paths()` in `steps/paths.py` and nowhere else.
- **D — Compute steps are fatal.** No try/except around the admixture work.
- **E — `threads` / `num_gpus` / `batch_size` must continue to thread through** `cmd_pipeline` → `run_pipeline` → the admixture step → the backend. `detect_cluster.sh` feeds `$CLUSTER_CPUS` / `$CLUSTER_GPUS` here, so dropping one silently changes cluster behaviour.
- **F — `project` naming everywhere; no dead params.**
- **Behaviour-preserving for the real path.** `run_pipeline()` still returns the same loose `results` dict: `admixture_dir`, `admixture_checkpoints_dir`, `fit_q_files`, `project_q_files`, `q_files`. `PipelineResult` is PR 5.
- **Out of scope, do not touch:** the admixture *visualization* block (the bar plot) and the other three viz blocks — PR 5 owns them; `_get_embedding_model` — PR 5 deletes it; anything under `src/manifold_genetics/admixture/`.
- **Style.** `black` (line length 100) and `isort` (profile black) are the binding gates. `flake8` is NOT a working gate — `setup.cfg` shadows `.flake8`, ~573 pre-existing violations, mostly E501 from black's own 100-column formatting (issue #71). Ignore E501; never edit `.flake8` or `setup.cfg`.
- **Tests.** Baseline on `main` after #74: **487 passed, 7 skipped, 7 deselected**, and `tests/integration/test_generic_pipeline.py` **6 passed**. Run on a compute node if an allocation exists (`srun --jobid=<N> --overlap --ntasks=1 --time=40:00 bash -c '... && source .venv/bin/activate && pytest ...'`); otherwise on the login node, **one pytest process at a time, in the foreground** — concurrent runs exhaust the node's thread limit and produce spurious `RuntimeError: can't start new thread` failures.

## The central decision: which semantics survive the unification

Today the orchestrator has two branches that are **not equivalent**:

| | fit-set Q files | project-set Q files |
|---|---|---|
| CLI path (`cmd_admixture`, what real runs execute) | `fit()` then `transform(fit_prefix)` | `transform(project_prefix)` |
| Backend branch (test-only injection) | `fit()` then **`fit_transform(fit_prefix)`** | `transform(project_prefix)` |

**Unify on the CLI semantics** — `fit()` then `transform()` twice. Rationale:

- It is what every real (non-test) run does today, so the real path is byte-for-byte preserved.
- `fit_transform` on a `NeuralAdmixtureBackend` re-runs training rather than reusing the models `fit()` just wrote; using it for the fit cohort was only ever harmless because no real run took that branch.
- For `PrecomputedAdmixtureBackend` and `FakeAdmixtureBackend`, `transform` and `fit_transform` both just materialise fixtures, so injected-backend *outputs* are unchanged.

The only visible change is the **sequence of method names an injected backend observes**: `["fit", "fit_transform", "transform"]` becomes `["fit", "transform", "transform"]`. That is a test-visible change only, and it is the point of the task — the spec calls this branch "two code paths for one step".

`AdmixtureBackend.fit_transform` stays on the ABC and stays implemented by all three backends; nothing in this PR removes it. It simply is no longer called by the pipeline.

## A latent bug this removes for free

The orchestrator's argv builder gates on truthiness for threads:

```python
if admix_threads:                    # 0 is falsy -> silently dropped
if admix_gpus is not None:           # correct
if admix_batch_size is not None:     # correct
```

Passing the values straight into `NeuralAdmixture(threads=..., num_gpus=..., batch_size=...)` removes the whole class of falsy-drop bugs. `num_gpus=0` (explicit CPU-only) already worked and must keep working; `threads=0` now also reaches the backend instead of being dropped. Pin both with tests.

## File Structure

**Create:**
- `src/manifold_genetics/pipeline/steps/admixture.py` — `AdmixtureStepResult`, `run_admixture()`, `run_admixture_step()`.
- `tests/unit/test_step_admixture.py`

**Modify:**
- `src/manifold_genetics/pipeline/steps/__init__.py` — re-export the new names.
- `src/manifold_genetics/cli.py` — `cmd_admixture` routed through the seam. **No `main()` changes.**
- `src/manifold_genetics/pipeline/orchestrator.py` — replace the two-branch admixture stage with one step call; delete `import subprocess` and the last argv builder.
- `tests/unit/test_orchestrator.py` — replace `TestPipelineRunAdmixtureFlags` (6 argv tests) with step-behaviour tests; update `_FakeBackend` expectations.
- `tests/unit/test_cli_main.py` — repoint `test_cmd_admixture_fit_project`'s monkeypatch.

---

### Task 1: `steps/admixture.py` — the seam and the step

**Files:**
- Create: `src/manifold_genetics/pipeline/steps/admixture.py`
- Create: `tests/unit/test_step_admixture.py`
- Modify: `src/manifold_genetics/pipeline/steps/__init__.py`

**Interfaces:**
- Consumes: `IOConfig`, `AdmixtureConfig` (fields `.k_min`, `.k_max`, `.threads`, `.num_gpus`, `.batch_size`) from `manifold_genetics.pipeline.config`; `admixture_output_paths(io, admix)` from `steps.paths`, whose keys are `"dir"`, `"checkpoints_dir"`, `"fit_prefix"`, `"project_prefix"`, `"fit_q_files"`, `"project_q_files"`; `NeuralAdmixture` and `AdmixtureBackend` from `manifold_genetics.admixture`.
- Produces:
  - `AdmixtureStepResult` — frozen dataclass with `q_prefix: Path` (the project prefix, the spec's field name), `k_values: tuple[int, ...]`, `dir: Path`, `checkpoints_dir: Path`, `fit_prefix: Path`, `fit_q_files: dict`, `project_q_files: dict`, `skipped: bool = False`.
  - `run_admixture(fit_plink, project_plink, *, checkpoints_dir, fit_output, project_output, k_min=2, k_max=10, force=False, threads=None, num_gpus=None, batch_size=None, model_name="fit", backend=None) -> tuple[dict, dict]` returning `(fit_q_files, project_q_files)`.
  - `run_admixture_step(io: IOConfig, admix: AdmixtureConfig, *, backend: AdmixtureBackend | None = None) -> AdmixtureStepResult`

Pure addition — nothing is wired up in this task.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_step_admixture.py`:

```python
"""Tests for the in-process admixture step (pipeline/steps/admixture.py).

This step replaces BOTH a `subprocess.run(["manifold-genetics", "admixture", ...])`
call and a special-case branch that called an injected backend directly. The two
were not equivalent — the backend branch used fit_transform() for the fit cohort
where the CLI path used transform(). These tests pin the unified sequence:
fit() once, then transform() on each cohort.
"""

from pathlib import Path

import pytest

from manifold_genetics.pipeline.config import AdmixtureConfig, IOConfig
from manifold_genetics.pipeline.steps.admixture import (
    AdmixtureStepResult,
    run_admixture,
    run_admixture_step,
)
from manifold_genetics.pipeline.steps.paths import admixture_output_paths

MODULE = "manifold_genetics.pipeline.steps.admixture"


class FakeBackend:
    """Records the call sequence an injected backend observes."""

    def __init__(self):
        self.calls = []

    def fit(self, plink_prefix, output_dir, model_name="fit"):
        self.calls.append(("fit", str(plink_prefix), str(output_dir), model_name))

    def transform(self, plink_prefix, output_prefix):
        self.calls.append(("transform", str(plink_prefix), str(output_prefix)))
        return {2: Path(f"{output_prefix}.2.csv")}

    def fit_transform(self, plink_prefix, output_prefix):
        self.calls.append(("fit_transform", str(plink_prefix), str(output_prefix)))
        return {2: Path(f"{output_prefix}.2.csv")}


class FakeNeuralAdmixture:
    """Stand-in for the NeuralAdmixture wrapper. Records construction kwargs."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeNeuralAdmixture.instances.append(self)

    def fit(self, plink_prefix, output_dir=None, model_name="fit"):
        self.calls.append(("fit", str(plink_prefix), str(output_dir), model_name))

    def transform(self, plink_prefix, output_prefix=None):
        self.calls.append(("transform", str(plink_prefix), str(output_prefix)))
        return {2: Path(f"{output_prefix}.2.csv")}

    def fit_transform(self, plink_prefix, output_prefix=None):
        raise AssertionError("the admixture step must not call fit_transform")


@pytest.fixture
def fake_admixture(monkeypatch):
    FakeNeuralAdmixture.instances = []
    monkeypatch.setattr(f"{MODULE}.NeuralAdmixture", FakeNeuralAdmixture)
    return FakeNeuralAdmixture


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
# run_admixture — the seam shared with `manifold-genetics admixture`
# ---------------------------------------------------------------------------


class TestRunAdmixture:
    def test_fits_once_then_transforms_each_cohort(self, tmp_path, fake_admixture):
        """The unified sequence. Using fit_transform for the fit cohort — as the
        old backend branch did — would retrain on a real backend."""
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            k_min=2,
            k_max=3,
        )

        (inst,) = fake_admixture.instances
        assert [c[0] for c in inst.calls] == ["fit", "transform", "transform"]
        assert inst.calls[0][1] == "fitset"
        assert inst.calls[1][1] == "fitset"
        assert inst.calls[2][1] == "projectset"

    def test_returns_both_q_file_maps(self, tmp_path, fake_admixture):
        fit_q, project_q = run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
        )
        assert fit_q == {2: Path(f"{tmp_path / 'fit'}.2.csv")}
        assert project_q == {2: Path(f"{tmp_path / 'project'}.2.csv")}

    def test_creates_checkpoints_and_output_directories(self, tmp_path, fake_admixture):
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "deep" / "ckpt",
            fit_output=tmp_path / "nested" / "fit",
            project_output=tmp_path / "nested" / "project",
        )
        assert (tmp_path / "deep" / "ckpt").is_dir()
        assert (tmp_path / "nested").is_dir()

    def test_construction_kwargs_forwarded(self, tmp_path, fake_admixture):
        """Constraint E: detect_cluster.sh feeds threads/gpus/batch size here."""
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            k_min=3,
            k_max=7,
            force=True,
            threads=8,
            num_gpus=2,
            batch_size=256,
        )
        k = fake_admixture.instances[0].kwargs
        assert k["k_min"] == 3
        assert k["k_max"] == 7
        assert k["force"] is True
        assert k["threads"] == 8
        assert k["num_gpus"] == 2
        assert k["batch_size"] == 256

    def test_zero_gpus_is_forwarded_not_dropped(self, tmp_path, fake_admixture):
        """num_gpus=0 explicitly requests CPU-only. The old argv builder handled
        this correctly; passing values directly must not regress it."""
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            num_gpus=0,
        )
        assert fake_admixture.instances[0].kwargs["num_gpus"] == 0

    def test_zero_threads_is_forwarded_not_dropped(self, tmp_path, fake_admixture):
        """The old argv builder used `if admix_threads:` and silently dropped 0."""
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            threads=0,
        )
        assert fake_admixture.instances[0].kwargs["threads"] == 0

    def test_model_name_defaults_to_fit_and_is_overridable(self, tmp_path, fake_admixture):
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
        )
        assert fake_admixture.instances[0].calls[0][3] == "fit"

        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            model_name="custom",
        )
        assert fake_admixture.instances[1].calls[0][3] == "custom"

    def test_injected_backend_is_passed_to_the_wrapper(self, tmp_path, fake_admixture):
        """The unification point: an injected backend goes through the SAME
        NeuralAdmixture wrapper the real path uses, not a separate branch."""
        backend = FakeBackend()
        run_admixture(
            "fitset",
            "projectset",
            checkpoints_dir=tmp_path / "ckpt",
            fit_output=tmp_path / "fit",
            project_output=tmp_path / "project",
            backend=backend,
        )
        assert fake_admixture.instances[0].kwargs["backend"] is backend


# ---------------------------------------------------------------------------
# run_admixture_step — config in, typed result out
# ---------------------------------------------------------------------------


class TestRunAdmixtureStep:
    def test_writes_to_the_documented_layout(self, tmp_path, fake_admixture):
        """Spec constraint B: downstream scripts read these exact paths."""
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=2, k_max=3)

        run_admixture_step(io, cfg)

        paths = admixture_output_paths(io, cfg)
        assert paths["dir"].is_dir()
        assert paths["checkpoints_dir"].is_dir()
        (inst,) = fake_admixture.instances
        assert inst.calls[1][2] == str(paths["fit_prefix"])
        assert inst.calls[2][2] == str(paths["project_prefix"])

    def test_result_carries_the_documented_fields(self, tmp_path, fake_admixture):
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=2, k_max=4)

        result = run_admixture_step(io, cfg)

        paths = admixture_output_paths(io, cfg)
        assert isinstance(result, AdmixtureStepResult)
        assert result.q_prefix == paths["project_prefix"]
        assert result.fit_prefix == paths["fit_prefix"]
        assert result.dir == paths["dir"]
        assert result.checkpoints_dir == paths["checkpoints_dir"]
        assert result.k_values == (2, 3, 4)
        assert result.fit_q_files == paths["fit_q_files"]
        assert result.project_q_files == paths["project_q_files"]
        assert result.skipped is False

    def test_fits_on_fit_cohort_and_transforms_both(self, tmp_path, fake_admixture):
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=2, k_max=3)

        run_admixture_step(io, cfg)

        (inst,) = fake_admixture.instances
        assert [c[0] for c in inst.calls] == ["fit", "transform", "transform"]
        assert inst.calls[0][1] == "data/fit"
        assert inst.calls[1][1] == "data/fit"
        assert inst.calls[2][1] == "data/project"

    def test_config_values_reach_the_wrapper(self, tmp_path, fake_admixture):
        io = make_io(tmp_path)
        cfg = AdmixtureConfig(k_min=2, k_max=5, threads=4, num_gpus=0, batch_size=400)

        run_admixture_step(io, cfg)

        k = fake_admixture.instances[0].kwargs
        assert k["k_min"] == 2
        assert k["k_max"] == 5
        assert k["threads"] == 4
        assert k["num_gpus"] == 0
        assert k["batch_size"] == 400

    def test_injected_backend_takes_the_same_path(self, tmp_path, fake_admixture):
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=2, k_max=3)
        backend = FakeBackend()

        run_admixture_step(io, cfg, backend=backend)

        assert fake_admixture.instances[0].kwargs["backend"] is backend

    def test_q_file_maps_track_the_k_range(self, tmp_path, fake_admixture):
        io, cfg = make_io(tmp_path), AdmixtureConfig(k_min=3, k_max=5)

        result = run_admixture_step(io, cfg)

        assert sorted(result.project_q_files) == [3, 4, 5]
        assert result.project_q_files[4].name == "project.4.csv"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_step_admixture.py -q -p no:cacheprovider
```

Expected: collection error — `ModuleNotFoundError: No module named 'manifold_genetics.pipeline.steps.admixture'`.

- [ ] **Step 3: Write `src/manifold_genetics/pipeline/steps/admixture.py`**

```python
"""In-process admixture step.

``run_admixture()`` is the seam shared by the ``manifold-genetics admixture``
subcommand and ``run_admixture_step()``. Both go through the ``NeuralAdmixture``
wrapper, which constructs a real ``NeuralAdmixtureBackend`` when no backend is
injected — so a test backend and the real one take the SAME path. That replaces
the orchestrator's old special-case branch, which called an injected backend
directly and used ``fit_transform`` for the fit cohort where the CLI path used
``transform``.

The real external tool still runs in its own process: ``NeuralAdmixtureBackend``
shells out to ``neural-admixture train`` / ``infer``, one child per K. That
boundary lives in L1 and is unchanged here.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from ...admixture import AdmixtureBackend, NeuralAdmixture
from ..config import AdmixtureConfig, IOConfig
from .paths import admixture_output_paths

logger = logging.getLogger(__name__)

__all__ = [
    "AdmixtureStepResult",
    "admixture_output_paths",
    "run_admixture",
    "run_admixture_step",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class AdmixtureStepResult:
    """Typed outputs of the admixture step.

    ``q_prefix`` is the project cohort's prefix — ``<prefix>.{k}.csv`` — which is
    what the metrics step consumes.
    """

    q_prefix: Path
    k_values: Tuple[int, ...]
    dir: Path
    checkpoints_dir: Path
    fit_prefix: Path
    fit_q_files: Dict[int, Path] = field(default_factory=dict)
    project_q_files: Dict[int, Path] = field(default_factory=dict)
    skipped: bool = False


def run_admixture(
    fit_plink: PathLike,
    project_plink: PathLike,
    *,
    checkpoints_dir: PathLike,
    fit_output: PathLike,
    project_output: PathLike,
    k_min: int = 2,
    k_max: int = 10,
    force: bool = False,
    threads: Optional[int] = None,
    num_gpus: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_name: str = "fit",
    backend: Optional[AdmixtureBackend] = None,
) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    """Train admixture models on the fit cohort and infer on both cohorts.

    ``backend`` is passed straight through to ``NeuralAdmixture``; ``None`` means
    the real ``NeuralAdmixtureBackend``. Values like ``threads=0`` / ``num_gpus=0``
    are forwarded as given — they are meaningful, not absent.

    Returns ``(fit_q_files, project_q_files)``, each mapping K to a CSV path.
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    Path(fit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(project_output).parent.mkdir(parents=True, exist_ok=True)

    admix = NeuralAdmixture(
        k_min=k_min,
        k_max=k_max,
        force=force,
        threads=threads,
        num_gpus=num_gpus,
        batch_size=batch_size,
        backend=backend,
    )

    admix.fit(fit_plink, output_dir=checkpoints_dir, model_name=model_name)
    fit_q_files = admix.transform(fit_plink, output_prefix=fit_output)
    project_q_files = admix.transform(project_plink, output_prefix=project_output)

    return fit_q_files, project_q_files


def run_admixture_step(
    io: IOConfig,
    admix: AdmixtureConfig,
    *,
    backend: Optional[AdmixtureBackend] = None,
) -> AdmixtureStepResult:
    """Run admixture for the configured K range and report where it wrote."""
    paths = admixture_output_paths(io, admix)

    logger.info(f"Running admixture K={admix.k_min}..{admix.k_max}")
    run_admixture(
        io.fit_plink,
        io.project_plink,
        checkpoints_dir=paths["checkpoints_dir"],
        fit_output=paths["fit_prefix"],
        project_output=paths["project_prefix"],
        k_min=admix.k_min,
        k_max=admix.k_max,
        threads=admix.threads,
        num_gpus=admix.num_gpus,
        batch_size=admix.batch_size,
        backend=backend,
    )

    return AdmixtureStepResult(
        q_prefix=paths["project_prefix"],
        k_values=tuple(range(admix.k_min, admix.k_max + 1)),
        dir=paths["dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        fit_prefix=paths["fit_prefix"],
        fit_q_files=paths["fit_q_files"],
        project_q_files=paths["project_q_files"],
    )
```

Note `run_admixture_step` reports the Q-file maps from `admixture_output_paths`, not from the backend's return values — the path helper is the contract (constraint B), and the two agree by construction.

- [ ] **Step 4: Re-export from `steps/__init__.py`**

Add, keeping imports and `__all__` alphabetically ordered:

```python
from .admixture import (
    AdmixtureStepResult,
    run_admixture,
    run_admixture_step,
)
```

plus `"AdmixtureStepResult"`, `"run_admixture"`, `"run_admixture_step"` in `__all__`. `admixture_output_paths` is already exported from `.paths` — do not add a second export of that name.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/unit/test_step_admixture.py tests/unit/test_step_paths.py -q -p no:cacheprovider
```

- [ ] **Step 6: Lint and commit**

```bash
black src/ tests/ && isort src/ tests/
git add src/manifold_genetics/pipeline/steps/admixture.py \
        src/manifold_genetics/pipeline/steps/__init__.py \
        tests/unit/test_step_admixture.py
git commit -m "feat(pipeline): add in-process admixture step

run_admixture() routes both callers through the NeuralAdmixture wrapper, so an
injected test backend and the real NeuralAdmixtureBackend take the same path.
Unifies on the CLI semantics: fit() once, then transform() on each cohort.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 2: Route `cmd_admixture` through the seam

**Files:**
- Modify: `src/manifold_genetics/cli.py` (`cmd_admixture`)
- Test: `tests/unit/test_cli_main.py` (`test_cmd_admixture_fit_project`)

**Interfaces:** consumes `run_admixture` from Task 1. Produces nothing new.

Argument resolution stays in `cmd_admixture`: `args.fit_plink or args.input`, `args.project_plink or fit_prefix`, the checkpoint-dir fallback chain `args.neuraladmixture_output_dir or args.output or Path.cwd() / "admixture_outputs"`, and the "must provide --fit-output and --project-output" check.

- [ ] **Step 1: Update the CLI test**

`test_cmd_admixture_fit_project` in `tests/unit/test_cli_main.py` currently patches `mg_cli.NeuralAdmixture`. The wrapper is now constructed inside `steps.admixture`. Repoint it:

```python
_STEP_ADMIXTURE = "manifold_genetics.pipeline.steps.admixture"
```

and change the `monkeypatch.setattr(mg_cli, "NeuralAdmixture", FakeAdmix)` line to
`monkeypatch.setattr(f"{_STEP_ADMIXTURE}.NeuralAdmixture", FakeAdmix)`. Keep the rest of the test as-is, then add:

```python
def test_cmd_admixture_forwards_cluster_resources(monkeypatch, tmp_path):
    """Constraint E: --threads / --num-gpus / --neuraladmixture-batch-size must
    reach the backend; detect_cluster.sh feeds them."""
    seen = {}

    class FakeAdmix:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def fit(self, *a, **k):
            pass

        def transform(self, plink_prefix, output_prefix=None):
            return {2: Path(f"{output_prefix}.2.csv")}

    monkeypatch.setattr(f"{_STEP_ADMIXTURE}.NeuralAdmixture", FakeAdmix)
    rc = mg_cli.main(
        [
            "admixture",
            "--fit-plink",
            "fit",
            "--project-plink",
            "proj",
            "--fit-output",
            str(tmp_path / "fit"),
            "--project-output",
            str(tmp_path / "project"),
            "--k-min",
            "2",
            "--k-max",
            "3",
            "--threads",
            "8",
            "--num-gpus",
            "0",
            "--neuraladmixture-batch-size",
            "256",
        ]
    )
    assert rc == 0
    assert seen["threads"] == 8
    assert seen["num_gpus"] == 0
    assert seen["batch_size"] == 256
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_cli_main.py -k admixture -q -p no:cacheprovider
```

Expected: FAIL — the real `NeuralAdmixture` is still constructed in `cli.py`.

- [ ] **Step 3: Rewrite `cmd_admixture`**

Replace the body from the `admix = NeuralAdmixture(` construction through the two `transform` calls with:

```python
    fit_q_files, project_q_files = run_admixture(
        fit_prefix,
        project_prefix,
        checkpoints_dir=checkpoint_dir,
        fit_output=fit_output_path,
        project_output=project_output_path,
        k_min=args.k_min,
        k_max=args.k_max,
        force=args.force,
        threads=args.threads,
        num_gpus=args.num_gpus,
        batch_size=getattr(args, "neuraladmixture_batch_size", None),
        model_name=args.model_name,
    )
```

Keep the three `print(...)` lines and `return 0` exactly as they are. `run_admixture` creates the checkpoint and output parent directories, so the explicit `mkdir` calls above them become redundant — remove them only if you confirm nothing between them and the call depends on the directories existing.

Add `run_admixture` to the existing `from .pipeline.steps import ...` line. Then check whether `NeuralAdmixture` still has a use in `cli.py` (`grep -n '\bNeuralAdmixture\b' src/manifold_genetics/cli.py`); if the only hits are the import and help text, delete `from .admixture import NeuralAdmixture`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/unit/test_cli_main.py -q -p no:cacheprovider
pytest -m "not slow and not network" -q -p no:cacheprovider
```

- [ ] **Step 5: Lint and commit**

```bash
black src/ tests/ && isort src/ tests/
git add src/manifold_genetics/cli.py tests/unit/test_cli_main.py
git commit -m "refactor(cli): route cmd_admixture through the shared seam

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 3: Orchestrator calls `run_admixture_step`; delete the last subprocess hop

**Files:**
- Modify: `src/manifold_genetics/pipeline/orchestrator.py`
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:** consumes `run_admixture_step(io, admix, *, backend=None)` and `AdmixtureConfig`. Produces nothing new; `results` keys `admixture_dir`, `admixture_checkpoints_dir`, `fit_q_files`, `project_q_files`, `q_files` keep their meanings.

This deletes the **last** `subprocess.run` in `run()`, so `import subprocess` goes with it.

- [ ] **Step 1: Rewrite the admixture tests**

In `tests/unit/test_orchestrator.py`:

1. Delete `class TestPipelineRunAdmixtureFlags` entirely (its section banner too) — all six tests assert flags in an argv list that no longer exists. Their intent is preserved by `test_step_admixture.py::TestRunAdmixture::test_construction_kwargs_forwarded`, `test_zero_gpus_is_forwarded_not_dropped` and `test_zero_threads_is_forwarded_not_dropped`, plus the new orchestrator test below.

2. Add a `stub_admixture_step` helper next to `stub_pca_step` and `stub_embedding_step`:

```python
def stub_admixture_step(monkeypatch):
    """Replace the in-process admixture step with a fake that writes its Q files.

    Returns the list of (io, admix_config, backend) it was called with.
    """
    import manifold_genetics.pipeline.orchestrator as orch
    from manifold_genetics.pipeline.steps.admixture import AdmixtureStepResult
    from manifold_genetics.pipeline.steps.paths import admixture_output_paths

    calls = []

    def fake_run_admixture_step(io, admix, *, backend=None):
        calls.append((io, admix, backend))
        paths = admixture_output_paths(io, admix)
        paths["checkpoints_dir"].mkdir(parents=True, exist_ok=True)
        for q in list(paths["fit_q_files"].values()) + list(paths["project_q_files"].values()):
            write_pca_csv(q, 2)
        return AdmixtureStepResult(
            q_prefix=paths["project_prefix"],
            k_values=tuple(range(admix.k_min, admix.k_max + 1)),
            dir=paths["dir"],
            checkpoints_dir=paths["checkpoints_dir"],
            fit_prefix=paths["fit_prefix"],
            fit_q_files=paths["fit_q_files"],
            project_q_files=paths["project_q_files"],
        )

    monkeypatch.setattr(orch, "run_admixture_step", fake_run_admixture_step)
    return calls
```

3. Add a new class:

```python
# ---------------------------------------------------------------------------
# TestPipelineRunAdmixtureInProcess — the last subprocess hop is gone
# ---------------------------------------------------------------------------


class TestPipelineRunAdmixtureInProcess:
    """run() calls run_admixture_step() directly, for the real backend and for an
    injected one alike. The old code had two branches here — the injected-backend
    one used fit_transform() for the fit cohort where the CLI path used
    transform() — which is exactly the divergence this unification removes."""

    def _run(self, pipeline, monkeypatch, **kwargs):
        calls = stub_admixture_step(monkeypatch)
        pipeline.run(
            skip_pca=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_admixture_visualization=True,
            skip_metrics=True,
            **kwargs,
        )
        assert len(calls) == 1, f"Expected one admixture step call, got {len(calls)}"
        return calls[0]

    def test_cluster_resources_reach_the_step(self, tmp_path, monkeypatch):
        """Constraint E. admix_gpus=0 and admix_threads=0 are meaningful values,
        not absent ones — the old argv builder dropped threads=0 as falsy."""
        pipeline = make_pipeline(tmp_path)
        io, cfg, backend = self._run(
            pipeline, monkeypatch, admix_threads=0, admix_gpus=0, admix_batch_size=256
        )
        assert cfg.threads == 0
        assert cfg.num_gpus == 0
        assert cfg.batch_size == 256

    def test_k_range_reaches_the_step(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        io, cfg, backend = self._run(pipeline, monkeypatch, k_min=3, k_max=6)
        assert cfg.k_min == 3
        assert cfg.k_max == 6

    def test_injected_backend_is_forwarded_not_branched_on(self, tmp_path, monkeypatch):
        """One code path: the backend is a parameter, not a branch."""
        backend = _FakeBackend()
        pipeline = make_pipeline(tmp_path, admixture_backend=backend)
        io, cfg, seen_backend = self._run(pipeline, monkeypatch)
        assert seen_backend is backend

    def test_no_backend_passes_none(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        io, cfg, seen_backend = self._run(pipeline, monkeypatch)
        assert seen_backend is None

    def test_results_expose_admixture_paths(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        stub_admixture_step(monkeypatch)
        results = pipeline.run(
            k_min=2,
            k_max=3,
            skip_pca=True,
            skip_embedding=True,
            skip_pca_visualization=True,
            skip_admixture_visualization=True,
            skip_metrics=True,
        )
        admix_dir = pipeline.output_dir / "admixture"
        assert results["admixture_dir"] == admix_dir
        assert results["admixture_checkpoints_dir"] == admix_dir / "checkpoints"
        assert sorted(results["fit_q_files"]) == [2, 3]
        assert sorted(results["project_q_files"]) == [2, 3]
        assert results["q_files"] == results["project_q_files"]
```

4. `TestPipelineRunFullFlow` uses `_FakeBackend` and asserts `backend.calls == ["fit", "fit_transform", "transform"]`. Change that expectation to `["fit", "transform", "transform"]` — that is the unification, deliberately observable. Those tests also call `stub_pca_step` / `stub_embedding_step`; add `stub_admixture_step(monkeypatch)` alongside, and delete the now-unused `capture_subprocess` helper and its `admix_calls` / `pca_calls` / `embed_calls` companions if nothing references them any more (check first — `test_pca_no_longer_shells_out` and `test_embedding_no_longer_shells_out` still use `capture_subprocess` and their respective filters, so most likely only `admix_calls` becomes dead).

5. Add to `TestPipelineRunFullFlow` (or the new class) one test asserting the pipeline no longer shells out at all:

```python
    def test_pipeline_makes_no_subprocess_calls(self, tmp_path, monkeypatch):
        """After PR 4 no stage shells out to our own CLI. subprocess.run is gone
        from orchestrator.py entirely, so patching it is no longer even possible —
        assert instead that the module no longer imports it."""
        import manifold_genetics.pipeline.orchestrator as orch

        assert not hasattr(orch, "subprocess"), (
            "orchestrator must no longer import subprocess — every stage runs in-process"
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_orchestrator.py -q -p no:cacheprovider
```

Expected: FAIL — `orchestrator` has no attribute `run_admixture_step`.

- [ ] **Step 3: Wire the orchestrator**

Add to the imports:

```python
from .config import AdmixtureConfig, EmbeddingConfig, IOConfig, PCAConfig
from .steps.admixture import run_admixture_step
```

Replace the whole admixture stage — from `admix_dir = self.output_dir / "admixture"` down to and including `results["q_files"] = project_q_files`, i.e. both branches and the argv builder — with:

```python
            admix_cfg = AdmixtureConfig(
                k_min=k_min,
                k_max=k_max,
                threads=admix_threads,
                num_gpus=admix_gpus,
                batch_size=admix_batch_size,
            )

            admix_result = run_admixture_step(io, admix_cfg, backend=self.admixture_backend)

            admix_dir = admix_result.dir
            results["admixture_dir"] = admix_dir
            results["admixture_checkpoints_dir"] = admix_result.checkpoints_dir
            results["fit_q_files"] = admix_result.fit_q_files
            results["project_q_files"] = admix_result.project_q_files
            results["q_files"] = admix_result.project_q_files
```

`admix_dir` must remain a local — the admixture-visualization block below it uses `admix_dir / "project"` as the bar plot's `q_prefix`. Leave that block untouched (PR 5 owns it).

Then delete `import subprocess` from the top of the file and confirm it is gone: `grep -n 'subprocess' src/manifold_genetics/pipeline/orchestrator.py` must print nothing.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/unit/test_orchestrator.py -q -p no:cacheprovider
pytest -m "not slow and not network" -q -p no:cacheprovider
pytest tests/integration/test_generic_pipeline.py -q -p no:cacheprovider
```

`test_generic_pipeline.py` injects `PrecomputedAdmixtureBackend`, so it exercises the unified path with a real injected backend end to end. It must stay at 6 passed — that is the single strongest check on this task.

- [ ] **Step 5: Lint and commit**

```bash
black src/ tests/ && isort src/ tests/
git add src/manifold_genetics/pipeline/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "refactor(pipeline): run admixture in-process and delete the backend branch

Removes the last subprocess hop to our own CLI, and with it the special-case
branch that called an injected backend directly. Both callers now go through
NeuralAdmixture, unified on the CLI's fit-then-transform-twice semantics.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 4: Verify against the constraints and open the PR

**Files:** none modified unless a check fails.

- [ ] **Step 1: CLI surface frozen (constraint A)**

```bash
git diff origin/main -- src/manifold_genetics/cli.py | grep -E '^[-+].*add_(parser|argument)'
```
Expected: no output.

- [ ] **Step 2: No subprocess hops remain**

```bash
grep -n 'subprocess' src/manifold_genetics/pipeline/orchestrator.py
grep -rn '"manifold-genetics"' src/manifold_genetics/pipeline/
```
Expected: no output from either. This is the milestone — `run_pipeline()` no longer spawns its own CLI for any stage.

- [ ] **Step 3: No admixture path literals outside `steps/paths.py`**

```bash
grep -rn 'admixture/checkpoints\|"admixture"' src/manifold_genetics/ --include=*.py | grep -v steps/paths.py
```
Inspect any hit; help text and docstrings are fine, path construction is not.

- [ ] **Step 4: Full check**

```bash
black --check src/ tests/ && isort --check src/ tests/
pytest -m "not slow and not network" -q -p no:cacheprovider
pytest tests/integration/test_generic_pipeline.py -q -p no:cacheprovider
```

- [ ] **Step 5: Push and open the PR**

The body must cover: the commit-by-commit summary; the fit_transform → transform unification and why the CLI semantics won; the falsy-`threads=0` bug it removes; that `AdmixtureBackend.fit_transform` remains on the ABC but is no longer called by the pipeline; and that this removes the **last** subprocess hop, leaving PR 5 (viz + `PipelineResult` + `run()` finalize) as the remainder.

---

## Self-Review

**Spec coverage (migration table row 4 — "Admixture step"):**

| Spec requirement | Task |
|---|---|
| `steps/admixture.py` | 1 |
| `run_admixture_step(..., backend=None)` | 1 |
| Replace the last compute subprocess call | 3 |
| **Delete the special-case backend branch (unified)** | 3 |
| `cmd_admixture` → step fn | 2 |
| Record the process-isolation finding in the PR body | 4 (module docstring also states it) |
| Constraint A (frozen CLI) | 4 |
| Constraint B (output paths) | 1, 4 |
| Constraint D (compute steps fatal) | 1 |
| Constraint E (threads/gpus/batch through to backend) | 1, 2, 3 |

**Deviations, both deliberate:**
1. `AdmixtureStepResult` carries more than the spec's sketch (`q_prefix`, `k_values`, `skipped`): it adds `dir`, `checkpoints_dir`, `fit_prefix`, `fit_q_files`, `project_q_files`, because `run_pipeline()`'s existing `results` dict exposes all of them and this PR must not change that contract.
2. The step reports Q-file maps from `admixture_output_paths` rather than from the backend's return values, so the documented layout stays the single source of truth (constraint B). The two agree by construction.

**Deferred to PR 5:** the four viz steps (including the admixture bar plot, which stays inline here), `PipelineResult`, `build_configs()` wiring into `Pipeline.__init__`, `_get_embedding_model` deletion, and the `--skip-<x>`-with-no-prior-outputs clear error.

**Type consistency:** `AdmixtureStepResult` field names are used identically in Tasks 1 and 3. `AdmixtureConfig(.k_min, .k_max, .threads, .num_gpus, .batch_size)` matches `pipeline/config.py`. `admixture_output_paths` keys `"dir"`, `"checkpoints_dir"`, `"fit_prefix"`, `"project_prefix"`, `"fit_q_files"`, `"project_q_files"` match `steps/paths.py` on `main`.
