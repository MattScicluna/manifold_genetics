# Pipeline Step Layer — PR 5b: orchestrator finalize

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the migration. Replace `run_pipeline()`'s loose `results` dict with a typed `PipelineResult`, wire `build_configs()` into `Pipeline.__init__`, reduce `run()` to an explicit top-to-bottom sequence, make skipped-step errors legible, delete the last dead code, and add the parametrized contract test that pins the output tree across all three pipeline modes.

**Architecture:** `run()` currently threads eighteen string keys through 255 lines of interleaved compute, visualization and skip handling. 5b turns that into a sequence of step calls returning typed results, assembled into one frozen `PipelineResult`. The step layer already exists and is unchanged — this PR is about the orchestrator's own shape and its public return type.

**Tech Stack:** Python 3.10–3.12, pandas, pytest, `uv`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-30-pipeline-step-layer-design.md` (migration row 5, second half). PRs 1–5a = #68, #70, #74, #76, #77 (all merged). **Base this branch on `main` after #78 merges** — #78 is the two-line lint fix that turns `main`'s CI green again.

## Global Constraints

- **A — The CLI surface is frozen.** No flag renames, removals, or additions; nothing below `main()` in `cli.py` changes. `cmd_pipeline`'s body changes only where it reads the result.
- **B — Output paths are a contract.** Unchanged by this PR, and now *pinned* by the new contract test across all three modes.
- **C — A skipped step still supplies its output paths downstream; if `--skip-<x>` is set and no prior outputs exist, raise a clear error** rather than a bare `FileNotFoundError` deep in a later step. This is the constraint 5b finally satisfies.
- **D — Visualization is non-fatal, compute is fatal.** Already delivered in 5a via `_run_viz`; preserve it, and fold `failed_viz_steps` into `PipelineResult.failed_steps`.
- **E — `threads` / `num_gpus` / `batch_size` reach the backend.** Preserved through `build_configs()`.
- **F — `project` naming everywhere; no dead params.**
- **Breaking change, sanctioned:** `run_pipeline()` returns `PipelineResult` instead of `Dict`. The user has confirmed **nothing outside this repository calls `run_pipeline()`**, and no script under `examples/` reads its return value — they all shell out to `manifold-genetics pipeline`. So **no `.to_legacy_dict()` shim**, per the spec.
- **Style — read this carefully, an earlier ruling here was wrong.** The binding gate is the exact command `.github/workflows/ci.yml` runs:

  ```
  flake8 src tests --max-line-length=88 --extend-ignore=E501,E203,W503,F841,F541
  ```

  It matches neither `.flake8` nor `setup.cfg` (the drift tracked in #71), so running flake8 locally with either config gives a useless answer — ~573 violations instead of the handful CI cares about. **E501 and F841 are ignored; F401 is not**, and an unused import failed CI on #77 (fixed by #78). Every task must run CI's command verbatim before committing and see it clean. `black --check` (line length 100) and `isort --check` (profile black) remain gates too. Never edit `.flake8` or `setup.cfg` here — reconciling them is #71's job.
- **Tests.** Baseline after #77: **523 passed, 7 skipped, 7 deselected**; integration `test_generic_pipeline.py` **6 passed**. Compute node: `srun --jobid=<JOBID> --overlap --ntasks=1 --time=40:00 bash -c 'cd <repo> && source .venv/bin/activate && pytest ... -q -p no:cacheprovider'` — ~60s for the fast suite. One pytest at a time, foreground.

## The full surface, mapped before planning

**Every `results` key** that becomes a `PipelineResult` field (18, plus two nested under `admixture_figures`):

```
admixture_checkpoints_dir  embedding_figures   fit_pca_file   pca_figures        project_pca_file
admixture_dir              embedding_file      fit_q_files    pca_file           project_q_files
embedding_coords           failed_viz_steps    metrics        pca_coords         q_files
fit_embedding_figures      fit_embedding_file  projection_plot
admixture_figures = {"bars": …, "admixture_colored_embedding": …}
```

**Everything that reads a result** — the complete consumer list, verified by grep, not assumed:

| Consumer | Sites | Action |
|---|---|---|
| `src/manifold_genetics/cli.py` `cmd_pipeline` | ~6 (`results["metrics"]`, `results.get("failed_viz_steps")`) | rewrite against `PipelineResult` |
| `tests/unit/test_orchestrator.py` | 25 `results["…"]` | rewrite against attributes |
| `tests/integration/test_generic_pipeline.py` | 15 `results["…"]` | rewrite against attributes |

No other file in `src/`, `tests/` or `examples/` subscripts a pipeline result.

## File Structure

**Create:**
- `src/manifold_genetics/pipeline/result.py` — `PipelineResult`.
- `tests/unit/test_pipeline_result.py`
- `tests/integration/test_pipeline_contract.py` — the parametrized 3-mode output-layout test.

**Modify:**
- `src/manifold_genetics/pipeline/orchestrator.py` — `build_configs()` into `__init__`; `run()` rewritten; skip errors; delete `_get_embedding_model`, `_io_config()` and the now-unused `..embeddings` imports.
- `src/manifold_genetics/pipeline/runner.py` — return type and docstring.
- `src/manifold_genetics/pipeline/steps/viz.py` — `EmbeddingVizResult` split; drop dead `VizStepResult.failed`; drop `run_pca_viz_step`'s unused `viz` param.
- `src/manifold_genetics/cli.py` — `cmd_pipeline` summary against the typed result.
- `tests/unit/test_orchestrator.py`, `tests/integration/test_generic_pipeline.py`, `tests/unit/test_step_viz.py`.

---

### Task 1: `PipelineResult` + the viz result split

Two changes that are cheap together and both pure additions to the type layer.

**Files:** Create `src/manifold_genetics/pipeline/result.py`, `tests/unit/test_pipeline_result.py`; modify `steps/viz.py`, `tests/unit/test_step_viz.py`, `pipeline/__init__.py`.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class PipelineResult:
    """Typed outputs of a pipeline run. Replaces the loose results dict."""

    # Compute step results, None when the step was skipped
    pca: Optional[PCAStepResult] = None
    admixture: Optional[AdmixtureStepResult] = None
    embedding: Optional[EmbeddingStepResult] = None
    geographic_metrics: Optional[MetricsStepResult] = None
    admixture_metrics: Optional[MetricsStepResult] = None

    # Figures, by family
    pca_figures: Tuple[Path, ...] = ()
    fit_embedding_figures: Tuple[Path, ...] = ()
    embedding_figures: Tuple[Path, ...] = ()
    projection_plot: Optional[Path] = None
    admixture_figures: Mapping[str, Path] = field(default_factory=dict)

    # Steps whose failure was tolerated (constraint D)
    failed_steps: Tuple[str, ...] = ()

    @property
    def figures(self) -> Tuple[Path, ...]:
        """Every figure produced, in stage order."""

    @property
    def metrics(self) -> dict:
        """The metrics mapping, in the shape the CLI summary prints:
        {"geographic": {...}, "admixture": {"2": {...}}}. Empty when neither ran."""
```

The old dict's `pca_file` / `fit_pca_file` / `project_pca_file` / `q_files` / `fit_q_files` / `project_q_files` / `embedding_file` / `fit_embedding_file` / `pca_coords` / `embedding_coords` / `admixture_dir` / `admixture_checkpoints_dir` are all reachable through the step results (`result.pca.project_pca`, `result.admixture.q_files`, and so on) — **do not duplicate them as top-level fields.** That collapse is the point of the change. Where a test used `results["pca_file"]`, it becomes `result.pca.project_pca`.

`failed_viz_steps` becomes `failed_steps`, keeping 5a's substep names (`"projection_plot"` alongside `"embedding_viz"`).

**Viz result split** — the spec says "one frozen dataclass per step", but 5a left one `VizStepResult` shared by four, with three embedding-only fields bolted on. Give the embedding viz step its own type:

```python
@dataclass(frozen=True)
class EmbeddingVizResult:
    fit_figures: Tuple[Path, ...] = ()
    project_figures: Tuple[Path, ...] = ()
    projection_plot: Optional[Path] = None
    failed_substeps: Tuple[str, ...] = ()

    @property
    def figures(self) -> Tuple[Path, ...]: ...
```

and reduce `VizStepResult` to `figures: Tuple[Path, ...] = ()`. **Delete `VizStepResult.failed`** — no step ever sets it and nothing reads it; failure is signalled by raising, which `_run_viz` catches. Also drop `run_pca_viz_step`'s unused `viz: VizConfig` parameter (`run_admixture_embedding_viz_step` already takes none, so the symmetry argument was never applied consistently).

Tests: `PipelineResult` defaults are all empty/None; `figures` aggregates in stage order; `metrics` returns the documented shape and `{}` when neither metric ran; the frozen dataclass does not raise on `__eq__` (watch for DataFrame-carrying step results — mark any such field `compare=False`, as `PCAStepResult` already does).

- [ ] Write the failing tests → run → implement → run → lint → commit.

---

### Task 2: `build_configs()` into `Pipeline.__init__`, and legible skip errors

**Files:** `orchestrator.py`, `runner.py`, `tests/unit/test_orchestrator.py`, `tests/unit/test_runner.py`.

`Pipeline.__init__` currently re-implements the labels/colormap resolution that `build_configs()` already does, and `_io_config()` repackages the attributes for the step layer on every call. Replace both: `__init__` keeps its current keyword signature (constraint: `Pipeline(...)` construction must keep working exactly as today), calls `build_configs()`, and stores the six sub-configs. `_io_config()` is deleted.

Two wrinkles the implementer must handle rather than discover:

1. **`build_configs()` takes the `run()`-time parameters too** (`n_pcs`, `k_min`/`k_max`, `embedding`, the skip flags…), but `Pipeline.__init__` does not receive them — they arrive at `run()`. Resolve by having `__init__` build the IO/viz configs it can, and `run()` build the per-stage configs from its own arguments; or by having `__init__` accept and store them with `run()` overriding. **Pick one, state it in the report, and keep both `Pipeline(...).run(...)` and `run_pipeline(...)` working with their current signatures.**
2. **`run_pipeline()` currently duplicates the labels/colormap validation** that `build_configs()` performs (`runner.py`, the two `if not labels …` blocks). Once `__init__` calls `build_configs()`, that duplication is dead — remove it, but confirm the error messages a caller sees are unchanged, since `tests/unit/test_runner.py` asserts on them.

**Constraint C** — the payoff. Today `run()` populates PCA paths from disk when `skip_pca=True` and lets the embedding stage raise a generic `RuntimeError` if neither file is there. Make each skipped step's resolution explicit and legible: a helper per skipped compute stage that returns the step result built from `*_output_paths()` when the files exist, and otherwise raises naming **which** flag was set, **which** files were expected, and **what to do**. Cover PCA, admixture and embedding. Add tests: for each, `--skip-<x>` with prior outputs succeeds and the downstream stage consumes them; `--skip-<x>` without them raises the clear error.

- [ ] Write the failing tests → run → implement → run → lint → commit.

---

### Task 3: `run()` becomes an explicit sequence; delete the dead code

**Files:** `orchestrator.py`, `cli.py`, `tests/unit/test_orchestrator.py`, `tests/integration/test_generic_pipeline.py`.

Rewrite `run()` to the shape the spec sketches — block per stage, each block a gate plus a step call plus a result assignment, ending in one `PipelineResult(...)` construction. Target ~130 lines. The spec's sketch in "### L3 — orchestrator" is the model; follow its ordering and its `_run_viz(name, failed, fn)` usage, which already exists from 5a.

Delete, having confirmed each is unreferenced:
- `_get_embedding_model` (dead since PR 3, labelled as such) and the `..embeddings` `PHATE`/`UMAP`/`TSNE`/`DiffusionMap` imports it was the last user of.
- `TestGetEmbeddingModel` in `tests/unit/test_orchestrator.py`.
- `_io_config()` if Task 2 has not already removed it.

Fix the two **unguarded `.figures[0]` indexes** flagged in 5a's review (`orchestrator.py` around the admixture-bars and admixture-coloured-embedding result mappings). Both sit *outside* `_run_viz`, so an `IndexError` there would be fatal on a path constraint D is meant to cover. Take the path from `figure_output_paths()` or a named result field instead of indexing.

Rewrite the 25 `results["…"]` sites in `tests/unit/test_orchestrator.py` and the 15 in `tests/integration/test_generic_pipeline.py` against attributes. Add the test 5a's review said was missing: **assert `fit_embedding_figures` is empty and `projection_plot` is `None` in project-only mode** — the existing stubs always populate them, so the absence condition has never been exercised.

Update `cmd_pipeline` to print from the typed result, preserving its current output exactly, including the `⚠ N visualization step(s) failed: …` line now driven by `result.failed_steps`.

- [ ] Write the failing tests → run → implement → run → lint → commit.

---

### Task 4: The parametrized output-layout contract test

**Files:** Create `tests/integration/test_pipeline_contract.py`.

This is the spec's **T2** mitigation and the strongest guarantee in the whole migration: `test_pipeline_output_layout[projection|subsample|transform]` runs a full `Pipeline(...).run()` on the checked-in fixtures with `PrecomputedAdmixtureBackend` and stubbed plotting, then asserts the **complete output tree** for that mode.

The three modes, taken from `examples/_shared/run_pipeline.sh`:

| Mode | `embedding_input` | Shape |
|---|---|---|
| `projection` | `both` | cross-cohort: separate fit/project labels and colormaps, fit + project embeddings, projection plot |
| `subsample` | `fit` | within-cohort, fit subset only |
| `transform` | `project` | within-cohort, project set only |

Use the `cross_cohort_fixtures` conftest fixture (labels, colormaps, geographic — from PR 1, so far unused) with `fit_plink_files` / `project_plink_files`. Assert the full tree per mode:

```
pca/fit_pca_{n}.csv, pca/project_pca_{n}.csv, pca/flashpca_outputs/
admixture/checkpoints/, admixture/fit.{k}.csv, admixture/project.{k}.csv
embeddings/{method}_2d.csv         (+ {method}_fit_2d.csv only in projection mode)
figures/pca/**, figures/embeddings/**, figures/admixture/project_bars.png,
figures/admixture/project_admixture_colored_embedding.png
metrics/geographic.json, metrics/admixture.json
```

Assert both **presence and absence** — `{method}_fit_2d.csv` and the projection plot must NOT exist in `subsample` or `transform` mode. An assertion that only checks presence would pass against a pipeline that writes everything in every mode, which is exactly the branch bug T2 exists to catch.

Mark `@pytest.mark.integration` (these run in CI — `integration` is not deselected; only `slow` and `network` are).

- [ ] Write the tests → run → fix whatever they catch → lint → commit.

---

### Task 5: Verify and open the PR

- [ ] **CLI surface frozen:** `git diff origin/main -- src/manifold_genetics/cli.py | grep -E '^[-+].*add_(parser|argument)'` empty.
- [ ] **The migration's end state:** `grep -rn 'subprocess' src/manifold_genetics/pipeline/` empty; `run()` at or near ~130 lines; `_get_embedding_model` gone; `results\[` gone from `src/`.
- [ ] **No dict left in the public API:** `run_pipeline` and `Pipeline.run` both annotated `-> PipelineResult`.
- [ ] **Full check:** `black --check`, `isort --check`, fast suite, integration suite.
- [ ] **Push and open the PR.** Body covers: the breaking return-type change and why no shim (nothing outside the repo calls it — confirmed by the maintainer); `run()`'s before/after line count; constraint C finally satisfied, with the error messages shown; the contract test and what its absence assertions catch; the dead code removed; and a note that this completes the five-PR migration, with a table of what each PR did.

---

## Self-Review

**Spec coverage (migration row 5, second half):**

| Spec requirement | Task |
|---|---|
| `PipelineResult` replacing the results dict | 1, 3 |
| No `.to_legacy_dict()` | 1 (maintainer confirmed no external callers) |
| `build_configs()` in `__init__`, validation consolidated | 2 |
| `run()` as an explicit sequence, ~130 lines | 3 |
| Constraint C — legible skip errors | 2 |
| Delete `_get_embedding_model`, unused imports | 3 |
| `test_pipeline_output_layout[3 modes]` (T2) | 4 |
| Update the internal consumers | 3 |
| Carried 5a cleanup: viz result split, dead `failed`, unused `viz` param, `.figures[0]`, absence test | 1, 3 |

**Risks:** Task 3 is the largest single diff in the migration — 40 test sites rewritten plus `run()` restructured. Mitigation: Tasks 1 and 2 land the type and the configs first, so Task 3 is mostly mechanical translation against a type that already exists and is tested. Task 4 is written last deliberately: it is the independent check on everything before it, and if the restructure broke a mode's branch, the absence assertions are what catch it.

**Type consistency:** `PipelineResult`'s step-result fields use `PCAStepResult`, `AdmixtureStepResult`, `EmbeddingStepResult`, `MetricsStepResult` as they exist on `main` after #77. `EmbeddingVizResult` replaces the three bolted-on fields added to `VizStepResult` in 5a. `failed_steps` supersedes 5a's `failed_viz_steps` key.
