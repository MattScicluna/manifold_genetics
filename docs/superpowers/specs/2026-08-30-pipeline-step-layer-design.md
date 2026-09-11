# Design: in-process pipeline step layer (A-lite)

**Date:** 2026-08-30
**Status:** approved for planning
**Depends on:** PR #66 (drop dead output-dir params), PR #67 (`transform` → `project` naming) merged to `main` first

## Problem

The pipeline orchestrator (`Pipeline.run()` in `src/manifold_genetics/pipeline/orchestrator.py`)
executes each stage by building an argv list and shelling out to the project's *own* CLI:

```
run_pipeline()                          [our Python, process A]
 └ subprocess("manifold-genetics pca")  [spawns process B — our CLI]
    └ cmd_pca(args)                      [our Python, process B]
       └ PCA().fit()                     [our Python, process B]
          └ subprocess("flashpca …")     [spawns process C — real binary]
```

The call graph is **API → CLI → API**: `run_pipeline` (library) spawns `manifold-genetics
pca` (CLI) which calls `cmd_pca` which calls the `PCA` class (library). Process B does no
work that needs a separate process — it reads a CSV, calls flashpca, writes a CSV.

Consequences today:

- **Brittle tests.** Every pipeline step is tested by asserting the argv list the
  orchestrator builds (`"--project-input" in cmd`). Change-detectors, not contract checks.
- **Coarse errors.** `subprocess.run(check=True)` yields `CalledProcessError` with the real
  traceback buried in the child's stderr.
- **The abstraction already broke once.** `Pipeline.run()` has a special-case branch that
  bypasses the CLI and calls a backend in-process, purely so tests can inject a fake
  admixture backend — two code paths for one step.
- **Cold-start cost.** Each subprocess re-imports pandas / numba / torch.
- **Naming drift.** `cmd_*` and the orchestrator marshal arguments independently, with no
  shared seam — how `transform_pca_*.csv` diverged from `--project-output` (fixed in #67;
  structural cause remains).
- The 500-line `run()` also carries checkpoint logic and interleaved visualization for
  every stage, with inter-step data dependencies passed implicitly via local scope.

## Goals

1. Remove the API → CLI → API inversion. The orchestrator calls step logic **in-process**.
2. One shared seam that both the CLI subcommands and the orchestrator call, so they cannot
   drift.
3. Shrink `Pipeline.run()` to an explicit, readable sequence of step calls.
4. Replace argv-assertion tests with tests that call step functions directly.
5. Keep the CLI and the Python API — both, unchanged for users (except `run_pipeline()`'s
   return type).

## Non-goals

- Changing what any stage computes.
- A workflow-DAG engine, a step registry, `--dry-run` / `--only` / `--resume-from`, or a
  plugin system. The pipeline is a fixed linear sequence with `--skip-*` flags; that is the
  entire step-control requirement. (These are addable later by making the explicit blocks
  in `run()` conditional — no infrastructure needed now.)
- Touching how `flashpca` / `neural-admixture` / `plink` are invoked — those stay as
  subprocess calls to real external software (see "Process isolation").
- Refactoring the compute primitives (`PCA`, `NeuralAdmixture`, `PHATE`, `compute_*`,
  plotting functions).

## Hard constraints

| # | Constraint | Why |
|---|---|---|
| A | The `manifold-genetics pipeline` CLI surface is frozen — no flag renames, removals, or additions in this work. | The example `run_pipeline.sh` scripts only touch this surface. |
| B | Output paths are a contract. Every step writes to the exact path the subprocess version did (`output_dir/pca/project_pca_{n}.csv`, `output_dir/admixture/project.{k}.csv`, `output_dir/embeddings/{method}_2d.csv`, `output_dir/figures/**`, `output_dir/metrics/*.json`). | Downstream example scripts (`geosketch_phate`, `60k_random`, `copy_to_manifoldGenetics`) read these paths; checkpoint/idempotency depends on them. |
| C | With a step skipped, `run()` still supplies its output paths to downstream steps; if `--skip-<x>` is set and no prior outputs exist, it raises a clear error (not a bare `FileNotFoundError` deep in a later step). | Example scripts pass `--skip-pca` etc.; embedding must still find the PCA output, or fail legibly. |
| D | Visualization steps are non-fatal: a failure is caught, logged as a warning, recorded in `PipelineResult.failed_steps`, and the pipeline continues. Compute steps are fatal. | Today `plot_projection` is wrapped in try/except; a broken plot must not fail `manifold-genetics pipeline`. |
| E | `threads` / `num_gpus` / `batch_size` continue to thread through `cmd_pipeline` → `run_pipeline` → admixture step → backend. | `detect_cluster.sh` feeds `$CLUSTER_CPUS` / `$CLUSTER_GPUS` here. |
| F | `project` naming everywhere; no dead params. | Carries #66 and #67 forward. |

## Process isolation (verified)

`flashpca` (C++ binary) and `neural-admixture` (separate pip package) run in their own
processes with their own memory / GPU allocation. **They still do** — that boundary lives
in L1 and this work does not touch it:

- `PCA` shells out to `flashpca` per operation; pure pandas I/O otherwise.
- `NeuralAdmixtureBackend` shells out to `neural-admixture train` / `infer` — one child
  process per K — and does pure file I/O otherwise. Its only in-process `torch` usage is a
  single `torch.cuda.is_available()` call in `_resolve_num_gpus` (no tensors, no compute).
  `torch` is already imported on every CLI invocation via `cli.py → NeuralAdmixture →
  backend`, so running the step in-process is not a regression.

Calling `PCA().fit()` / `backend.transform()` in-process **is** "calling the underlying
API"; the real external tools still get their own process inside those calls. No step
needs a dedicated in-process isolation boundary. If one ever does, a single step can be
made to shell out to `manifold-genetics <step>` in its `run()` body without disturbing the
rest.

## Design

### Layers

```
      CLI (cli.py)                         Python API
  manifold-genetics …              from manifold_genetics import …
        │  cmd_* = argparse → sub-config(s)     │
        ▼                                       ▼
  L3  Orchestrator   run_pipeline() / Pipeline(...).run()
        │            — an explicit sequence of step calls
        ▼
  L2  Step layer (NEW)   run_<x>_step(...)  +  <x>_output_paths(...)
        │                in-process; no subprocess to our own CLI
        ▼
  L1  Compute primitives (UNCHANGED)   PCA · NeuralAdmixture(+backends) · PHATE/UMAP/…
                                       compute_*_preservation · plotting functions
```

Every dependency points down. The CLI is never in the middle of the API's call path.

### L2 — step layer

New package `src/manifold_genetics/pipeline/steps/`: `pca.py`, `admixture.py`,
`embedding.py`, `metrics.py`, `viz.py`. Each compute-step module exports **two** functions:

```python
def pca_output_paths(io: IOConfig, pca: PCAConfig) -> dict[str, Path]:
    """Pure. No I/O. The single source of truth for where this step writes."""

def run_pca_step(io: IOConfig, pca: PCAConfig) -> PCAStepResult:
    """Idempotent: instantiate the L1 class, do the work, return a typed result.
    Skips recomputation when outputs are already valid."""
```

- **`*_output_paths`** — pure. Every place that needs to know where a step writes calls it:
  the step itself, the orchestrator's `--skip-*` handling, and tests. Path-layout knowledge
  lives in exactly one function per step.
- **`run_*_step`** — instantiates the L1 class and does the work. **Idempotent for free**:
  `PCA(force=…)` and `NeuralAdmixtureBackend` already implement skip-when-outputs-exist, so
  the step inherits it. The one extra guard the orchestrator does today (PCA: existing file
  has the wrong dim-column count → force recompute) moves *into* `run_pca_step`.
- **Upstream artifacts are passed as explicit typed parameters**, not looked up from a
  shared dict:

  ```python
  def run_embedding_step(io: IOConfig, emb: EmbeddingConfig, *, pca: PCAStepResult) -> EmbeddingStepResult
  def run_admixture_step(io: IOConfig, admix: AdmixtureConfig, *, backend: AdmixtureBackend | None = None) -> AdmixtureStepResult
  ```

  The dependency is a signature fact — the type checker and the reader both see it.
  Exception: the **metrics** steps take plain `Path` arguments (embedding CSV, Q-file
  prefix, output JSON) rather than result objects — matching `cmd_metrics_*`, and keeping
  them decoupled from *when* the embedding / admixture steps get converted during
  migration:

  ```python
  def run_geographic_metrics_step(embedding_csv: Path, geo_csv: Path, out: Path, **opts) -> MetricsStepResult
  def run_admixture_metrics_step(embedding_csv: Path, q_prefix: Path, k_range: range, out: Path, **opts) -> MetricsStepResult
  ```

`run_*_step` is directly callable by `cmd_*` with no adapter.

### Result types

One frozen dataclass per step, holding just that step's typed outputs:

```python
@dataclass(frozen=True)
class PCAStepResult:
    fit_pca: Path
    project_pca: Path
    coords_df: "pd.DataFrame | None" = None   # populated where current code returns it
    skipped: bool = False

@dataclass(frozen=True)
class AdmixtureStepResult:
    q_prefix: Path            # <dir>/project  → <dir>/project.{k}.csv
    k_values: tuple[int, ...]
    skipped: bool = False

@dataclass(frozen=True)
class EmbeddingStepResult:
    embedding_file: Path
    fit_embedding_file: Path | None = None
    skipped: bool = False

@dataclass(frozen=True)
class MetricsStepResult:
    path: Path
    values: dict            # parsed JSON
    skipped: bool = False

@dataclass(frozen=True)
class VizStepResult:
    figures: tuple[Path, ...] = ()
    failed: bool = False
```

No `StepResult`/`outputs: dict` base and no `StepSpec` — the earlier draft's uniform
contract was ceremony that only the (now removed) generic loop consumed.

### Config

Validated sub-config dataclasses — `IOConfig`, `PCAConfig`, `AdmixtureConfig`,
`EmbeddingConfig`, `VizConfig` — built and validated in **one place** (`build_configs()`,
called by `Pipeline.__init__` / `run_pipeline`), consolidating the validation currently
split between `cmd_pipeline` and `run_pipeline`. `EmbeddingConfig` absorbs the ~30-line
per-method params assembly currently inline in `cmd_pipeline`.

No frozen nested `PipelineConfig` god-object. The orchestrator holds the five sub-configs
and hands each step the one or two it needs.

`Pipeline.__init__` keeps a kwargs signature close to today's (`fit_plink_prefix`,
`project_plink_prefix`, `labels`/`fit_labels`/…, plus the `run()` params folded in or kept
on `run()`), so `Pipeline(...)` construction and `run_pipeline(...)` both keep working.
`run_pipeline` is a thin wrapper over `Pipeline`.

### L3 — orchestrator

`Pipeline.run()` becomes an explicit top-to-bottom sequence (~130 lines):

```python
def run(self) -> PipelineResult:
    io, pca, admix, emb, viz = self._configs
    s = self.skips
    failed: list[str] = []

    # ---- PCA ----
    if s.skip_pca:
        r_pca = _skipped_pca(io, pca)          # paths from pca_output_paths(); raises if absent
    else:
        r_pca = run_pca_step(io, pca)

    if not s.skip_pca_visualization:
        _run_viz("pca_viz", failed, lambda: run_pca_viz_step(io, viz, r_pca))

    # ---- Admixture ----
    r_admix = None
    if not s.skip_admixture:
        r_admix = run_admixture_step(io, admix, backend=self.admixture_backend)
        if not s.skip_admixture_visualization:
            _run_viz("admixture_viz", failed, lambda: run_admixture_viz_step(io, viz, r_admix))

    # ---- Embedding ----
    r_emb = None
    if not s.skip_embedding:
        r_emb = run_embedding_step(io, emb, pca=r_pca)
        if not s.skip_embedding_visualization:
            _run_viz("embedding_viz", failed, lambda: run_embedding_viz_step(io, viz, r_emb))
        if r_admix and not s.skip_admixture_visualization:
            _run_viz("admixture_embedding_viz", failed,
                     lambda: run_admixture_embedding_viz_step(io, viz, r_emb, r_admix))

    # ---- Metrics ----
    r_geo = r_admix_metrics = None
    if not s.skip_metrics and r_emb is not None:
        if io.geographic_coords:
            r_geo = run_geographic_metrics_step(
                r_emb.embedding_file, io.geographic_coords, io.output_dir / "metrics/geographic.json")
        if r_admix is not None:
            r_admix_metrics = run_admixture_metrics_step(
                r_emb.embedding_file, r_admix.q_prefix, range(admix.k_min, admix.k_max + 1),
                io.output_dir / "metrics/admixture.json")

    return PipelineResult(pca=r_pca, admixture=r_admix, embedding=r_emb,
                          geographic_metrics=r_geo, admixture_metrics=r_admix_metrics,
                          failed_steps=tuple(failed))
```

`_run_viz(name, failed, fn)` — the one shared helper: `try: fn() except Exception as e:
logger.warning(...); failed.append(name)`. Non-fatal viz is *visible here*, not hidden in a
decorator (constraint D).

The block order **is** the dependency graph — greppable, and adding or reordering a step is
a local edit in this one function.

### Results

`PipelineResult` — frozen dataclass replacing the loose `results` dict:
`.pca`, `.admixture`, `.embedding`, `.geographic_metrics`, `.admixture_metrics`,
`.figures` (aggregated), `.failed_steps`. **Breaking change to `run_pipeline()`'s return
type** (0.1.0, unreleased; the three internal consumers — `cmd_pipeline`'s summary print,
two integration tests — are updated in the same PR). No `.to_legacy_dict()` unless an
external consumer surfaces.

### CLI

- `cmd_pipeline`: parse args → `build_configs(args)` → `Pipeline(...).run()` → summary
  print (which now also shouts about `result.failed_steps`).
- `cmd_pca` / `cmd_admixture` / `cmd_embed` / `cmd_metrics_geographic` /
  `cmd_metrics_admixture`: parse args → build sub-config(s) → call `run_*_step(...)` →
  print. Arg resolution (`args.fit_plink or args.input`, output defaulting) stays in
  `cmd_*`.
- `cmd_plot` / `cmd_plot_pca` / `cmd_plot_admixture` / `cmd_plot_admixture_embedding` /
  `cmd_plot_projection` → `run_*_viz_step(...)`.
- **No subparser changes.** Every flag identical (constraint A).
- Deleted from `orchestrator.py`: the five `subprocess.run([...])` calls and argv builders,
  `import subprocess`, `_get_embedding_model`, the unused `PHATE/UMAP/TSNE/DiffusionMap`
  imports.

### New public capability

`from manifold_genetics.pipeline.steps import run_pca_step` — a notebook can run one step
in-process and get a typed result, without argparse or the full orchestrator.

## Downstream: the example scripts

The three per-dataset `run_pipeline.sh` wrappers → `examples/_shared/run_pipeline.sh` →
which builds one `manifold-genetics pipeline …` command and `eval`s it. **The scripts touch
only the `manifold-genetics pipeline` CLI surface** — they never call `pca`/`embed`
directly and never read pipeline outputs back in (the per-dataset scripts `echo` a
hardcoded filename list, which is cosmetic).

A-lite changes only what happens *inside* `run_pipeline()`. Against each constraint:

- **A** — `cmd_pipeline` and every subparser unchanged. ✓
- **B** — output paths are now defined in exactly one function per step
  (`*_output_paths`), and `test_pipeline_output_layout` (parametrized over modes, below)
  asserts the full tree. Stronger guarantee than today. ✓
- **C** — a skipped step's paths come from `*_output_paths`; `_skipped_pca` etc. raise a
  clear error if `--skip-<x>` is set with no prior outputs. This replaces today's implicit
  "resolve from disk, hope it's there." ✓
- **D** — `_run_viz` wrapper; failures land in `result.failed_steps` and the summary. ✓
- **E** — `threads` / `num_gpus` / `batch_size` flow `cmd_pipeline` → `build_configs` →
  `AdmixtureConfig` → `run_admixture_step` → backend. ✓
- **F** — #66/#67 land first; `_shared/run_pipeline.sh` already updated for the
  `--projection-plot-project-column` rename and `--flashpca-output-dir` removal on those
  branches. A-lite adds no flag changes. ✓

No circular paths: the sequence is strictly linear and no step consumes a later step's
output.

## Known design tensions

**T1 — non-fatal viz can still mask a real bug (constraint D).** Generalizing the
`plot_projection` try/except to all four viz steps means a genuine code error in, say,
`plot_admixture_bar_grid` becomes a `logger.warning` + green exit. Mitigation: every viz
failure is recorded in `PipelineResult.failed_steps` and the CLI summary prints a loud
`⚠ N visualization step(s) failed: …` line. Residual risk: a cluster user ignores the
summary in a long log. A `--strict` flag that promotes viz failures to fatal is deferred
(YAGNI until asked for).

**T2 — the contract test must cover all three example modes.** `test_pipeline_output_layout`
on one fixture path would miss branch bugs in `subsample` mode (`--embedding-input fit`)
and `projection` mode (cross-cohort: separate colormaps, the `fit_embedding` +
`--projection-plot-*-column` path). Mitigation: parametrize the test over
`projection` / `subsample` / `transform`, which requires cross-cohort fixtures (separate
fit/project labels + colormaps in `tests/fixtures`). Residual cost: those fixtures are new
and need maintaining.

**T3 — `run()` is an explicit 9-block sequence, not data-driven.** Adding or reordering a
step is a manual edit to one function rather than appending to a list. Accepted: steps
change rarely, and the explicit form is easier to review and debug (real function names in
tracebacks, visible control flow) than a list of specs walked by a generic loop. If step
manipulation ever becomes common, the blocks can be lifted into a list at that point.

*(An earlier draft of this design used a `StepSpec` + generic-loop + three-functions-per-step
structure. That introduced: path-layout logic duplicated across `run`/`current`/`resolve`;
string-keyed `upstream["pca"]` lookups with no declared dependencies; and ~27 closure
adapters bridging narrow signatures to a uniform loop signature. A-lite's single
`*_output_paths` helper, explicit typed upstream parameters, and explicit block sequence
remove all three.)*

## Testing

**Shift:** delete argv-assertion tests; replace with calling `run_*_step(...)` against a
monkeypatched L1 class / small CSVs and asserting the file written + (embedding modes)
which input was fed where via the fake's recorded calls. `admixture_backend` injection is
retained as `run_admixture_step(io, admix, *, backend=None)`; tests pass
`PrecomputedAdmixtureBackend`.

**Per-step tests:** `pca_output_paths` returns the documented paths for representative
configs; `run_pca_step` writes them, is idempotent on a second call, and force-recomputes
on a dim-count mismatch. Analogous for the other steps.

**New contract tests:**

- `test_pipeline_output_layout[projection|subsample|transform]` — full `Pipeline(...).run()`
  on `tests/fixtures` + `PrecomputedAdmixtureBackend` + Agg-stubbed plotting; assert the
  complete output tree matches the documented layout for that mode. *(constraint B, T2)*
- `test_skip_pca_then_embedding` — `--skip-pca`, pre-place `project_pca_50.csv`, assert
  embedding runs and consumes it; and `--skip-pca` with no prior outputs raises the clear
  error. *(constraint C)*
- `test_viz_failure_records_and_continues` — make one viz step raise; assert the pipeline
  completes, `result.failed_steps` contains it, a warning is logged, and compute outputs
  exist. *(constraint D, T1)*

CI marker rules unchanged (`not slow and not network`).

## Migration plan

Each PR: green CI, independently reviewable, branched off `main` after #66/#67 land.

| # | Scope | Risk |
|---|---|---|
| 1 | Sub-config dataclasses + `build_configs()` (validation consolidation) + `*_output_paths()` for all steps. `Pipeline`/`run_pipeline` build configs but still run the existing subprocess body. Pure addition; tests for config building + path helpers. | low |
| 2 | PCA step + metrics steps: `steps/pca.py`, `steps/metrics.py`. Replace those `subprocess.run` calls in `run()`; point `cmd_pca` / `cmd_metrics_*` at the step functions; rewrite their tests. Grouped and safe together because the metrics steps take plain `Path` args — the orchestrator passes whatever produced the embedding CSV / Q prefix (still the subprocess steps at this point). | low |
| 3 | Embedding step: `steps/embedding.py`. The three-mode `--embedding-input` → IO mapping; explicit `pca: PCAStepResult` param. Replace subprocess call; `cmd_embed` → step fn; port the mode tests to step-behaviour tests. | med |
| 4 | Admixture step: `steps/admixture.py`, `run_admixture_step(..., backend=None)`. Replace the last compute subprocess call **and** delete the special-case backend branch (unified). `cmd_admixture` → step fn. Record the process-isolation finding in the PR body. | med |
| 5 | Viz steps + orchestrator finalize: `steps/viz.py` × 4 wired into `run()` + `cmd_plot*`; `_run_viz` helper; `run()` becomes the clean explicit sequence; delete `import subprocess`, argv builders, `_get_embedding_model`; `run_pipeline` returns `PipelineResult`; update the three internal consumers; add `test_pipeline_output_layout` parametrized + `test_viz_failure_records_and_continues`. | **high — main review** |
| 6 | *(optional)* Cleanup: any transitional shim, dead helpers, doc / README touch-ups. | low |

~5 PRs (+ optional cleanup).

## Risks

- **PR 5 is the cutover.** Mitigation: PRs 2–4 each land behaviour-preserving with per-step
  tests in place, so PR 5 is mostly deletion + wiring + the parametrized contract test.
- **Embedding three-mode mapping (PR 3)** is the subtlest logic. Mitigation: the existing
  mode tests become step-behaviour tests and must pass unchanged in intent.
- **`PipelineResult` return-type change** could surprise an external caller. Mitigation:
  0.1.0 / unreleased; grep for consumers; add `.to_legacy_dict()` only if one appears.
- **Cross-cohort fixtures (T2)** are new test infrastructure. Mitigation: build them in
  PR 1 alongside the path-helper tests so later PRs can rely on them.
- **#66 / #67 not merged first.** Mitigation: gate the whole effort on those landing;
  carry F as a constraint.

## Open questions

None outstanding. Spec-location note: `docs/` is gitignored in this repo, so this file is
local-only unless a `!docs/superpowers/specs/` exception is added to `.gitignore`.
