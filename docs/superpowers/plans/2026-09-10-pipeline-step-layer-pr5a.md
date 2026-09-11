# Pipeline Step Layer — PR 5a: visualization steps

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the four inline visualization blocks in `Pipeline.run()` into `steps/viz.py`, route the `cmd_plot*` CLI handlers through the same functions, and make non-fatal viz failure explicit and uniform via a `_run_viz` helper.

**Architecture:** `steps/viz.py` gains one `run_*_viz_step()` per figure family plus a `VizStepResult`. `steps/paths.py` gains `figure_output_paths()`, so the `figures/**` layout joins the other output contracts. `Pipeline.run()` calls each step through `_run_viz(name, failed, fn)`, which catches, logs and records the failure — spec constraint D — so a broken plot can never fail the pipeline. The `results` dict keeps every key it has today; `PipelineResult` is PR 5b.

**Tech Stack:** Python 3.10–3.12, matplotlib (Agg in tests), pytest, `uv`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-30-pipeline-step-layer-design.md` (migration table row 5, split into 5a/5b by agreement — 5a is the viz half). PRs 1–4 = #68, #70, #74, #76. Base this branch on `main` **after #76 merges**.

## Global Constraints

- **A — The CLI surface is frozen.** No flag renames, removals, or additions in `src/manifold_genetics/cli.py`; nothing below `main()` changes.
- **B — Output paths are a contract.** Figures live at `output_dir/figures/pca/pca_pairs_by_{col}.png`, `output_dir/figures/embeddings/**`, `output_dir/figures/admixture/project_bars.png`, and `output_dir/figures/admixture/project_admixture_colored_embedding.png`. After this PR that layout comes from `figure_output_paths()` in `steps/paths.py` and nowhere else.
- **D — Visualization steps are non-fatal; compute steps are fatal.** Today only `plot_projection` is wrapped in try/except. This PR generalises that to all four viz steps via `_run_viz`, records each failure in the returned results, and prints a loud summary line. A viz failure must never fail `manifold-genetics pipeline`.
- **F — `project` naming everywhere; no dead params.**
- **Behaviour-preserving.** `run_pipeline()` still returns the same loose `results` dict with the same keys: `pca_figures`, `fit_embedding_figures`, `embedding_figures`, `projection_plot`, and `admixture_figures` (a dict with `bars` and `admixture_colored_embedding`). One key is **added**: `failed_viz_steps`, a tuple — see "The one new key" below.
- **Out of scope, do not touch:** `PipelineResult`, `build_configs()` wiring into `Pipeline.__init__`, the `run()` rewrite into an explicit sequence, constraint-C skip errors, deleting `_get_embedding_model`, and the parametrized 3-mode contract test. **All of those are PR 5b.** Also leave the compute stages (PCA, admixture, embedding, metrics) alone.
- **Style.** `black` (line length 100) and `isort` (profile black) are the binding gates. `flake8` is NOT a working gate — `setup.cfg` shadows `.flake8`, ~573 pre-existing violations, mostly E501 from black's own formatting (issue #71). Ignore E501; never edit `.flake8` or `setup.cfg`.
- **Tests.** Baseline on `main` after #76: **499 passed, 7 skipped, 7 deselected**, integration `test_generic_pipeline.py` **6 passed**. Run on the compute node — allocation **JOBID <id> on <node>**, which does the whole fast suite in ~45s: `srun --jobid=<JOBID> --overlap --ntasks=1 --time=40:00 bash -c 'cd <repo> && source .venv/bin/activate && pytest ... -q -p no:cacheprovider'`. One pytest at a time, in the foreground; never background a run and wait on it.

## A latent crash this PR fixes structurally

`orchestrator.py:234` imports `read_colormap` **inside** `if pca_file.exists():`, which is nested inside `if not skip_pca_visualization:`. That makes the name function-local for the whole of `run()`, and `orchestrator.py:287` (the admixture bar-plot block) then uses it. So:

```bash
manifold-genetics pipeline ... --skip-pca-visualization     # UnboundLocalError
```

This is **issue #72**, pre-existing and unrelated to the migration. Splitting each viz block into its own module-level function with its own top-level imports removes it by construction. Task 4 adds the regression test for the flag combination and the PR closes #72.

## The one new key, and why it is not a contract break

Constraint D requires viz failures to be *visible*. The spec's design records them on `PipelineResult.failed_steps`, but `PipelineResult` is PR 5b. Rather than build the machinery twice, this PR adds a single new key to the existing dict:

```python
results["failed_viz_steps"] = tuple(failed)   # e.g. ("admixture_viz",)
```

Adding a key is backward compatible — every existing consumer reads by name. PR 5b folds it into `PipelineResult.failed_steps` and drops the key. `cmd_pipeline` prints a loud summary line when the tuple is non-empty.

## File Structure

**Create:**
- `src/manifold_genetics/pipeline/steps/viz.py` — `VizStepResult`, `run_pca_viz_step()`, `run_embedding_viz_step()`, `run_admixture_viz_step()`, `run_admixture_embedding_viz_step()`.
- `tests/unit/test_step_viz.py`

**Modify:**
- `src/manifold_genetics/pipeline/steps/paths.py` — add `figure_output_paths()`.
- `src/manifold_genetics/pipeline/steps/__init__.py` — re-export.
- `src/manifold_genetics/pipeline/orchestrator.py` — replace the four inline blocks with `_run_viz`-wrapped step calls; delete the function-scoped `read_colormap` import.
- `src/manifold_genetics/cli.py` — `cmd_plot`, `cmd_plot_pca`, `cmd_plot_projection`, `cmd_plot_admixture`, `cmd_plot_admixture_embedding` routed at the viz steps where they map cleanly; `cmd_pipeline` gains the failed-viz summary line. **No `main()` changes.**
- `tests/unit/test_orchestrator.py` — rework `_stub_plots`; add non-fatal-viz tests.
- `tests/unit/test_cli_main.py` — repoint **7** `mg_cli` monkeypatches (lines ~619, 639, 659, 660, 683, 702, 728).

**The test surface, mapped up front** (this is the step the last two PRs skipped, and it cost a cycle each time). Files patching a viz symbol through a *namespace that this PR changes*:

| File | What breaks | Action |
|---|---|---|
| `tests/unit/test_cli_main.py` | 7 `monkeypatch.setattr(mg_cli, "<viz fn>", ...)` at lines 619, 639, 659, 660, 683, 702, 728 | repoint to wherever the symbol is now called |
| `tests/unit/test_orchestrator.py` | `_stub_plots` patches 4 names on `orch` plus 2 by dotted path | rewrite to patch `steps.viz` |

Files patching `manifold_genetics.visualization.*` **directly** are unaffected and must not be touched: `tests/unit/test_visualization.py`, `tests/unit/test_plotting_coverage.py`, `tests/unit/test_plotting_projection.py`, `tests/unit/test_io_helpers.py`, `tests/unit/test_io.py`.

---

### Task 1: `figure_output_paths()` + `steps/viz.py`

**Files:**
- Modify: `src/manifold_genetics/pipeline/steps/paths.py`
- Create: `src/manifold_genetics/pipeline/steps/viz.py`
- Create: `tests/unit/test_step_viz.py`
- Modify: `src/manifold_genetics/pipeline/steps/__init__.py`

**Interfaces:**
- Consumes: `IOConfig`, `VizConfig` (fields `.admix_group_column`, `.admix_within_group_order`, `.projection_plot_fit_column`, `.projection_plot_project_column`), `PCAConfig`, `AdmixtureConfig`, `EmbeddingConfig` from `pipeline.config`; `PCAStepResult`, `EmbeddingStepResult`, `AdmixtureStepResult` from their step modules; `visualize`, `plot_pca_pairs`, `plot_projection`, `plot_admixture_bar_grid`, `plot_admixture_embedding_grid` from `manifold_genetics.visualization`; `read_colormap` from `manifold_genetics.utils.io`.
- Produces:
  - `figure_output_paths(io: IOConfig) -> Dict[str, Path]` with keys `"root"` (`output_dir/figures`), `"pca"`, `"embeddings"`, `"admixture"`, plus `"admixture_bars"` (`admixture/project_bars.png`) and `"admixture_colored_embedding"` (`admixture/project_admixture_colored_embedding.png`). Pure, no I/O.
  - `VizStepResult(figures: tuple[Path, ...] = (), failed: bool = False)` — frozen dataclass, per the spec.
  - `run_pca_viz_step(io, viz, *, pca_file: Path, n_pcs: int) -> VizStepResult`
  - `run_embedding_viz_step(io, viz, *, embedding: EmbeddingStepResult, method: str) -> VizStepResult`
  - `run_admixture_viz_step(io, viz, *, admixture: AdmixtureStepResult) -> VizStepResult`
  - `run_admixture_embedding_viz_step(io, *, embedding: EmbeddingStepResult, admixture: AdmixtureStepResult) -> VizStepResult`

Each step must reproduce its inline block **exactly**: same plotting function, same keyword arguments, same figure filenames, same iteration over colormap columns, same `subsample_per_group=300`, same group-column fallback to the first colormap key. Read the current blocks before writing anything — `orchestrator.py` lines ~220-250 (PCA viz), ~282-305 (admixture bars), ~342-406 (embedding viz + projection plot), ~410-434 (admixture-coloured embedding).

Two structural notes:
- `run_embedding_viz_step` covers the *whole* of today's "Step 4", including the projection-plot sub-block. Keep the projection plot's existing internal try/except for now — it is inside the step, and `_run_viz` will wrap the step as well; belt and braces here is deliberate, because the projection plot failing must not lose the fit/project figures already produced.
- `run_admixture_embedding_viz_step` takes no `VizConfig` — its block reads nothing from it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_step_viz.py`. Use `matplotlib` in Agg mode is unnecessary because every plotting function is stubbed; patch them at `manifold_genetics.pipeline.steps.viz.<name>`.

```python
"""Tests for the in-process visualization steps (pipeline/steps/viz.py).

These four steps are extracted verbatim from inline blocks in Pipeline.run().
The tests assert the extraction is faithful: same plotting function, same
kwargs, same figure paths. Every plotting function is stubbed — nothing here
renders a figure.

Note the asymmetry with the compute steps: visualization is NON-FATAL by spec
constraint D. These step functions may raise; the orchestrator's _run_viz
wrapper is what catches. That contract is tested in test_orchestrator.py.
"""

from pathlib import Path

import pytest

from manifold_genetics.pipeline.config import (
    AdmixtureConfig,
    EmbeddingConfig,
    IOConfig,
    VizConfig,
)
from manifold_genetics.pipeline.steps.admixture import AdmixtureStepResult
from manifold_genetics.pipeline.steps.embedding import EmbeddingStepResult
from manifold_genetics.pipeline.steps.paths import figure_output_paths
from manifold_genetics.pipeline.steps.viz import (
    VizStepResult,
    run_admixture_embedding_viz_step,
    run_admixture_viz_step,
    run_embedding_viz_step,
    run_pca_viz_step,
)

MODULE = "manifold_genetics.pipeline.steps.viz"

COLORMAP = {"Population": {"A": "#000000"}, "Region": {"X": "#ffffff"}}


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


@pytest.fixture
def stub_colormap(monkeypatch):
    monkeypatch.setattr(f"{MODULE}.read_colormap", lambda p: dict(COLORMAP))


@pytest.fixture
def calls(monkeypatch):
    """Record every plotting call as (name, kwargs)."""
    recorded = []

    def rec(name, ret):
        def _fn(**kwargs):
            recorded.append((name, kwargs))
            return ret
        return _fn

    monkeypatch.setattr(f"{MODULE}.plot_pca_pairs", rec("plot_pca_pairs", Path("pca.png")))
    monkeypatch.setattr(f"{MODULE}.visualize", rec("visualize", [Path("emb.png")]))
    monkeypatch.setattr(f"{MODULE}.plot_projection", rec("plot_projection", Path("proj.png")))
    monkeypatch.setattr(
        f"{MODULE}.plot_admixture_bar_grid", rec("plot_admixture_bar_grid", None)
    )
    monkeypatch.setattr(
        f"{MODULE}.plot_admixture_embedding_grid",
        rec("plot_admixture_embedding_grid", None),
    )
    return recorded


# ---------------------------------------------------------------------------
# figure_output_paths — the figures/** layout contract
# ---------------------------------------------------------------------------


def test_figure_output_paths_match_the_documented_layout(tmp_path):
    """Spec constraint B. Downstream scripts and users look for these by name."""
    io = make_io(tmp_path)
    p = figure_output_paths(io)
    root = io.output_dir / "figures"
    assert p["root"] == root
    assert p["pca"] == root / "pca"
    assert p["embeddings"] == root / "embeddings"
    assert p["admixture"] == root / "admixture"
    assert p["admixture_bars"] == root / "admixture" / "project_bars.png"
    assert (
        p["admixture_colored_embedding"]
        == root / "admixture" / "project_admixture_colored_embedding.png"
    )


def test_figure_output_paths_is_pure(tmp_path):
    """No directory may be created by asking where figures go."""
    io = make_io(tmp_path)
    figure_output_paths(io)
    assert not (io.output_dir / "figures").exists()


# ---------------------------------------------------------------------------
# run_pca_viz_step
# ---------------------------------------------------------------------------


class TestPcaVizStep:
    def test_plots_one_grid_per_colormap_column(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        pca_file = tmp_path / "project_pca_10.csv"

        result = run_pca_viz_step(io, VizConfig(), pca_file=pca_file, n_pcs=10)

        names = [c[0] for c in calls]
        assert names == ["plot_pca_pairs", "plot_pca_pairs"], "one grid per colormap column"
        cols = [c[1]["label_column"] for c in calls]
        assert cols == ["Population", "Region"]
        assert isinstance(result, VizStepResult)
        assert result.failed is False
        assert len(result.figures) == 2

    def test_forwards_the_documented_kwargs(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        pca_file = tmp_path / "project_pca_10.csv"

        run_pca_viz_step(io, VizConfig(), pca_file=pca_file, n_pcs=10)

        _, kw = calls[0]
        assert kw["pca_coords"] == pca_file
        assert kw["labels"] == io.project_labels
        assert kw["colormap"] == COLORMAP
        assert kw["n_pcs"] == 10
        assert kw["output_path"] == figure_output_paths(io)["pca"] / "pca_pairs_by_Population.png"
        assert kw["title"] == "PCA Pairs by Population"

    def test_creates_the_figures_directory(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_pca_viz_step(io, VizConfig(), pca_file=tmp_path / "p.csv", n_pcs=3)
        assert figure_output_paths(io)["pca"].is_dir()


# ---------------------------------------------------------------------------
# run_embedding_viz_step
# ---------------------------------------------------------------------------


class TestEmbeddingVizStep:
    def _emb(self, tmp_path, with_fit=True):
        return EmbeddingStepResult(
            embedding_file=tmp_path / "phate_2d.csv",
            fit_embedding_file=(tmp_path / "phate_fit_2d.csv") if with_fit else None,
        )

    def test_project_only_when_there_is_no_fit_embedding(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_embedding_viz_step(
            io, VizConfig(), embedding=self._emb(tmp_path, with_fit=False), method="phate"
        )

        vis = [c for c in calls if c[0] == "visualize"]
        assert len(vis) == 1, "single-input modes produce only the project figures"
        assert vis[0][1]["dataset_prefix"] == "project_"

    def test_fit_and_project_when_both_embeddings_exist(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_embedding_viz_step(io, VizConfig(), embedding=self._emb(tmp_path), method="phate")

        vis = [c for c in calls if c[0] == "visualize"]
        assert [v[1]["dataset_prefix"] for v in vis] == ["fit_", "project_"]
        assert vis[0][1]["labels"] == io.fit_labels
        assert vis[0][1]["colormap"] == io.fit_colormap
        assert vis[1][1]["labels"] == io.project_labels
        assert vis[1][1]["colormap"] == io.project_colormap

    def test_projection_plot_only_when_both_columns_configured(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)

        run_embedding_viz_step(io, VizConfig(), embedding=self._emb(tmp_path), method="phate")
        assert not [c for c in calls if c[0] == "plot_projection"]

        calls.clear()
        viz = VizConfig(
            projection_plot_fit_column="Population", projection_plot_project_column="Region"
        )
        run_embedding_viz_step(io, viz, embedding=self._emb(tmp_path), method="phate")
        proj = [c for c in calls if c[0] == "plot_projection"]
        assert len(proj) == 1
        assert proj[0][1]["fit_label_column"] == "Population"
        assert proj[0][1]["project_label_column"] == "Region"
        assert proj[0][1]["output_path"].name == (
            "phate_projection_fit_Population_project_Region.png"
        )

    def test_projection_plot_failure_does_not_lose_the_other_figures(
        self, tmp_path, calls, stub_colormap, monkeypatch
    ):
        """The projection plot keeps its own try/except: losing it must not lose
        the fit and project figures already produced."""
        def boom(**kwargs):
            raise RuntimeError("projection exploded")

        monkeypatch.setattr(f"{MODULE}.plot_projection", boom)
        io = make_io(tmp_path)
        viz = VizConfig(
            projection_plot_fit_column="Population", projection_plot_project_column="Region"
        )

        result = run_embedding_viz_step(io, viz, embedding=self._emb(tmp_path), method="phate")

        assert result.figures, "fit/project figures must survive a projection-plot failure"


# ---------------------------------------------------------------------------
# run_admixture_viz_step
# ---------------------------------------------------------------------------


class TestAdmixtureVizStep:
    def _admix(self, tmp_path):
        d = tmp_path / "out" / "admixture"
        return AdmixtureStepResult(
            q_prefix=d / "project",
            k_values=(2, 3),
            dir=d,
            checkpoints_dir=d / "checkpoints",
            fit_prefix=d / "fit",
        )

    def test_bar_grid_kwargs_and_path(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_admixture_viz_step(io, VizConfig(), admixture=self._admix(tmp_path))

        (name, kw), = [c for c in calls if c[0] == "plot_admixture_bar_grid"]
        assert kw["q_prefix"] == self._admix(tmp_path).q_prefix
        assert kw["labels"] == io.project_labels
        assert kw["colormap"] == COLORMAP
        assert kw["subsample_per_group"] == 300
        assert list(kw["k_values"]) == [2, 3]
        assert kw["output_path"] == figure_output_paths(io)["admixture_bars"]

    def test_group_column_defaults_to_first_colormap_key(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_admixture_viz_step(io, VizConfig(admix_group_column=None), admixture=self._admix(tmp_path))
        assert calls[0][1]["group_column"] == "Population"

    def test_explicit_group_column_wins(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_admixture_viz_step(
            io, VizConfig(admix_group_column="Region"), admixture=self._admix(tmp_path)
        )
        assert calls[0][1]["group_column"] == "Region"

    def test_within_group_order_forwarded(self, tmp_path, calls, stub_colormap):
        io = make_io(tmp_path)
        run_admixture_viz_step(
            io, VizConfig(admix_within_group_order="tree"), admixture=self._admix(tmp_path)
        )
        assert calls[0][1]["within_group_order"] == "tree"


# ---------------------------------------------------------------------------
# run_admixture_embedding_viz_step
# ---------------------------------------------------------------------------


class TestAdmixtureEmbeddingVizStep:
    def test_grid_kwargs_and_path(self, tmp_path, calls):
        io = make_io(tmp_path)
        d = tmp_path / "out" / "admixture"
        admix = AdmixtureStepResult(
            q_prefix=d / "project",
            k_values=(2, 3, 4),
            dir=d,
            checkpoints_dir=d / "checkpoints",
            fit_prefix=d / "fit",
        )
        emb = EmbeddingStepResult(embedding_file=tmp_path / "phate_2d.csv")

        result = run_admixture_embedding_viz_step(io, embedding=emb, admixture=admix)

        (name, kw), = [c for c in calls if c[0] == "plot_admixture_embedding_grid"]
        assert kw["embedding"] == emb.embedding_file
        assert kw["q_prefix"] == admix.q_prefix
        assert list(kw["k_values"]) == [2, 3, 4]
        assert kw["output_path"] == figure_output_paths(io)["admixture_colored_embedding"]
        assert result.figures == (figure_output_paths(io)["admixture_colored_embedding"],)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_step_viz.py -q -p no:cacheprovider
```
Expected: collection error — no `steps.viz`, and `figure_output_paths` missing from `steps.paths`.

- [ ] **Step 3: Add `figure_output_paths()` to `steps/paths.py`**

```python
def figure_output_paths(io: IOConfig) -> Dict[str, Path]:
    """Where each figure family is written. Pure; creates nothing."""
    root = io.output_dir / "figures"
    admixture = root / "admixture"
    return {
        "root": root,
        "pca": root / "pca",
        "embeddings": root / "embeddings",
        "admixture": admixture,
        "admixture_bars": admixture / "project_bars.png",
        "admixture_colored_embedding": admixture / "project_admixture_colored_embedding.png",
    }
```

- [ ] **Step 4: Write `src/manifold_genetics/pipeline/steps/viz.py`**

Extract each block verbatim. The module docstring should state that these steps are non-fatal by contract and that the orchestrator's `_run_viz` wrapper is what enforces it. Imports go at module level — that is what fixes issue #72.

Structure each step as: resolve paths via `figure_output_paths(io)`, `mkdir(parents=True, exist_ok=True)` the directory it writes into, do exactly what the inline block did, and return `VizStepResult(figures=tuple(paths))`. Reproduce the existing `logger.info` messages where they carry information.

For `run_embedding_viz_step`, preserve the inline block's three sub-parts in order (fit figures when a fit embedding exists, project figures always, projection plot when both columns are configured) and keep the projection plot's `try/except Exception` with its `logger.warning`.

- [ ] **Step 5: Re-export from `steps/__init__.py`**

Add `figure_output_paths` from `.paths` and the four `run_*_viz_step` names plus `VizStepResult` from `.viz`, keeping imports and `__all__` alphabetical.

- [ ] **Step 6: Run the tests to verify they pass, then lint and commit**

```bash
pytest tests/unit/test_step_viz.py tests/unit/test_step_paths.py -q -p no:cacheprovider
black src/ tests/ && isort src/ tests/
git add src/manifold_genetics/pipeline/steps/viz.py \
        src/manifold_genetics/pipeline/steps/paths.py \
        src/manifold_genetics/pipeline/steps/__init__.py \
        tests/unit/test_step_viz.py
git commit -m "feat(pipeline): add in-process visualization steps

Extracts the four inline viz blocks from Pipeline.run() into steps/viz.py, and
adds figure_output_paths() so the figures/** layout is a declared contract like
the others. Module-level imports here remove the function-scoped read_colormap
that made --skip-pca-visualization crash (issue #72).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 2: Route the `cmd_plot*` handlers through the viz steps

**Files:**
- Modify: `src/manifold_genetics/cli.py` — `cmd_plot_pca`, `cmd_plot_admixture`, `cmd_plot_admixture_embedding`
- Test: `tests/unit/test_cli_main.py` — repoint the affected monkeypatches

**Interfaces:** consumes the step functions from Task 1. Produces nothing new.

**Judgement call, stated explicitly:** not every `cmd_plot*` maps onto a viz step. Route only those that do, and leave the rest calling the plotting functions directly:

| Handler | Action |
|---|---|
| `cmd_plot_pca` | route through `run_pca_viz_step` if its args can be expressed as `IOConfig` + `VizConfig`; otherwise leave it, and say so in the report |
| `cmd_plot_admixture` | same assessment against `run_admixture_viz_step` |
| `cmd_plot_admixture_embedding` | same against `run_admixture_embedding_viz_step` |
| `cmd_plot` | leave alone — it is `visualize()` on an arbitrary embedding CSV with free-form output dir, which `run_embedding_viz_step` cannot express without breaking constraint A |
| `cmd_plot_projection` | leave alone — free-form output path, same reason |
| `cmd_plot_knn_composition` | leave alone — no pipeline step covers it |

This mirrors the `cmd_pca` decision in PR 3: the seam is shared where the shapes genuinely match, and forcing a fixed-layout step to serve a free-form CLI flag set would break the frozen surface. **If a handler's flags cannot be expressed through the step without changing the CLI, leave it and record why in your report** — that is the expected outcome for at least `cmd_plot`, `cmd_plot_projection` and `cmd_plot_knn_composition`.

- [ ] **Step 1: Repoint the affected tests**

In `tests/unit/test_cli_main.py`, seven monkeypatches target `mg_cli`:

| Line | Symbol | Used by |
|---|---|---|
| ~619 | `visualize` | `cmd_plot` |
| ~639 | `plot_pca_pairs` | `cmd_plot_pca` |
| ~659 | `read_colormap` | `cmd_plot_admixture` |
| ~660 | `plot_admixture_bar_grid` | `cmd_plot_admixture` |
| ~683 | `plot_admixture_embedding_grid` | `cmd_plot_admixture_embedding` |
| ~702 | `plot_knn_composition` | `cmd_plot_knn_composition` |
| ~728 | `plot_projection` | `cmd_plot_projection` |

For each handler you route through a step, repoint its patches to `manifold_genetics.pipeline.steps.viz.<name>`. For each handler you leave alone, the patch stays on `mg_cli`. Do not change a patch whose handler you did not touch.

- [ ] **Step 2-5:** run the tests to see them fail, make the changes, re-run, lint, commit. Full fast suite must be green; the count changes only by tests you add.

```bash
git commit -m "refactor(cli): route the plot subcommands through the viz steps

Only the handlers whose flags fit the step layer's fixed figure layout are
routed; cmd_plot, cmd_plot_projection and cmd_plot_knn_composition keep calling
the plotting functions directly, since expressing their free-form output paths
through a step would require changing the frozen CLI surface.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 3: Orchestrator calls the viz steps through `_run_viz`

**Files:**
- Modify: `src/manifold_genetics/pipeline/orchestrator.py`
- Modify: `src/manifold_genetics/cli.py` (`cmd_pipeline` summary line only)
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes the four step functions and `figure_output_paths`.
- Produces: `_run_viz(name: str, failed: list, fn) -> VizStepResult | None` — a module-level helper in `orchestrator.py`:

```python
def _run_viz(name: str, failed: list, fn):
    """Run a visualization step, converting any failure into a warning.

    Spec constraint D: visualization is non-fatal. A broken plot must never fail
    `manifold-genetics pipeline`, but it must not vanish silently either — the
    step name lands in the returned results so the CLI can shout about it.
    """
    try:
        return fn()
    except Exception as e:
        logger.warning(f"Visualization step {name!r} failed: {e}")
        failed.append(name)
        return None
```

Replace each inline block with a guarded call, preserving today's gating conditions exactly:

| Step name | Gate (unchanged) |
|---|---|
| `pca_viz` | `not skip_pca_visualization` **and** the project PCA file exists |
| `admixture_viz` | `not skip_admixture` and `not skip_admixture_visualization` |
| `embedding_viz` | `not skip_visualization` and `not skip_embedding` |
| `admixture_embedding_viz` | `not skip_admixture_visualization` and `not skip_embedding` and `not skip_admixture`, and both results present |

Populate the same `results` keys from each `VizStepResult`, add `results["failed_viz_steps"] = tuple(failed)`, and delete the function-scoped `from ..utils.io import read_colormap` import.

In `cmd_pipeline`, after the existing summary prints, add:

```python
    if results.get("failed_viz_steps"):
        names = ", ".join(results["failed_viz_steps"])
        print(f"\n⚠ {len(results['failed_viz_steps'])} visualization step(s) failed: {names}")
```

- [ ] **Step 1: Rework the tests**

Rewrite `TestPipelineRunFullFlow._stub_plots` to patch the four step functions on the orchestrator module (`orch.run_pca_viz_step` etc.) returning `VizStepResult`, rather than patching plotting functions. Then add a class:

```python
class TestPipelineVizIsNonFatal:
    """Spec constraint D: a broken plot must not fail the pipeline, and must not
    vanish silently either."""

    def test_failing_viz_step_does_not_fail_the_pipeline(self, tmp_path, monkeypatch):
        ...  # make one viz step raise; assert run() returns and compute outputs exist

    def test_failure_is_recorded_in_results(self, tmp_path, monkeypatch):
        ...  # assert results["failed_viz_steps"] == ("admixture_viz",)

    def test_a_warning_is_logged(self, tmp_path, monkeypatch, caplog):
        ...  # assert the step name appears in a WARNING record

    def test_other_viz_steps_still_run_after_one_fails(self, tmp_path, monkeypatch):
        ...  # one raises; assert the others' results keys are still populated
```

and a regression test for issue #72:

```python
def test_skip_pca_visualization_with_admixture_visualization_on(tmp_path, monkeypatch):
    """Regression, issue #72: read_colormap used to be imported inside the PCA-viz
    branch, so this flag combination raised UnboundLocalError from the admixture
    bar-plot block."""
```

- [ ] **Steps 2-5:** fail, implement, pass, lint, commit. Run the full fast suite and `tests/integration/test_generic_pipeline.py` (must stay 6 passed).

---

### Task 4: Verify, close #72, and open the PR

- [ ] **Step 1: CLI surface frozen** — `git diff origin/main -- src/manifold_genetics/cli.py | grep -E '^[-+].*add_(parser|argument)'` must be empty.
- [ ] **Step 2: No figure path literals outside `steps/paths.py`** — `grep -rn 'figures/' src/manifold_genetics/ --include=*.py | grep -v steps/paths.py`; inspect each hit (help text and docstrings are fine).
- [ ] **Step 3: `run()` shrank** — report the before/after line count; it should drop by roughly 130 lines.
- [ ] **Step 4: Full check** — `black --check`, `isort --check`, fast suite, integration suite.
- [ ] **Step 5: Push and open the PR.** Body covers: the four extractions; that constraint D is now uniform rather than projection-plot-only; the new `failed_viz_steps` key and why adding a key is not a break; which `cmd_plot*` handlers were routed and which deliberately were not, with the reason; and **`Closes #72`**, with the flag combination that used to crash and the regression test that now covers it.

---

## Self-Review

**Spec coverage (migration row 5, viz half):**

| Spec requirement | Task |
|---|---|
| `steps/viz.py` × 4 wired into `run()` | 1, 3 |
| `_run_viz` helper; constraint D uniform | 3 |
| `cmd_plot*` → viz steps | 2 (where shapes match; documented where not) |
| Figure layout as a declared contract | 1 |
| `test_viz_failure_records_and_continues` | 3 |

**Deferred to PR 5b:** `PipelineResult` (and folding `failed_viz_steps` into `.failed_steps`), `build_configs()` into `Pipeline.__init__`, the `run()` rewrite into the explicit sequence, constraint-C skip errors, deleting `_get_embedding_model` and its now-unused embedding imports, and `test_pipeline_output_layout[projection|subsample|transform]` with cross-cohort fixtures.

**Deviations, both deliberate:** (1) `failed_viz_steps` is added to the loose dict rather than building `PipelineResult` early — avoids implementing the failure-recording machinery twice. (2) Only some `cmd_plot*` handlers route through steps; the rest cannot without breaking constraint A, exactly as `cmd_pca` could not in PR 3.

**Type consistency:** `VizStepResult(figures, failed)` matches the spec's sketch. `AdmixtureStepResult.q_prefix` / `.k_values` and `EmbeddingStepResult.embedding_file` / `.fit_embedding_file` match the dataclasses on `main`. `figure_output_paths` keys are used identically in Tasks 1 and 3.
