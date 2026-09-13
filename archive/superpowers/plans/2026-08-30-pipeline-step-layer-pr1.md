# Pipeline Step Layer — PR 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the validated config dataclasses and the pure output-path helpers that the in-process step layer will be built on — a behaviour-neutral foundation PR.

**Architecture:** New module `pipeline/config.py` holds six frozen sub-config dataclasses plus a `build_configs()` factory that does argument-shape validation (the "shared vs separate labels/colormap" rule, valid enum values, sane ranges). New package `pipeline/steps/` holds `paths.py` with one pure `<step>_output_paths()` function per compute step — the single source of truth for where each stage writes. Nothing in `orchestrator.py`, `cli.py`, or `runner.py` changes; this PR only adds tested building blocks.

**Tech Stack:** Python 3.10–3.12, `dataclasses`, `pytest`. Package uses `uv`; line length 100 (`black`); `flake8` with `--extend-ignore=E501,E203,W503,F841,F541`.

**Spec:** `docs/superpowers/specs/2026-08-30-pipeline-step-layer-design.md`

## Global Constraints

- **Branch off `main` only after PR #66 (dead param removal) and PR #67 (`transform`→`project` rename) are merged.** This plan assumes the post-#67 tree: the second cohort is named `project` everywhere (`project_pca_{n}.csv`, `project.{k}.csv`, `projection_plot_project_column`, fixture files `tests/fixtures/admixture/project.{2,3}.csv`, conftest fixture `project_plink_files`).
- **No changes to `orchestrator.py`, `cli.py`, `runner.py`, or any subparser.** Pure addition.
- **`build_configs()` does argument-shape validation only — no filesystem access.** File-content validation (`validate_labels_csv` etc.) stays where it is today; it moves in a later PR.
- Frozen dataclasses (`@dataclass(frozen=True)`).
- Output-path helpers are **pure**: no I/O, no `mkdir`, deterministic from their arguments.
- Every new `.py` file must pass `black --check` and `isort --check-only` and `flake8 src tests --max-line-length=88 --extend-ignore=E501,E203,W503,F841,F541`.
- Run the fast suite before each commit: `uv run pytest -m "not slow and not network" -q`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/manifold_genetics/pipeline/config.py` (create) | Six frozen sub-config dataclasses (`IOConfig`, `PCAConfig`, `AdmixtureConfig`, `EmbeddingConfig`, `VizConfig`, `SkipConfig`), the `PipelineConfigs` container, and `build_configs()`. |
| `src/manifold_genetics/pipeline/steps/__init__.py` (create) | Empty package marker (re-exports added in later PRs). |
| `src/manifold_genetics/pipeline/steps/paths.py` (create) | `pca_output_paths`, `admixture_output_paths`, `embedding_output_paths`, `metrics_output_paths` — pure functions returning `dict[str, Path]`. |
| `tests/unit/test_pipeline_config.py` (create) | Tests for the dataclasses and `build_configs()`. |
| `tests/unit/test_step_paths.py` (create) | Tests for the four path helpers against the documented layout. |
| `tests/fixtures/cross_cohort/` (create) | `fit_labels.csv`, `project_labels.csv`, `fit_colormap.json`, `project_colormap.json`, `geographic.csv` — for the parametrized contract test that lands in PR 5. |
| `tests/conftest.py` (modify) | Add `cross_cohort_fixtures` fixture returning paths to the above. |
| `tests/fixtures/cross_cohort/README.md` (create) | One paragraph explaining what these are for. |

---

## Task 1: Sub-config dataclasses

**Files:**
- Create: `src/manifold_genetics/pipeline/config.py`
- Test: `tests/unit/test_pipeline_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `IOConfig(fit_plink: Path, project_plink: Path, output_dir: Path, fit_labels: Path, project_labels: Path, fit_colormap: Path, project_colormap: Path, geographic_coords: Path | None = None)` — frozen.
  - `PCAConfig(n_pcs: int = 50, force: bool = False)` — frozen.
  - `AdmixtureConfig(k_min: int = 2, k_max: int = 10, threads: int | None = None, num_gpus: int | None = None, batch_size: int | None = None)` — frozen.
  - `EmbeddingConfig(method: str = "phate", input_mode: str = "both", params: dict = <factory>)` — frozen; `params` via `field(default_factory=dict)`, treated read-only.
  - `VizConfig(admix_group_column: str | None = None, admix_within_group_order: str | None = "chron", projection_plot_fit_column: str | None = None, projection_plot_project_column: str | None = None)` — frozen.
  - `SkipConfig(skip_pca=False, skip_admixture=False, skip_embedding=False, skip_pca_visualization=False, skip_embedding_visualization=False, skip_admixture_visualization=False, skip_metrics=False)` — all `bool`, frozen.
  - `PipelineConfigs(io: IOConfig, pca: PCAConfig, admixture: AdmixtureConfig, embedding: EmbeddingConfig, viz: VizConfig, skips: SkipConfig)` — frozen.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pipeline_config.py`:

```python
"""Tests for pipeline sub-config dataclasses and build_configs()."""

import dataclasses
from pathlib import Path

import pytest

from manifold_genetics.pipeline.config import (
    AdmixtureConfig,
    EmbeddingConfig,
    IOConfig,
    PCAConfig,
    PipelineConfigs,
    SkipConfig,
    VizConfig,
)


def _io(**overrides):
    base = dict(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=Path("out"),
        fit_labels=Path("fit_labels.csv"),
        project_labels=Path("project_labels.csv"),
        fit_colormap=Path("fit_cmap.json"),
        project_colormap=Path("project_cmap.json"),
    )
    base.update(overrides)
    return IOConfig(**base)


class TestSubConfigs:
    def test_ioconfig_holds_paths_and_defaults_geographic_to_none(self):
        io = _io()
        assert io.fit_plink == Path("data/fit")
        assert io.geographic_coords is None

    def test_all_configs_are_frozen(self):
        for cfg in (
            _io(),
            PCAConfig(),
            AdmixtureConfig(),
            EmbeddingConfig(),
            VizConfig(),
            SkipConfig(),
        ):
            first_field = next(iter(dataclasses.fields(cfg))).name
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(cfg, first_field, "x")

    def test_defaults_match_current_pipeline(self):
        assert PCAConfig().n_pcs == 50
        assert PCAConfig().force is False
        assert (AdmixtureConfig().k_min, AdmixtureConfig().k_max) == (2, 10)
        assert AdmixtureConfig().threads is None
        assert EmbeddingConfig().method == "phate"
        assert EmbeddingConfig().input_mode == "both"
        assert EmbeddingConfig().params == {}
        assert VizConfig().admix_within_group_order == "chron"
        assert SkipConfig().skip_pca is False

    def test_embedding_params_default_is_not_shared_between_instances(self):
        a = EmbeddingConfig()
        b = EmbeddingConfig()
        assert a.params is not b.params

    def test_pipeline_configs_container_groups_the_six(self):
        pc = PipelineConfigs(
            io=_io(),
            pca=PCAConfig(),
            admixture=AdmixtureConfig(),
            embedding=EmbeddingConfig(),
            viz=VizConfig(),
            skips=SkipConfig(),
        )
        assert isinstance(pc.io, IOConfig)
        assert isinstance(pc.skips, SkipConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_pipeline_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'manifold_genetics.pipeline.config'`

- [ ] **Step 3: Write minimal implementation**

Create `src/manifold_genetics/pipeline/config.py`:

```python
"""
Resolved, validated configuration for the manifold-genetics pipeline.

Six frozen sub-config dataclasses plus ``build_configs()``, which turns the
loose keyword arguments of ``run_pipeline`` / the ``pipeline`` CLI subcommand
into these typed objects and performs argument-shape validation (no filesystem
access).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_EMBEDDING_METHODS = ("phate", "umap", "tsne", "diffusion_map")
_EMBEDDING_INPUT_MODES = ("fit", "project", "both")


@dataclass(frozen=True)
class IOConfig:
    """Input data locations, the output directory, and per-cohort labels/colormaps.

    After ``build_configs`` runs, ``fit_*`` and ``project_*`` are always both set
    (to the shared value when the caller passed only ``labels`` / ``colormap``).
    """

    fit_plink: Path
    project_plink: Path
    output_dir: Path
    fit_labels: Path
    project_labels: Path
    fit_colormap: Path
    project_colormap: Path
    geographic_coords: Optional[Path] = None


@dataclass(frozen=True)
class PCAConfig:
    n_pcs: int = 50
    force: bool = False


@dataclass(frozen=True)
class AdmixtureConfig:
    k_min: int = 2
    k_max: int = 10
    threads: Optional[int] = None
    num_gpus: Optional[int] = None
    batch_size: Optional[int] = None


@dataclass(frozen=True)
class EmbeddingConfig:
    method: str = "phate"
    input_mode: str = "both"
    params: dict = field(default_factory=dict)  # method-specific; treated read-only


@dataclass(frozen=True)
class VizConfig:
    admix_group_column: Optional[str] = None
    admix_within_group_order: Optional[str] = "chron"
    projection_plot_fit_column: Optional[str] = None
    projection_plot_project_column: Optional[str] = None


@dataclass(frozen=True)
class SkipConfig:
    skip_pca: bool = False
    skip_admixture: bool = False
    skip_embedding: bool = False
    skip_pca_visualization: bool = False
    skip_embedding_visualization: bool = False
    skip_admixture_visualization: bool = False
    skip_metrics: bool = False


@dataclass(frozen=True)
class PipelineConfigs:
    io: IOConfig
    pca: PCAConfig
    admixture: AdmixtureConfig
    embedding: EmbeddingConfig
    viz: VizConfig
    skips: SkipConfig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_pipeline_config.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Format, lint, full fast suite**

Run:
```bash
uv run black src/manifold_genetics/pipeline/config.py tests/unit/test_pipeline_config.py
uv run isort src/manifold_genetics/pipeline/config.py tests/unit/test_pipeline_config.py
uv run flake8 src/manifold_genetics/pipeline/config.py tests/unit/test_pipeline_config.py --max-line-length=88 --extend-ignore=E501,E203,W503,F841,F541
uv run pytest -m "not slow and not network" -q
```
Expected: no lint output; suite green.

- [ ] **Step 6: Commit**

```bash
git add src/manifold_genetics/pipeline/config.py tests/unit/test_pipeline_config.py
git commit -m "feat(pipeline): add frozen sub-config dataclasses"
```

---

## Task 2: `build_configs()` factory + argument-shape validation

**Files:**
- Modify: `src/manifold_genetics/pipeline/config.py` (append `build_configs`)
- Test: `tests/unit/test_pipeline_config.py` (append a `TestBuildConfigs` class)

**Interfaces:**
- Consumes: the dataclasses from Task 1.
- Produces:
  ```python
  def build_configs(
      *,
      fit_plink, project_plink, output_dir,
      labels=None, colormap=None,
      fit_labels=None, project_labels=None, fit_colormap=None, project_colormap=None,
      geographic_coords=None,
      n_pcs=50, force_pca=False,
      k_min=2, k_max=10, admix_threads=None, admix_gpus=None, admix_batch_size=None,
      embedding="phate", embedding_input="both", embedding_params=None,
      admix_group_column=None, admix_within_group_order="chron",
      projection_plot_fit_column=None, projection_plot_project_column=None,
      skip_pca=False, skip_admixture=False, skip_embedding=False,
      skip_pca_visualization=False, skip_embedding_visualization=False,
      skip_admixture_visualization=False, skip_metrics=False,
  ) -> PipelineConfigs
  ```
  All path-like inputs may be `str` or `Path`; the returned configs hold `Path`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pipeline_config.py`:

```python
from manifold_genetics.pipeline.config import build_configs


def _required(**overrides):
    base = dict(
        fit_plink="data/fit",
        project_plink="data/project",
        output_dir="out",
        labels="labels.csv",
        colormap="cmap.json",
    )
    base.update(overrides)
    return base


class TestBuildConfigs:
    def test_shared_labels_and_colormap_fan_out_to_both_cohorts(self):
        pc = build_configs(**_required())
        assert pc.io.fit_labels == Path("labels.csv")
        assert pc.io.project_labels == Path("labels.csv")
        assert pc.io.fit_colormap == Path("cmap.json")
        assert pc.io.project_colormap == Path("cmap.json")

    def test_separate_labels_and_colormaps_pass_through(self):
        pc = build_configs(
            fit_plink="f",
            project_plink="p",
            output_dir="out",
            fit_labels="fl.csv",
            project_labels="pl.csv",
            fit_colormap="fc.json",
            project_colormap="pc.json",
        )
        assert pc.io.fit_labels == Path("fl.csv")
        assert pc.io.project_labels == Path("pl.csv")
        assert pc.io.fit_colormap == Path("fc.json")
        assert pc.io.project_colormap == Path("pc.json")

    def test_str_paths_become_path_objects(self):
        pc = build_configs(**_required())
        assert isinstance(pc.io.fit_plink, Path)
        assert isinstance(pc.io.output_dir, Path)

    def test_geographic_coords_optional(self):
        assert build_configs(**_required()).io.geographic_coords is None
        pc = build_configs(**_required(geographic_coords="geo.csv"))
        assert pc.io.geographic_coords == Path("geo.csv")

    def test_scalar_params_land_in_the_right_sub_config(self):
        pc = build_configs(
            **_required(
                n_pcs=30,
                force_pca=True,
                k_min=3,
                k_max=6,
                admix_threads=8,
                admix_gpus=1,
                admix_batch_size=400,
                embedding="umap",
                embedding_input="fit",
                embedding_params={"n_neighbors": 20},
                admix_group_column="region",
                projection_plot_fit_column="Population",
                projection_plot_project_column="ancestry",
                skip_metrics=True,
            )
        )
        assert (pc.pca.n_pcs, pc.pca.force) == (30, True)
        assert (pc.admixture.k_min, pc.admixture.k_max) == (3, 6)
        assert pc.admixture.threads == 8
        assert pc.admixture.num_gpus == 1
        assert pc.admixture.batch_size == 400
        assert pc.embedding.method == "umap"
        assert pc.embedding.input_mode == "fit"
        assert pc.embedding.params == {"n_neighbors": 20}
        assert pc.viz.admix_group_column == "region"
        assert pc.viz.projection_plot_project_column == "ancestry"
        assert pc.skips.skip_metrics is True

    def test_missing_labels_raises_with_the_current_message(self):
        with pytest.raises(ValueError, match="Must provide either 'labels'"):
            build_configs(
                fit_plink="f", project_plink="p", output_dir="out", colormap="c.json"
            )

    def test_only_one_of_fit_project_labels_raises(self):
        with pytest.raises(ValueError, match="only one of 'fit_labels'"):
            build_configs(
                fit_plink="f",
                project_plink="p",
                output_dir="out",
                fit_labels="fl.csv",
                colormap="c.json",
            )

    def test_missing_colormap_raises(self):
        with pytest.raises(ValueError, match="Must provide either 'colormap'"):
            build_configs(
                fit_plink="f", project_plink="p", output_dir="out", labels="l.csv"
            )

    def test_unknown_embedding_method_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding method: 'wavelet'"):
            build_configs(**_required(embedding="wavelet"))

    def test_unknown_embedding_input_mode_raises(self):
        with pytest.raises(ValueError, match="embedding_input"):
            build_configs(**_required(embedding_input="sideways"))

    def test_k_min_greater_than_k_max_raises(self):
        with pytest.raises(ValueError, match="k_min .* k_max"):
            build_configs(**_required(k_min=8, k_max=3))

    def test_non_positive_n_pcs_raises(self):
        with pytest.raises(ValueError, match="n_pcs"):
            build_configs(**_required(n_pcs=0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_pipeline_config.py::TestBuildConfigs -q`
Expected: FAIL — `ImportError: cannot import name 'build_configs'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/manifold_genetics/pipeline/config.py`:

```python
def _as_path(value) -> Optional[Path]:
    return None if value is None else Path(value)


def build_configs(
    *,
    fit_plink,
    project_plink,
    output_dir,
    labels=None,
    colormap=None,
    fit_labels=None,
    project_labels=None,
    fit_colormap=None,
    project_colormap=None,
    geographic_coords=None,
    n_pcs: int = 50,
    force_pca: bool = False,
    k_min: int = 2,
    k_max: int = 10,
    admix_threads: Optional[int] = None,
    admix_gpus: Optional[int] = None,
    admix_batch_size: Optional[int] = None,
    embedding: str = "phate",
    embedding_input: str = "both",
    embedding_params: Optional[dict] = None,
    admix_group_column: Optional[str] = None,
    admix_within_group_order: Optional[str] = "chron",
    projection_plot_fit_column: Optional[str] = None,
    projection_plot_project_column: Optional[str] = None,
    skip_pca: bool = False,
    skip_admixture: bool = False,
    skip_embedding: bool = False,
    skip_pca_visualization: bool = False,
    skip_embedding_visualization: bool = False,
    skip_admixture_visualization: bool = False,
    skip_metrics: bool = False,
) -> PipelineConfigs:
    """Resolve loose pipeline kwargs into validated ``PipelineConfigs``.

    Argument-shape validation only; no filesystem access.
    """
    # --- labels: shared OR both separate ---
    if labels is not None:
        eff_fit_labels = fit_labels if fit_labels is not None else labels
        eff_project_labels = project_labels if project_labels is not None else labels
    elif fit_labels is not None and project_labels is not None:
        eff_fit_labels, eff_project_labels = fit_labels, project_labels
    else:
        raise ValueError(
            "Must provide either 'labels' (used for both fit and project) OR both "
            "'fit_labels' and 'project_labels'. Providing only one of 'fit_labels' or "
            "'project_labels' without 'labels' is not allowed."
        )

    # --- colormap: shared OR both separate ---
    if colormap is not None:
        eff_fit_colormap = fit_colormap if fit_colormap is not None else colormap
        eff_project_colormap = (
            project_colormap if project_colormap is not None else colormap
        )
    elif fit_colormap is not None and project_colormap is not None:
        eff_fit_colormap, eff_project_colormap = fit_colormap, project_colormap
    else:
        raise ValueError(
            "Must provide either 'colormap' (used for both fit and project) OR both "
            "'fit_colormap' and 'project_colormap'. Providing only one of "
            "'fit_colormap' or 'project_colormap' without 'colormap' is not allowed."
        )

    # --- enum / range checks ---
    if embedding not in _EMBEDDING_METHODS:
        raise ValueError(
            f"Unknown embedding method: {embedding!r}. "
            f"Choose from: {', '.join(_EMBEDDING_METHODS)}"
        )
    if embedding_input not in _EMBEDDING_INPUT_MODES:
        raise ValueError(
            f"Unknown embedding_input: {embedding_input!r}. "
            f"Choose from: {', '.join(_EMBEDDING_INPUT_MODES)}"
        )
    if k_min > k_max:
        raise ValueError(f"k_min ({k_min}) must be <= k_max ({k_max})")
    if n_pcs <= 0:
        raise ValueError(f"n_pcs must be a positive integer, got {n_pcs}")

    io = IOConfig(
        fit_plink=Path(fit_plink),
        project_plink=Path(project_plink),
        output_dir=Path(output_dir),
        fit_labels=Path(eff_fit_labels),
        project_labels=Path(eff_project_labels),
        fit_colormap=Path(eff_fit_colormap),
        project_colormap=Path(eff_project_colormap),
        geographic_coords=_as_path(geographic_coords),
    )
    return PipelineConfigs(
        io=io,
        pca=PCAConfig(n_pcs=n_pcs, force=force_pca),
        admixture=AdmixtureConfig(
            k_min=k_min,
            k_max=k_max,
            threads=admix_threads,
            num_gpus=admix_gpus,
            batch_size=admix_batch_size,
        ),
        embedding=EmbeddingConfig(
            method=embedding,
            input_mode=embedding_input,
            params=dict(embedding_params or {}),
        ),
        viz=VizConfig(
            admix_group_column=admix_group_column,
            admix_within_group_order=admix_within_group_order,
            projection_plot_fit_column=projection_plot_fit_column,
            projection_plot_project_column=projection_plot_project_column,
        ),
        skips=SkipConfig(
            skip_pca=skip_pca,
            skip_admixture=skip_admixture,
            skip_embedding=skip_embedding,
            skip_pca_visualization=skip_pca_visualization,
            skip_embedding_visualization=skip_embedding_visualization,
            skip_admixture_visualization=skip_admixture_visualization,
            skip_metrics=skip_metrics,
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_pipeline_config.py -q`
Expected: PASS (all TestSubConfigs + TestBuildConfigs)

- [ ] **Step 5: Format, lint, full fast suite**

Run:
```bash
uv run black src/manifold_genetics/pipeline/config.py tests/unit/test_pipeline_config.py
uv run isort src/manifold_genetics/pipeline/config.py tests/unit/test_pipeline_config.py
uv run flake8 src/manifold_genetics/pipeline/config.py tests/unit/test_pipeline_config.py --max-line-length=88 --extend-ignore=E501,E203,W503,F841,F541
uv run pytest -m "not slow and not network" -q
```
Expected: no lint output; suite green.

- [ ] **Step 6: Commit**

```bash
git add src/manifold_genetics/pipeline/config.py tests/unit/test_pipeline_config.py
git commit -m "feat(pipeline): add build_configs() with argument-shape validation"
```

---

## Task 3: pure output-path helpers

**Files:**
- Create: `src/manifold_genetics/pipeline/steps/__init__.py`
- Create: `src/manifold_genetics/pipeline/steps/paths.py`
- Test: `tests/unit/test_step_paths.py`

**Interfaces:**
- Consumes: `IOConfig`, `PCAConfig`, `AdmixtureConfig`, `EmbeddingConfig` from Task 1.
- Produces:
  - `pca_output_paths(io: IOConfig, pca: PCAConfig) -> dict[str, Path]` — keys `"fit_pca"`, `"project_pca"`, `"flashpca_dir"`.
  - `admixture_output_paths(io: IOConfig, admix: AdmixtureConfig) -> dict[str, Path]` — keys `"dir"`, `"checkpoints_dir"`, `"fit_prefix"`, `"project_prefix"`. Per-K CSVs are `<prefix>.parent / f"{<prefix>.name}.{k}.csv"`.
  - `embedding_output_paths(io: IOConfig, emb: EmbeddingConfig) -> dict[str, Path]` — key `"embedding"` always; key `"fit_embedding"` only when `emb.input_mode == "both"`.
  - `metrics_output_paths(io: IOConfig) -> dict[str, Path]` — keys `"geographic"`, `"admixture"`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_step_paths.py`:

```python
"""Tests for the pure output-path helpers.

These paths are a contract (spec constraint B): every downstream example script
and the pipeline's own checkpoint logic depend on them. If a path here changes,
that is a deliberate, breaking decision.
"""

from pathlib import Path

from manifold_genetics.pipeline.config import (
    AdmixtureConfig,
    EmbeddingConfig,
    IOConfig,
    PCAConfig,
)
from manifold_genetics.pipeline.steps.paths import (
    admixture_output_paths,
    embedding_output_paths,
    metrics_output_paths,
    pca_output_paths,
)

OUT = Path("/work/results")


def _io():
    return IOConfig(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=OUT,
        fit_labels=Path("fl.csv"),
        project_labels=Path("pl.csv"),
        fit_colormap=Path("fc.json"),
        project_colormap=Path("pc.json"),
    )


def test_pca_paths_match_documented_layout():
    p = pca_output_paths(_io(), PCAConfig(n_pcs=50))
    assert p["fit_pca"] == OUT / "pca" / "fit_pca_50.csv"
    assert p["project_pca"] == OUT / "pca" / "project_pca_50.csv"
    assert p["flashpca_dir"] == OUT / "pca" / "flashpca_outputs"


def test_pca_paths_track_n_pcs():
    assert pca_output_paths(_io(), PCAConfig(n_pcs=20))["project_pca"] == (
        OUT / "pca" / "project_pca_20.csv"
    )


def test_admixture_paths_match_documented_layout():
    p = admixture_output_paths(_io(), AdmixtureConfig(k_min=2, k_max=5))
    assert p["dir"] == OUT / "admixture"
    assert p["checkpoints_dir"] == OUT / "admixture" / "checkpoints"
    assert p["fit_prefix"] == OUT / "admixture" / "fit"
    assert p["project_prefix"] == OUT / "admixture" / "project"


def test_admixture_per_k_csv_derives_from_project_prefix():
    p = admixture_output_paths(_io(), AdmixtureConfig(k_min=2, k_max=3))
    prefix = p["project_prefix"]
    k3 = prefix.parent / f"{prefix.name}.3.csv"
    assert k3 == OUT / "admixture" / "project.3.csv"


def test_embedding_paths_both_mode_has_fit_embedding():
    p = embedding_output_paths(_io(), EmbeddingConfig(method="phate", input_mode="both"))
    assert p["embedding"] == OUT / "embeddings" / "phate_2d.csv"
    assert p["fit_embedding"] == OUT / "embeddings" / "phate_fit_2d.csv"


def test_embedding_paths_single_mode_has_no_fit_embedding():
    for mode in ("fit", "project"):
        p = embedding_output_paths(
            _io(), EmbeddingConfig(method="umap", input_mode=mode)
        )
        assert p["embedding"] == OUT / "embeddings" / "umap_2d.csv"
        assert "fit_embedding" not in p


def test_metrics_paths_match_documented_layout():
    p = metrics_output_paths(_io())
    assert p["geographic"] == OUT / "metrics" / "geographic.json"
    assert p["admixture"] == OUT / "metrics" / "admixture.json"


def test_helpers_do_no_io(tmp_path):
    io = IOConfig(
        fit_plink=Path("data/fit"),
        project_plink=Path("data/project"),
        output_dir=tmp_path / "never_created",
        fit_labels=Path("fl.csv"),
        project_labels=Path("pl.csv"),
        fit_colormap=Path("fc.json"),
        project_colormap=Path("pc.json"),
    )
    pca_output_paths(io, PCAConfig())
    admixture_output_paths(io, AdmixtureConfig())
    embedding_output_paths(io, EmbeddingConfig())
    metrics_output_paths(io)
    assert not (tmp_path / "never_created").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_step_paths.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'manifold_genetics.pipeline.steps'`

- [ ] **Step 3: Write minimal implementation**

Create `src/manifold_genetics/pipeline/steps/__init__.py`:

```python
"""In-process pipeline steps.

Each compute step lives in its own module and exposes a pure
``<step>_output_paths()`` helper plus (added in later PRs) a ``run_<step>_step()``
function. Both the ``manifold-genetics`` CLI subcommands and the pipeline
orchestrator call these directly, in-process.
"""
```

Create `src/manifold_genetics/pipeline/steps/paths.py`:

```python
"""Pure output-path helpers — the single source of truth for where each pipeline
step writes. No I/O; deterministic from the config arguments.

These paths are a contract: downstream example scripts and the pipeline's
checkpoint logic depend on them (spec constraint B).
"""

from pathlib import Path
from typing import Dict

from ..config import AdmixtureConfig, EmbeddingConfig, IOConfig, PCAConfig


def pca_output_paths(io: IOConfig, pca: PCAConfig) -> Dict[str, Path]:
    d = io.output_dir / "pca"
    return {
        "fit_pca": d / f"fit_pca_{pca.n_pcs}.csv",
        "project_pca": d / f"project_pca_{pca.n_pcs}.csv",
        "flashpca_dir": d / "flashpca_outputs",
    }


def admixture_output_paths(io: IOConfig, admix: AdmixtureConfig) -> Dict[str, Path]:
    d = io.output_dir / "admixture"
    return {
        "dir": d,
        "checkpoints_dir": d / "checkpoints",
        "fit_prefix": d / "fit",
        "project_prefix": d / "project",
    }


def embedding_output_paths(io: IOConfig, emb: EmbeddingConfig) -> Dict[str, Path]:
    d = io.output_dir / "embeddings"
    paths = {"embedding": d / f"{emb.method}_2d.csv"}
    if emb.input_mode == "both":
        paths["fit_embedding"] = d / f"{emb.method}_fit_2d.csv"
    return paths


def metrics_output_paths(io: IOConfig) -> Dict[str, Path]:
    d = io.output_dir / "metrics"
    return {"geographic": d / "geographic.json", "admixture": d / "admixture.json"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_step_paths.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Format, lint, full fast suite**

Run:
```bash
uv run black src/manifold_genetics/pipeline/steps/ tests/unit/test_step_paths.py
uv run isort src/manifold_genetics/pipeline/steps/ tests/unit/test_step_paths.py
uv run flake8 src/manifold_genetics/pipeline/steps/ tests/unit/test_step_paths.py --max-line-length=88 --extend-ignore=E501,E203,W503,F841,F541
uv run pytest -m "not slow and not network" -q
```
Expected: no lint output; suite green.

- [ ] **Step 6: Commit**

```bash
git add src/manifold_genetics/pipeline/steps/ tests/unit/test_step_paths.py
git commit -m "feat(pipeline): add pure output-path helpers for compute steps"
```

---

## Task 4: cross-cohort test fixtures

**Files:**
- Create: `tests/fixtures/cross_cohort/fit_labels.csv`
- Create: `tests/fixtures/cross_cohort/project_labels.csv`
- Create: `tests/fixtures/cross_cohort/fit_colormap.json`
- Create: `tests/fixtures/cross_cohort/project_colormap.json`
- Create: `tests/fixtures/cross_cohort/geographic.csv`
- Create: `tests/fixtures/cross_cohort/README.md`
- Modify: `tests/conftest.py` (add `cross_cohort_fixtures` fixture)
- Test: `tests/unit/test_step_paths.py` (append `TestCrossCohortFixtures`)

**Why:** the PR 5 contract test runs the pipeline in all three example modes
(`projection` / `subsample` / `transform`). `projection` mode is genuinely
cross-cohort — different labels *and* colormaps for fit vs project, and a
geographic file for the metrics step. These fixtures exist so PR 5 can rely on
them; this task only checks they are well-formed.

The sample IDs must match the existing admixture fixtures
(`tests/fixtures/admixture/{fit,project}.{2,3}.csv`), which use `SAMPLE_000` …
`SAMPLE_049` — 50 samples each (see `tests/integration/test_integration.py`).

**Interfaces:**
- Produces: `cross_cohort_fixtures` pytest fixture → `dict[str, Path]` with keys
  `fit_labels`, `project_labels`, `fit_colormap`, `project_colormap`, `geographic`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_step_paths.py`:

```python
import json

import pandas as pd


class TestCrossCohortFixtures:
    def test_fixture_provides_five_readable_paths(self, cross_cohort_fixtures):
        assert set(cross_cohort_fixtures) == {
            "fit_labels",
            "project_labels",
            "fit_colormap",
            "project_colormap",
            "geographic",
        }
        for p in cross_cohort_fixtures.values():
            assert p.exists(), p

    def test_labels_have_sample_id_and_distinct_group_columns(self, cross_cohort_fixtures):
        fit = pd.read_csv(cross_cohort_fixtures["fit_labels"])
        proj = pd.read_csv(cross_cohort_fixtures["project_labels"])
        assert "sample_id" in fit.columns and "sample_id" in proj.columns
        # fit uses "Population"; project uses "self_described_ancestry" — mirrors the
        # HGDP -> UKBB projection example.
        assert "Population" in fit.columns
        assert "self_described_ancestry" in proj.columns
        assert len(fit) == 50 and len(proj) == 50

    def test_colormaps_key_on_their_cohort_label_column(self, cross_cohort_fixtures):
        fit_cmap = json.loads(cross_cohort_fixtures["fit_colormap"].read_text())
        proj_cmap = json.loads(cross_cohort_fixtures["project_colormap"].read_text())
        assert "Population" in fit_cmap
        assert "self_described_ancestry" in proj_cmap

    def test_geographic_has_coords_for_project_samples(self, cross_cohort_fixtures):
        geo = pd.read_csv(cross_cohort_fixtures["geographic"])
        assert {"sample_id", "latitude", "longitude"} <= set(geo.columns)
        assert len(geo) == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_step_paths.py::TestCrossCohortFixtures -q`
Expected: FAIL — `fixture 'cross_cohort_fixtures' not found`

- [ ] **Step 3: Create the fixture data files**

Generate the five files with this one-off script (run from repo root, then delete the script — it is not committed):

```python
# scratch_gen_fixtures.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

d = Path("tests/fixtures/cross_cohort")
d.mkdir(parents=True, exist_ok=True)
ids = [f"SAMPLE_{i:03d}" for i in range(50)]
rng = np.random.default_rng(0)

fit_pops = ["FrenchBasque", "Yoruba", "Han", "Karitiana", "Papuan"]
pd.DataFrame(
    {"sample_id": ids, "Population": [fit_pops[i % len(fit_pops)] for i in range(50)]}
).to_csv(d / "fit_labels.csv", index=False)

proj_anc = ["European", "African", "East Asian", "South Asian", "Admixed"]
pd.DataFrame(
    {
        "sample_id": ids,
        "self_described_ancestry": [proj_anc[i % len(proj_anc)] for i in range(50)],
    }
).to_csv(d / "project_labels.csv", index=False)

(d / "fit_colormap.json").write_text(
    json.dumps(
        {
            "Population": {
                "FrenchBasque": "#1f77b4",
                "Yoruba": "#ff7f0e",
                "Han": "#2ca02c",
                "Karitiana": "#d62728",
                "Papuan": "#9467bd",
            }
        },
        indent=2,
    )
)
(d / "project_colormap.json").write_text(
    json.dumps(
        {
            "self_described_ancestry": {
                "European": "#1f77b4",
                "African": "#ff7f0e",
                "East Asian": "#2ca02c",
                "South Asian": "#d62728",
                "Admixed": "#7f7f7f",
            }
        },
        indent=2,
    )
)

pd.DataFrame(
    {
        "sample_id": ids,
        "latitude": rng.uniform(-55, 70, 50).round(4),
        "longitude": rng.uniform(-160, 175, 50).round(4),
    }
).to_csv(d / "geographic.csv", index=False)
print("wrote", *sorted(p.name for p in d.iterdir()))
```

Run: `uv run python scratch_gen_fixtures.py && rm scratch_gen_fixtures.py`

Then create `tests/fixtures/cross_cohort/README.md`:

```markdown
# Cross-cohort test fixtures

50-sample fit/project label + colormap + geographic files for the parametrized
pipeline contract test (`test_pipeline_output_layout`, PR 5). Sample IDs
(`SAMPLE_000`…`SAMPLE_049`) match `tests/fixtures/admixture/`.

- `fit_labels.csv` — `sample_id`, `Population` (HGDP-style)
- `project_labels.csv` — `sample_id`, `self_described_ancestry` (UKBB-style)
- `fit_colormap.json` / `project_colormap.json` — keyed on the above columns
- `geographic.csv` — `sample_id`, `latitude`, `longitude`

Regenerate with the throwaway script in the PR-1 plan (Task 4, Step 3).
```

- [ ] **Step 4: Add the conftest fixture**

In `tests/conftest.py`, add (near the other fixture definitions):

```python
@pytest.fixture
def cross_cohort_fixtures():
    """Paths to the checked-in cross-cohort label/colormap/geographic fixtures."""
    d = Path(__file__).parent / "fixtures" / "cross_cohort"
    return {
        "fit_labels": d / "fit_labels.csv",
        "project_labels": d / "project_labels.csv",
        "fit_colormap": d / "fit_colormap.json",
        "project_colormap": d / "project_colormap.json",
        "geographic": d / "geographic.csv",
    }
```

(`Path` is already imported in `conftest.py`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_step_paths.py::TestCrossCohortFixtures -q`
Expected: PASS (4 tests)

- [ ] **Step 6: Format, lint, full fast suite**

Run:
```bash
uv run black tests/conftest.py tests/unit/test_step_paths.py
uv run isort tests/conftest.py tests/unit/test_step_paths.py
uv run flake8 tests/conftest.py tests/unit/test_step_paths.py --max-line-length=88 --extend-ignore=E501,E203,W503,F841,F541
uv run pytest -m "not slow and not network" -q
```
Expected: no lint output; suite green.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/cross_cohort/ tests/conftest.py tests/unit/test_step_paths.py
git commit -m "test: add cross-cohort fixtures for the pipeline contract test"
```

---

## Task 5: open the PR

- [ ] **Step 1: Push and open a draft PR**

```bash
git push -u origin <branch>
~/bin/gh pr create --draft --base main \
  --title "refactor(pipeline): step-layer foundation — configs + path helpers" \
  --body "PR 1 of the pipeline step-layer refactor (spec: docs/superpowers/specs/2026-08-30-pipeline-step-layer-design.md).

Pure addition, zero behaviour change:
- \`pipeline/config.py\` — six frozen sub-config dataclasses + \`build_configs()\` (argument-shape validation).
- \`pipeline/steps/paths.py\` — pure \`<step>_output_paths()\` helpers; single source of truth for the output-path contract (spec constraint B).
- \`tests/fixtures/cross_cohort/\` — label/colormap/geographic fixtures for the parametrized contract test in PR 5.

Nothing in \`orchestrator.py\` / \`cli.py\` / \`runner.py\` changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 2: Confirm CI is green**, then leave the PR for review.

---

## Subsequent PRs (own plan each, written when its predecessor merges)

Full bite-sized plans are deferred so each can reference the real signatures the previous PR landed.

- **PR 2 — PCA + metrics steps.** `steps/pca.py` (`run_pca_step(io, pca) -> PCAStepResult`, idempotent, folds in the dim-count force-recompute check) and `steps/metrics.py` (`run_geographic_metrics_step(embedding_csv, geo_csv, out, **opts)`, `run_admixture_metrics_step(embedding_csv, q_prefix, k_range, out, **opts)`). Replace the three matching `subprocess.run` calls in `Pipeline.run()`; point `cmd_pca` / `cmd_metrics_geographic` / `cmd_metrics_admixture` at the step functions. Rewrite `test_orchestrator.py` PCA/metrics argv-assertion tests as step-behaviour tests.
- **PR 3 — embedding step.** `steps/embedding.py` (`run_embedding_step(io, emb, *, pca: PCAStepResult) -> EmbeddingStepResult`) carrying the three `--embedding-input` modes. Replace the `embed` subprocess call; `cmd_embed` → step fn. Port the mode-mapping tests.
- **PR 4 — admixture step.** `steps/admixture.py` (`run_admixture_step(io, admix, *, backend=None) -> AdmixtureStepResult`). Replace the last compute subprocess call **and** delete the special-case backend branch in `Pipeline.run()` (unified — the step always goes through a backend, default `NeuralAdmixtureBackend`). `cmd_admixture` → step fn. PR body records the verified process-isolation finding (only in-process torch use is `cuda.is_available()`).
- **PR 5 — viz steps + orchestrator cutover.** `steps/viz.py` × 4 + their `*_output_paths` (these take the loaded colormap dict, since figure names depend on its columns) + `_run_viz` non-fatal wrapper. `Pipeline.run()` becomes the explicit ~130-line block sequence; delete `import subprocess`, argv builders, `_get_embedding_model`. `run_pipeline` returns `PipelineResult`; update `cmd_pipeline`'s summary + the two integration tests. Add `test_pipeline_output_layout[projection|subsample|transform]` (uses the Task-4 fixtures) and `test_viz_failure_records_and_continues`. **Highest-review PR.**

  > Bridge the rename at the `run_pipeline` boundary: `run_pipeline(skip_visualization=…)`
  > is a public kwarg with ~8 live call sites (`tests/unit/test_orchestrator.py`,
  > `tests/integration/test_generic_pipeline.py`) — map it to
  > `build_configs(skip_embedding_visualization=…)`, keeping the public kwarg name.
- **PR 6 (optional) — cleanup.** Drop any transitional shim, dead helpers, doc/README touch-ups.

---

## Self-Review

**Spec coverage (PR 1 scope only):**
- Sub-config dataclasses (spec "Config" section) → Task 1. ✓
- `build_configs()` validation consolidation → Task 2 (argument-shape subset; file-content validation explicitly deferred per Global Constraints). ✓
- `*_output_paths()` for compute steps (spec "L2 — step layer") → Task 3. Viz path helpers deferred to PR 5 because they need the colormap dict (noted in Subsequent PRs). ✓
- Cross-cohort fixtures for T2 (spec "Known design tensions", "Risks") → Task 4. ✓
- "Nothing in orchestrator/cli/runner changes" (spec migration table PR 1) → enforced by Global Constraints; no task touches them. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step has a full code block. The `**opts` in the deferred PR-2 metrics signatures is a forward reference in the non-bite-sized "Subsequent PRs" section, not a plan step. ✓

**Type consistency:**
- `IOConfig` field names identical across Task 1 (definition), Task 2 (`build_configs` constructs it), Task 3 (`_io()` helper), Task 4 (conftest fixture keys mirror the label/colormap names). ✓
- `pca_output_paths` keys `"fit_pca"`/`"project_pca"`/`"flashpca_dir"` — same in Task 3 impl and test. ✓
- `admixture_output_paths` keys `"dir"`/`"checkpoints_dir"`/`"fit_prefix"`/`"project_prefix"` — same in impl and test; per-K derivation shown identically in the interface block and the test. ✓
- `embedding_output_paths` `"fit_embedding"` present only for `input_mode == "both"` — impl and both tests agree. ✓
- `EmbeddingConfig.params` default: `field(default_factory=dict)` in Task 1, asserted `== {}` and "not shared between instances" in Task 1 test, `dict(embedding_params or {})` in Task 2. Consistent. ✓
