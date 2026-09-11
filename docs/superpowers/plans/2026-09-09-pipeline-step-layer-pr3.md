# Pipeline Step Layer — PR 3: embedding step

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the embedding stage out of `subprocess.run(["manifold-genetics", "embed", ...])` and into an in-process step function that both `Pipeline.run()` and `cmd_embed` call.

**Architecture:** Add `src/manifold_genetics/pipeline/steps/embedding.py` exposing three things: `build_embedding_model(method, params)` — the single place a method string plus a params dict becomes an `EmbeddingBase` instance; `run_embedding(...)` — the seam shared by the CLI handler and the step, reproducing today's fit/transform call sequence exactly; and `run_embedding_step(io, emb, *, pca) -> EmbeddingStepResult`, which maps the three `--embedding-input` modes onto the PCA outputs and writes to the documented paths. `cmd_embed` and the orchestrator are then both pointed at it, and the duplicated argparse→params assembly in `cmd_embed` / `cmd_pipeline` is collapsed into one helper.

**Tech Stack:** Python 3.10–3.12, pandas, pytest, `uv`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-30-pipeline-step-layer-design.md` (migration table row 3). PR 1 = #68, PR 2 = #70, both merged; `main` is at `ae190f6`.

## Global Constraints

- **A — The CLI surface is frozen.** No flag renames, removals, or additions to any `add_parser` / `add_argument` call in `src/manifold_genetics/cli.py`. Nothing below `main()` may change.
- **B — Output paths are a contract.** The embedding stage writes `output_dir/embeddings/{method}_2d.csv`, and in `both` mode also `output_dir/embeddings/{method}_fit_2d.csv`. These come from `embedding_output_paths()` in `src/manifold_genetics/pipeline/steps/paths.py` and nowhere else.
- **C — With a step skipped, `run()` still supplies its output paths downstream.** The embedding stage's existing `RuntimeError` when no PCA files are present must keep firing, with the same message. Do not add new skip-time error behaviour — that is PR 5.
- **D — Compute steps are fatal.** No try/except around the embedding work.
- **F — `project` naming everywhere; no dead params.**
- **Behaviour-preserving.** `run_pipeline()` still returns the same loose `results` dict with the same keys: `embedding_file`, `embedding_coords`, and `fit_embedding_file` (present only in `both` mode). `PipelineResult` is PR 5.
- **Out of scope, do not touch:** the admixture stage of `run()` (PR 4), the four visualization blocks and `_get_embedding_model` (PR 5), anything under `src/manifold_genetics/embeddings/`.
- **Style.** `black` (line length 100) and `isort` (profile black) are the binding gates. `flake8` is NOT a working gate in this repo — `setup.cfg` shadows `.flake8`, and `main` carries ~573 violations, most of them E501 produced by black's own 100-column formatting (tracked in issue #71). Ignore E501; never edit `.flake8` or `setup.cfg`.
- **Tests.** `uv run pytest -m "not slow and not network"` green at the end of every task. Baseline on `main` @ `ae190f6`: **457 passed, 7 skipped, 7 deselected**. The login node is slow and shared — run on a compute node:
  `srun --jobid=<JOBID> --overlap --ntasks=1 --time=40:00 bash -c 'cd <repo> && source .venv/bin/activate && pytest -m "not slow and not network" -q -p no:cacheprovider'`

## Two traps this PR exists to avoid

Both are pinned by tests in Task 1. Read this section before writing any code.

**Trap 1 — `n_landmark` must be passed explicitly.** `PHATE.__init__` declares `n_landmark: Optional[int] = 2000`. Today's real pipeline path goes orchestrator → argv → `cmd_embed`, and `cmd_embed` *always* passes `n_landmark=` (as `None` when the user did not set it), which disables landmarking. The orchestrator's other helper, `_get_embedding_model`, instead does `defaults.update(params)` and would leave PHATE's own `2000` in place. **Reproduce `cmd_embed`'s behaviour, not `_get_embedding_model`'s.** `_get_embedding_model` is dead code that PR 5 deletes; do not use it as a reference.

**Trap 2 — modes `fit` and `project` are not `fit_transform`.** When the orchestrator omits `--project-input`, `cmd_embed` sets `project_input = fit_input` and then calls `model.fit(X)` followed by `model.transform(X)` on the same data. It does not call `fit_transform`. Reproduce fit-then-transform.

## File Structure

**Create:**
- `src/manifold_genetics/pipeline/steps/embedding.py` — `EmbeddingStepResult`, `build_embedding_model()`, `run_embedding()`, `run_embedding_step()`.
- `tests/unit/test_step_embedding.py`

**Modify:**
- `src/manifold_genetics/pipeline/steps/__init__.py` — re-export the new names.
- `src/manifold_genetics/cli.py` — new `_embedding_params_from_args()` helper; `cmd_embed` routed through the seam; `cmd_pipeline` routed through the helper. **No `main()` changes.**
- `src/manifold_genetics/pipeline/orchestrator.py` — replace the embed argv builder + `subprocess.run` with `run_embedding_step()`.
- `tests/unit/test_cli_main.py`, `tests/test_cli.py` — repoint the embedding-model monkeypatches.
- `tests/unit/test_orchestrator.py` — port `TestPipelineRunEmbeddingInputMode` from argv assertions to step-behaviour assertions.

---

### Task 1: `steps/embedding.py` — model builder, seam, and step

**Files:**
- Create: `src/manifold_genetics/pipeline/steps/embedding.py`
- Create: `tests/unit/test_step_embedding.py`
- Modify: `src/manifold_genetics/pipeline/steps/__init__.py`

**Interfaces:**
- Consumes: `IOConfig`, `EmbeddingConfig` from `manifold_genetics.pipeline.config` (`EmbeddingConfig` has `.method`, `.input_mode`, `.params`); `embedding_output_paths(io, emb) -> Dict[str, Path]` from `steps.paths` (key `"embedding"` always, `"fit_embedding"` only when `input_mode == "both"`); `PCAStepResult` from `steps.pca` (fields `.fit_pca`, `.project_pca`); `PHATE`, `UMAP`, `TSNE`, `DiffusionMap` from `manifold_genetics.embeddings`, each with `fit(X)` and `transform(X) -> pd.DataFrame`.
- Produces:
  - `EmbeddingStepResult(embedding_file: Path, fit_embedding_file: Path | None = None, coords_df: pd.DataFrame | None = None, skipped: bool = False)` — frozen dataclass, `coords_df` declared `compare=False`.
  - `build_embedding_model(method: str, params: Mapping) -> EmbeddingBase`
  - `run_embedding(fit_input, project_input=None, *, project_output, fit_output=None, method="phate", params=None) -> pd.DataFrame`
  - `run_embedding_step(io: IOConfig, emb: EmbeddingConfig, *, pca: PCAStepResult) -> EmbeddingStepResult`

Pure addition — nothing is wired up in this task.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_step_embedding.py`:

```python
"""Tests for the in-process embedding step (pipeline/steps/embedding.py).

This step replaces a `subprocess.run(["manifold-genetics", "embed", ...])` call,
so these assert behaviour — which CSV is fitted, which is transformed, which
file each result lands in, and exactly which kwargs reach the model — rather
than an argv list.

Two behaviours are load-bearing and pinned below:
  * `n_landmark` is always passed to PHATE, as None when unset, because PHATE's
    own default is 2000 and the CLI path has always overridden it.
  * Modes 'fit' and 'project' call fit() then transform() on the same input;
    they do not call fit_transform().
"""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.config import EmbeddingConfig, IOConfig
from manifold_genetics.pipeline.steps.embedding import (
    EmbeddingStepResult,
    build_embedding_model,
    run_embedding,
    run_embedding_step,
)
from manifold_genetics.pipeline.steps.paths import embedding_output_paths
from manifold_genetics.pipeline.steps.pca import PCAStepResult

MODULE = "manifold_genetics.pipeline.steps.embedding"


def coords(n_rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n_rows)],
            "dim_1": [0.1] * n_rows,
            "dim_2": [0.2] * n_rows,
        }
    )


class FakeEmbed:
    """Stand-in for PHATE/UMAP/TSNE/DiffusionMap. Records construction and calls."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeEmbed.instances.append(self)

    def fit(self, X):
        self.calls.append(("fit", str(X)))
        return self

    def transform(self, X):
        self.calls.append(("transform", str(X)))
        return coords()

    def fit_transform(self, *a, **k):
        raise AssertionError("fit_transform must not be used by the embedding step")


@pytest.fixture
def fake_models(monkeypatch):
    FakeEmbed.instances = []
    for name in ("PHATE", "UMAP", "TSNE", "DiffusionMap"):
        monkeypatch.setattr(f"{MODULE}.{name}", FakeEmbed)
    return FakeEmbed


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


def make_pca_result(tmp_path) -> PCAStepResult:
    d = tmp_path / "out" / "pca"
    d.mkdir(parents=True, exist_ok=True)
    fit_pca, project_pca = d / "fit_pca_10.csv", d / "project_pca_10.csv"
    coords().to_csv(fit_pca, index=False)
    coords().to_csv(project_pca, index=False)
    return PCAStepResult(fit_pca=fit_pca, project_pca=project_pca)


# ---------------------------------------------------------------------------
# build_embedding_model — method string + params -> model instance
# ---------------------------------------------------------------------------


class TestBuildEmbeddingModel:
    """The single place a method name becomes a model. If the kwargs it passes
    drift from what the CLI used to pass, every embedding silently changes."""

    def test_phate_always_passes_n_landmark_even_when_absent_from_params(self, fake_models):
        """PHATE's own default is n_landmark=2000. The CLI path has always passed
        n_landmark explicitly (None when unset), disabling landmarking. Omitting
        it here would silently switch landmarking on for every pipeline run."""
        build_embedding_model("phate", {"knn": 25})

        (inst,) = fake_models.instances
        assert "n_landmark" in inst.kwargs, "n_landmark must be passed explicitly"
        assert inst.kwargs["n_landmark"] is None

    def test_phate_forwards_all_supported_params(self, fake_models):
        build_embedding_model(
            "phate",
            {
                "knn": 7,
                "t": 3,
                "n_landmark": 200,
                "random_landmarking": True,
                "embed_batch_size": 64,
            },
        )
        k = fake_models.instances[0].kwargs
        assert k["n_components"] == 2
        assert k["knn"] == 7
        assert k["t"] == 3
        assert k["n_landmark"] == 200
        assert k["random_landmarking"] is True
        assert k["embed_batch_size"] == 64

    def test_phate_defaults_match_the_cli_defaults(self, fake_models):
        """Drift here changes results between `manifold-genetics embed` and the pipeline."""
        build_embedding_model("phate", {})
        k = fake_models.instances[0].kwargs
        assert k["knn"] == 25
        assert k["t"] == "auto"
        assert k["n_landmark"] is None
        assert k["random_landmarking"] is False
        assert k["embed_batch_size"] is None

    def test_umap_params_and_defaults(self, fake_models):
        build_embedding_model("umap", {})
        k = fake_models.instances[0].kwargs
        assert k["n_components"] == 2
        assert k["n_neighbors"] == 15
        assert k["min_dist"] == 0.1

        build_embedding_model("umap", {"n_neighbors": 40, "min_dist": 0.3})
        k = fake_models.instances[1].kwargs
        assert k["n_neighbors"] == 40
        assert k["min_dist"] == 0.3

    def test_tsne_params_and_defaults(self, fake_models):
        build_embedding_model("tsne", {})
        assert fake_models.instances[0].kwargs["perplexity"] == 30
        build_embedding_model("tsne", {"perplexity": 12})
        assert fake_models.instances[1].kwargs["perplexity"] == 12

    def test_diffusion_map_params_and_defaults(self, fake_models):
        build_embedding_model("diffusion_map", {})
        assert fake_models.instances[0].kwargs["knn"] == 25
        build_embedding_model("diffusion_map", {"knn": 9})
        assert fake_models.instances[1].kwargs["knn"] == 9

    def test_unknown_method_raises_valueerror(self, fake_models):
        with pytest.raises(ValueError, match="[Uu]nknown"):
            build_embedding_model("wavelet", {})

    def test_none_params_treated_as_empty(self, fake_models):
        build_embedding_model("tsne", None)
        assert fake_models.instances[0].kwargs["perplexity"] == 30


# ---------------------------------------------------------------------------
# run_embedding — the seam shared with `manifold-genetics embed`
# ---------------------------------------------------------------------------


class TestRunEmbedding:
    def test_fits_on_fit_input_and_transforms_project_input(self, tmp_path, fake_models):
        out = tmp_path / "emb.csv"
        run_embedding("fit.csv", "proj.csv", project_output=out, method="umap")

        (inst,) = fake_models.instances
        assert inst.calls == [("fit", "fit.csv"), ("transform", "proj.csv")]
        assert out.exists()

    def test_writes_fit_output_when_requested(self, tmp_path, fake_models):
        fit_out, proj_out = tmp_path / "fit.csv", tmp_path / "proj.csv"
        run_embedding(
            "fit_in.csv", "proj_in.csv", fit_output=fit_out, project_output=proj_out, method="umap"
        )

        (inst,) = fake_models.instances
        assert inst.calls == [
            ("fit", "fit_in.csv"),
            ("transform", "fit_in.csv"),
            ("transform", "proj_in.csv"),
        ]
        assert fit_out.exists() and proj_out.exists()

    def test_absent_project_input_reuses_fit_input(self, tmp_path, fake_models):
        """Modes 'fit'/'project': fit and transform the SAME csv — not fit_transform.
        FakeEmbed.fit_transform raises, so a regression to fit_transform fails here."""
        out = tmp_path / "emb.csv"
        run_embedding("only.csv", None, project_output=out, method="umap")

        (inst,) = fake_models.instances
        assert inst.calls == [("fit", "only.csv"), ("transform", "only.csv")]

    def test_creates_parent_directories(self, tmp_path, fake_models):
        out = tmp_path / "nested" / "deeper" / "emb.csv"
        run_embedding("a.csv", None, project_output=out, method="umap")
        assert out.exists()

    def test_returns_the_projected_coordinates(self, tmp_path, fake_models):
        df = run_embedding("a.csv", None, project_output=tmp_path / "e.csv", method="umap")
        assert list(df.columns) == ["sample_id", "dim_1", "dim_2"]

    def test_params_reach_the_model(self, tmp_path, fake_models):
        run_embedding(
            "a.csv", None, project_output=tmp_path / "e.csv", method="tsne", params={"perplexity": 5}
        )
        assert fake_models.instances[0].kwargs["perplexity"] == 5


# ---------------------------------------------------------------------------
# run_embedding_step — the three --embedding-input modes
# ---------------------------------------------------------------------------


class TestRunEmbeddingStep:
    """embedding_input='fit'|'project'|'both' decides which PCA file is fitted and
    which is projected. Getting it wrong embeds the wrong cohort, or embeds the
    project cohort with its own geometry instead of the fit cohort's."""

    def test_mode_both_fits_fit_pca_and_projects_project_pca(self, tmp_path, fake_models):
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="both", params={"knn": 5})

        result = run_embedding_step(io, emb, pca=pca)

        (inst,) = fake_models.instances
        assert inst.calls == [
            ("fit", str(pca.fit_pca)),
            ("transform", str(pca.fit_pca)),
            ("transform", str(pca.project_pca)),
        ]
        paths = embedding_output_paths(io, emb)
        assert result.embedding_file == paths["embedding"]
        assert result.fit_embedding_file == paths["fit_embedding"]
        assert result.embedding_file.exists()
        assert result.fit_embedding_file.exists()

    def test_mode_fit_uses_fit_pca_only(self, tmp_path, fake_models):
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="fit", params={"knn": 5})

        result = run_embedding_step(io, emb, pca=pca)

        (inst,) = fake_models.instances
        assert inst.calls == [("fit", str(pca.fit_pca)), ("transform", str(pca.fit_pca))]
        assert result.fit_embedding_file is None, "no separate fit embedding in single-input modes"
        assert result.embedding_file.exists()

    def test_mode_project_uses_project_pca_only(self, tmp_path, fake_models):
        """The project PCA must be the FITTED input here. Passing the fit PCA
        instead would embed project samples using fit-cohort geometry."""
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="project", params={"knn": 5})

        result = run_embedding_step(io, emb, pca=pca)

        (inst,) = fake_models.instances
        assert inst.calls == [("fit", str(pca.project_pca)), ("transform", str(pca.project_pca))]
        assert result.fit_embedding_file is None
        assert result.embedding_file.exists()

    def test_writes_the_documented_output_layout(self, tmp_path, fake_models):
        """Spec constraint B: downstream example scripts read these exact paths."""
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="umap", input_mode="both", params={})

        run_embedding_step(io, emb, pca=pca)

        d = io.output_dir / "embeddings"
        assert (d / "umap_2d.csv").exists()
        assert (d / "umap_fit_2d.csv").exists()

    def test_result_carries_coords_and_is_not_skipped(self, tmp_path, fake_models):
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="umap", input_mode="fit", params={})

        result = run_embedding_step(io, emb, pca=pca)

        assert isinstance(result, EmbeddingStepResult)
        assert isinstance(result.coords_df, pd.DataFrame)
        assert result.skipped is False

    def test_config_params_reach_the_model(self, tmp_path, fake_models):
        io, pca = make_io(tmp_path), make_pca_result(tmp_path)
        emb = EmbeddingConfig(method="phate", input_mode="fit", params={"knn": 3, "n_landmark": 50})

        run_embedding_step(io, emb, pca=pca)

        k = fake_models.instances[0].kwargs
        assert k["knn"] == 3
        assert k["n_landmark"] == 50
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
srun --jobid=<JOBID> --overlap --ntasks=1 --time=40:00 bash -c 'cd <repo> && source .venv/bin/activate && pytest tests/unit/test_step_embedding.py -q -p no:cacheprovider'
```

Expected: collection error — `ModuleNotFoundError: No module named 'manifold_genetics.pipeline.steps.embedding'`.

- [ ] **Step 3: Write `src/manifold_genetics/pipeline/steps/embedding.py`**

```python
"""In-process embedding step.

``run_embedding()`` is the seam shared by the ``manifold-genetics embed``
subcommand and ``run_embedding_step()``, and ``build_embedding_model()`` is the
one place a method name plus a params dict becomes a model — so the CLI and the
orchestrator cannot drift in how an embedding is parameterised (spec goal 2).

The defaults here reproduce what the CLI path has always produced. In
particular PHATE receives ``n_landmark`` explicitly (``None`` when unset),
because PHATE's own default is 2000 and ``cmd_embed`` has always overridden it.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

import pandas as pd

from ...embeddings import PHATE, TSNE, UMAP, DiffusionMap
from ..config import EmbeddingConfig, IOConfig
from .paths import embedding_output_paths
from .pca import PCAStepResult

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingStepResult",
    "build_embedding_model",
    "embedding_output_paths",
    "run_embedding",
    "run_embedding_step",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class EmbeddingStepResult:
    """Typed outputs of the embedding step.

    ``fit_embedding_file`` is set only in ``both`` mode. ``coords_df`` is
    excluded from comparison: DataFrame ``==`` is elementwise and would make a
    generated ``__eq__`` raise.
    """

    embedding_file: Path
    fit_embedding_file: Optional[Path] = None
    coords_df: Optional[pd.DataFrame] = field(default=None, compare=False)
    skipped: bool = False


def build_embedding_model(method: str, params: Optional[Mapping] = None):
    """Construct the embedding model for ``method`` from ``params``.

    Every parameter the CLI path passes is passed here too, including ones whose
    value is ``None`` — omitting them would silently fall back to the embedding
    class's own defaults, which differ (notably ``PHATE(n_landmark=2000)``).
    """
    p = dict(params or {})

    if method == "phate":
        return PHATE(
            n_components=2,
            knn=p.get("knn", 25),
            t=p.get("t", "auto"),
            n_landmark=p.get("n_landmark"),
            random_landmarking=p.get("random_landmarking", False),
            embed_batch_size=p.get("embed_batch_size"),
        )
    if method == "umap":
        return UMAP(
            n_components=2,
            n_neighbors=p.get("n_neighbors", 15),
            min_dist=p.get("min_dist", 0.1),
        )
    if method == "tsne":
        return TSNE(n_components=2, perplexity=p.get("perplexity", 30))
    if method == "diffusion_map":
        return DiffusionMap(n_components=2, knn=p.get("knn", 25))

    raise ValueError(
        f"Unknown embedding method: {method}. Choose from: phate, umap, tsne, diffusion_map"
    )


def run_embedding(
    fit_input: PathLike,
    project_input: Optional[PathLike] = None,
    *,
    project_output: PathLike,
    fit_output: Optional[PathLike] = None,
    method: str = "phate",
    params: Optional[Mapping] = None,
) -> pd.DataFrame:
    """Fit an embedding on ``fit_input`` and project ``project_input`` into it.

    ``project_input=None`` means "project the fitted dataset itself" — fit then
    transform the same CSV. This is deliberately not ``fit_transform``.

    Returns the DataFrame written to ``project_output``.
    """
    if project_input is None:
        project_input = fit_input

    model = build_embedding_model(method, params)
    model.fit(fit_input)

    if fit_output:
        fit_output = Path(fit_output)
        fit_output.parent.mkdir(parents=True, exist_ok=True)
        model.transform(fit_input).to_csv(fit_output, index=False)

    embedding = model.transform(project_input)
    project_output = Path(project_output)
    project_output.parent.mkdir(parents=True, exist_ok=True)
    embedding.to_csv(project_output, index=False)
    return embedding


def run_embedding_step(
    io: IOConfig, emb: EmbeddingConfig, *, pca: PCAStepResult
) -> EmbeddingStepResult:
    """Embed the PCA coordinates according to ``emb.input_mode``.

    * ``fit``     — fit and project the fit cohort's PCA coordinates.
    * ``project`` — fit and project the project cohort's PCA coordinates.
    * ``both``    — fit on the fit cohort, apply that embedding to the project
      cohort, and additionally write the fit cohort's own embedding.
    """
    paths = embedding_output_paths(io, emb)
    embedding_file = paths["embedding"]

    if emb.input_mode == "fit":
        logger.info("Embedding fit PCA coordinates only (mode: fit-only)")
        fit_input, project_input = pca.fit_pca, None
    elif emb.input_mode == "project":
        logger.info("Embedding project PCA coordinates only (mode: project-only)")
        fit_input, project_input = pca.project_pca, None
    else:  # "both"
        logger.info("Embedding: fit on fit PCA, apply to project PCA (mode: both)")
        fit_input, project_input = pca.fit_pca, pca.project_pca

    fit_embedding_file = paths.get("fit_embedding") if project_input is not None else None

    coords = run_embedding(
        fit_input,
        project_input,
        fit_output=fit_embedding_file,
        project_output=embedding_file,
        method=emb.method,
        params=emb.params,
    )

    return EmbeddingStepResult(
        embedding_file=embedding_file,
        fit_embedding_file=fit_embedding_file,
        coords_df=coords,
    )
```

- [ ] **Step 4: Re-export from `steps/__init__.py`**

Add to the existing import block and `__all__`, keeping both alphabetically ordered:

```python
from .embedding import (
    EmbeddingStepResult,
    build_embedding_model,
    run_embedding,
    run_embedding_step,
)
```

and the four names `"EmbeddingStepResult"`, `"build_embedding_model"`, `"run_embedding"`, `"run_embedding_step"` into `__all__`. Leave `embedding_output_paths` exported from `.paths` as it already is — do not add a second export of the same name.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/unit/test_step_embedding.py tests/unit/test_step_paths.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
black src/ tests/ && isort src/ tests/
git add src/manifold_genetics/pipeline/steps/embedding.py \
        src/manifold_genetics/pipeline/steps/__init__.py \
        tests/unit/test_step_embedding.py
git commit -m "feat(pipeline): add in-process embedding step

build_embedding_model() is the single place a method name plus params becomes a
model, and run_embedding() is the seam cmd_embed and the step share. PHATE
receives n_landmark explicitly so the pipeline keeps disabling landmarking by
default, as the CLI path always has.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 2: Route `cmd_embed` and `cmd_pipeline` through the shared code

**Files:**
- Modify: `src/manifold_genetics/cli.py` — add `_embedding_params_from_args()`; rewrite `cmd_embed`; replace `cmd_pipeline`'s inline params assembly.
- Test: `tests/unit/test_cli_main.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `build_embedding_model`, `run_embedding` from Task 1.
- Produces: `_embedding_params_from_args(args) -> dict` — module-private helper in `cli.py`, used by both `cmd_embed` and `cmd_pipeline`.

`cmd_embed` and `cmd_pipeline` today each contain a near-identical ~25-line block turning argparse values into an embedding params dict (`t` string→int, `n_landmark` string→int, the `random_landmarking` guard). That duplication is the drift risk this task removes. Argparse-shape resolution stays in `cli.py` per the spec — the helper is CLI-layer, not step-layer.

**One behavioural subtlety to preserve:** the `pipeline` subparser has no `--min-dist` flag, while the `embed` subparser does. The shared helper must therefore read it as `getattr(args, "min_dist", 0.1)`, which yields exactly today's behaviour for both callers (`cmd_pipeline` never set `min_dist`, and the orchestrator's argv builder defaulted it to `0.1`).

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_cli_main.py`, replace `test_cmd_embed_fit_project` with the version below and add the two new tests after it. The old test patched `mg_cli.PHATE` etc.; the models are now constructed inside `steps.embedding`.

```python
_STEP_EMBEDDING = "manifold_genetics.pipeline.steps.embedding"


def _fake_embed_models(monkeypatch, calls):
    class FakeEmbed:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def fit(self, X):
            calls.append(("fit", str(X)))
            return self

        def transform(self, X):
            calls.append(("transform", str(X)))
            return pd.DataFrame({"sample_id": ["s1"], "dim_1": [0.1], "dim_2": [0.2]})

    for name in ("PHATE", "UMAP", "TSNE", "DiffusionMap"):
        monkeypatch.setattr(f"{_STEP_EMBEDDING}.{name}", FakeEmbed)
    return FakeEmbed


def test_cmd_embed_fit_project(monkeypatch, tmp_path, stub_validation):
    calls = []
    _fake_embed_models(monkeypatch, calls)

    out = tmp_path / "emb.csv"
    rc = mg_cli.main(
        [
            "embed",
            "--method",
            "umap",
            "--fit-input",
            str(tmp_path / "fit.csv"),
            "--project-input",
            str(tmp_path / "proj.csv"),
            "--project-output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert ("fit", str(tmp_path / "fit.csv")) in calls
    assert ("transform", str(tmp_path / "proj.csv")) in calls


def test_cmd_embed_phate_passes_n_landmark_explicitly(monkeypatch, tmp_path, stub_validation):
    """PHATE defaults n_landmark=2000; the CLI has always overridden it with None.
    Losing that would silently enable landmarking for every default run."""
    calls = []
    _fake_embed_models(monkeypatch, calls)

    rc = mg_cli.main(
        [
            "embed",
            "--method",
            "phate",
            "--input",
            str(tmp_path / "in.csv"),
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )
    assert rc == 0
    init_kwargs = next(k for tag, k in calls if tag == "init")
    assert "n_landmark" in init_kwargs
    assert init_kwargs["n_landmark"] is None


def test_cmd_embed_random_landmarking_without_n_landmark_raises(
    monkeypatch, tmp_path, stub_validation
):
    calls = []
    _fake_embed_models(monkeypatch, calls)

    with pytest.raises(ValueError, match="random-landmarking"):
        mg_cli.main(
            [
                "embed",
                "--method",
                "phate",
                "--input",
                str(tmp_path / "in.csv"),
                "--output",
                str(tmp_path / "out.csv"),
                "--random-landmarking",
            ]
        )
```

In `tests/test_cli.py`, `test_cli_embed_fit_project` patches `mg_cli.PHATE` / `UMAP` / `TSNE` / `DiffusionMap`. Repoint those four `monkeypatch.setattr` calls to `f"{_STEP_EMBEDDING}.{name}"` (define the constant locally in that file, or inline the dotted string). Change nothing else about that test — its `FakeEmbed.fit_transform` guard must stay, since it is what proves the fit-then-transform path.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_cli_main.py -k embed -q -p no:cacheprovider
```

Expected: FAIL — the real embedding classes are still constructed in `cli.py`, so the fakes are never used and `calls` stays empty.

- [ ] **Step 3: Add the shared params helper**

Add to `src/manifold_genetics/cli.py`, next to the other module-level helpers (near `_resolve_k_values`):

```python
def _embedding_params_from_args(args) -> dict:
    """Turn argparse values into the embedding params dict.

    Shared by ``cmd_embed`` and ``cmd_pipeline`` so the two cannot drift. Reads
    optional attributes with ``getattr`` because the two subparsers do not
    define an identical flag set — the ``pipeline`` parser has no ``--min-dist``.
    """
    method = getattr(args, "method", None) or getattr(args, "embedding", None)
    params: dict = {}

    if method == "phate":
        t_param = args.t
        if isinstance(t_param, str) and t_param != "auto":
            t_param = int(t_param)

        n_landmark = None
        if args.n_landmark is not None:
            if isinstance(args.n_landmark, str):
                if args.n_landmark.lower() != "none":
                    n_landmark = int(args.n_landmark)
            else:
                n_landmark = args.n_landmark

        if args.random_landmarking and n_landmark is None:
            raise ValueError("random-landmarking requires --n-landmark to be set")

        params = {
            "knn": args.knn,
            "t": t_param,
            "n_landmark": n_landmark,
            "random_landmarking": args.random_landmarking,
            "embed_batch_size": getattr(args, "embed_batch_size", None),
        }
    elif method == "umap":
        params = {
            "n_neighbors": args.n_neighbors,
            "min_dist": getattr(args, "min_dist", 0.1),
        }
    elif method == "tsne":
        params = {"perplexity": args.perplexity}
    elif method == "diffusion_map":
        params = {"knn": args.knn}

    return params
```

- [ ] **Step 4: Rewrite `cmd_embed`**

Replace the body of `cmd_embed` in `src/manifold_genetics/cli.py` with:

```python
def cmd_embed(args):
    """Run embedding command."""
    setup_logging(args.verbose)

    fit_input = args.fit_input or args.input
    project_input = args.project_input or fit_input

    # Validate inputs (after resolving fit/project)
    if fit_input is not None:
        validate_embedding_csv(fit_input)
    if project_input is not None and project_input != fit_input:
        validate_embedding_csv(project_input)

    if fit_input is None:
        raise ValueError("Please provide --input or --fit-input for embedding fit.")
    if args.output is None and args.project_output is None:
        raise ValueError("Please provide --output or --project-output for embedding transform.")

    project_output = args.project_output or args.output
    fit_output = args.fit_output

    if args.method not in ("phate", "umap", "tsne", "diffusion_map"):
        print(f"Unknown method: {args.method}")
        return 1

    embedding = run_embedding(
        fit_input,
        project_input,
        fit_output=fit_output,
        project_output=project_output,
        method=args.method,
        params=_embedding_params_from_args(args),
    )

    print(f"Embedding fit on {fit_input} and projected {project_input}")
    if fit_output:
        print(f"Fit embedding saved to: {fit_output}")
    print(f"Projected embedding saved to: {project_output}")
    # Report shape excluding sample_id column
    n_samples = embedding.shape[0]
    n_dims = embedding.shape[1] - 1  # Exclude sample_id column
    print(f"Shape: ({n_samples}, {n_dims}) [excluding sample_id column]")
    return 0
```

Note the unknown-method check now happens **before** any work, so `cmd_embed` still returns `1` rather than raising — `test_cmd_embed_unknown_method_returns_1` depends on that. `run_embedding` creates the output parent directories, so the two explicit `mkdir` calls the old body performed are no longer needed.

Then replace `cmd_pipeline`'s inline params block — everything from `embedding_params = {}` down to the end of the `elif args.embedding == "diffusion_map":` branch — with:

```python
    embedding_params = _embedding_params_from_args(args)
```

Finally update the imports: add `run_embedding` to the existing `from .pipeline.steps import ...` line. Then check whether `PHATE`, `UMAP`, `TSNE`, `DiffusionMap` still have any use in `cli.py` — run `grep -nE '\b(PHATE|UMAP|TSNE|DiffusionMap)\b' src/manifold_genetics/cli.py`. If the only remaining hits are the import line itself and help text, delete `from .embeddings import PHATE, TSNE, UMAP, DiffusionMap`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/unit/test_cli_main.py tests/test_cli.py -q -p no:cacheprovider
pytest -m "not slow and not network" -q -p no:cacheprovider
```

Expected: PASS. Net new tests vs. the 457 baseline: Task 1 added 21, this task adds 2 → 480.

- [ ] **Step 6: Lint and commit**

```bash
black src/ tests/ && isort src/ tests/
git add src/manifold_genetics/cli.py tests/unit/test_cli_main.py tests/test_cli.py
git commit -m "refactor(cli): route cmd_embed through the shared embedding seam

Also collapses the duplicated argparse-to-params assembly in cmd_embed and
cmd_pipeline into one _embedding_params_from_args() helper.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF"
```

---

### Task 3: Orchestrator calls `run_embedding_step`

**Files:**
- Modify: `src/manifold_genetics/pipeline/orchestrator.py` — replace the embed argv builder and its `subprocess.run` with a step call.
- Test: `tests/unit/test_orchestrator.py` — port `TestPipelineRunEmbeddingInputMode` to step-behaviour assertions.

**Interfaces:**
- Consumes: `run_embedding_step(io, emb, *, pca) -> EmbeddingStepResult`, `EmbeddingConfig` from `..config`.
- Produces: nothing new. `results["embedding_file"]`, `results["embedding_coords"]` and (in `both` mode only) `results["fit_embedding_file"]` keep their current meanings.

The orchestrator has no `PCAStepResult` in scope when `skip_pca=True` — it only has `results["fit_pca_file"]` / `results["project_pca_file"]` resolved from disk. Construct one from those entries rather than changing the skip logic.

- [ ] **Step 1: Rewrite the mode tests**

In `tests/unit/test_orchestrator.py`, replace the whole of `class TestPipelineRunEmbeddingInputMode` (its section banner comment included) with:

```python
# ---------------------------------------------------------------------------
# TestPipelineRunEmbeddingInputMode — which PCA file feeds the embedder
# ---------------------------------------------------------------------------


def stub_embedding_step(monkeypatch):
    """Replace the in-process embedding step with a fake that writes its outputs.

    Returns the list of (io, emb, pca) it was called with.
    """
    import manifold_genetics.pipeline.orchestrator as orch
    from manifold_genetics.pipeline.steps.embedding import EmbeddingStepResult

    calls = []

    def fake_run_embedding_step(io, emb, *, pca):
        calls.append((io, emb, pca))
        paths = orch.embedding_output_paths(io, emb)
        write_pca_csv(paths["embedding"], 2)
        fit_file = paths.get("fit_embedding")
        if fit_file is not None:
            write_pca_csv(fit_file, 2)
        return EmbeddingStepResult(
            embedding_file=paths["embedding"],
            fit_embedding_file=fit_file,
            coords_df=pd.read_csv(paths["embedding"]),
        )

    monkeypatch.setattr(orch, "run_embedding_step", fake_run_embedding_step)
    return calls


class TestPipelineRunEmbeddingInputMode:
    """embedding_input='fit'|'project'|'both' controls which PCA coordinates the
    embedding step receives.

    'fit':     embed the fit cohort's PCA standalone
    'project': embed the project cohort's PCA standalone
    'both':    fit on the fit PCA, apply to the project PCA, write both files

    Wrong mode silently embeds the wrong dataset.
    """

    def _setup(self, pipeline, n_pcs):
        pca_dir = pipeline.output_dir / "pca"
        fit_file = pca_dir / f"fit_pca_{n_pcs}.csv"
        project_file = pca_dir / f"project_pca_{n_pcs}.csv"
        write_pca_csv(fit_file, n_pcs)
        write_pca_csv(project_file, n_pcs)
        return fit_file, project_file

    def _run(self, pipeline, monkeypatch, n_pcs, mode, method="phate"):
        calls = stub_embedding_step(monkeypatch)
        capture_subprocess(monkeypatch)
        pipeline.run(
            n_pcs=n_pcs,
            embedding=method,
            embedding_params={"knn": 5},
            embedding_input=mode,
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )
        assert len(calls) == 1, f"Expected one embedding step call, got {len(calls)}"
        return calls[0]

    def test_mode_both_passes_both_pca_files(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        fit_file, project_file = self._setup(pipeline, n_pcs)

        io, emb, pca = self._run(pipeline, monkeypatch, n_pcs, "both")

        assert emb.input_mode == "both"
        assert pca.fit_pca == fit_file
        assert pca.project_pca == project_file

    def test_mode_fit_passes_the_fit_pca(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        fit_file, _ = self._setup(pipeline, n_pcs)

        io, emb, pca = self._run(pipeline, monkeypatch, n_pcs, "fit")

        assert emb.input_mode == "fit"
        assert pca.fit_pca == fit_file

    def test_mode_project_passes_the_project_pca(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        _, project_file = self._setup(pipeline, n_pcs)

        io, emb, pca = self._run(pipeline, monkeypatch, n_pcs, "project")

        assert emb.input_mode == "project"
        assert pca.project_pca == project_file

    def test_method_and_params_reach_the_step(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)

        io, emb, pca = self._run(pipeline, monkeypatch, n_pcs, "both", method="umap")

        assert emb.method == "umap"
        assert emb.params["knn"] == 5

    def test_embedding_no_longer_shells_out(self, tmp_path, monkeypatch):
        """Re-introducing a subprocess hop would restore the API -> CLI -> API
        inversion this refactor removed."""
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)

        stub_embedding_step(monkeypatch)
        calls = capture_subprocess(monkeypatch)
        pipeline.run(
            n_pcs=n_pcs,
            embedding="phate",
            embedding_params={"knn": 5},
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )
        assert embed_calls(calls) == [], f"Embedding must run in-process, got: {calls}"

    def test_results_expose_embedding_paths(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)
        stub_embedding_step(monkeypatch)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=n_pcs,
            embedding="phate",
            embedding_params={"knn": 5},
            embedding_input="both",
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )

        emb_dir = pipeline.output_dir / "embeddings"
        assert results["embedding_file"] == emb_dir / "phate_2d.csv"
        assert results["fit_embedding_file"] == emb_dir / "phate_fit_2d.csv"
        assert isinstance(results["embedding_coords"], pd.DataFrame)

    def test_single_input_mode_reports_no_fit_embedding_file(self, tmp_path, monkeypatch):
        pipeline = make_pipeline(tmp_path)
        n_pcs = 10
        self._setup(pipeline, n_pcs)
        stub_embedding_step(monkeypatch)
        capture_subprocess(monkeypatch)

        results = pipeline.run(
            n_pcs=n_pcs,
            embedding="phate",
            embedding_params={"knn": 5},
            embedding_input="fit",
            skip_pca=True,
            skip_admixture=True,
            skip_visualization=True,
            skip_pca_visualization=True,
            skip_metrics=True,
        )
        assert "fit_embedding_file" not in results
```

`TestPipelineRunFullFlow` also drives the embedding stage — its `_writer` currently writes the embedding CSVs on the `"embed"` subprocess call. Drop that branch (leaving `_writer` with nothing to do, so delete `_writer` and its `on_call=` usages entirely if no branch remains), and add `stub_embedding_step(monkeypatch)` alongside the existing `stub_pca_step(...)` call in each of those tests.

`TestPipelineRunMissingPCA::test_embedding_succeeds_when_cached_pca_files_present` likewise relies on the subprocess writing the embedding file — give it `stub_embedding_step(monkeypatch)` too. Do NOT weaken `test_embedding_raises_runtime_error_when_pca_files_absent`; that guard must still fire (constraint C).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/test_orchestrator.py -q -p no:cacheprovider
```

Expected: FAIL — `AttributeError: module 'manifold_genetics.pipeline.orchestrator' has no attribute 'run_embedding_step'`.

- [ ] **Step 3: Wire the orchestrator**

Add to the imports in `src/manifold_genetics/pipeline/orchestrator.py` (merging into the existing lines rather than duplicating a module):

```python
from .config import EmbeddingConfig, IOConfig, PCAConfig
from .steps.embedding import run_embedding_step
from .steps.paths import embedding_output_paths, metrics_output_paths, pca_output_paths
from .steps.pca import PCAStepResult, run_pca_step
```

Replace the embedding stage body — from `embedding_dir = self.output_dir / "embeddings"` through `embedding_coords = pd.read_csv(embedding_file)` and the `results[...]` assignments that follow — with:

```python
            emb_cfg = EmbeddingConfig(
                method=embedding,
                input_mode=embedding_input,
                params=dict(embedding_params or {}),
            )

            # PCA outputs may have come from the step or been resolved from disk
            # under --skip-pca; either way the embedding step takes them as a result.
            pca_result = PCAStepResult(
                fit_pca=results.get("fit_pca_file"),
                project_pca=results.get("project_pca_file"),
            )

            emb_result = run_embedding_step(io, emb_cfg, pca=pca_result)

            embedding_file = emb_result.embedding_file
            results["embedding_file"] = embedding_file
            results["embedding_coords"] = emb_result.coords_df

            if emb_result.fit_embedding_file is not None:
                results["fit_embedding_file"] = emb_result.fit_embedding_file
```

Keep the `RuntimeError` guard above it exactly as it is, and keep the logging banner. `embedding_file` stays a local because the later visualization and metrics blocks read it.

`import subprocess` must remain — the admixture stage still uses it. Check whether `import pandas as pd` still has a use in `orchestrator.py` after this (`grep -n 'pd\.' src/manifold_genetics/pipeline/orchestrator.py`); if it has none, remove the import.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/unit/test_orchestrator.py -q -p no:cacheprovider
pytest -m "not slow and not network" -q -p no:cacheprovider
pytest tests/integration/test_generic_pipeline.py -q -p no:cacheprovider
```

Expected: the fast suite passes; `test_generic_pipeline.py` stays at 6/6.

- [ ] **Step 5: Lint and commit**

```bash
black src/ tests/ && isort src/ tests/
git add src/manifold_genetics/pipeline/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "refactor(pipeline): run the embedding stage in-process

Replaces the embed argv builder and its subprocess hop with run_embedding_step().
The three --embedding-input modes are now tested by which PCA coordinates reach
the step rather than by asserting flags in a command list.

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

Expected: **no output.**

- [ ] **Step 2: Only the admixture hop remains**

```bash
grep -n 'subprocess.run' src/manifold_genetics/pipeline/orchestrator.py
grep -n '"embed",' src/manifold_genetics/pipeline/orchestrator.py
```

Expected: exactly one `subprocess.run` (the admixture stage), and no `"embed",` argv element.

- [ ] **Step 3: No embedding path literals outside `steps/paths.py`**

```bash
grep -rn '_2d\.csv' src/manifold_genetics/ --include=*.py | grep -v steps/paths.py
```

Expected: no output outside help text / docstrings; inspect any hit rather than deleting it blindly.

- [ ] **Step 4: Full check**

```bash
black --check src/ tests/ && isort --check src/ tests/
pytest -m "not slow and not network" -q -p no:cacheprovider
pytest tests/integration/test_generic_pipeline.py -q -p no:cacheprovider
```

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin HEAD
~/bin/gh pr create --base main --title "feat(pipeline): run the embedding stage in-process (step layer PR 3)" --body "<see below>"
```

PR body must cover: the commit-by-commit summary; the `n_landmark` trap and how it is pinned; that modes `fit`/`project` are fit-then-transform rather than `fit_transform`; the `_embedding_params_from_args` de-duplication; and that only the admixture hop now remains (PR 4).

---

## Self-Review

**Spec coverage (migration table row 3 — "Embedding step"):**

| Spec requirement | Task |
|---|---|
| `steps/embedding.py` | 1 |
| The three-mode `--embedding-input` → IO mapping | 1 (step), 3 (orchestrator) |
| Explicit `pca: PCAStepResult` parameter | 1 |
| Replace the embed subprocess call | 3 |
| `cmd_embed` → step fn | 2 |
| Port the mode tests to step-behaviour tests | 3 |
| `EmbeddingConfig` absorbs the per-method params assembly | 2 (as `_embedding_params_from_args`, a CLI-layer helper — see deviation below) |
| Constraint A (frozen CLI) | 4 |
| Constraint B (output paths) | 1, 4 |
| Constraint C (skip still yields paths; `RuntimeError` preserved) | 3 |
| Constraint D (compute steps fatal) | 1 |

**Deviations from the spec, both deliberate:**
1. `EmbeddingStepResult` gains a `coords_df` field the spec's sketch omits, mirroring `PCAStepResult` — the orchestrator's `results["embedding_coords"]` needs it, and returning it avoids re-reading the CSV just written.
2. The per-method params assembly lands in `cli.py` as `_embedding_params_from_args()` rather than inside `EmbeddingConfig`. It reads an argparse `Namespace`, which is CLI-shape knowledge; the spec itself says arg resolution stays in `cmd_*`. The de-duplication the spec wanted is achieved either way.

**Deferred to later PRs:** the admixture step and its backend branch (PR 4); `_get_embedding_model` deletion, `import subprocess` removal, the four viz steps, `PipelineResult`, and `build_configs()` wiring (PR 5).

**Type consistency:** `EmbeddingStepResult(embedding_file, fit_embedding_file, coords_df, skipped)` is used with those exact field names in Tasks 1 and 3. `EmbeddingConfig(.method, .input_mode, .params)` matches `pipeline/config.py` on `main`. `embedding_output_paths` keys `"embedding"` / `"fit_embedding"` match `steps/paths.py`, and the `"fit_embedding"` key is absent unless `input_mode == "both"` — which is why Task 1 and Task 3 both use `paths.get("fit_embedding")` rather than indexing.
