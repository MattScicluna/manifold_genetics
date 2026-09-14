# acquire / preprocess / subsample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `init` with three cohort-directory commands — `acquire` (per-cohort fetch + labels + config), `preprocess` (generic SNP filtering via the shipped shell), `subsample` (generic sample selection) — reproducing every `examples/*/prepare_data.sh` flow.

**Architecture:** A *cohort directory* (`config.yaml` + colormap(s) + `data/{fit,project}_subset.*` + labels) is the unit of exchange. `preprocess` and `subsample` each read one (or two) and write one; `run` consumes one. The 1,038-line `preprocess_cross_projection.sh` moves into the wheel and is orchestrated from Python, which resolves tools through `ToolResolver` and passes them in explicitly. `acquire` is `init` renamed, plus a workbench-archive layout for HGDP and a port of `download_aou_data.sh`.

**Tech Stack:** Python 3.10–3.12, argparse, pandas, PyYAML, subprocess → bash + plink2 + plink 1.9; pytest with `integration`/`network`/`requires_private_data` markers.

**Spec:** `archive/superpowers/specs/2026-09-14-acquire-preprocess-design.md`

## Global Constraints

- Python `>=3.10,<3.13`; no new runtime dependencies (`pyyaml`, `pandas`, `platformdirs` are already present).
- Naming rule: CLI says `fit`/`project`; the shell keeps `reference`/`biobank`. `--skip-biobank-maf` surfaces as `--skip-project-maf`, `--reference-has-chr-prefix` as `--fit-has-chr-prefix`.
- A command always rewrites the labels in its output directory from its input; it never reuses a `labels.csv` found there.
- Label files must **cover** the `.fam` they pair with; extra rows are allowed.
- Old scripts under `examples/` are kept. `examples/_shared/preprocessing/*.sh` become symlinks into the package so there is one copy.
- Code style: black (line length 100), isort profile black, flake8. Run `black src/ tests/ && isort src/ tests/ && flake8 src/ tests/` before every commit.
- Fast tests (`uv run pytest -m "not slow and not network"`) must stay fast: anything that runs the real shell is `integration` and skips when plink2/plink are not resolvable.
- Compute nodes have no internet. Anything that downloads runs on the login node.
- Commit after every task with the attribution trailer:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_014iaf45iEjPXpn7KPVn52EF
  ```
- Work on branch `feat/acquire-preprocess` (exists; spec is its first commit).

## File structure

| path | responsibility |
|---|---|
| `src/manifold_genetics/preprocessing/__init__.py` | re-exports `preprocess`, `PreprocessOptions`, `subsample` |
| `src/manifold_genetics/preprocessing/preprocess_cross_projection.sh`, `common.sh` | the shell, moved here (git mv); gains `--plink2/--plink/--python/--min-common-snps` |
| `src/manifold_genetics/preprocessing/flags.py` | pure: `PreprocessOptions`, `PRESET_FLAGS`, `shell_argv()` |
| `src/manifold_genetics/preprocessing/cohort.py` | cohort-directory I/O: `Side`, `read_side()`, `filter_labels_to_fam()`, `write_cohort_config()` |
| `src/manifold_genetics/preprocessing/runner.py` | `preprocess()` — orchestration |
| `src/manifold_genetics/preprocessing/subsample.py` | `parse_group()`, `select_by_groups()`, `subsample()` |
| `src/manifold_genetics/preprocessing/references.py` | `install_harmonisation_references()` for `setup --preprocessing` |
| `src/manifold_genetics/scaffold.py` | `init_*` → `acquire_*`; `_detect_hgdp_layout()`; `--archive` accepting `gs://` |
| `src/manifold_genetics/aou.py` | `acquire_aou()` — port of `download_aou_data.sh` |
| `src/manifold_genetics/cli.py` | `preprocess`, `subsample`, `acquire` parsers; `setup --preprocessing`; `init` removed |
| `tests/unit/test_preprocess_flags.py`, `test_preprocess_cohort.py`, `test_subsample.py`, `test_shipped_shell.py`, `test_aou_acquire.py` | fast unit tests |
| `tests/integration/test_preprocess_synthetic.py`, `test_hgdp_idempotence.py`, `test_ukbb_preprocess_idempotence.py` | real shell, real plink |
| `docs/preprocessing.md`, `docs/cli.md`, `docs/quickstart.md`, `docs/tutorial.ipynb`, `README.md`, `CHANGELOG.md`, `mkdocs.yml` | docs |

---

### Task 1: Ship the shell inside the package

**Files:**
- Move: `examples/_shared/preprocessing/preprocess_cross_projection.sh` → `src/manifold_genetics/preprocessing/preprocess_cross_projection.sh`
- Move: `examples/_shared/preprocessing/common.sh` → `src/manifold_genetics/preprocessing/common.sh`
- Create: `src/manifold_genetics/preprocessing/__init__.py`
- Create: symlinks `examples/_shared/preprocessing/preprocess_cross_projection.sh`, `examples/_shared/preprocessing/common.sh`
- Test: `tests/unit/test_shipped_shell.py`

**Interfaces:**
- Produces: `manifold_genetics.preprocessing.SHELL_SCRIPT: Path` — absolute path of the shipped script.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_shipped_shell.py
"""The filtering shell ships inside the wheel and is syntactically valid.

It lived in examples/, which the wheel does not carry, so `preprocess` on an
installed copy would have had nothing to run.
"""

import shutil
import subprocess

import pytest

from manifold_genetics import preprocessing


def test_the_shell_ships_with_the_package():
    assert preprocessing.SHELL_SCRIPT.is_file()
    assert (preprocessing.SHELL_SCRIPT.parent / "common.sh").is_file()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not on PATH")
def test_the_shell_parses():
    for name in ("preprocess_cross_projection.sh", "common.sh"):
        subprocess.run(["bash", "-n", str(preprocessing.SHELL_SCRIPT.parent / name)], check=True)


def test_the_example_copies_are_links_to_the_shipped_one(tmp_path):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    link = repo / "examples/_shared/preprocessing/preprocess_cross_projection.sh"
    if not link.exists():
        pytest.skip("not running from a checkout")
    assert link.is_symlink()
    assert link.resolve() == preprocessing.SHELL_SCRIPT.resolve()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_shipped_shell.py -v`
Expected: FAIL — `ImportError: cannot import name 'preprocessing'`.

- [ ] **Step 3: Move the scripts and create the package**

```bash
mkdir -p src/manifold_genetics/preprocessing
git mv examples/_shared/preprocessing/preprocess_cross_projection.sh src/manifold_genetics/preprocessing/
git mv examples/_shared/preprocessing/common.sh src/manifold_genetics/preprocessing/
rm -rf examples/_shared/preprocessing/.ipynb_checkpoints
ln -s ../../../src/manifold_genetics/preprocessing/preprocess_cross_projection.sh examples/_shared/preprocessing/preprocess_cross_projection.sh
ln -s ../../../src/manifold_genetics/preprocessing/common.sh examples/_shared/preprocessing/common.sh
git add examples/_shared/preprocessing
```

```python
# src/manifold_genetics/preprocessing/__init__.py
"""SNP filtering and sample selection between `acquire` and `run`.

The filtering itself is `preprocess_cross_projection.sh`, shipped here rather
than reimplemented: it produced the published figures and cannot be validated
against UK Biobank or All of Us from a development machine, so a Python port
would diverge silently. Python assembles its arguments and owns everything
around it -- tool resolution, labels, the output config.
"""

from pathlib import Path

SHELL_SCRIPT = Path(__file__).resolve().parent / "preprocess_cross_projection.sh"

__all__ = ["SHELL_SCRIPT"]
```

Confirm the wheel carries `.sh` files (hatchling includes every non-ignored file under `packages`):

```bash
uv build --wheel -o "$CLAUDE_JOB_DIR/tmp/wheel" >/dev/null && unzip -l "$CLAUDE_JOB_DIR"/tmp/wheel/*.whl | grep preprocessing/
```
Expected: both `.sh` files listed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_shipped_shell.py -v`
Expected: 3 passed.

Also confirm the old wrappers still resolve their sources through the links:
```bash
bash -c 'source examples/_shared/preprocessing/common.sh && type print_status'
```
Expected: `print_status is a function`.

- [ ] **Step 5: Commit**

```bash
git add -A src/manifold_genetics/preprocessing examples/_shared/preprocessing tests/unit/test_shipped_shell.py
git commit -m "feat(preprocessing): ship the filtering shell inside the package"
```

---

### Task 2: Untie the shell from the checkout

**Files:**
- Modify: `src/manifold_genetics/preprocessing/preprocess_cross_projection.sh` (arg parsing ~L74–214, tool lookup ~L276–285, `python3` at ~L409 and ~L797, common-SNP abort ~L788)
- Modify: `src/manifold_genetics/preprocessing/common.sh` (`find_plink2` ~L70, `find_plink` ~L129)
- Test: `tests/unit/test_shipped_shell.py`

**Interfaces:**
- Produces: shell flags `--plink2 PATH`, `--plink PATH`, `--python PATH`, `--min-common-snps N` (default 50000). `PYTHON` replaces every `python3`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_shipped_shell.py`:

```python
NEW_FLAGS = ["--plink2", "--plink", "--python", "--min-common-snps"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not on PATH")
def test_help_lists_the_flags_python_passes():
    out = subprocess.run(
        ["bash", str(preprocessing.SHELL_SCRIPT), "--help"], capture_output=True, text=True
    )
    assert out.returncode == 0
    for flag in NEW_FLAGS:
        assert flag in out.stdout, f"{flag} missing from --help"


def test_the_shell_no_longer_reaches_for_src():
    text = preprocessing.SHELL_SCRIPT.read_text()
    assert "sys.path.insert" not in text
    assert "python3 " not in text and "python3\n" not in text, "python3 must go through ${PYTHON}"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_shipped_shell.py -v`
Expected: the two new tests FAIL.

- [ ] **Step 3: Edit the shell**

In `preprocess_cross_projection.sh`, in the defaults block after `TOOLS_DIR=...`:

```bash
# Set by `manifold-genetics preprocess`, which resolves the tools itself. Left
# empty, the script searches the way it always has (bin/, modules, PATH).
PLINK2="${PLINK2:-}"
PLINK="${PLINK:-}"
PYTHON="${PYTHON:-python3}"
MIN_COMMON_SNPS="${MIN_COMMON_SNPS:-50000}"
```

In the `case` block, before `-h|--help)`:

```bash
        --plink2)
            PLINK2="$2"
            shift 2
            ;;
        --plink)
            PLINK="$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --min-common-snps)
            MIN_COMMON_SNPS="$2"
            shift 2
            ;;
```

In `--help`, after the `--threads` line:

```bash
            echo "  --plink2 PATH             plink2 binary (default: search bin/, modules, PATH)"
            echo "  --plink PATH              plink v1.9 binary (default: search as above)"
            echo "  --python PATH             Python with manifold_genetics importable (default: python3)"
            echo "  --min-common-snps N       Abort below this many shared SNPs (default: 50000)"
```

Replace the tool lookup (the `find_plink2 "$PROJECT_ROOT"` / `find_plink "$PROJECT_ROOT"` block):

```bash
print_status "Looking for plink2..."
if [[ -n "$PLINK2" ]]; then
    print_success "Using plink2 given on the command line: ${PLINK2}"
elif ! find_plink2 "$PROJECT_ROOT"; then
    exit 1
fi

print_status "Looking for plink (v1.9)..."
if [[ -n "$PLINK" ]]; then
    print_success "Using plink given on the command line: ${PLINK}"
elif ! find_plink "$PROJECT_ROOT"; then
    exit 1
fi
```

Replace `python3 -m manifold_genetics.utils.filter_duplicates` with `"${PYTHON}" -m manifold_genetics.utils.filter_duplicates`, and `python3 << EOF` with `"${PYTHON}" << EOF`. In that heredoc delete the two lines:

```python
# Add package to path
sys.path.insert(0, str(Path("${PROJECT_ROOT}/src")))
```

Replace the hard-coded abort:

```bash
    if [[ $COMMON_SNP_COUNT -lt $MIN_COMMON_SNPS ]]; then
        print_error "Fewer than ${MIN_COMMON_SNPS} common SNPs! Data may be incompatible."
        exit 1
    elif [[ $COMMON_SNP_COUNT -lt 100000 ]]; then
```

In `common.sh`, `find_plink2` currently starts with `PLINK2=""`; leave it — the caller only reaches it when `PLINK2` is empty. Same for `find_plink`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_shipped_shell.py -v`
Expected: 5 passed.

Sanity-check the old wrapper still runs its help through the link:
```bash
bash examples/_shared/preprocessing/preprocess_cross_projection.sh --help | head -3
```

- [ ] **Step 5: Commit**

```bash
git add src/manifold_genetics/preprocessing tests/unit/test_shipped_shell.py
git commit -m "feat(preprocessing): let the caller name plink2, plink and python"
```

---

### Task 3: Flag assembly (`flags.py`)

**Files:**
- Create: `src/manifold_genetics/preprocessing/flags.py`
- Test: `tests/unit/test_preprocess_flags.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class PreprocessOptions:
      preset: Optional[str] = None            # "intersect-only" | "harmonise" | None
      maf: Optional[float] = None; geno: Optional[float] = None
      ld_window: Optional[int] = None; ld_step: Optional[int] = None; ld_r2: Optional[float] = None
      skip_wrayner: bool = False; skip_giab: bool = False; skip_hla: bool = False
      skip_ld_prune: bool = False; skip_dedup: bool = False
      skip_maf: bool = False; skip_project_maf: bool = False
      fit_has_chr_prefix: bool = False; cleanup: bool = False
      threads: Optional[int] = None; memory: Optional[int] = None
      temp_dir: Optional[Path] = None; tools_dir: Optional[Path] = None
      min_common_snps: Optional[int] = None
  PRESET_FLAGS: Dict[str, Dict[str, bool]]
  def shell_argv(fit_plink: Path, project_plink: Path, output_dir: Path, options: PreprocessOptions,
                 *, plink2: str, plink: str, python: str) -> List[str]
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_preprocess_flags.py
"""The argv handed to the shell, preset by preset.

Each preset must reproduce exactly the flags one of the old wrappers passed;
those wrappers are the reference until each flow has been rerun for real.
"""

from pathlib import Path

import pytest

from manifold_genetics.preprocessing.flags import PRESET_FLAGS, PreprocessOptions, shell_argv

TOOLS = dict(plink2="/t/plink2", plink="/t/plink", python="/t/python")


def _argv(options):
    return shell_argv(Path("/d/ref"), Path("/d/bio"), Path("/o/data"), options, **TOOLS)


def _flags(argv):
    """The boolean flags, in order, after the positional/tool part."""
    return [a for a in argv if a.startswith("--skip") or a in ("--cleanup", "--reference-has-chr-prefix")]


def test_always_names_the_sides_the_output_and_the_tools():
    argv = _argv(PreprocessOptions())
    assert argv[:2] == ["bash", str(__import__("manifold_genetics.preprocessing", fromlist=["SHELL_SCRIPT"]).SHELL_SCRIPT)]
    for flag, value in [
        ("--reference-plink", "/d/ref"),
        ("--biobank-plink", "/d/bio"),
        ("--output-dir", "/o/data"),
        ("--plink2", "/t/plink2"),
        ("--plink", "/t/plink"),
        ("--python", "/t/python"),
    ]:
        assert argv[argv.index(flag) + 1] == value


def test_intersect_only_reproduces_the_generic_wrapper():
    argv = _argv(PreprocessOptions(preset="intersect-only"))
    assert _flags(argv) == [
        "--skip-wrayner", "--skip-giab", "--skip-hla", "--skip-ld-prune", "--skip-dedup", "--skip-maf",
    ]


def test_harmonise_reproduces_the_aou_wrapper():
    argv = _argv(PreprocessOptions(preset="harmonise", fit_has_chr_prefix=True))
    assert _flags(argv) == ["--skip-biobank-maf", "--cleanup", "--reference-has-chr-prefix"]


def test_no_preset_with_ukbb_flags_reproduces_the_ukbb_wrapper():
    argv = _argv(PreprocessOptions(skip_wrayner=True, skip_project_maf=True))
    assert _flags(argv) == ["--skip-wrayner", "--skip-biobank-maf"]


def test_explicit_flags_add_to_a_preset():
    argv = _argv(PreprocessOptions(preset="harmonise", skip_wrayner=True))
    assert "--skip-wrayner" in _flags(argv)


def test_numeric_options_are_passed_only_when_set():
    assert "--maf" not in _argv(PreprocessOptions())
    argv = _argv(PreprocessOptions(maf=0.05, geno=0.1, ld_window=150, ld_step=1, ld_r2=0.05,
                                   threads=8, memory=4000, min_common_snps=100))
    for flag, value in [("--maf", "0.05"), ("--geno", "0.1"), ("--ld-window", "150"),
                        ("--ld-step", "1"), ("--ld-r2", "0.05"), ("--threads", "8"),
                        ("--memory", "4000"), ("--min-common-snps", "100")]:
        assert argv[argv.index(flag) + 1] == value


def test_an_unknown_preset_is_rejected_by_name():
    with pytest.raises(ValueError, match="harmonize"):
        _argv(PreprocessOptions(preset="harmonize"))


def test_the_preset_table_only_names_real_options():
    fields = set(PreprocessOptions.__dataclass_fields__)
    for name, flags in PRESET_FLAGS.items():
        assert set(flags) <= fields, name
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_preprocess_flags.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/manifold_genetics/preprocessing/flags.py
"""Turn a preset and overrides into the shell's argv. Pure, so it is testable
against the exact flags each old wrapper passed."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

from . import SHELL_SCRIPT

# Bundles of boolean options, each reproducing one of the wrappers in examples/.
# `--reference-has-chr-prefix` is a property of the data, not of a preset, so it
# is never part of a bundle.
PRESET_FLAGS: Dict[str, Dict[str, bool]] = {
    # examples/generic/hgdp_1kgp_proj/prepare_data.sh: no external tools at all.
    "intersect-only": {
        "skip_wrayner": True,
        "skip_giab": True,
        "skip_hla": True,
        "skip_ld_prune": True,
        "skip_dedup": True,
        "skip_maf": True,
    },
    # examples/aou/hgdp_1kgp_proj/prepare_data.sh: everything on, MAF on the
    # reference only, intermediates deleted as they are consumed.
    "harmonise": {"skip_project_maf": True, "cleanup": True},
}

# option name -> shell flag. The shell keeps its reference/biobank vocabulary.
_BOOL_FLAGS = {
    "skip_wrayner": "--skip-wrayner",
    "skip_giab": "--skip-giab",
    "skip_hla": "--skip-hla",
    "skip_ld_prune": "--skip-ld-prune",
    "skip_dedup": "--skip-dedup",
    "skip_maf": "--skip-maf",
    "skip_project_maf": "--skip-biobank-maf",
    "cleanup": "--cleanup",
    "fit_has_chr_prefix": "--reference-has-chr-prefix",
}
_VALUE_FLAGS = {
    "maf": "--maf",
    "geno": "--geno",
    "ld_window": "--ld-window",
    "ld_step": "--ld-step",
    "ld_r2": "--ld-r2",
    "threads": "--threads",
    "memory": "--memory",
    "temp_dir": "--temp-dir",
    "tools_dir": "--tools-dir",
    "min_common_snps": "--min-common-snps",
}


@dataclass(frozen=True)
class PreprocessOptions:
    preset: Optional[str] = None
    maf: Optional[float] = None
    geno: Optional[float] = None
    ld_window: Optional[int] = None
    ld_step: Optional[int] = None
    ld_r2: Optional[float] = None
    skip_wrayner: bool = False
    skip_giab: bool = False
    skip_hla: bool = False
    skip_ld_prune: bool = False
    skip_dedup: bool = False
    skip_maf: bool = False
    skip_project_maf: bool = False
    fit_has_chr_prefix: bool = False
    cleanup: bool = False
    threads: Optional[int] = None
    memory: Optional[int] = None
    temp_dir: Optional[Path] = None
    tools_dir: Optional[Path] = None
    min_common_snps: Optional[int] = None

    def effective_flags(self) -> Dict[str, bool]:
        """Boolean options after the preset is applied; an explicit True always wins."""
        if self.preset is not None and self.preset not in PRESET_FLAGS:
            raise ValueError(
                f"Unknown preset {self.preset!r}. Choose from: {', '.join(sorted(PRESET_FLAGS))}"
            )
        flags = dict(PRESET_FLAGS.get(self.preset, {}))
        for name in _BOOL_FLAGS:
            if getattr(self, name):
                flags[name] = True
        return flags


def shell_argv(
    fit_plink: Path,
    project_plink: Path,
    output_dir: Path,
    options: PreprocessOptions,
    *,
    plink2: str,
    plink: str,
    python: str,
) -> List[str]:
    """The command that runs the shipped shell on one pair of PLINK prefixes."""
    argv = [
        "bash",
        str(SHELL_SCRIPT),
        "--reference-plink", str(fit_plink),
        "--biobank-plink", str(project_plink),
        "--output-dir", str(output_dir),
        "--plink2", plink2,
        "--plink", plink,
        "--python", python,
    ]
    for name, flag in _VALUE_FLAGS.items():
        value = getattr(options, name)
        if value is not None:
            argv += [flag, str(value)]
    effective = options.effective_flags()
    for name, flag in _BOOL_FLAGS.items():
        if effective.get(name):
            argv.append(flag)
    return argv


__all__ = ["PRESET_FLAGS", "PreprocessOptions", "shell_argv"]
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_preprocess_flags.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifold_genetics/preprocessing/flags.py tests/unit/test_preprocess_flags.py
git commit -m "feat(preprocessing): assemble the shell argv from presets and overrides"
```

---

### Task 4: Cohort-directory I/O (`cohort.py`)

**Files:**
- Create: `src/manifold_genetics/preprocessing/cohort.py`
- Test: `tests/unit/test_preprocess_cohort.py`

**Interfaces:**
- Consumes: `manifold_genetics.pipeline.configfile.load_config(path) -> dict` (resolved absolute paths under keys `fit_plink`, `project_plink`, `labels`, `fit_labels`, `project_labels`, `colormap`, `fit_colormap`, `project_colormap`).
- Produces:
  ```python
  @dataclass(frozen=True)
  class Side:            # one half of a cohort directory
      plink: Path; labels: Path; colormap: Path
  @dataclass(frozen=True)
  class CohortConfig:
      path: Path; raw: dict; preset: Optional[str]; fit: Side; project: Side; shared_labels: bool
  def read_cohort(config_path) -> CohortConfig
  def read_fam_ids(prefix: Path) -> List[str]                 # IIDs, in .fam order
  def filter_labels_to_fam(labels: Path, prefix: Path, out: Path) -> int  # rows written; raises if not covered
  def write_cohort_config(out_dir: Path, *, based_on: CohortConfig, preset: str, data: Dict[str, str],
                          visualization: Optional[Dict[str, str]] = None, written_by: str) -> Path
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_preprocess_cohort.py
"""Reading and writing the cohort directory -- the unit `preprocess`, `subsample`
and `run` exchange."""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing.cohort import (
    filter_labels_to_fam,
    read_cohort,
    read_fam_ids,
    write_cohort_config,
)
from manifold_genetics.scaffold import init_synthetic


@pytest.fixture
def cohort(tmp_path):
    init_synthetic(tmp_path)
    return read_cohort(tmp_path / "config.yaml")


def test_a_whole_cohort_config_uses_one_label_file_for_both_sides(cohort):
    assert cohort.preset == "whole_cohort"
    assert cohort.shared_labels
    assert cohort.fit.labels == cohort.project.labels == cohort.path.parent / "data" / "labels.csv"
    assert cohort.fit.colormap == cohort.project.colormap


def test_a_projection_config_keeps_the_sides_apart(tmp_path):
    (tmp_path / "data").mkdir()
    for name in ("a", "b"):
        for ext in ("bed", "bim", "fam"):
            (tmp_path / "data" / f"{name}.{ext}").write_text("")
        (tmp_path / "data" / f"{name}.csv").write_text("sample_id,x\n")
        (tmp_path / f"{name}.json").write_text("{}")
    (tmp_path / "config.yaml").write_text(
        "preset: projection\ndata:\n  fit_plink: data/a\n  project_plink: data/b\n"
        "  fit_labels: data/a.csv\n  project_labels: data/b.csv\n"
        "  fit_colormap: a.json\n  project_colormap: b.json\n  output_dir: outputs\n"
    )
    cohort = read_cohort(tmp_path / "config.yaml")
    assert not cohort.shared_labels
    assert cohort.fit.labels.name == "a.csv" and cohort.project.labels.name == "b.csv"
    assert cohort.fit.colormap.name == "a.json" and cohort.project.colormap.name == "b.json"


def test_fam_ids_come_back_in_file_order(cohort):
    ids = read_fam_ids(cohort.fit.plink)
    fam = pd.read_csv(f"{cohort.fit.plink}.fam", sep=r"\s+", header=None, dtype=str)
    assert ids == list(fam[1])


def test_labels_are_filtered_to_the_fam_and_keep_their_columns(cohort, tmp_path):
    out = tmp_path / "filtered.csv"
    n = filter_labels_to_fam(cohort.project.labels, cohort.fit.plink, out)
    written = pd.read_csv(out, dtype=str)
    assert n == len(written) == len(read_fam_ids(cohort.fit.plink))
    assert list(written.columns) == list(pd.read_csv(cohort.project.labels, nrows=0).columns)


def test_labels_that_do_not_cover_the_fam_are_an_error(cohort, tmp_path):
    partial = tmp_path / "partial.csv"
    pd.read_csv(cohort.project.labels).iloc[:5].to_csv(partial, index=False)
    with pytest.raises(ValueError, match="not in"):
        filter_labels_to_fam(partial, cohort.fit.plink, tmp_path / "out.csv")


def test_the_written_config_is_accepted_by_the_loader_and_carries_settings(cohort, tmp_path):
    out = tmp_path / "next"
    out.mkdir()
    path = write_cohort_config(
        out,
        based_on=cohort,
        preset="projection",
        data={
            "fit_plink": "data/fit_subset",
            "project_plink": "data/project_subset",
            "fit_labels": "data/fit_labels.csv",
            "project_labels": "data/project_labels.csv",
            "fit_colormap": "colormap_fit.json",
            "project_colormap": "colormap_project.json",
            "output_dir": "outputs",
        },
        visualization={"projection_plot_fit_column": "branch", "projection_plot_project_column": "branch"},
        written_by="manifold-genetics preprocess",
    )
    text = path.read_text()
    assert text.startswith("# Written by `manifold-genetics preprocess`")
    loaded = load_config(path)
    assert loaded["fit_plink"] == (out / "data/fit_subset").resolve()
    assert loaded["n_pcs"] == cohort.raw["pca"]["n_pcs"], "pca settings must carry forward"
    assert loaded["projection_plot_fit_column"] == "branch"
    assert "labels" not in loaded, "the old shared-label key must not survive a projection rewrite"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_preprocess_cohort.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/manifold_genetics/preprocessing/cohort.py
"""The cohort directory: config.yaml, colormap(s), data/ with genotypes and labels.

`acquire` writes one; `preprocess` and `subsample` read one and write another;
`run` consumes one. Everything here is about reading that layout through the
config and writing it back with the paths moved."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from ..pipeline.configfile import load_config
from ..utils.io import PathLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Side:
    """One half of a cohort: its genotypes, the labels that cover them, the colours."""

    plink: Path
    labels: Path
    colormap: Path


@dataclass(frozen=True)
class CohortConfig:
    path: Path
    raw: dict
    preset: Optional[str]
    fit: Side
    project: Side
    shared_labels: bool


def read_cohort(config_path: PathLike) -> CohortConfig:
    """Read a config and resolve both sides.

    A side's labels are ``<side>_labels`` when given, else the shared ``labels``;
    colormaps likewise. That is the precedence ``run`` applies.
    """
    path = Path(config_path).expanduser().resolve()
    resolved = load_config(path)
    raw = yaml.safe_load(path.read_text()) or {}

    def side(name: str) -> Side:
        labels = resolved.get(f"{name}_labels") or resolved.get("labels")
        colormap = resolved.get(f"{name}_colormap") or resolved.get("colormap")
        if labels is None or colormap is None:
            raise ValueError(f"{path} names no labels or colormap for the {name} side")
        return Side(plink=Path(resolved[f"{name}_plink"]), labels=Path(labels), colormap=Path(colormap))

    return CohortConfig(
        path=path,
        raw=raw,
        preset=raw.get("preset"),
        fit=side("fit"),
        project=side("project"),
        shared_labels="labels" in resolved and "fit_labels" not in resolved,
    )


def read_fam_ids(prefix: PathLike) -> List[str]:
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, dtype=str, usecols=[0, 1])
    return list(fam[1])


def filter_labels_to_fam(labels: PathLike, prefix: PathLike, out: PathLike) -> int:
    """Write the rows of ``labels`` for the samples in ``prefix``.fam, in .fam order.

    Raises:
        ValueError: a sample in the .fam has no label row. A label file must
            cover its genotypes; a filtered copy that silently dropped samples
            would produce a figure with grey points and no error.
    """
    ids = read_fam_ids(prefix)
    frame = pd.read_csv(labels, dtype={"sample_id": str}, low_memory=False)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{labels} has no sample_id column")
    missing = set(ids) - set(frame["sample_id"])
    if missing:
        example = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"{len(missing)} of {len(ids)} samples in {prefix}.fam are not in {labels} "
            f"(e.g. {example}). Labels must cover the genotypes they describe."
        )
    kept = frame.set_index("sample_id").loc[ids].reset_index()
    kept = kept.drop_duplicates("sample_id")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False)
    return len(kept)


_CARRIED_SECTIONS = ("pca", "admixture", "embedding", "visualization", "skip")


def write_cohort_config(
    out_dir: Path,
    *,
    based_on: CohortConfig,
    preset: str,
    data: Dict[str, str],
    visualization: Optional[Dict[str, str]] = None,
    written_by: str,
) -> Path:
    """Write ``out_dir/config.yaml``: the input's settings with a new data section.

    The data section is replaced wholesale so that no stale key survives -- a
    ``labels`` left beside ``fit_labels`` is exactly the shape ``run`` rejects.
    """
    document: Dict[str, object] = {"preset": preset, "data": dict(data)}
    for section in _CARRIED_SECTIONS:
        if section in based_on.raw and based_on.raw[section]:
            document[section] = dict(based_on.raw[section])
    if visualization:
        document.setdefault("visualization", {})
        document["visualization"].update(visualization)  # type: ignore[union-attr]

    header = (
        f"# Written by `{written_by}` from {based_on.path}.\n"
        "#\n"
        "#   manifold-genetics run config.yaml --dry-run   # print the settings, do nothing\n"
        "#   manifold-genetics run config.yaml             # do the work\n"
        "#\n"
        "# Paths are relative to this file.\n\n"
    )
    path = out_dir / "config.yaml"
    path.write_text(header + yaml.safe_dump(document, sort_keys=False))
    logger.info("Wrote %s", path)
    return path


__all__ = ["CohortConfig", "Side", "filter_labels_to_fam", "read_cohort", "read_fam_ids", "write_cohort_config"]
```

Check `PathLike` exists in `utils/io.py` (`grep -n "^PathLike" src/manifold_genetics/utils/io.py`); if it does not, define `PathLike = Union[str, Path]` locally.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_preprocess_cohort.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/manifold_genetics/preprocessing/cohort.py tests/unit/test_preprocess_cohort.py
git commit -m "feat(preprocessing): read and write cohort directories"
```

---

### Task 5: `preprocess()` orchestration and the synthetic round trip

**Files:**
- Create: `src/manifold_genetics/preprocessing/runner.py`
- Modify: `src/manifold_genetics/preprocessing/__init__.py`
- Modify: `src/manifold_genetics/utils/tools.py` — add `resolve_plink1()` (mirrors `resolve_plink2`, env var `PLINK1_PATH`, binary name `plink`, downloads via `_download_plink1`)
- Test: `tests/integration/test_preprocess_synthetic.py`, `tests/unit/test_preprocess_runner.py`

**Interfaces:**
- Consumes: `shell_argv`, `PreprocessOptions`, `read_cohort`, `filter_labels_to_fam`, `write_cohort_config`, `ToolResolver.resolve_plink2()`, new `ToolResolver.resolve_plink1()`.
- Produces:
  ```python
  def preprocess(fit_config: PathLike, out_dir: PathLike, *, project_config: Optional[PathLike] = None,
                 options: PreprocessOptions = PreprocessOptions(), force: bool = False,
                 runner: Callable[[List[str]], None] = _run) -> Path   # returns out_dir/config.yaml
  ```

- [ ] **Step 1: Write the failing unit test (shell stubbed)**

```python
# tests/unit/test_preprocess_runner.py
"""What `preprocess` does around the shell, with the shell replaced by a stub
that writes the two outputs the real one writes."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess
from manifold_genetics.scaffold import init_synthetic


def _fake_shell(argv):
    """Copy reference -> fit_subset and biobank -> project_subset, dropping the last variant."""
    ref = Path(argv[argv.index("--reference-plink") + 1])
    bio = Path(argv[argv.index("--biobank-plink") + 1])
    out = Path(argv[argv.index("--output-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    for src, name in ((ref, "fit_subset"), (bio, "project_subset")):
        for ext in ("bed", "fam"):
            shutil.copy(f"{src}.{ext}", out / f"{name}.{ext}")
        lines = Path(f"{src}.bim").read_text().splitlines()[:-1]
        (out / f"{name}.bim").write_text("\n".join(lines) + "\n")


@pytest.fixture
def tools(monkeypatch):
    from manifold_genetics.utils import tools as t

    monkeypatch.setattr(t.ToolResolver, "resolve_plink2", lambda self: "/stub/plink2")
    monkeypatch.setattr(t.ToolResolver, "resolve_plink1", lambda self: "/stub/plink")


def test_one_config_keeps_the_cohort_shape(tmp_path, tools):
    init_synthetic(tmp_path / "in")
    config = preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)

    loaded = load_config(config)
    assert loaded["fit_plink"] == (tmp_path / "out/data/fit_subset").resolve()
    assert loaded["labels"] == (tmp_path / "out/data/labels.csv").resolve()
    assert (tmp_path / "out/colormap.json").exists()
    labels = pd.read_csv(tmp_path / "out/data/labels.csv", dtype=str)
    fam = pd.read_csv(tmp_path / "out/data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
    assert set(labels["sample_id"]) >= set(fam[1])


def test_two_configs_make_a_projection(tmp_path, tools):
    init_synthetic(tmp_path / "a", seed=1)
    init_synthetic(tmp_path / "b", seed=2)
    config = preprocess(
        tmp_path / "a/config.yaml", tmp_path / "out", project_config=tmp_path / "b/config.yaml",
        runner=_fake_shell,
    )
    loaded = load_config(config)
    assert loaded["embedding_input"] == "both", "projection preset expected"
    assert loaded["fit_labels"].name == "fit_labels.csv"
    assert loaded["project_labels"].name == "project_labels.csv"
    assert loaded["fit_colormap"].name == "colormap_fit.json"
    assert loaded["projection_plot_fit_column"] == "branch"


def test_the_shell_gets_the_resolved_tools(tmp_path, tools):
    seen = {}

    def spy(argv):
        seen["argv"] = argv
        _fake_shell(argv)

    init_synthetic(tmp_path / "in")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=spy)
    argv = seen["argv"]
    assert argv[argv.index("--plink2") + 1] == "/stub/plink2"
    assert argv[argv.index("--plink") + 1] == "/stub/plink"
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "out/data")


def test_refuses_to_overwrite_a_config_without_force(tmp_path, tools):
    init_synthetic(tmp_path / "in")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    with pytest.raises(FileExistsError):
        preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell, force=True)


def test_labels_are_rewritten_not_reused(tmp_path, tools):
    init_synthetic(tmp_path / "in")
    (tmp_path / "out/data").mkdir(parents=True)
    (tmp_path / "out/data/labels.csv").write_text("sample_id,branch\nSTALE,0\n")
    preprocess(tmp_path / "in/config.yaml", tmp_path / "out", runner=_fake_shell)
    assert "STALE" not in (tmp_path / "out/data/labels.csv").read_text()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_preprocess_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'preprocess'`.

- [ ] **Step 3: Add `resolve_plink1` to `ToolResolver`**

In `src/manifold_genetics/utils/tools.py`, after `resolve_plink2`, copy its structure:

```python
    def resolve_plink1(self) -> str:
        """Resolve plink v1.9, needed by the preprocessing shell for LD pruning
        and the WRayner checker. Same chain as plink2: $PLINK1_PATH, a loaded
        module, PATH, then download."""
        env_path = os.environ.get("PLINK1_PATH")
        if env_path and Path(env_path).exists():
            logger.info(f"Using plink (v1.9) from PLINK1_PATH: {env_path}")
            return env_path
        for candidate in (self.download_dir / "plink",):
            if candidate.exists():
                return str(candidate)
        path = shutil.which("plink")
        if path:
            return path
        logger.info("plink v1.9 not found, downloading...")
        return self._download_plink1()
```

Read `resolve_plink2` first and match its exact structure (module-system check included) rather than this sketch, then adjust the sketch to it.

- [ ] **Step 4: Implement `runner.py`**

```python
# src/manifold_genetics/preprocessing/runner.py
"""`preprocess`: one cohort directory in, one out, SNPs filtered in between."""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

from ..utils.io import PathLike
from ..utils.tools import ToolResolver
from .cohort import CohortConfig, filter_labels_to_fam, read_cohort, write_cohort_config
from .flags import PreprocessOptions, shell_argv

logger = logging.getLogger(__name__)


def _run(argv: List[str]) -> None:
    logger.info("Running: %s", " ".join(argv))
    subprocess.run(argv, check=True)


def _first_colormap_column(colormap: Path) -> Optional[str]:
    import json

    keys = list(json.loads(colormap.read_text()))
    return keys[0] if keys else None


def preprocess(
    fit_config: PathLike,
    out_dir: PathLike,
    *,
    project_config: Optional[PathLike] = None,
    options: PreprocessOptions = PreprocessOptions(),
    force: bool = False,
    runner: Callable[[List[str]], None] = _run,
) -> Path:
    """Filter one cohort, or intersect two, into a new cohort directory.

    One config: its own fit and project sets are the two sides; the output keeps
    the input's preset and label layout. Two configs: the first's fit side and
    the second's project side; the output is a ``projection`` with separate
    labels and colormaps per side.

    Args:
        runner: Executes the shell argv. Replaced in tests.

    Returns:
        The config written.

    Raises:
        FileExistsError: ``out_dir/config.yaml`` exists and ``force`` is False.
        subprocess.CalledProcessError: the shell failed; its own output says why.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    config_path = out_dir / "config.yaml"
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} exists. Pass --force to overwrite it.")

    fit_cohort = read_cohort(fit_config)
    project_cohort = read_cohort(project_config) if project_config else fit_cohort
    fit, project = fit_cohort.fit, project_cohort.project
    two_sided = project_config is not None or not fit_cohort.shared_labels

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    resolver = ToolResolver()
    runner(
        shell_argv(
            fit.plink, project.plink, data_dir, options,
            plink2=resolver.resolve_plink2(), plink=resolver.resolve_plink1(), python=sys.executable,
        )
    )
    for name in ("fit_subset", "project_subset"):
        if not (data_dir / f"{name}.bed").exists():
            raise RuntimeError(f"the shell finished without writing {data_dir / name}.bed")

    data = {"fit_plink": "data/fit_subset", "project_plink": "data/project_subset"}
    visualization = None
    if two_sided:
        filter_labels_to_fam(fit.labels, data_dir / "fit_subset", data_dir / "fit_labels.csv")
        filter_labels_to_fam(project.labels, data_dir / "project_subset", data_dir / "project_labels.csv")
        shutil.copy(fit.colormap, out_dir / "colormap_fit.json")
        shutil.copy(project.colormap, out_dir / "colormap_project.json")
        data.update(
            fit_labels="data/fit_labels.csv", project_labels="data/project_labels.csv",
            fit_colormap="colormap_fit.json", project_colormap="colormap_project.json",
        )
        preset = "projection" if project_config is not None else (fit_cohort.preset or "whole_cohort")
        if preset == "projection":
            visualization = {
                "projection_plot_fit_column": _first_colormap_column(fit.colormap),
                "projection_plot_project_column": _first_colormap_column(project.colormap),
            }
    else:
        # One label file covering both sides: filter it to their union, fit first.
        _filter_shared_labels(fit.labels, data_dir)
        shutil.copy(fit.colormap, out_dir / "colormap.json")
        data.update(labels="data/labels.csv", colormap="colormap.json")
        preset = fit_cohort.preset or "whole_cohort"
    data["output_dir"] = "outputs"

    return write_cohort_config(
        out_dir, based_on=fit_cohort, preset=preset, data=data,
        visualization=visualization, written_by="manifold-genetics preprocess",
    )


def _filter_shared_labels(labels: Path, data_dir: Path) -> None:
    import pandas as pd

    from .cohort import read_fam_ids

    ids = list(dict.fromkeys(read_fam_ids(data_dir / "fit_subset") + read_fam_ids(data_dir / "project_subset")))
    frame = pd.read_csv(labels, dtype={"sample_id": str}, low_memory=False)
    missing = set(ids) - set(frame["sample_id"])
    if missing:
        raise ValueError(f"{len(missing)} samples in the output have no row in {labels}")
    frame.set_index("sample_id").loc[ids].reset_index().to_csv(data_dir / "labels.csv", index=False)


__all__ = ["preprocess"]
```

Update `__init__.py`:

```python
from .flags import PreprocessOptions  # noqa: E402
from .runner import preprocess  # noqa: E402

__all__ = ["SHELL_SCRIPT", "PreprocessOptions", "preprocess"]
```
(`SHELL_SCRIPT` must be defined before these imports; `flags.py` imports it.)

- [ ] **Step 5: Run the unit tests**

Run: `uv run pytest tests/unit/test_preprocess_runner.py -v`
Expected: 5 passed.

- [ ] **Step 6: Write the integration round trip**

```python
# tests/integration/test_preprocess_synthetic.py
"""The real shell on the simulated cohort: the round trip that proves the
cohort directory survives `preprocess`, and settles whether the intersection
script accepts two subsets of one dataset as its two sides (spec issue 9)."""

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess
from manifold_genetics.scaffold import init_synthetic
from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tools():
    resolver = ToolResolver()
    try:
        return resolver.resolve_plink2(), resolver.resolve_plink1()
    except ToolNotFoundError as exc:
        pytest.skip(f"plink binaries unavailable: {exc}")


def _variants(prefix):
    return {tuple(line.split()[i] for i in (0, 3, 4, 5)) for line in open(f"{prefix}.bim")}


def _samples(prefix):
    return [line.split()[1] for line in open(f"{prefix}.fam")]


def test_intersect_only_on_a_whole_cohort_is_lossless(tmp_path, tools):
    init_synthetic(tmp_path / "in")
    config = preprocess(
        tmp_path / "in/config.yaml", tmp_path / "out",
        options=PreprocessOptions(preset="intersect-only", min_common_snps=100, memory=2000, threads=2),
    )
    before, after = load_config(tmp_path / "in/config.yaml"), load_config(config)
    for side in ("fit_plink", "project_plink"):
        assert _samples(after[side]) == _samples(before[side])
        assert _variants(after[side]) == _variants(before[side])


def test_two_cohorts_intersect_to_the_shared_variants(tmp_path, tools):
    init_synthetic(tmp_path / "a", seed=1)
    init_synthetic(tmp_path / "b", seed=2)
    config = preprocess(
        tmp_path / "a/config.yaml", tmp_path / "out", project_config=tmp_path / "b/config.yaml",
        options=PreprocessOptions(preset="intersect-only", min_common_snps=100, memory=2000, threads=2),
    )
    loaded = load_config(config)
    assert _variants(loaded["fit_plink"]) == _variants(loaded["project_plink"])


def test_the_output_dry_runs(tmp_path, tools):
    from manifold_genetics.cli import main

    init_synthetic(tmp_path / "in")
    config = preprocess(
        tmp_path / "in/config.yaml", tmp_path / "out",
        options=PreprocessOptions(preset="intersect-only", min_common_snps=100, memory=2000, threads=2),
    )
    assert main(["run", str(config), "--dry-run"]) == 0
```

Check `main()` accepts an argv list (`grep -n "def main" src/manifold_genetics/cli.py`); if it reads `sys.argv`, use `monkeypatch.setattr(sys, "argv", [...])`.

- [ ] **Step 7: Run it on the compute node**

The login node is fine for the synthetic cohort (1000 variants), but the user's allocation is faster:
```bash
JOBID=$(squeue --me -h -o %i | head -1)
srun --jobid=$JOBID --overlap --ntasks=1 --time=10:00 bash -c \
  'cd /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics && source .venv/bin/activate && pytest tests/integration/test_preprocess_synthetic.py -v'
```
Expected: 3 passed. If the shell aborts, read its output: the likely faults are the synthetic `.bim` (chr `1`, alleles A/G — acceptable) and `--memory` (lower it).

- [ ] **Step 8: Commit**

```bash
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/
git add src/manifold_genetics/preprocessing src/manifold_genetics/utils/tools.py tests/
git commit -m "feat(preprocessing): preprocess() maps one cohort directory to another"
```

---

### Task 6: `preprocess` on the command line

**Files:**
- Modify: `src/manifold_genetics/cli.py` — add `cmd_preprocess` beside `cmd_init` (~L439) and its parser beside `init_parser` (~L1293)
- Test: `tests/unit/test_cli_main.py` (add to `SUBCOMMANDS`; add a dispatch test)

**Interfaces:**
- Consumes: `preprocess`, `PreprocessOptions`.
- Produces: `manifold-genetics preprocess FIT_CONFIG [PROJECT_CONFIG] --out DIR [flags]`.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_cli_main.py`, add `"preprocess"` to `SUBCOMMANDS`, and:

```python
def test_preprocess_passes_configs_options_and_out(monkeypatch, tmp_path):
    seen = {}

    def fake(fit_config, out_dir, *, project_config=None, options=None, force=False):
        seen.update(fit=fit_config, project=project_config, out=out_dir, options=options, force=force)
        return Path(out_dir) / "config.yaml"

    monkeypatch.setattr("manifold_genetics.preprocessing.preprocess", fake)
    rc = mg_cli.main([
        "preprocess", "a.yaml", "b.yaml", "--out", str(tmp_path), "--preset", "harmonise",
        "--skip-wrayner", "--fit-has-chr-prefix", "--maf", "0.02", "--threads", "4",
    ])
    assert rc == 0
    assert seen["fit"] == "a.yaml" and seen["project"] == "b.yaml"
    assert seen["options"].preset == "harmonise"
    assert seen["options"].skip_wrayner and seen["options"].fit_has_chr_prefix
    assert seen["options"].maf == 0.02 and seen["options"].threads == 4
```

Follow the file's existing pattern for how `main` is invoked and how `cmd_*` imports are patched (it patches at the point of use; `cmd_preprocess` must import `from . import preprocessing` and call `preprocessing.preprocess(...)` so the monkeypatch lands).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_cli_main.py -k preprocess -v`
Expected: FAIL — argparse `invalid choice: 'preprocess'`.

- [ ] **Step 3: Implement**

```python
def cmd_preprocess(args):
    """Filter one cohort, or intersect two, into a new cohort directory."""
    setup_logging(args.verbose)

    from . import preprocessing
    from .preprocessing.flags import PreprocessOptions

    options = PreprocessOptions(
        preset=args.preset,
        maf=args.maf, geno=args.geno,
        ld_window=args.ld_window, ld_step=args.ld_step, ld_r2=args.ld_r2,
        skip_wrayner=args.skip_wrayner, skip_giab=args.skip_giab, skip_hla=args.skip_hla,
        skip_ld_prune=args.skip_ld_prune, skip_dedup=args.skip_dedup,
        skip_maf=args.skip_maf, skip_project_maf=args.skip_project_maf,
        fit_has_chr_prefix=args.fit_has_chr_prefix, cleanup=args.cleanup,
        threads=args.threads, memory=args.memory,
        temp_dir=Path(args.temp_dir) if args.temp_dir else None,
        tools_dir=Path(args.tools_dir) if args.tools_dir else None,
        min_common_snps=args.min_common_snps,
    )
    try:
        config = preprocessing.preprocess(
            args.fit_config, args.out, project_config=args.project_config,
            options=options, force=args.force,
        )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Error: the filtering script exited with status {exc.returncode}; see its output above.",
              file=sys.stderr)
        return 1

    print(f"\nWrote {config}")
    print("\nNext:")
    print(f"  manifold-genetics run {config} --dry-run")
    print(f"  manifold-genetics run {config}")
    return 0
```

Parser (description text is the user-facing doc; keep it accurate):

```python
    pre_parser = subparsers.add_parser(
        "preprocess",
        help="Filter SNPs of one cohort, or intersect two, into a new cohort directory",
        description=(
            "Optional. Reads a cohort directory (what `acquire` writes), filters its\n"
            "SNPs with the shell that produced the published figures, and writes a new\n"
            "cohort directory in the same layout -- so its output can go to `run`, to\n"
            "`subsample`, or to another `preprocess`. Samples are never removed here.\n\n"
            "  preprocess cohort/config.yaml --out filtered/\n"
            "      one cohort: its own fit and project sets are the two sides\n"
            "  preprocess ref/config.yaml biobank/config.yaml --out proj/\n"
            "      two cohorts: intersect them; the output is a projection\n\n"
            "Presets bundle the skip flags:\n"
            "  intersect-only   no external tools: indels, missingness, intersection\n"
            "  harmonise        everything on, including WRayner/TOPMed checks\n"
            "                   (downloads references: run `setup --preprocessing` first)\n"
            "  (none)           defaults; add --skip-* flags as needed\n\n"
            "Needs bash, plink2 and plink v1.9 (fetched by `setup`)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pre_parser.add_argument("fit_config", help="Config of the cohort to fit on")
    pre_parser.add_argument("project_config", nargs="?", help="Config of the cohort to project (optional)")
    pre_parser.add_argument("--out", required=True, help="Directory to write the new cohort into")
    pre_parser.add_argument("--preset", choices=["intersect-only", "harmonise"])
    pre_parser.add_argument("--maf", type=float, help="MAF threshold (shell default 0.01)")
    pre_parser.add_argument("--geno", type=float, help="Missingness threshold (shell default 0.05)")
    pre_parser.add_argument("--ld-window", type=int, help="LD pruning window, kb (shell default 150)")
    pre_parser.add_argument("--ld-step", type=int, help="LD pruning step (shell default 1)")
    pre_parser.add_argument("--ld-r2", type=float, help="LD pruning r² (shell default 0.05)")
    for flag, text in [
        ("skip-wrayner", "Skip WRayner/TOPMed harmonisation"),
        ("skip-giab", "Skip GIAB difficult-region exclusion"),
        ("skip-hla", "Skip HLA/MHC exclusion"),
        ("skip-ld-prune", "Skip LD pruning"),
        ("skip-dedup", "Skip reference deduplication"),
        ("skip-maf", "Skip MAF filtering on both sides"),
        ("skip-project-maf", "Skip MAF filtering on the project side only"),
        ("fit-has-chr-prefix", "The fit set's chromosomes already carry a 'chr' prefix"),
        ("cleanup", "Delete intermediates as they are consumed (large cohorts)"),
    ]:
        pre_parser.add_argument(f"--{flag}", action="store_true", help=text)
    pre_parser.add_argument("--threads", type=int, help="Threads (default: SLURM_CPUS_PER_TASK or 4)")
    pre_parser.add_argument("--memory", type=int, help="plink memory limit, MB (shell default 100000)")
    pre_parser.add_argument("--temp-dir", help="Scratch directory (default: OUT/data/temp)")
    pre_parser.add_argument("--tools-dir", help="Where harmonisation references live (default: tool cache)")
    pre_parser.add_argument("--min-common-snps", type=int, help="Abort below this many shared SNPs (50000)")
    pre_parser.add_argument("--force", action="store_true", help="Overwrite an existing config.yaml")
    pre_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    pre_parser.set_defaults(func=cmd_preprocess)
```

Add `import subprocess` at the top of `cli.py` if absent.

- [ ] **Step 4: Run the CLI tests**

Run: `uv run pytest tests/unit/test_cli_main.py tests/unit/test_docs_match_cli.py -v`
Expected: all pass. `test_docs_match_cli.py` may require the new command to appear in `docs/cli.md` — if it fails, add a stub section now (full docs in Task 12):

```markdown
## preprocess

Filter SNPs of one cohort, or intersect two, into a new cohort directory. See [Preprocessing](preprocessing.md).
```

- [ ] **Step 5: Commit**

```bash
git add src/manifold_genetics/cli.py tests/unit/test_cli_main.py docs/cli.md
git commit -m "feat(cli): preprocess command"
```

---

### Task 7: Harmonisation references — `setup --preprocessing`

**Files:**
- Create: `src/manifold_genetics/preprocessing/references.py`
- Modify: `src/manifold_genetics/preprocessing/preprocess_cross_projection.sh` — the three download sites (~L437–446 GIAB, ~L547–563 WRayner, ~L569–585 TOPMed) already skip when the file exists; only the default `TOOLS_DIR` changes
- Modify: `src/manifold_genetics/preprocessing/runner.py` — default `tools_dir`
- Modify: `src/manifold_genetics/cli.py` — `cmd_setup`, `setup_parser`
- Test: `tests/unit/test_preprocess_references.py`

**Interfaces:**
- Produces:
  ```python
  HARMONISATION_REFERENCES: Dict[str, Tuple[str, str]]   # name -> (url, relative path under tools dir)
  def default_tools_dir() -> Path                          # ToolResolver().download_dir / "preprocessing"
  def install_harmonisation_references(tools_dir: Optional[Path] = None, fetch=fetch_url) -> Dict[str, Path]
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_preprocess_references.py
"""`setup --preprocessing` fetches what the harmonise preset otherwise downloads
mid-run -- which a compute node without internet cannot do."""

import gzip
import io
import zipfile
from pathlib import Path

from manifold_genetics.preprocessing.references import (
    HARMONISATION_REFERENCES,
    install_harmonisation_references,
)


def _fake_fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if url.endswith(".zip"):
        with zipfile.ZipFile(destination, "w") as zf:
            zf.writestr("HRC-1000G-check-bim.pl", "#!/usr/bin/perl\nprint SH \"$plink --bfile\";\n")
    elif ".bed.gz" in url:
        destination.write_bytes(gzip.compress(b"chr1\t1\t2\n"))
    else:
        destination.write_bytes(b"data")


def test_places_each_reference_where_the_shell_looks(tmp_path):
    placed = install_harmonisation_references(tmp_path, fetch=_fake_fetch)
    assert set(placed) == {"giab", "wrayner", "topmed"}
    assert placed["giab"] == tmp_path / "giab/GRCh38_alldifficultregions.bed"
    assert placed["giab"].read_text() == "chr1\t1\t2\n", "the bed is stored gunzipped, as the shell writes it"
    assert placed["wrayner"] == tmp_path / "wrayner/HRC-1000G-check-bim.pl"
    assert placed["topmed"] == tmp_path / "topmed/bravo-dbsnp-all.hrc_format.tab.gz"


def test_is_idempotent(tmp_path):
    calls = []

    def counting(url, destination):
        calls.append(url)
        _fake_fetch(url, destination)

    install_harmonisation_references(tmp_path, fetch=counting)
    install_harmonisation_references(tmp_path, fetch=counting)
    assert len(calls) == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_preprocess_references.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Copy the three URLs verbatim from the shell (`grep -n "GIAB_URL=\|TOPMED_REF_URL=\|wrayner/tools" src/manifold_genetics/preprocessing/preprocess_cross_projection.sh`).

```python
# src/manifold_genetics/preprocessing/references.py
"""The files the `harmonise` preset needs and would otherwise download mid-run."""

import gzip
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from ..utils.tools import ToolResolver, fetch_url

GIAB_URL = "<copy from shell>"
WRAYNER_URL = "https://www.chg.ox.ac.uk/~wrayner/tools/HRC-1000G-check-bim-v4.3.0.zip"
TOPMED_URL = "<copy from shell>"

# name -> (url, path relative to the tools dir, as the shell expects it)
HARMONISATION_REFERENCES: Dict[str, Tuple[str, str]] = {
    "giab": (GIAB_URL, "giab/GRCh38_alldifficultregions.bed"),
    "wrayner": (WRAYNER_URL, "wrayner/HRC-1000G-check-bim.pl"),
    "topmed": (TOPMED_URL, "topmed/bravo-dbsnp-all.hrc_format.tab.gz"),
}


def default_tools_dir() -> Path:
    return ToolResolver().download_dir / "preprocessing"


def install_harmonisation_references(
    tools_dir: Optional[Path] = None, fetch: Callable[[str, Path], None] = fetch_url
) -> Dict[str, Path]:
    """Fetch each reference into ``tools_dir`` in the layout the shell reads.

    Idempotent: a file already in place is left alone. The GIAB bed is stored
    gunzipped and the WRayner checker unzipped, matching what the shell does
    when it fetches them itself.
    """
    tools_dir = Path(tools_dir) if tools_dir else default_tools_dir()
    placed = {}
    for name, (url, relative) in HARMONISATION_REFERENCES.items():
        target = tools_dir / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == "giab":
                packed = target.with_suffix(".bed.gz")
                fetch(url, packed)
                with gzip.open(packed, "rb") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                packed.unlink()
            elif name == "wrayner":
                packed = target.parent / "HRC-1000G-check-bim.zip"
                fetch(url, packed)
                with zipfile.ZipFile(packed) as zf:
                    zf.extractall(target.parent)
                # The shell comments out the VCF line of the checker; do the same
                # so a prefetched copy behaves like a shell-fetched one.
                text = target.read_text()
                target.write_text(
                    text.replace(
                        'print SH "$plink --bfile $newfile --real-ref-alleles --recode vcf',
                        '#print SH "$plink --bfile $newfile --real-ref-alleles --recode vcf',
                    )
                )
            else:
                fetch(url, target)
        placed[name] = target
    return placed


__all__ = ["HARMONISATION_REFERENCES", "default_tools_dir", "install_harmonisation_references"]
```

Check the shell's own WRayner step (`sed -n '547,566p' src/manifold_genetics/preprocessing/preprocess_cross_projection.sh`) for the exact unzip layout and the `sed` it applies, and match them.

In `runner.py`, before `shell_argv(...)`: if `options.tools_dir is None`, use `default_tools_dir()` — build a new `PreprocessOptions` via `dataclasses.replace(options, tools_dir=default_tools_dir())`.

In `cli.py`:

```python
def cmd_setup(args):
    """Download external tools (plink2, flashpca, optional plink v1.9)."""
    setup_logging(args.verbose)

    resolver = ToolResolver()
    tools = resolver.install_tools(include_plink1=not args.skip_plink1)
    if args.preprocessing:
        from .preprocessing.references import install_harmonisation_references

        tools.update(install_harmonisation_references())

    print("External tools installed:")
    for name, path in tools.items():
        print(f"  - {name}: {path}")
    return 0
```

```python
    setup_parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="Also fetch the GIAB, WRayner and TOPMed references the harmonise preset needs (~1 GB)",
    )
```
And extend the `setup` description with a line about it.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_preprocess_references.py tests/unit/test_cli_main.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/manifold_genetics/preprocessing src/manifold_genetics/cli.py tests/unit/test_preprocess_references.py
git commit -m "feat(setup): --preprocessing prefetches the harmonisation references"
```

---

### Task 8: `acquire` — rename `init`, HGDP archive layouts

**Files:**
- Modify: `src/manifold_genetics/scaffold.py` — rename `init_synthetic/init_hgdp/init_custom/init_aou` → `acquire_*`; `_extract_hgdp_archive`; `_download_hgdp_archive`; the `# Written by` headers in `_SYNTHETIC_CONFIG`, `_HGDP_CONFIG`, `_CUSTOM_CONFIG`, `_AOU_CONFIG`
- Modify: `src/manifold_genetics/cli.py` — `cmd_init` → `cmd_acquire`, parser name `acquire`, `init` removed
- Modify: every test importing `init_*` (`grep -rln "init_synthetic\|init_hgdp\|init_custom\|init_aou" tests/`), `tests/unit/test_cli_main.py`
- Test: `tests/unit/test_scaffold.py` (new class `TestHgdpLayouts`)

**Interfaces:**
- Produces: `acquire_synthetic(out_dir, force=False, seed=0)`, `acquire_hgdp(out_dir, force=False, download=True, plink2=None, archive=None)`, `acquire_custom(...)`, `acquire_aou(out_dir, force=False)` — same signatures as the `init_*` they replace. `archive` may be a local path or a `gs://` URL. New private: `_detect_hgdp_layout(raw_dir) -> str` returning `"public"` or `"workbench"`, `_normalise_workbench_layout(raw_dir, plink)`.

- [ ] **Step 1: Rename mechanically**

```bash
git grep -l "init_synthetic\|init_hgdp\|init_custom\|init_aou" -- src tests | xargs sed -i \
  -e 's/init_synthetic/acquire_synthetic/g' -e 's/init_hgdp/acquire_hgdp/g' \
  -e 's/init_custom/acquire_custom/g' -e 's/init_aou/acquire_aou/g'
sed -i 's/manifold-genetics init /manifold-genetics acquire /g' src/manifold_genetics/scaffold.py
```
In `cli.py`: rename `cmd_init` → `cmd_acquire`, `init_parser` → `acquire_parser`, `add_parser("init"` → `add_parser("acquire"`, the two `init custom` error strings → `acquire custom`, and in the description replace "Scaffold a working pipeline in one command." with "Get a cohort into a cohort directory: genotypes, labels, colormap, config." Update `SUBCOMMANDS` in `test_cli_main.py` if `init` is listed; add `acquire`.

Run: `uv run pytest tests/unit/test_scaffold.py tests/unit/test_cli_main.py tests/unit/test_colormap_warning.py tests/integration/test_init_runs.py -q`
Expected: all pass (`git mv tests/integration/test_init_runs.py tests/integration/test_acquire_runs.py`).

- [ ] **Step 2: Write the failing layout tests**

Add to `tests/unit/test_scaffold.py`:

```python
class TestHgdpLayouts:
    """The workbench ships a different HGDP+1KGP archive from the public one."""

    def _workbench_archive(self, tmp_path):
        raw = tmp_path / "src"
        raw.mkdir()
        (raw / "extractedChrAllUnpruned.bim").write_text("1\trs1\t0\t100\tA\tG\n22\trs2\t0\t200\tC\tT\n")
        (raw / "extractedChrAllUnpruned.fam").write_text(
            "forReferenceYoruba\tS1\t0\t0\t0\t-9\nforReferenceFrench\tS2\t0\t0\t0\t-9\n"
        )
        (raw / "extractedChrAllUnpruned.bed").write_bytes(b"\x6c\x1b\x01\x00\x00")
        archive = tmp_path / "1KGPHGDP.tar.gz"
        import tarfile

        with tarfile.open(archive, "w:gz") as tar:
            for f in raw.iterdir():
                tar.add(f, arcname=f.name)
        return archive

    def test_detects_the_public_layout(self, tmp_path):
        from manifold_genetics.scaffold import _detect_hgdp_layout

        (tmp_path / "full_dataset.bed").write_bytes(b"")
        assert _detect_hgdp_layout(tmp_path) == "public"

    def test_detects_the_workbench_layout(self, tmp_path):
        from manifold_genetics.scaffold import _detect_hgdp_layout

        (tmp_path / "extractedChrAllUnpruned.bed").write_bytes(b"")
        assert _detect_hgdp_layout(tmp_path) == "workbench"

    def test_an_unknown_layout_names_both_it_looked_for(self, tmp_path):
        from manifold_genetics.scaffold import _detect_hgdp_layout

        with pytest.raises(FileNotFoundError, match="full_dataset.*extractedChrAllUnpruned"):
            _detect_hgdp_layout(tmp_path)

    def test_the_workbench_archive_is_normalised_to_the_public_layout(self, tmp_path):
        from manifold_genetics.scaffold import _extract_hgdp_archive

        raw = tmp_path / "raw"
        _extract_hgdp_archive(self._workbench_archive(tmp_path), raw)
        bim = (raw / "full_dataset.bim").read_text().splitlines()
        assert bim[0].startswith("chr1\t") and bim[1].startswith("chr22\t")
        fam = (raw / "full_dataset.fam").read_text().splitlines()
        assert fam[0].startswith("Yoruba\t"), "forReference prefix stripped"
        metadata = pd.read_csv(raw / "metadata.csv")
        assert list(metadata["project_meta.sample_id"]) == ["S1", "S2"]
        assert list(metadata["Population"]) == ["Yoruba", "French"]

    def test_a_gs_url_is_fetched_with_gsutil(self, tmp_path, monkeypatch):
        from manifold_genetics import scaffold

        calls = []
        monkeypatch.setattr(scaffold.subprocess, "run", lambda argv, **kw: calls.append(argv))
        monkeypatch.setattr(scaffold, "_extract_hgdp_archive", lambda a, r: None)
        monkeypatch.setattr(scaffold, "_run_plink2_keep", lambda *a, **k: None)
        with pytest.raises(FileNotFoundError):  # nothing was really extracted
            scaffold.acquire_hgdp(tmp_path, archive="gs://bucket/1KGPHGDP.tar.gz")
        assert calls and calls[0][:2] == ["gsutil", "cp"]
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/unit/test_scaffold.py::TestHgdpLayouts -v`
Expected: FAIL — `_detect_hgdp_layout` not found.

- [ ] **Step 4: Implement**

In `scaffold.py`:

```python
import subprocess  # at top, if not present

_PUBLIC_PREFIX = "full_dataset"
_WORKBENCH_PREFIX = "extractedChrAllUnpruned"
_WORKBENCH_FID_PREFIX = "forReference"


def _detect_hgdp_layout(raw_dir: Path) -> str:
    """Which HGDP+1KGP archive was unpacked here.

    "public" is the Dropbox archive `acquire hgdp` fetches: ``full_dataset.*``,
    already filtered and LD-pruned, with a ``metadata.csv``. "workbench" is the
    archive kept beside All of Us: ``extractedChrAllUnpruned.*``, unfiltered,
    chromosomes without a ``chr`` prefix, and the population carried in the FID
    as ``forReference<Population>``. They are not the same panel and the log
    line says which one this is.
    """
    if (raw_dir / f"{_PUBLIC_PREFIX}.bed").exists():
        return "public"
    if (raw_dir / f"{_WORKBENCH_PREFIX}.bed").exists():
        return "workbench"
    raise FileNotFoundError(
        f"{raw_dir} holds neither {_PUBLIC_PREFIX}.bed (the public archive) nor "
        f"{_WORKBENCH_PREFIX}.bed (the workbench archive)."
    )


def _normalise_workbench_layout(raw_dir: Path) -> None:
    """Rewrite the workbench archive into the public layout, as
    examples/aou/hgdp_1kgp_proj/prepare_data.sh steps 3 and 14 did: 'chr' on
    every chromosome, the population out of the FID and into metadata.csv."""
    src = raw_dir / _WORKBENCH_PREFIX
    dst = raw_dir / _PUBLIC_PREFIX
    bim = pd.read_csv(f"{src}.bim", sep=r"\s+", header=None, dtype=str)
    bim[0] = bim[0].where(bim[0].str.startswith("chr"), "chr" + bim[0])
    bim.to_csv(f"{dst}.bim", sep="\t", header=False, index=False)
    fam = pd.read_csv(f"{src}.fam", sep=r"\s+", header=None, dtype=str)
    population = fam[0].str.replace(f"^{_WORKBENCH_FID_PREFIX}", "", regex=True)
    fam[0] = population
    fam.to_csv(f"{dst}.fam", sep="\t", header=False, index=False)
    Path(f"{src}.bed").rename(f"{dst}.bed")
    pd.DataFrame({"project_meta.sample_id": fam[1], "Population": population}).to_csv(
        raw_dir / "metadata.csv", index=False
    )
    logger.warning(
        "Workbench HGDP+1KGP archive: unfiltered, %d samples. Not the public panel; "
        "run `preprocess --preset harmonise --fit-has-chr-prefix` before fitting on it.",
        len(fam),
    )
```

In `_extract_hgdp_archive(archive, raw_dir)`: after extraction, `if _detect_hgdp_layout(raw_dir) == "workbench": _normalise_workbench_layout(raw_dir)`. Read the existing function first — it may extract into a subdirectory; the tar in the workbench archive extracts to `<TEMP>/1KGPHGDP/` per the old script (`REF_DIR`), so search `raw_dir` and one level down for the prefix and move files up.

In `acquire_hgdp`, before the `local = Path(archive) ...` line:

```python
        if archive and str(archive).startswith("gs://"):
            fetched = data_dir / Path(str(archive)).name
            if not fetched.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Fetching %s with gsutil", archive)
                subprocess.run(["gsutil", "cp", str(archive), str(fetched)], check=True)
            archive = fetched
```

`hgdp_subsets(metadata)` and `_write_hgdp_labels` expect the public metadata columns (`hgdp_subsets` uses relatedness/QC columns; `_write_hgdp_labels` needs `_HGDP_REGION_COLUMN`). For the workbench layout neither exists. In `acquire_hgdp`, branch on the layout:

```python
    layout = _detect_hgdp_layout(raw_dir)
    metadata = pd.read_csv(raw_dir / "metadata.csv")
    if layout == "public":
        fit_ids, project_ids = hgdp_subsets(metadata)
    else:
        # The workbench panel carries no relatedness or QC columns; the old
        # AoU flow fitted on every sample, and so does this.
        fit_ids = project_ids = metadata["project_meta.sample_id"]
    ...
    if layout == "public":
        _write_hgdp_labels(metadata, project_ids, data_dir / "labels.csv", out_dir / "colormap.json")
    else:
        labels = metadata.rename(columns={"project_meta.sample_id": "sample_id"})
        labels.to_csv(data_dir / "labels.csv", index=False)
        _write_generated_colormap(labels, out_dir / "colormap.json")
```

Note the spec's decision 2 said fit on the 3,400 unrelated; that is only possible when the metadata has the columns to say who is unrelated, which the workbench archive lacks — so the workbench layout fits on all, exactly as the old script did. Record this in the spec's issue 2 as an amendment ("accepted difference applies to the public panel only").

- [ ] **Step 5: Run all scaffold tests**

Run: `uv run pytest tests/unit/test_scaffold.py tests/unit/test_cli_main.py tests/integration/test_acquire_runs.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/
git add -A src tests
git commit -m "feat(acquire): rename init to acquire; accept the workbench HGDP archive"
```

---

### Task 9: HGDP idempotence — the real-data sanity check

**Files:**
- Create: `tests/integration/test_hgdp_idempotence.py`

**Interfaces:**
- Consumes: `acquire_hgdp`, `preprocess`, `PreprocessOptions`.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_hgdp_idempotence.py
"""The public HGDP+1KGP panel is already filtered and LD-pruned, so filtering it
again must change nothing. This is the end-to-end check that `acquire`,
`preprocess` and the shell agree on what a cohort directory is -- on real data,
without private access.

Needs the archive: set MG_HGDP_ARCHIVE to a local hgdp_1kgp_full.tar.gz, or
run with network on a login node."""

import os
from pathlib import Path

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import PreprocessOptions, preprocess
from manifold_genetics.scaffold import acquire_hgdp
from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _variants(prefix):
    return {tuple(line.split()[i] for i in (0, 3, 4, 5)) for line in open(f"{prefix}.bim")}


def _samples(prefix):
    return [line.split()[1] for line in open(f"{prefix}.fam")]


@pytest.fixture(scope="module")
def hgdp(tmp_path_factory):
    try:
        ToolResolver().resolve_plink2()
        ToolResolver().resolve_plink1()
    except ToolNotFoundError as exc:
        pytest.skip(str(exc))
    archive = os.environ.get("MG_HGDP_ARCHIVE")
    if archive is None:
        pytest.skip("set MG_HGDP_ARCHIVE to the downloaded hgdp_1kgp_full.tar.gz")
    out = tmp_path_factory.mktemp("hgdp")
    return acquire_hgdp(out, archive=Path(archive), download=False)


@pytest.mark.parametrize("preset,extra", [("intersect-only", {}), (None, {"skip_wrayner": True, "skip_project_maf": True})])
def test_filtering_the_filtered_panel_changes_nothing(hgdp, tmp_path, preset, extra):
    before = load_config(hgdp)
    after = load_config(
        preprocess(
            hgdp, tmp_path / (preset or "ukbb-flags"),
            options=PreprocessOptions(preset=preset, threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
                                      memory=16000, **extra),
        )
    )
    for side in ("fit_plink", "project_plink"):
        assert _samples(after[side]) == _samples(before[side])
        lost = _variants(before[side]) - _variants(after[side])
        assert not lost, f"{len(lost)} of {len(_variants(before[side]))} variants lost on {side}"
        assert _variants(after[side]) == _variants(before[side])
```

The second parameter set is the UKBB flag set (default GIAB/HLA/LD-prune/dedup on). If it strips variants — HLA exclusion on a panel that was built `.noHLA` should strip none; LD pruning at the same r² on an already-pruned panel should strip few but possibly not zero — report the count rather than force it to zero: change the assertion to a tolerance only after seeing the number, and record it in the spec.

- [ ] **Step 2: Run on the compute node**

```bash
JOBID=$(squeue --me -h -o %i | head -1)
srun --jobid=$JOBID --overlap --ntasks=1 --time=60:00 bash -c \
  'cd /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics && source .venv/bin/activate && \
   MG_HGDP_ARCHIVE=examples/hgdp_1kgp/data/hgdp_1kgp_full.tar.gz \
   pytest tests/integration/test_hgdp_idempotence.py -v -s 2>&1 | tail -60'
```
Expected: `intersect-only` passes exactly. Record the UKBB-flags result (exact, or the number lost) in the spec under issue 1.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_hgdp_idempotence.py archive/superpowers/specs/2026-09-14-acquire-preprocess-design.md
git commit -m "test: filtering the public HGDP panel again changes nothing"
```

---

### Task 10: `subsample`

**Files:**
- Create: `src/manifold_genetics/preprocessing/subsample.py`
- Modify: `src/manifold_genetics/preprocessing/__init__.py`, `src/manifold_genetics/cli.py`
- Test: `tests/unit/test_subsample.py`, `tests/unit/test_cli_main.py`

**Interfaces:**
- Consumes: `read_cohort`, `read_fam_ids`, `filter_labels_to_fam`, `write_cohort_config`, `ToolResolver.resolve_plink2()`, `scaffold._run_plink2_keep(bfile, keep, out, plink2)`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class Group: column: str; pattern: str; count: int
  def parse_group(spec: str) -> Group                       # "COLUMN=PATTERN:COUNT"
  def select_by_groups(labels: pd.DataFrame, groups: Sequence[Group], *, include_rest: bool, seed: int) -> List[str]
  def subsample(config: PathLike, out_dir: PathLike, *, groups=(), include_rest=False, seed=42,
                fit_samples: Optional[PathLike] = None, force=False,
                keep_runner: Callable = _run_plink2_keep) -> Path
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_subsample.py
"""Choosing the fit samples of a cohort by label counts or a list."""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing.subsample import Group, parse_group, select_by_groups, subsample
from manifold_genetics.scaffold import acquire_synthetic


class TestParseGroup:
    def test_column_pattern_count(self):
        assert parse_group("race_ethnicity=White|European:10000") == Group("race_ethnicity", "White|European", 10000)

    def test_pattern_may_contain_colons_and_equals(self):
        assert parse_group("col=a=b:c:5") == Group("col", "a=b:c", 5)

    @pytest.mark.parametrize("bad", ["nocolumn:5", "col=:5", "col=x", "col=x:0", "col=x:five"])
    def test_malformed_specs_are_rejected(self, bad):
        with pytest.raises(ValueError, match="COLUMN=PATTERN:COUNT"):
            parse_group(bad)


@pytest.fixture
def labels():
    return pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(10)],
        "group": ["British"] * 4 + ["Irish"] * 3 + ["Other"] * 3,
    })


class TestSelectByGroups:
    def test_takes_count_from_each_group_and_all_when_fewer(self, labels):
        chosen = select_by_groups(labels, [Group("group", "British", 2), Group("group", "Irish", 5)],
                                  include_rest=False, seed=0)
        kinds = labels.set_index("sample_id").loc[chosen, "group"]
        assert (kinds == "British").sum() == 2 and (kinds == "Irish").sum() == 3
        assert "Other" not in set(kinds)

    def test_include_rest_appends_the_unmatched(self, labels):
        chosen = select_by_groups(labels, [Group("group", "British", 1)], include_rest=True, seed=0)
        kinds = labels.set_index("sample_id").loc[chosen, "group"]
        assert (kinds == "Other").sum() == 3 and (kinds == "Irish").sum() == 0

    def test_matching_is_case_insensitive_regex(self, labels):
        chosen = select_by_groups(labels, [Group("group", "brit|iri", 10)], include_rest=False, seed=0)
        assert len(chosen) == 7

    def test_later_groups_do_not_reuse_samples(self, labels):
        chosen = select_by_groups(labels, [Group("group", "British", 4), Group("group", "B", 10)],
                                  include_rest=False, seed=0)
        assert len(chosen) == len(set(chosen)) == 4

    def test_seed_makes_it_reproducible(self, labels):
        a = select_by_groups(labels, [Group("group", "British", 2)], include_rest=False, seed=7)
        b = select_by_groups(labels, [Group("group", "British", 2)], include_rest=False, seed=7)
        assert a == b

    def test_an_unknown_column_is_named(self, labels):
        with pytest.raises(ValueError, match="nope.*group"):
            select_by_groups(labels, [Group("nope", "x", 1)], include_rest=False, seed=0)


def _fake_keep(bfile, keep, out, plink2):
    ids = [line.split()[1] for line in Path(keep).read_text().splitlines()]
    fam = {l.split()[1]: l for l in Path(f"{bfile}.fam").read_text().splitlines()}
    Path(f"{out}.fam").write_text("".join(fam[i] + "\n" for i in ids))
    for ext in ("bed", "bim"):
        Path(f"{out}.{ext}").write_bytes(Path(f"{bfile}.{ext}").read_bytes())


class TestSubsample:
    @pytest.fixture
    def cohort(self, tmp_path, monkeypatch):
        from manifold_genetics.utils import tools

        monkeypatch.setattr(tools.ToolResolver, "resolve_plink2", lambda self: "/stub/plink2")
        acquire_synthetic(tmp_path / "in")
        return tmp_path / "in/config.yaml"

    def test_writes_a_subsample_cohort(self, cohort, tmp_path):
        config = subsample(cohort, tmp_path / "out", groups=[Group("branch", "^0$", 3)],
                           include_rest=False, keep_runner=_fake_keep)
        loaded = load_config(config)
        assert loaded["embedding_input"] == "fit", "subsample preset expected"
        fit = [l.split()[1] for l in open(f"{loaded['fit_plink']}.fam")]
        assert len(fit) == 3
        project = [l.split()[1] for l in open(f"{loaded['project_plink']}.fam")]
        assert len(project) > 3, "the project side is the whole cohort"
        fit_labels = pd.read_csv(loaded["fit_labels"], dtype=str)
        assert list(fit_labels["sample_id"]) == fit

    def test_a_sample_list_is_honoured_verbatim(self, cohort, tmp_path):
        fam = [l.split() for l in open(tmp_path / "in/data/project_subset.fam")]
        keep = tmp_path / "keep.txt"
        keep.write_text("".join(f"{f[0]} {f[1]}\n" for f in fam[:5]))
        config = subsample(cohort, tmp_path / "out", fit_samples=keep, keep_runner=_fake_keep)
        fit = [l.split()[1] for l in open(f"{load_config(config)['fit_plink']}.fam")]
        assert fit == [f[1] for f in fam[:5]]

    def test_needs_exactly_one_way_of_choosing(self, cohort, tmp_path):
        with pytest.raises(ValueError, match="one of"):
            subsample(cohort, tmp_path / "out", keep_runner=_fake_keep)
        with pytest.raises(ValueError, match="one of"):
            subsample(cohort, tmp_path / "out", groups=[Group("branch", "0", 1)],
                      fit_samples=tmp_path / "x", keep_runner=_fake_keep)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_subsample.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/manifold_genetics/preprocessing/subsample.py
"""`subsample`: choose the fit samples of a cohort. Samples only, never SNPs."""

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pandas as pd

from ..scaffold import _run_plink2_keep
from ..utils.io import PathLike
from ..utils.tools import ToolResolver
from .cohort import filter_labels_to_fam, read_cohort, read_fam_ids, write_cohort_config

logger = logging.getLogger(__name__)

_GROUP_FORMAT = "COLUMN=PATTERN:COUNT, e.g. race_ethnicity=White|European:10000"


@dataclass(frozen=True)
class Group:
    column: str
    pattern: str
    count: int


def parse_group(spec: str) -> Group:
    column, sep, rest = spec.partition("=")
    pattern, sep2, count_text = rest.rpartition(":")
    if not sep or not sep2 or not column or not pattern:
        raise ValueError(f"--group {spec!r} is not of the form {_GROUP_FORMAT}")
    try:
        count = int(count_text)
    except ValueError:
        raise ValueError(f"--group {spec!r}: COUNT must be an integer ({_GROUP_FORMAT})")
    if count <= 0:
        raise ValueError(f"--group {spec!r}: COUNT must be positive ({_GROUP_FORMAT})")
    return Group(column, pattern, count)


def select_by_groups(
    labels: pd.DataFrame, groups: Sequence[Group], *, include_rest: bool, seed: int
) -> List[str]:
    """Sample IDs chosen group by group; a sample is taken at most once.

    This is examples/aou/shared/select_samples.py with the column named rather
    than guessed, which is what makes it serve UK Biobank and All of Us alike.
    """
    taken: List[str] = []
    taken_set: set = set()
    matched: set = set()
    for group in groups:
        if group.column not in labels.columns:
            raise ValueError(
                f"column {group.column!r} is not in the labels (have: {', '.join(labels.columns)})"
            )
        mask = labels[group.column].astype(str).str.contains(group.pattern, case=False, regex=True, na=False)
        matched.update(labels.loc[mask, "sample_id"])
        candidates = labels.loc[mask & ~labels["sample_id"].isin(taken_set), "sample_id"]
        if len(candidates) > group.count:
            chosen = candidates.sample(n=group.count, random_state=seed)
            logger.info("%s=%s: %d available, taking %d", group.column, group.pattern, len(candidates), group.count)
        else:
            chosen = candidates
            logger.info("%s=%s: %d available, taking all", group.column, group.pattern, len(candidates))
        taken.extend(chosen)
        taken_set.update(chosen)
    if include_rest:
        rest = labels.loc[~labels["sample_id"].isin(matched), "sample_id"]
        logger.info("rest: %d samples matched no group", len(rest))
        taken.extend(rest)
    return taken


def subsample(
    config: PathLike,
    out_dir: PathLike,
    *,
    groups: Sequence[Group] = (),
    include_rest: bool = False,
    seed: int = 42,
    fit_samples: Optional[PathLike] = None,
    force: bool = False,
    keep_runner: Callable = _run_plink2_keep,
) -> Path:
    """Write a cohort directory whose fit set is a chosen subset of the project set.

    The input's fit side is dropped: given a projection (HGDP fit, biobank
    project), the output fits on a subset of the biobank.

    Raises:
        ValueError: neither or both of ``groups`` and ``fit_samples`` given.
        FileExistsError: ``out_dir/config.yaml`` exists and ``force`` is False.
    """
    if bool(groups) == (fit_samples is not None):
        raise ValueError("choose fit samples by exactly one of --group or --fit-samples")
    out_dir = Path(out_dir).expanduser().resolve()
    config_path = out_dir / "config.yaml"
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} exists. Pass --force to overwrite it.")

    cohort = read_cohort(config)
    project = cohort.project
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    keep = data_dir / "fit_samples.txt"
    if fit_samples is not None:
        shutil.copy(fit_samples, keep)
    else:
        labels = pd.read_csv(project.labels, dtype={"sample_id": str}, low_memory=False)
        available = set(read_fam_ids(project.plink))
        labels = labels[labels["sample_id"].isin(available)]
        chosen = select_by_groups(labels, groups, include_rest=include_rest, seed=seed)
        fam = pd.read_csv(f"{project.plink}.fam", sep=r"\s+", header=None, dtype=str, usecols=[0, 1])
        fid = dict(zip(fam[1], fam[0]))
        keep.write_text("".join(f"{fid[s]}\t{s}\n" for s in chosen))
        logger.info("Selected %d of %d samples for the fit set", len(chosen), len(available))

    keep_runner(project.plink, keep, data_dir / "fit_subset", ToolResolver().resolve_plink2())
    for ext in ("bed", "bim", "fam"):
        target = data_dir / f"project_subset.{ext}"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(Path(f"{project.plink}.{ext}").resolve())

    filter_labels_to_fam(project.labels, data_dir / "fit_subset", data_dir / "fit_labels.csv")
    filter_labels_to_fam(project.labels, data_dir / "project_subset", data_dir / "project_labels.csv")
    shutil.copy(project.colormap, out_dir / "colormap_fit.json")
    shutil.copy(project.colormap, out_dir / "colormap_project.json")

    return write_cohort_config(
        out_dir, based_on=cohort, preset="subsample",
        data={
            "fit_plink": "data/fit_subset", "project_plink": "data/project_subset",
            "fit_labels": "data/fit_labels.csv", "project_labels": "data/project_labels.csv",
            "fit_colormap": "colormap_fit.json", "project_colormap": "colormap_project.json",
            "output_dir": "outputs",
        },
        written_by="manifold-genetics subsample",
    )


__all__ = ["Group", "parse_group", "select_by_groups", "subsample"]
```

Symlinking `project_subset` rather than copying: the biobank `.bed` is ~15 GB. `load_config` resolves symlinks fine. Note this in the CLI description.

Export from `__init__.py`: `from .subsample import Group, parse_group, subsample`.

CLI, modelled on Task 6:

```python
def cmd_subsample(args):
    """Choose the fit samples of a cohort."""
    setup_logging(args.verbose)

    from . import preprocessing

    try:
        groups = [preprocessing.parse_group(g) for g in (args.group or [])]
        config = preprocessing.subsample(
            args.config, args.out, groups=groups, include_rest=args.include_rest, seed=args.seed,
            fit_samples=args.fit_samples, force=args.force,
        )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"\nWrote {config}")
    print("\nNext:")
    print(f"  manifold-genetics run {config} --dry-run")
    print(f"  manifold-genetics run {config}")
    return 0
```

Parser: positional `config`; `--out` required; `--group` (`action="append"`, metavar `COLUMN=PATTERN:COUNT`); `--include-rest`; `--seed` (int, default 42); `--fit-samples`; `--force`; `--verbose`. Description:

```
Optional. Reads a cohort directory and writes one whose fit set is a chosen
subset of its project set; the project set is linked, not copied. The output
uses the `subsample` preset (fit on the subset, embed the subset, landmarked).

  --group COLUMN=PATTERN:COUNT   take COUNT samples whose COLUMN matches PATTERN
                                 (case-insensitive regex); repeatable; a sample is
                                 taken once. --include-rest adds every unmatched sample.
  --fit-samples FILE             a FID IID list chosen elsewhere.

  subsample proj/config.yaml --out 10k/ \
      --group "race_ethnicity=White|European:10000" \
      --group "race_ethnicity=Black or African American:10000" --include-rest
```

Add `"subsample"` to `SUBCOMMANDS` in `test_cli_main.py` and a dispatch test mirroring Task 6's (assert `groups` parsed and `include_rest` forwarded).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_subsample.py tests/unit/test_cli_main.py tests/unit/test_docs_match_cli.py -v`
Expected: pass (add a `## subsample` stub to `docs/cli.md` if the docs test demands it).

- [ ] **Step 5: Extend the synthetic integration test**

Append to `tests/integration/test_preprocess_synthetic.py`:

```python
def test_preprocess_then_subsample_then_dry_run(tmp_path, tools):
    from manifold_genetics.cli import main
    from manifold_genetics.preprocessing import Group, subsample

    init_synthetic(tmp_path / "in")
    filtered = preprocess(
        tmp_path / "in/config.yaml", tmp_path / "filtered",
        options=PreprocessOptions(preset="intersect-only", min_common_snps=100, memory=2000, threads=2),
    )
    config = subsample(filtered, tmp_path / "sub", groups=[Group("branch", ".", 20)])
    assert main(["run", str(config), "--dry-run"]) == 0
```
(`init_synthetic` is now `acquire_synthetic` after Task 8 — use the new name.)

Run on the compute node as in Task 5 step 7. Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/
git add -A src tests docs/cli.md
git commit -m "feat: subsample chooses a cohort's fit samples by label counts or a list"
```

---

### Task 11: UKBB reproduction (`requires_private_data`)

**Files:**
- Create: `tests/integration/test_ukbb_preprocess_idempotence.py`

**Interfaces:**
- Consumes: `acquire_custom`, `preprocess`, `PreprocessOptions`, `subsample`.

- [ ] **Step 1: Write the test**

The intersected UKBB cohort at `examples/ukbb/hgdp_1kgp_proj/data/` (fit 3,340 × 120,849; project 486,748 × 120,849) was produced by the UKBB flag set. Filtering it again with the same flags must reproduce it — the same idempotence argument as HGDP, on the biobank that is reachable.

```python
# tests/integration/test_ukbb_preprocess_idempotence.py
"""examples/ukbb/hgdp_1kgp_proj/data was produced by preprocess_cross_projection.sh
with --skip-wrayner --skip-biobank-maf. Running `preprocess` with the same flags
on that output must reproduce it. The one real-biobank check available."""

import os
from pathlib import Path

import pytest

from manifold_genetics.pipeline.configfile import load_config
from manifold_genetics.preprocessing import Group, PreprocessOptions, preprocess, subsample
from manifold_genetics.scaffold import acquire_custom

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.requires_private_data]

UKBB = Path(__file__).resolve().parents[2] / "examples/ukbb/hgdp_1kgp_proj"


def _variants(prefix):
    return {tuple(line.split()[i] for i in (0, 3, 4, 5)) for line in open(f"{prefix}.bim")}


def _samples(prefix):
    return [line.split()[1] for line in open(f"{prefix}.fam")]


@pytest.fixture(scope="module")
def cohorts(tmp_path_factory):
    if not (UKBB / "data/project_subset.bed").exists():
        pytest.skip("UKBB intersected data is not on this machine")
    root = tmp_path_factory.mktemp("ukbb")
    ref = acquire_custom(root / "ref", fit_plink=UKBB / "data/fit_subset", labels=UKBB / "data/fit_labels.csv")
    bio = acquire_custom(root / "bio", fit_plink=UKBB / "data/project_subset", labels=UKBB / "data/project_labels.csv")
    return ref, bio


def test_the_ukbb_flags_reproduce_the_intersected_cohort(cohorts, tmp_path):
    ref, bio = cohorts
    config = preprocess(
        ref, tmp_path / "out", project_config=bio,
        options=PreprocessOptions(
            skip_wrayner=True, skip_project_maf=True,
            threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), memory=48000, cleanup=True,
            temp_dir=Path(os.environ.get("SLURM_TMPDIR", tmp_path)) / "temp",
        ),
    )
    after = load_config(config)
    assert _samples(after["fit_plink"]) == _samples(UKBB / "data/fit_subset")
    assert _samples(after["project_plink"]) == _samples(UKBB / "data/project_subset")
    before = _variants(UKBB / "data/project_subset")
    lost = before - _variants(after["project_plink"])
    assert not lost, f"{len(lost)} of {len(before)} variants lost"


def test_subsample_reproduces_the_wb_irish_selection(cohorts, tmp_path):
    keep = UKBB.parent / "10k_WB_5K_Irish/data/fit_samples.txt"
    if not keep.exists():
        pytest.skip("10k_WB_5K_Irish selection is not on this machine")
    _, bio = cohorts
    config = subsample(bio, tmp_path / "sub", fit_samples=keep)
    fit = _samples(load_config(config)["fit_plink"])
    expected = [line.split()[1] for line in open(keep)]
    assert sorted(fit) == sorted(expected)
```

`acquire_custom`'s `labels` for the biobank: `project_labels.csv` covers the 486k project samples. Check `_checked_labels`'s overlap rule accepts it (it should — it is the file the published run used).

- [ ] **Step 2: Run on the compute node**

The project `.bed` is ~15 GB; LD pruning and GIAB/HLA exclusion on 486k samples will take a while on 8 cores. Use `$SLURM_TMPDIR` for scratch.

```bash
JOBID=$(squeue --me -h -o %i | head -1)
srun --jobid=$JOBID --overlap --ntasks=1 --time=180:00 bash -c \
  'cd /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics && source .venv/bin/activate && \
   pytest tests/integration/test_ukbb_preprocess_idempotence.py -v -s 2>&1 | tail -80'
```
Expected: both pass. If variants are lost, print which step lost them from the shell's log in `temp/` and record it; do not loosen the assertion without understanding the step.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_ukbb_preprocess_idempotence.py
git commit -m "test: preprocess reproduces the UKBB intersected cohort"
```

---

### Task 12: `acquire aou` — port `download_aou_data.sh`

**Files:**
- Create: `src/manifold_genetics/aou.py`
- Modify: `src/manifold_genetics/scaffold.py` — `acquire_aou` delegates to `aou.acquire_aou`; `_AOU_CONFIG` comment updated
- Test: `tests/unit/test_aou_acquire.py`

**Interfaces:**
- Consumes: `aou_environment_problems()`, `_refuse_to_clobber`, `_write_generated_colormap`, `ToolResolver.resolve_plink1()` (the old script splits/merges with plink 1.9 — check `download_aou_data.sh` L120–160 for which binary).
- Produces: `acquire_aou(out_dir, force=False, *, runner=subprocess.run, query=pandas_gbq_read) -> Path`.

Read `examples/aou/shared/download_aou_data.sh` in full first. Port its steps in order; each is a function taking its inputs and a `runner`, so the sequence is testable without GCS:

1. `_fetch_plink(bucket_root, aou_dir, runner)` — `gsutil -m cp gs://…/plink/{arrays.bed,bim,fam}` (copy exact names from the script) unless present.
2. `_fix_fam(aou_dir)` — the FAM rewrite the script does (read the script: it adds a family ID; reproduce with pandas).
3. `_split_by_chromosome(prefix, aou_dir, plink, threads, runner)` — 22 `plink --chr N --make-bed` calls; skip when `extractedChr22.bed` exists.
4. `_merge_chromosomes(...)` — however the script produces `extractedChrAll`; reproduce.
5. `_fetch_metadata(cdr, meta_dir, query)` — the two BigQuery SQL strings copied verbatim from the script's embedded `extract_metadata.py`; written to `DemographicData.tsv` and `SocioeconomicZipCodes.tsv`. `query` defaults to `pandas_gbq.read_gbq`, imported lazily (the `aou` extra).
6. `_write_labels(meta_dir, fam, data_dir, out_dir)` — `person_id` → `sample_id`, filtered to the `.fam`, colormap from `race_ethnicity` via `_write_generated_colormap`.
7. Write `config.yaml` from `_AOU_CONFIG` with `fit_plink`/`project_plink` both `data/project_subset` (the cohort alone is `whole_cohort`; `preprocess` against the HGDP cohort turns it into a projection) — **change `_AOU_CONFIG`** to `preset: whole_cohort` with `labels`/`colormap` and drop the "not fetched" comment.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_aou_acquire.py
"""acquire_aou with GCS, plink and BigQuery replaced by stubs that leave behind
what the real ones do. Untestable for real until workbench access returns (#124);
these pin the sequence and the files."""

from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics import aou


@pytest.fixture
def workbench(monkeypatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "proj")
    monkeypatch.setenv("WORKSPACE_CDR", "cdr.v8")
    monkeypatch.setattr(aou, "aou_environment_problems", lambda: [])
    monkeypatch.setattr(aou.ToolResolver, "resolve_plink1", lambda self: "/stub/plink")


def _runner_factory(calls):
    def runner(argv, **kw):
        calls.append(argv)
        if argv[0] == "gsutil":
            for target in argv[argv.index("cp") + 1 :][1:] or [argv[-1]]:
                p = Path(target)
                p.mkdir(parents=True, exist_ok=True)
                (p / "arrays.fam").write_text("0 P1 0 0 0 -9\n0 P2 0 0 0 -9\n")
                (p / "arrays.bim").write_text("1 rs1 0 1 A G\n")
                (p / "arrays.bed").write_bytes(b"\x6c\x1b\x01")
        elif argv[0] == "/stub/plink":
            out = Path(argv[argv.index("--out") + 1])
            for ext in ("bed", "bim", "fam"):
                (out.parent / f"{out.name}.{ext}").write_bytes(b"x")
    return runner


def _query(sql, **kw):
    if "race" in sql:
        return pd.DataFrame({"person_id": [1, 2], "race_ethnicity": ["White", "Asian"]})
    return pd.DataFrame({"person_id": [1, 2], "zip": ["1", "2"]})


def test_writes_a_whole_cohort_directory(tmp_path, workbench):
    calls = []
    config = aou.acquire_aou(tmp_path, runner=_runner_factory(calls), query=_query)
    assert config == tmp_path / "config.yaml"
    assert calls[0][0] == "gsutil"
    assert any(a[0] == "/stub/plink" for a in calls)
    assert (tmp_path / "data/project_subset.bed").exists()
    labels = pd.read_csv(tmp_path / "data/labels.csv", dtype=str)
    assert list(labels.columns)[:2] == ["sample_id", "race_ethnicity"]
    assert (tmp_path / "colormap.json").exists()


def test_outside_the_workbench_it_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(aou, "aou_environment_problems", lambda: ["GOOGLE_PROJECT is not set"])
    with pytest.raises(EnvironmentError, match="GOOGLE_PROJECT"):
        aou.acquire_aou(tmp_path)
    assert not list(tmp_path.iterdir())
```

Adjust the stub file names once you have read the script (the real bucket file names replace `arrays.*`).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_aou_acquire.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Port the script**

Write `src/manifold_genetics/aou.py` following the seven functions above, each a direct transcription of the corresponding shell block with the shell's comments carried over as docstrings. `acquire_aou` runs them in order, skipping each whose output exists (mirror the script's own `if [[ -f ... ]]` checks). Move `aou_environment_problems`, `_AOU_REQUIRED_ENV`, `_AOU_REQUIRED_TOOLS` from `scaffold.py` into `aou.py` and re-import them in `scaffold.py` so existing tests keep passing. `scaffold.acquire_aou` becomes `from .aou import acquire_aou`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_aou_acquire.py tests/unit/test_scaffold.py -v`
Expected: pass. `TestInitAou`-style tests in `test_scaffold.py` that assert the old "not fetched" warning must be updated to the new behaviour.

- [ ] **Step 5: Commit**

```bash
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/
git add -A src tests
git commit -m "feat(acquire): aou fetches the cohort, ported from download_aou_data.sh"
```

Update `#124` with a comment: "`acquire aou` is written against `download_aou_data.sh`; run `acquire hgdp --archive gs://…`, `acquire aou`, `preprocess … --preset harmonise --fit-has-chr-prefix` when access returns."

---

### Task 13: `--geosketch`

**Files:**
- Modify: `src/manifold_genetics/preprocessing/subsample.py`, `src/manifold_genetics/cli.py`
- Test: `tests/unit/test_subsample.py`

**Interfaces:**
- Produces: `select_by_geosketch(pca: pd.DataFrame, n: int, seed: int, sketch=geosketch.gs) -> List[str]`; `subsample(..., geosketch: Optional[int] = None, pca: Optional[PathLike] = None)`.

- [ ] **Step 1: Read the old selection**

`sed -n '110,160p' examples/ukbb/geosketch_phate/prepare_data.sh` — note which columns of the PCA CSV it passes to `geosketch.gs` and any `replace`/`seed` arguments.

- [ ] **Step 2: Write the failing tests**

```python
class TestGeosketch:
    def test_chooses_n_ids_from_the_pca_table(self):
        from manifold_genetics.preprocessing.subsample import select_by_geosketch

        pca = pd.DataFrame({"sample_id": [f"S{i}" for i in range(50)], "dim_1": range(50), "dim_2": range(50)})
        chosen = select_by_geosketch(pca, 10, seed=0, sketch=lambda X, n, **kw: list(range(n)))
        assert chosen == [f"S{i}" for i in range(10)]

    def test_the_cli_way_is_exclusive_with_the_others(self, cohort, tmp_path):
        with pytest.raises(ValueError, match="one of"):
            subsample(cohort, tmp_path / "out", groups=[Group("branch", "0", 1)], geosketch=5,
                      pca=tmp_path / "pca.csv", keep_runner=_fake_keep)
```

- [ ] **Step 3: Implement**

```python
def select_by_geosketch(pca: pd.DataFrame, n: int, seed: int, sketch=None) -> List[str]:
    """Geometric sketching (Hie et al. 2019) on the PCA coordinates of a run."""
    if sketch is None:
        try:
            from geosketch import gs as sketch
        except ImportError:
            raise ImportError("--geosketch needs the geosketch extra: pip install 'manifold-genetics[geosketch]'")
    dims = [c for c in pca.columns if c.startswith("dim_")]
    index = sketch(pca[dims].to_numpy(), n, seed=seed, replace=False)
    return list(pca["sample_id"].iloc[index])
```

Match the keyword arguments to what the old script passed. In `subsample`, the exclusivity check becomes `sum(bool(groups), fit_samples is not None, geosketch is not None) == 1`; with `geosketch`, read `pca` (`dtype={"sample_id": str}`), restrict to the project `.fam`, call `select_by_geosketch`, write `keep` as in the group branch. CLI: `--geosketch N`, `--pca CSV`.

- [ ] **Step 4: Run, commit**

Run: `uv run pytest tests/unit/test_subsample.py tests/unit/test_cli_main.py -v`

```bash
git add -A src tests
git commit -m "feat(subsample): --geosketch selects geometrically from a PCA table"
```

---

### Task 14: Docs, README, tutorial, CHANGELOG, version

**Files:**
- Create: `docs/preprocessing.md`
- Modify: `docs/cli.md` (replace `init` section; full `acquire`, `preprocess`, `subsample`, `setup --preprocessing` sections), `docs/quickstart.md`, `docs/tutorial.ipynb` (every `init` → `acquire`), `README.md`, `mkdocs.yml` (nav: `- Preprocessing: preprocessing.md` after Quickstart), `CHANGELOG.md`, `pyproject.toml` (`version = "0.3.0"`), `src/manifold_genetics/__init__.py` if it carries `__version__`
- Test: `tests/unit/test_docs_match_cli.py`, `tests/unit/test_package_metadata.py` (whatever pins the version)

- [ ] **Step 1: Find every `init` mention**

```bash
git grep -n "manifold-genetics init\|init synthetic\|init hgdp\|init custom\|init aou" -- docs README.md CHANGELOG.md
```
Replace each with the `acquire` form. In the tutorial notebook, edit the JSON source cells (`python - <<EOF` with `json` load/dump, or `sed` on the `.ipynb` — keep the file otherwise byte-identical to avoid a noisy diff).

- [ ] **Step 2: Write `docs/preprocessing.md`**

Content, in this order, user-facing only (memory: docs stay minimal):

```markdown
# Preprocessing

Optional. If your genotypes are already filtered to the SNPs you want, skip this page: `acquire custom` and `run` are enough.

If you are starting from raw biobank PLINK files, two commands sit between `acquire` and `run`. Each reads a cohort directory and writes a new one, so they compose and can be re-run.

## The cohort directory

(the tree from the spec, five lines of explanation)

## preprocess — filter SNPs

(one-config and two-config forms; the preset table; what needs internet: `harmonise` → `setup --preprocessing` on a login node; needs bash, plink2, plink 1.9)

## subsample — choose the fit samples

(--group with the UKBB and AoU examples; --fit-samples; --geosketch)

## Worked examples

### UK Biobank projected onto HGDP+1KGP
(the three commands from the equivalence table)

### All of Us, in the Researcher Workbench
(the notebook cell: acquire hgdp --archive gs://…, acquire aou, preprocess --preset harmonise --fit-has-chr-prefix, run)

### A biobank subsample
(preprocess → subsample --group → run)
```

Every command in the page must match `--help` exactly; `test_docs_match_cli.py` is the check.

- [ ] **Step 3: CHANGELOG and version**

```markdown
## 0.3.0 — 2026-09-XX

### Changed
- `init` is now `acquire`. Same targets (`synthetic`, `hgdp`, `custom`, `aou`), same output; the name says what it does now that it fetches All of Us too.
- `acquire aou` fetches the cohort and its demographics inside the Researcher Workbench. It used to write a config and point at a shell script.
- `acquire hgdp --archive` accepts the workbench's HGDP+1KGP archive (a `gs://` URL or a local path) and normalises it to the public layout.

### Added
- `preprocess`: SNP filtering and cross-cohort intersection, shipped inside the package. The shell that produced the published figures, orchestrated from Python.
- `subsample`: choose a cohort's fit samples by label counts, a list, or geosketch.
- `setup --preprocessing`: prefetch the GIAB, WRayner and TOPMed references for the `harmonise` preset, for machines without internet.
```

- [ ] **Step 4: Build the docs and run the full fast suite**

```bash
uv run mkdocs build --strict 2>&1 | tail -5
uv run pytest -m "not slow and not network" -q 2>&1 | tail -5
```
Expected: clean build; all pass.

- [ ] **Step 5: Commit**

```bash
git add -A docs README.md CHANGELOG.md mkdocs.yml pyproject.toml src/manifold_genetics/__init__.py
git commit -m "docs: preprocessing page; init becomes acquire; 0.3.0"
```

---

### Task 15: Finish

- [ ] Run the complete verification: fast suite, `black --check`, `isort --check`, `flake8`, `mkdocs build --strict`, and the two integration files on the compute node (Tasks 5, 9, 10, 11).
- [ ] Open the PR against `main` with `~/bin/gh pr create`, body summarising the three commands, linking #123, #124, #126, and stating which flows were reproduced for real (synthetic, HGDP, UKBB) and which await the workbench (AoU).
- [ ] Close #123 via the PR; comment on #124 with the AoU command sequence.

## Self-review

**Spec coverage.** Cohort directory → Task 4. `acquire` targets and workbench layout → Tasks 8, 12. `preprocess` one/two-config, presets, flag naming → Tasks 3, 5, 6. `subsample` three modes → Tasks 10, 13. Issue 4's three fixes → Tasks 2, 7. `setup --preprocessing` → Task 7. Equivalence table: generic/UKBB (Task 11), AoU (Task 12, deferred run), 10k_WBH/10k_WB_5K_Irish (Tasks 10, 11), geosketch (Task 13), HGDP (Task 9). Idempotence test → Task 9. Docs, CHANGELOG, 0.3.0, `init` removal → Task 14. Old scripts kept → Task 1 (symlinks). Order of work matches the spec's.

**Amendment to the spec found while planning:** decision 2 (fit on 3,400 unrelated) only applies to the public archive; the workbench archive has no relatedness metadata, so `acquire hgdp` on it fits on every sample, as the old script did. Task 8 records this in the spec.

**Type consistency.** `PreprocessOptions` field names match between Tasks 3, 5, 6, 9, 11. `read_cohort` → `CohortConfig.fit/.project: Side` used identically in Tasks 5 and 10. `filter_labels_to_fam(labels, prefix, out)` argument order consistent. `_run_plink2_keep(bfile, keep, out, plink2)` matches `scaffold.py:626`. `acquire_*` names consistent after Task 8; Task 4/5 tests written before the rename use `init_synthetic` and are renamed by the `sed` in Task 8 step 1.
