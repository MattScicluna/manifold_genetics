# Design: manifold-genetics as a publishable package

**Date:** 2026-09-10
**Status:** approved, in progress
**Goal:** `pip install manifold-genetics` on any platform, then run the HGDP+1KGP
example end to end. UKBB and AoU examples runnable by those with data access.

## The finding that shapes everything

`src/` has exactly **one** hard external dependency: the `flashpca` binary
(`pca/flashpca.py:60`). `plink2` appears only in the examples' data-preparation
shell scripts, never in the library. `neural-admixture` is pip-installable and
already an optional extra.

The shipped binary is `flashpca_x86-64` — Linux x86-64 only. That single fact
explains three separate symptoms:

- the package cannot run after a plain `pip install` on macOS or Windows;
- CI runs unit tests only, deselecting `integration`, because the runner has no
  binary;
- the "real data" tests that would catch a scientific regression are exactly the
  ones that never run automatically.

Removing that dependency is therefore the keystone. It is scheduled first among
the substantive phases and everything else is hygiene or packaging ceremony.

## Decisions

| Question | Decision |
|---|---|
| PCA on a pip-installed package | Pure-Python backend by default; flashpca stays as an opt-in accelerator behind the same interface |
| Example layer | Config-driven: `manifold-genetics run config.yaml`; bash reduced to optional SLURM submitters |
| UKBB/AoU tests | In-repo, marker-gated, module-level skip when data is absent |
| Release | Public PyPI, TestPyPI dry run first |

## Phases

Phase 0 is done. Phases 1 and 2 are independent and may proceed in parallel.
Phase 3 depends on both.

### Phase 0 — repo hygiene (done)

`docs/` un-ignored; cluster paths and the SLURM account genericised;
`submit_batch.sh` de-Narvalised; `project.urls` placeholders fixed; `dist/`
ignored; migration note relocated under `docs/migrations/`.

### Phase 1 — `PCABackend` plugin, with a pure-Python implementation

Mirror the plugin idiom the repo already uses for admixture
(`admixture/backends/base.py` plus `neural`/`precomputed`/`fake`):

```
pca/backends/base.py          PCABackend ABC: fit() -> model artefacts, project() -> coords
pca/backends/flashpca.py      existing subprocess code, moved behind the ABC
pca/backends/sklearn.py       bed-reader -> standardise -> randomized SVD
```

**The correctness bar is parity with flashpca's projection semantics.** flashpca
is invoked with no `--stand` flag, so it applies its default standardisation, and
its `.loadings` / `.meansd` outputs *are* the definition of "project a second
cohort onto a reference PCA". Getting the standardisation wrong yields
projections that look plausible and are wrong.

The standardisation will be determined **empirically** from real `.meansd` files
already on disk, by regressing the recorded means and SDs against per-variant
allele frequencies computed from the same `.bed`. It will not be guessed from
documentation.

Oracle: the checked-in HGDP reference outputs, which
`tests/integration/test_hgdp_reproducibility.py` already pins and which flashpca
has reproduced bit-for-bit.

**Gate:** the default backend switches only once per-PC correlation against those
references clears the agreed threshold on real data, with signs aligned (PC sign
is arbitrary; flipped PCs are correct and must be tolerated by the comparison,
not counted as failure). `FlashPCABackend` remains indefinitely as a fallback and
a reference implementation.

Memory: projection is `X_std @ loadings`, which chunks cleanly over samples — the
486,748-sample cohort must never be materialised whole. Fit uses randomized SVD
over a chunked reader.

#### Measured contract (2026-09-10)

Determined empirically, not assumed. ``bed-reader`` was dropped: the reader is
about forty lines of numpy and adds no dependency, so cross-platform wheel
coverage stopped being a question.

| property | convention | agreement with flashpca |
|---|---|---|
| dosage | count of A1 (``.bim`` col 5) | 5e-7; counting A2 is wrong by up to 1.97 |
| mean | over non-missing genotypes | 5e-7 |
| sd | ``sqrt(mean * (1 - mean/2))`` (binom2) | 1.7e-6 |
| eigenvalues | ``S**2 / n_variants`` | exact |
| loadings | right singular vectors, unit-norm | corr 1.0 |
| PC (fit) | ``U * sqrt(eigenvalues)`` | ratio sd 1.7e-6 |
| PC (project) | ``X_new @ loadings / sqrt(n_variants)`` | 5e-8, signed |

Two findings worth carrying forward:

**sklearn's randomized_svd defaults are not adequate here.** A genotype spectrum
has a long flat tail -- on HGDP, PC13-PC20 span 2.81 to 2.28 with gaps as small
as 0.010 -- and near-equal eigenvalues are where a randomized SVD loses trailing
components. At ``n_iter=10, n_oversamples=10`` the first twelve PCs matched to
1e-13 while PC20's loading correlation was 0.993: a plausible-looking wrong
answer. ``n_iter=20, n_oversamples=40`` reaches 1.5e-7, flashpca's own text
precision, in 36s against an exact SVD's 147s.

**Fit memory was the real blocker to switching the default**, not accuracy. The
straightforward fit materialises the standardised matrix: 82 GB for the 60,000
sample geosketch cohort. A streaming randomized range finder bounds peak memory
to ``O((n_variants + n_samples) * l)`` -- about 110 MB at those sizes,
independent of cohort size -- at the cost of streaming the ``.bed`` once per
half-iteration.

**Exit:** integration tests pass on a GitHub ubuntu runner with no binary present.

### Phase 2 — config-driven examples

`examples/_shared/run_pipeline.sh` is ~400 lines that re-declare an argument
surface `cli.py` already owns. That duplication is the source of the ugliness and
of the drift this repo keeps finding (a mode default every caller had to
correct; one script silently taking the most expensive landmarking path).

Replace with `manifold-genetics run config.yaml`, where the config file is a
serialisation of the `IOConfig` / `PCAConfig` / `EmbeddingConfig` objects that
**already exist** in `pipeline/config.py`. The three modes become named presets
in the package rather than bash functions, carrying the landmarking policy from
PR #82.

Each example keeps a `prepare_data.sh` — data preparation genuinely is shell
work (plink2, wget) — and gains a `config.yaml`. Geosketch stops breaking the
pattern: its only remaining difference is a preparation step, and
`_shared/select_samples_geosketch.py` already exists.

### Phase 3 — three-cohort test suite

Cohort-parametrised pipeline tests: HGDP (public, CI-runnable after Phase 1),
UKBB projection and subsample, AoU. A `requires_private_data` marker and a
conftest fixture resolving roots from `MG_UKBB_DATA` / `MG_AOU_DATA`, so no test
hardcodes a cluster path and all of them skip cleanly without access.

Assertions state science, in the manner of `test_hgdp_pipeline_real.py` (region
accuracy, geographic preservation), not smoke tests — coverage percentage is not
verification, and this repo has already found smoke tests hiding real bugs.

Expensive tests get a documented sbatch entry point.

### Phase 4 — packaging and release

Version to 0.2.0, leaving room before any 1.0 API commitment. `CHANGELOG.md`,
`py.typed`, tag-triggered trusted-publishing workflow, TestPyPI dry run. **The
acceptance gate is a clean-venv install on a machine that is not this cluster**,
running the HGDP example.

### Phase 5 — documentation

MkDocs Material — markdown-native, matching the existing `.md` docs. Pages:
install, quickstart, concepts, the three modes, CLI reference, API reference via
mkdocstrings, HPC guide, controlled-access data guide. Tutorial notebook on the
small HGDP subset, executed in CI so it cannot rot.

## Additions not in the original request

**`run_manifest.json` in every output directory** — package version, git sha,
resolved config, resolved tool versions, timings. For a scientific package this
is the difference between a results directory that is interpretable in six months
and one that is not. The stale-labels incident of 2026-09-10, where an example's
published figures coloured 40% of their points because a label file silently
belonged to a superseded sample selection, is precisely what a manifest catches.

**Finish `transform` -> `project` before 1.0.** It is a breaking change to output
paths; the migration note is now at `docs/migrations/`. Cheap now, expensive once
there are users.

**Three open defects** found 2026-09-10:
1. `examples/ukbb/geosketch_phate/data/fit_labels.csv` is stale — 40.6% overlap
   with the selection it claims to describe.
2. `validate_sample_id_overlap` (`utils/validation.py:438`) raises only at *zero*
   overlap, which is why (1) passed silently. It needs a minimum-overlap
   threshold.
3. `setup_logging` (`cli.py:48`) calls `logging.basicConfig`, setting the **root**
   logger, so `--verbose` enables DEBUG for numba and every other library — 665 KB
   of numba SSA dumps in two minutes of one run.

## Risks

**Phase 1 parity is the dominant risk.** A subtle divergence between
`SklearnPCABackend` and flashpca shifts every downstream result, and the tests
that would catch it are the expensive ones. Mitigation is structural: both
backends coexist, the default switch is gated on a real-data parity test, and the
existing HGDP references are treated as a contract.

Secondary: `bed-reader`'s cross-platform wheel coverage is unconfirmed. If it is
inadequate, the fallback is a small numpy `.bed` reader — the format is a 3-byte
magic header plus 2-bit-packed genotypes, readable with `np.fromfile` and
`np.unpackbits`, which would also remove the dependency entirely.

## Out of scope

Merging to `main`, and any publish to PyPI or TestPyPI, are explicitly reserved
for a human. Overnight autonomous work stops at "ready to tag".
