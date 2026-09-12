# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **PCA projection had no memory budget and was OOM-killed on any large cohort.**
  `fit()` chose a chunk size from `max_fit_memory_gb`; `project()` read every
  variant for every sample in one pass, and its `variant_chunk_size` escape hatch
  was set nowhere outside the test suite. Projecting UK Biobank's 486,748 samples
  onto a 120,849-variant panel asks for 0.9 TB, of which the 58.8 GB uint8 unpack
  buffer is the first to land — and because `np.empty` commits lazily, the
  allocation *succeeds* and the kernel kills the process while it faults the
  pages in. No MemoryError, no traceback, just a dead job.

  There is now a `max_project_memory_gb` (8 GB, matching the fit budget), so the
  same run reads 1,038 variants at a time in 117 passes. Note that resident
  memory was measured at 22 GB against that 8 GB setting, because freed chunks
  are not all returned to the OS between iterations; budget roughly three times
  the number you set.

  `docs/hpc.md` had asserted the opposite — "Projection chunks over samples and
  is never the constraint" — which is corrected.

### Changed

- **`run_pipeline` and `load_config` are now importable from the top level** and
  named in `__all__`, which is the public API. They were reachable only through
  `manifold_genetics.pipeline.runner`, a private-looking path, while the API
  reference presented them as the headline entry points and the tutorial
  notebook imported them that way. The reference page now documents only the
  supported surface; the PCA backends, the PLINK reader and the standardisation
  helpers are explicitly internal.
- **The documentation site is the seven pages a user needs** — home, install,
  quickstart, tutorial, configuration, command line, API. How the project is
  tested, released and developed is no longer published alongside them. A new
  quickstart carries the input and output formats end to end.

### Fixed (test suite)

- **The real-cohort suite could never have run on UK Biobank or All of Us.** Its
  `_coords` helper read pipeline CSVs with inferred dtypes while label files are
  read as strings, so all-digit biobank sample IDs came back as `int64` and every
  merge raised `ValueError: You are trying to merge on int64 and object columns`.
  HGDP hid it for a year: its identifiers are not numeric.
- **`_nearest` built a `(n_probe, n_reference, d)` intermediate** — 155.8 GB when
  probing 2,000 points against 486,748 at 20 PCs, which OOM-killed the run that
  was measuring it. It is now chunked over the query, with peak memory
  `chunk x n_reference x 8` bytes; measured peak for that comparison is 3.4 GB.
  The ordering is still a full stable sort rather than a partial selection,
  because `argpartition` breaks exact ties arbitrarily and duplicate or
  monozygotic samples sit at distance zero.
- **Neighbourhood preservation is asserted against chance**, like every other
  statistic in the suite. It was a raw fraction with a floor of 0.05, which UK
  Biobank failed at 0.011 while HGDP passed at 0.333 — even though against
  chance UK Biobank scores 182x and HGDP 45x, making the larger cohort the
  better embedding. Chance overlap is `k / (n - 1)`, so a raw floor can only be
  calibrated for one cohort size.
- The command-line page is now checked against the CLI it documents: a test
  fails if a subcommand is undocumented, or documented but absent.

## [0.2.1] - 2026-09-12

First public release.

0.2.0 was published first, to reserve the name. This is the release to install:
it adds the `whole_cohort` rename, per-user tool caching, and the leak
guardrails. An index never accepts a version twice, so each of these is a step
forward rather than a correction in place.

The headline is that `pip install manifold-genetics` gives you a working
pipeline. Until this release it did not: PCA required the `flashpca` binary,
which ships only as a Linux x86-64 executable and had to be fetched and placed
by hand.

The version number is deliberately low. The API is not frozen, the
`transform` -> `whole_cohort` rename landed days before this, and All of Us has
not yet been verified end to end -- so expect further breaking changes before
1.0.

### Added

- **Guardrails against committing controlled-access data**
  ([docs/working-with-agents.md](docs/working-with-agents.md)). A pre-commit hook and
  a CI job run `scripts/check_sensitive_files.py`, written against this
  repository's own two incidents rather than as a generic secret scanner: it
  flags files whose header declares them private, genotype containers, long
  columns of biobank identifiers, local agent and editor state, and absolute
  cluster paths. Public cohort identifiers (`HG00096`) deliberately do not
  trigger it. Exemptions live in `.sensitive-allow`, scoped to a single rule, so
  that overriding the check leaves a reviewable trace.
- **A pure-Python PCA backend**, selected by default. Reads PLINK 1 `.bed`
  directly (about forty lines of numpy, no new dependency) and computes a
  randomized SVD. It reproduces `flashpca`'s conventions rather than inventing
  its own — dosage as the count of A1, `binom2` standardisation, eigenvalues as
  `S²/n_variants`, projection through the reference cohort's statistics — to
  agreement of 5e-7 or better against checked-in reference outputs. The
  standardisation was determined empirically from real `.meansd` files, not from
  documentation.
- **`flashpca` remains available** as an opt-in accelerator via
  `--pca-backend flashpca`, and writes the same artefacts as the Python backend,
  so downstream code cannot tell which produced a result.
- **Streaming fit** for cohorts too large to hold densely. Selected
  automatically from `max_fit_memory_gb`: a 60,000-sample cohort needs 82 GB
  dense and about 110 MB streamed.
- **`manifold-genetics run config.yaml`** — runs a whole pipeline from a config
  file, with `--dry-run` to print the resolved settings first. Unknown keys are
  rejected with a suggestion rather than ignored.
- **A config file for every shipped example**, replacing 1,382 lines of shell.
- **`py.typed`**, so the annotations in this package are visible to type
  checkers in projects that depend on it.
- **A real-cohort test suite** (`docs/testing-real-cohorts.md`): fast preflight
  checks that a cohort's data agrees with the config describing it, and an
  end-to-end suite with chance-corrected assertions about the science. Includes
  a SLURM entry point.
- **A documentation site** built with MkDocs Material and published to GitHub
  Pages, including a tutorial notebook that runs the whole pipeline on a cohort
  it simulates as it goes. The notebook is *executed* during the build and the
  build runs in strict mode, so a tutorial that stopped working or a broken
  internal link fails CI rather than shipping.
- **A tag-triggered release workflow** (`docs/releasing.md`) using trusted
  publishing, so there is no API token in the repository and no way to publish
  from a laptop. It builds from a fresh checkout, verifies the package metadata
  and the distribution contents, and installs the wheel into a clean
  virtualenv on Linux and macOS before anything is uploaded.

### Changed

- **External tools are cached per user and fetched on first use.** They were
  downloaded to a directory computed from `__file__`, which in a git checkout is
  the repository's `bin/` and in a `pip install` is a directory *beside*
  site-packages: the wrong place, frequently not writable, and discarded on
  upgrade. Worse, the constructor created it eagerly, so merely building a
  `ToolResolver` — which `PCA(backend="flashpca")` does — left a stray directory
  behind.

  Now: `$MANIFOLD_GENETICS_TOOL_DIR` if set, else the checkout's `bin/` when
  running from one, else a per-user cache (`~/.cache/manifold-genetics/bin`).
  Nothing is created until something is actually downloaded.
  `manifold-genetics setup` stays, as the pre-fetch you want before submitting a
  job to a compute node with no internet.
- **The right binary is downloaded for the platform.** Every URL was
  `linux_x86_64`, so on macOS the resolver fetched a Linux binary and then
  failed to execute it — which reads as a corrupt download rather than as "there
  is no build for you". plink2 and plink now resolve per platform (Linux x86-64,
  macOS arm64 and Intel, Windows x64), and `flashpca`, which upstream publishes
  only for Linux x86-64, says exactly that and points at the in-process PCA
  backend that needs no binary.

- **The `transform` preset is now `whole_cohort`.** This project used
  `transform` for two things — the second cohort's dataset role and the
  sklearn-style method verb. The role was renamed to `project` earlier; this
  preset was the last holdout, naming a *mode* while every other use of the word
  names an *operation*. The rule is now: `fit` estimates, `transform` applies,
  `project` is the second cohort and its outputs, and a preset is named for the
  shape of the run.

  `transform` still works as a deprecated alias and warns, naming its
  replacement. It is deliberately not listed among the valid choices in the
  error message, because an advertised alias is a name people keep choosing.
- **PCA's default backend is now `python`.** `flashpca` produced the reference
  outputs and still reproduces them bit-for-bit; the change is about being
  installable, not about accuracy.
- **Landmarking defaults are consistent across large cohorts**: 10,000 random
  landmarks and `t=50` for every cohort above ~50,000 samples, whether the fit
  subset was chosen by majority-capping or by geometric sketching. How a subset
  was selected should not change how it is embedded. A matched comparison found
  the two give the same branch topology.
- **`--verbose` no longer sets the root logger.** It previously enabled DEBUG
  for every library in the process, which produced 665 KB of numba SSA dumps in
  two minutes of one run.
- **`manifold-genetics run --dry-run` now prints the whole effective call**,
  marking `(default)` on anything the package supplied rather than the config
  file. The admixture batch size is why: it is a workaround for an upstream bug,
  so it appears in no config and was previously invisible in the one place built
  for checking settings before a long run.

### Fixed

- **`manifold-genetics --version` disagreed with the package.** The number was
  typed into the argparse argument as a third copy and drifted a minor version
  behind `manifold_genetics.__version__`. It now reads the package. Caught by a
  TestPyPI rehearsal, which is what a rehearsal is for.

  A test pinned the stale literal, so it passed while the command was wrong; it
  now compares against the package. Two further tests assert that the CLI
  reports the declared version and that the version appears in exactly one
  source file.
- The wheel is now checked to contain no untracked module. It picks files by the
  same "everything not gitignored" rule that put untracked working files in a
  pre-release sdist; there are stale `.ipynb_checkpoints` copies of two real
  backends under `src/` today, kept out only by a `.gitignore` entry.
- **The declared license was MIT; the LICENSE file is BSD 3-Clause.** The
  LICENSE file is authoritative and the metadata now matches it. A release
  cannot be re-uploaded under the same version, so this would have been
  permanent.
- **Every `project.urls` entry pointed at `manifold-genetics` with a hyphen**,
  which is not the repository name — four 404s on the PyPI sidebar.
- **Neural admixture is always given a batch size** (400). Left unset it batches
  the entire dataset at once; that is a bug in the tool, and the workaround was
  previously applied only in `subsample` mode, leaving the All of Us projection
  example — the one running on the largest cohort — without it.
- **`validate_sample_id_overlap` now enforces a minimum overlap** (50% by
  default) instead of raising only at zero. Near-total mismatch between a label
  file and a genotype set previously passed silently, which is how one example's
  figures came to colour 40% of their points from a superseded sample selection.
- Example label files are regenerated when they no longer match their `.fam`,
  rather than being kept because they exist.
- **The source distribution contained untracked working files.** hatchling's
  default is to include everything not gitignored, which is not the same as
  everything tracked; a pre-release sdist carried local tool state and an
  example script whose header says "NOT for external users". Its contents are now
  declared explicitly and checked by a test.

### Removed

- `examples/_shared/run_pipeline.sh`, `examples/_shared/detect_cluster.sh` and
  the nine per-example `run_pipeline.sh` wrappers, superseded by config files.

[Unreleased]: https://github.com/MattScicluna/manifold_genetics/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/MattScicluna/manifold_genetics/releases/tag/v0.2.1
[0.2.0]: https://pypi.org/project/manifold-genetics/0.2.0/
