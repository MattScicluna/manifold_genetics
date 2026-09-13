# Installation

```bash
pip install manifold-genetics
```

Or from source, for an unreleased change:

```bash
pip install git+https://github.com/MattScicluna/manifold_genetics
```

Python 3.10–3.12, on Linux or macOS. No compiler, no external binary: the
default PCA backend reads PLINK `.bed` files and computes a randomized SVD in
process.

## Checking it works

A clone gets you the test suite, which is the quickest way to confirm the
install on a given machine — no data of your own, no network, about two minutes:

```bash
git clone https://github.com/MattScicluna/manifold_genetics
cd manifold_genetics
uv sync --frozen --extra dev

uv run pytest -m "not slow and not network"
```

Worth doing on any machine you have not run this on before. The suite covers the
in-process PCA backend, which on macOS is the only way PCA runs at all.

## Extras

```bash
pip install 'manifold-genetics[admixture]'   # torch + neural-admixture
pip install 'manifold-genetics[geosketch]'   # geometric-sketch subsetting
```

The core install is deliberately torch-free. Admixture is the one stage that
wants a GPU and a deep-learning stack, and most of what this package does — PCA,
embeddings, plots, metrics — does not, so making everyone install torch to get
them would be a poor trade. `manifold-genetics admixture`, and the admixture
stage of `pipeline` and `run`, need the `admixture` extra; they fail with a clear
message rather than a bare `ImportError` if it is missing.

`geosketch` is needed only to *select* a geometric sketch of a large cohort, in
`examples/_shared/select_samples_geosketch.py`. Nothing in the pipeline itself
imports it.

That selection runs on **PCA coordinates, not genotypes** — sketching 486,748
samples across 120,849 raw dosages would take far too long, and the PCA space is
what the sketch is meant to be representative of. So it needs an existing PCA
CSV as input, which is a preparation step, not a change to how the pipeline
works: the run that follows still starts from PLINK files like any other.

## External tools

`plink2` and `plink` are needed to **prepare** data, not to run the pipeline:
the example `prepare_data.sh` scripts use them to build fit and project subsets
from a full cohort, intersect variants across cohorts, and handle strand flips.

You do not normally have to fetch them yourself. The first time something needs
one, it is downloaded and cached per user — `~/.cache/manifold-genetics/bin` on
Linux, `~/Library/Caches/manifold-genetics/bin` on macOS — and reused
thereafter. The right build for your platform is selected; where upstream
publishes none (`flashpca` exists only for Linux x86-64) you get a message
saying so rather than a Linux binary that will not run.

```bash
manifold-genetics setup
```

is therefore optional: it pre-fetches everything in one go, which is what you
want **before** submitting a job, because compute nodes on most clusters have no
internet.

To keep the binaries somewhere else — a shared project directory, say — set:

```bash
export MANIFOLD_GENETICS_TOOL_DIR=/project/shared/manifold-tools
```

Resolution order for each tool is: that variable, then the module system, then
`PATH`, then the cache. In a git checkout the cache is the repository's `bin/`,
so an existing development setup keeps working unchanged.

`manifold-genetics setup` does not touch your Python environment. It only
fetches binaries.

## Developing on the repository

The clone above is all you need. To work on the documentation as well:

`uv sync --extra docs` adds the documentation toolchain, and `mkdocs serve`
then builds this site locally. The tutorial renders without its outputs, because
the build does not execute it — CI does that in a separate step. To see the
outputs locally, execute it first:

```bash
uv run jupyter nbconvert --to notebook --inplace --execute docs/tutorial.ipynb
```

Testing against real genotypes is a separate suite with its own entry points —
see `archive/testing-real-cohorts.md` in the repository.
