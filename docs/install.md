# Installation

```bash
pip install manifold-genetics
```

Or from source, for an unreleased change:

```bash
pip install git+https://github.com/MattScicluna/manifold_genetics
```

The code has been tested on Python 3.10–3.12, on Linux or macOS. 

## Checking it works

You can run the test suite to ensure that the code is working as expected.

Change below code to just pip. we did not introduce uv!!!
```bash
git clone https://github.com/MattScicluna/manifold_genetics
cd manifold_genetics
uv sync --frozen --extra dev

uv run pytest -m "not slow and not network"
```

## Extras

```bash
pip install 'manifold-genetics[admixture]'   # torch + neural-admixture
pip install 'manifold-genetics[geosketch]'   # geometric-sketch subsetting
```

Note that Admixture is much faster when a GPU is available

`geosketch` is needed only to *select* a geometric sketch of a large cohort.

NOTE we are deprecating examples so you can remove this line: in
`examples/_shared/select_samples_geosketch.py`. Nothing in the pipeline itself
imports it.

We add this somewhere else (stuff below)
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
one, it is automatically downloaded and cached per user — `~/.cache/manifold-genetics/bin` on
Linux, `~/Library/Caches/manifold-genetics/bin` on macOS — and reused
thereafter.
Note that if your computing environment has limited access to the internet (e.g. computing nodes on a cluster without internet access)
You can run the following when you have internet access to pre-fetches everything in one go:

```bash
manifold-genetics setup
```

To keep the binaries somewhere else — a shared project directory, say — set:

```bash
export MANIFOLD_GENETICS_TOOL_DIR=/project/shared/manifold-tools
```

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
