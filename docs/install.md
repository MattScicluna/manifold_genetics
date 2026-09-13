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

Run the test suite to confirm the install:

```bash
git clone https://github.com/MattScicluna/manifold_genetics
cd manifold_genetics
pip install -e '.[dev]'

pytest -m "not slow and not network"
```

No data of your own and no network; about two minutes.

## Extras

```bash
pip install 'manifold-genetics[admixture]'   # torch + neural-admixture
pip install 'manifold-genetics[geosketch]'   # geometric-sketch subsetting
```

Admixture is much faster when a GPU is available.

`geosketch` is needed only to *select* a geometric sketch of a large cohort, and
runs on PCA coordinates rather than genotypes. Nothing in the pipeline imports
it.

## External tools

`plink2` and `plink` are needed to **prepare** data, not to run the pipeline:
the example `prepare_data.sh` scripts use them to build fit and project subsets
from a full cohort, intersect variants across cohorts, and handle strand flips.

You do not normally have to fetch them yourself. The first time something needs
one it is downloaded and cached per user — `~/.cache/manifold-genetics/bin` on
Linux, `~/Library/Caches/manifold-genetics/bin` on macOS — and reused after that.

If the machine that runs the pipeline has no internet, as compute nodes on a
cluster often do not, pre-fetch them from one that does:

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
