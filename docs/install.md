# Installation

```bash
pip install manifold-genetics
```

Python 3.10–3.12, tested on Linux and macOS. No compiler, no external binary:
the default PCA backend reads PLINK `.bed` files and computes a randomized SVD
in process.

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

## External tools

`plink2` and `plink` are needed to **prepare** data, not to run the pipeline:
the example `prepare_data.sh` scripts use them to build fit and project subsets
from a full cohort, intersect variants across cohorts, and handle strand flips.

```bash
manifold-genetics setup
```

downloads them into `bin/`, along with `flashpca` if you want the accelerated
PCA backend. It needs internet, which on an HPC cluster means running it on a
login node — see [Running on a cluster](hpc.md).

`manifold-genetics setup` does not touch your Python environment. It only
fetches binaries.

## Developing on the repository

```bash
git clone https://github.com/MattScicluna/manifold_genetics
cd manifold_genetics

uv venv --python python3.11
uv sync --frozen --extra dev

uv run pytest -m "not slow and not network"      # ~2 minutes
```

`uv sync --extra docs` adds the documentation toolchain; `mkdocs serve` then
builds this site locally, executing the tutorial notebook as it goes.

Testing against real genotypes is a separate suite with its own entry points —
see [Testing against real cohorts](testing-real-cohorts.md).
