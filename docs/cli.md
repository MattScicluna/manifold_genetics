# Command line

```bash
manifold-genetics <command> --help
```

Every command prints its own options and a worked example. This page is about
which one to reach for.

## Running a whole pipeline

```bash
manifold-genetics run config.yaml [--dry-run] [--output DIR] [--skip-...] [-v]
```

The entry point for real work. Reads a [config file](configuration.md), applies
a few command-line overrides, runs the pipeline.

- `--dry-run` prints the resolved call and exits. Use it before anything long.
- `--output DIR` overrides `output_dir`, so one config can drive a verification
  run without being edited.
- `--skip-pca`, `--skip-admixture`, `--skip-metrics` add to the config's own
  skip settings.
- `-v` enables debug logging for this package only. It does not turn on debug
  for numba and every other library in the process, which is what it used to do.

```bash
manifold-genetics pipeline --fit-plink data/fit --project-plink data/project \
    --labels labels.csv --colormap colors.json --output results/ \
    --n-pcs 50 --k-min 2 --k-max 10 --embedding phate --knn 100
```

`pipeline` is the same thing with every setting on the command line. It predates
`run` and remains useful for one-off experiments; for anything you will want to
repeat or review, a config file is the better record.

## Individual stages

Each stage reads and writes the standard CSV — `sample_id` then `dim_1 … dim_n`
— so they compose, and you can start from the middle if you already have
components.

| command | in | out |
|---|---|---|
| `pca` | PLINK prefixes | PCA CSVs and a projectable model |
| `admixture` | PLINK prefixes | Q matrices per K |
| `embed` | PCA CSV | 2-D embedding CSV |
| `plot` | embedding CSV + labels + colormap | scatter per label column |
| `plot-pca` | PCA CSV | PC-pair grid |
| `plot-projection` | fit and project embeddings | both cohorts on one figure |
| `plot-admixture` | Q matrices | stacked bar plots |
| `plot-admixture-embedding` | embedding + Q | embedding coloured by component |
| `plot-knn-composition` | embedding + labels | neighbour label composition |
| `metrics-geographic` | embedding + coordinates | JSON |
| `metrics-admixture` | embedding + Q | JSON |

```bash
# PCA, choosing the backend explicitly
manifold-genetics pca --fit-plink data/fit --project-plink data/project \
    --fit-output out/fit_pca.csv --project-output out/project_pca.csv \
    --n-pcs 50 --pca-backend python

# An embedding of components you already have
manifold-genetics embed --method phate --fit-input out/project_pca.csv \
    --project-output out/phate_2d.csv --knn 100 --t 3
```

`--pca-backend` takes `python` (in process, the default) or `flashpca` (the
external binary). They agree to 1.5e-7 and write the same artefacts, so a model fitted by
either is readable by the other.

## Setup

```bash
manifold-genetics setup
```

Pre-fetches `plink2`, `plink` and `flashpca` into the per-user cache
(`~/.cache/manifold-genetics/bin`, or the checkout's `bin/` when you are running
from one). They are also fetched on first use, so this is optional — what it is
for is fetching them **before** submitting a job, because compute nodes usually
have no internet. Needed for data *preparation*, not for the pipeline itself;
see [Installation](install.md#external-tools).

## Exit codes and logging

Commands return 0 on success and non-zero on failure, with the reason on stderr
rather than a traceback for the errors a user can actually act on — a missing
config, an unknown key, absent input files.

Logging goes to stderr at INFO by default and DEBUG under `-v`, scoped to the
`manifold_genetics` logger.
