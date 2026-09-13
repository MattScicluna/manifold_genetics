# Command line

```bash
manifold-genetics <command> --help
```

Every command prints its own options and a worked example. This page is about
which one to reach for.

## Starting from nothing

```bash
manifold-genetics init synthetic     # simulate a cohort and a config beside it
manifold-genetics init hgdp          # fetch and prepare the real HGDP+1KGP cohort
manifold-genetics init custom        # a config for genotypes you already have
manifold-genetics init aou           # a config for All of Us (workbench only)
```

A `pip install` ships no data and no example config, so `init` is what gives you
something to run.

`synthetic` needs no network and takes seconds: it simulates 240 samples across
three groups, writes the PLINK triples, labels, colormap and `config.yaml`, and
the result runs end to end. It is the quickest way to confirm an installation
works.

`hgdp` downloads about 183 MB and prepares it with plink2 — 4,094 QC-passing
samples across seven genetic regions, fitted on the 3,400 that are also
unrelated. It needs internet, so on a cluster run it on a login node.

A TLS-intercepting proxy needs nothing from you: `init` falls back to `curl` and
`wget`, which use the system certificate store rather than Python's.

On a machine with no route out at all — a compute node, say — fetch the archive
elsewhere and point `init` at it:

```bash
manifold-genetics init hgdp --archive hgdp_1kgp_full.tar.gz
```

`--no-download` means never fetch: it still unpacks an archive already present.

`custom` takes `--fit-plink`, `--labels`, and optionally `--project-plink`,
`--preset` and `--n-pcs`. It writes a colour for every label value -- 22 of them
for UK Biobank's `self_described_ancestry` -- and refuses if fewer than half the
genotyped samples appear in the label file, which is the failure that otherwise
shows up as a figure colouring some of its points.

`aou` is the exception to the pattern: it does **not** fetch anything. All of Us
is controlled-access, lives in `gs://fc-aou-datasets-controlled`, and its
preparation is about 1,300 lines of workbench-specific shell. So it checks the
environment — `GOOGLE_PROJECT`, `WORKSPACE_CDR`, `gsutil`, `bq`, `plink2` — names
everything missing at once, and writes the config for the prepared cohort.

All four take `--out DIR`, and none overwrites an existing `config.yaml`
without `--force`.

## Running a whole pipeline

```bash
manifold-genetics run config.yaml [--dry-run] [--output DIR] [--skip-...] [-v]
```

The entry point for real work. Reads a [config file](formats.md#the-config-file), applies
a few command-line overrides, runs the pipeline.

- `--dry-run` prints the resolved call and exits. Use it before anything long.
- `--output DIR` overrides `output_dir`, so one config can drive a verification
  run without being edited.
- `--memory-gb GB` overrides the PCA memory budget, because how much memory
  you have is a property of the machine rather than of the analysis. Above
  the budget the fit streams: bounded memory, about nineteen times the wall
  clock. Lower it if a run is killed; raise it on a large node to keep a big
  cohort in memory.
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
see [Install](install.md#external-tools).

## Exit codes and logging

Commands return 0 on success and non-zero on failure, with the reason on stderr
rather than a traceback for the errors a user can actually act on — a missing
config, an unknown key, absent input files.

Logging goes to stderr at INFO by default and DEBUG under `-v`, scoped to the
`manifold_genetics` logger.
