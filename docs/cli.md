# Command line

```bash
manifold-genetics <command> --help
```

Every command prints its own options and a worked example.

## Starting from nothing

```bash
manifold-genetics init synthetic     # simulate a cohort and a config beside it
manifold-genetics init hgdp          # fetch and prepare the real HGDP+1KGP cohort
manifold-genetics init custom        # a config for genotypes you already have
manifold-genetics init aou           # a config for All of Us (workbench only)
```

`pip install` downloads no data and no example config, so `init` is what gives you
something to run.

`synthetic` needs no network and takes under a minute. It places 2,000 samples
along a branching tree — eight branches, some separated by unsampled gaps — then
draws 1,000 variants whose allele frequencies drift along it. No single variant
carries the tree; it is recoverable only from all of them together, which is the
situation the pipeline exists for. It writes the PLINK triples, labels, colormap
and `config.yaml`, plus `dla_tree_ground_truth.png` — the tree itself, to check
the embedding against.

`hgdp` downloads about 183 MB: a variant-processed cohort of 4,151 samples over
172,152 SNPs. From it, two `plink2 --keep` calls select the 4,094 that pass QC
and the 3,400 of those that are also unrelated, using the flags in the archive's
`metadata.csv`. Sample selection only — no MAF, missingness, indel or LD
filtering, because the archive arrives with that already done. It needs
internet, so on a cluster run it on a login node.

You can also point `init` at archived data:

```bash
manifold-genetics init hgdp --archive hgdp_1kgp_full.tar.gz
```

`--no-download` means never fetch: it still unpacks an archive already present.

`custom` takes `--fit-plink` and labels, plus optionally `--project-plink`,
`--preset` and `--n-pcs`. Labels are either one `--labels` describing both sets,
or `--fit-labels` and `--project-labels` when they differ, as in every UK
Biobank config. It writes a colour for every label value -- 22 of them
for UK Biobank's `self_described_ancestry` -- and refuses if fewer than half the
genotyped samples appear in the label file, which is the failure that otherwise
shows up as a figure colouring some of its points.

`aou` does **not** fetch anything: All of Us is controlled-access and prepared
by workbench-specific tooling. It checks the environment — `GOOGLE_PROJECT`,
`WORKSPACE_CDR`, `gsutil`, `bq`, `plink2` — names everything missing at once, and
writes the config. Run in the workbench, it reproduces the All of Us experiments
from the manuscript.

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

## preprocess

```bash
manifold-genetics preprocess fit/config.yaml --out filtered/
manifold-genetics preprocess fit/config.yaml project/config.yaml --out filtered/ --preset harmonise
```

Filter SNPs of one cohort, or intersect two, into a new cohort directory in the
same layout, so the result can go to `run`, `pipeline`, or another `preprocess`.
Samples are never removed here. See [Preprocessing](preprocessing.md).

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
