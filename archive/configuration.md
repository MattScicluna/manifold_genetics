# Configuration

A config file is a serialisation of `run_pipeline`'s keyword arguments. It
exists because the shell layer that used to hold these settings re-declared an
argument surface the CLI already owned, and that duplication is where this
project's drift kept coming from: a `subsample` default every caller had to
correct by hand, one script silently taking the most expensive landmarking path.

```bash
manifold-genetics run config.yaml --dry-run   # inspect
manifold-genetics run config.yaml             # execute
```

Two rules worth knowing up front:

- **Paths resolve relative to the config file**, not the working directory, so a
  config runs from anywhere.
- **Unknown keys are errors, not warnings**, with a suggested correction.
  Silently ignoring a typo is how a setting ends up not doing what the file
  says.

## A complete example

```yaml
preset: subsample

data:
  fit_plink: data/fit_subset          # prefix: .bed/.bim/.fam
  project_plink: data/project_subset
  fit_labels: data/fit_labels.csv     # or a single `labels:` for both
  project_labels: data/project_labels.csv
  colormap: ../../colormaps/ukbb.json
  geographic_coords: data/coords.csv  # optional; enables the geographic metric
  output_dir: outputs

pca:
  n_pcs: 20

admixture:
  k_min: 2
  k_max: 10

embedding:
  method: phate
  t: 100                              # overrides the preset's t

visualization:
  admix_group_column: self_described_ancestry

skip:
  admixture: true
```

## Presets

`preset` supplies the settings that follow from the shape of the run. Anything
in the file overrides them.

| preset | `embedding_input` | knn | t | `n_landmark` | `random_landmarking` |
|---|---|---|---|---|---|
| `projection` | `both` | 100 | 3 | none | false |
| `subsample` | `fit` | 500 | 50 | 10000 | true |
| `whole_cohort` | `project` | 100 | 3 | none | false |

`n_landmark` and `random_landmarking` are always set together: setting the first
alone selects a far more expensive code path. See
the preset you chose.

A preset is optional. Without one you set `embedding.input_mode` and the
embedding parameters yourself.

## Sections

### `data`

| key | meaning |
|---|---|
| `fit_plink` | PLINK prefix the model is fitted on — **required** |
| `project_plink` | PLINK prefix projected onto it — **required** |
| `output_dir` | where everything is written — **required** |
| `labels` | labels CSV used for both sets |
| `fit_labels`, `project_labels` | per-set overrides, for cross-cohort runs |
| `colormap` | colormap JSON used for both sets |
| `fit_colormap`, `project_colormap` | per-set overrides |
| `geographic_coords` | sample coordinates; enables the geographic metric |

A labels CSV has a `sample_id` column plus any number of label columns. A
colormap is keyed by label column, then by value:

```json
{
  "self_described_ancestry": {"British": "#9370DB", "Irish": "#8A2BE2"},
  "Population": {"GBR": "#9370DB"}
}
```

One file can therefore colour several groupings of the same cohort; a figure is
produced per column the colormap names.

### `pca`

| key | default | meaning |
|---|---|---|
| `n_pcs` | 50 | components to compute |
| `backend` | `python` | `python` (in process) or `flashpca` (external binary) |
| `force` | false | recompute even if output exists |
| `max_fit_memory_gb` | 8 | GB budget for the dense fit; above it the fit streams |
| `max_project_memory_gb` | 8 | GB budget for one projection chunk |

Set the two budgets to match the machine, not the cohort. They are the
difference between a fit that is held in memory and one that streams at roughly
nineteen times the wall clock — see [Memory on large cohorts](#memory-on-large-cohorts).

### `admixture`

| key | default | meaning |
|---|---|---|
| `k_min`, `k_max` | 2, 10 | range of K to fit |
| `threads` | auto | CPU threads |
| `num_gpus` | auto | GPUs |
| `batch_size` | 400 | see below |

**Do not remove `batch_size` from a run.** Left unset, neural-admixture batches
the entire dataset at once — a bug in the tool rather than a choice, and on a
large cohort it will exhaust memory. 400 is the workaround, and it is now a
package default, so no config needs to repeat it. `--dry-run` prints it as
`admix_batch_size 400 (default)`. Setting it to `null` is an explicit opt-out
for anyone investigating the upstream bug.

### `embedding`

| key | default | meaning |
|---|---|---|
| `method` | `phate` | `phate`, `umap`, `tsne` or `diffusion_map` |
| `input_mode` | from preset | `fit`, `project` or `both` |

Every other key is passed through to the method: `knn`, `t`, `n_landmark`,
`random_landmarking`, `decay`, `gamma`, `n_pca`, `embed_batch_size`,
`min_dist`, `n_neighbors`, `perplexity`. Which apply depends on the method.

`embed_batch_size` bounds peak memory when transforming a very large project
set; the UK Biobank projection example uses 60000 for 486,748 samples.

### `visualization`

| key | meaning |
|---|---|
| `admix_group_column` | label column the admixture bar plot groups by |
| `admix_within_group_order` | ordering within a group; `chron` by default |
| `projection_plot_fit_column` | label column colouring the fit set in the projection figure |
| `projection_plot_project_column` | and the project set |

### `skip`

`pca`, `admixture`, `embedding`, `pca_visualization`, `visualization`,
`admixture_visualization`, `metrics` — each a boolean.

```yaml
skip:
  admixture: true     # no GPU available
```

Command-line `--skip-*` flags add to whatever the file sets; they never clear
it.

## Checking before you run

`--dry-run` prints the full call `run_pipeline` would receive, with paths
resolved and `(default)` marking every setting the package supplied rather than
the file:

```
  admix_batch_size                400  (default)
  embedding_input                 project
  fit_plink                       /data/hgdp/fit_subset
  pca_backend                     python  (default)
```

On a cohort where a run takes hours, this is the cheapest check there is. The
other one, which reads the data rather than the config, is
`pytest tests/integration/test_cohort_preflight.py`, which cross-checks a
cohort's data against its config in seconds.

## Memory on large cohorts

PCA is where the arithmetic matters, and both halves of it are bounded by a
budget you can raise.

**Fitting** a dense model needs `n_samples × n_variants × 8` bytes:

| cohort | dense fit |
|---|---|
| 3,400 × 121k (HGDP+1KGP) | 3.3 GB |
| 60,000 × 121k (UK Biobank sketch) | 58 GB |
| 59,264 × 170k (UK Biobank capped) | 80 GB |

Above `max_fit_memory_gb` (8 GB by default) the backend switches to a streaming
fit, which bounds memory to roughly 110 MB regardless of cohort size — at about
nineteen times the wall clock. It is chosen automatically. If a run is
unexpectedly slow, this is the likely reason.

More memory alone is not the fix: asking SLURM for `--mem=256GB` changes nothing
unless you also raise `max_fit_memory_gb`, because the budget is what the backend
compares against, not the free memory on the node. Raise both together.

**Projecting** is bounded the same way by `max_project_memory_gb`, also 8 GB.
It sizes one chunk at roughly `17 × n_samples × chunk` bytes, so projecting UK
Biobank's 486,748 samples onto a 120,849-variant panel reads 1,038 variants at a
time, in 117 passes over the `.bed`.

This budget bounds the *chunk*, not the process: measured resident memory for
that run was about 22 GB, because freed chunks are not all returned to the OS
between iterations. Budget roughly three times the setting, and note that
without it the same run needs 0.9 TB and is killed outright.

Embedding a very large project set is bounded separately, by
`embedding.embed_batch_size`.

Admixture wants a GPU and, critically, a `batch_size` — left unset,
neural-admixture batches the whole dataset at once. The package supplies 400.
