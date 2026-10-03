# Nextflow workflow

`main.nf` runs `manifold-genetics pipeline` as a Nextflow workflow, so a
cohort can be processed as one job on Verily Workbench (or Google Batch, or a
laptop). It is the same pipeline as the CLI, with the same outputs.

## Two profiles

| profile | runs |
|---|---|
| `full` | PCA, admixture, embedding, all plots (including admixture), all metrics (including admixture) |
| `no_admixture` | PCA, embedding, standard plots, non-admixture metrics — passes `--skip-admixture` |

Both are the same workflow; the profile only sets `params.run_admixture`.

```bash
nextflow run main.nf -profile no_admixture \
    --fit_plink    gs://bucket/cohort/fit_subset \
    --project_plink gs://bucket/cohort/project_subset \
    --labels       gs://bucket/cohort/labels.csv \
    --colormap     gs://bucket/cohort/colormap.json \
    --outdir       gs://bucket/runs/2026-10-03
```

A PLINK prefix is the path without `.bed/.bim/.fam`; all three must exist.

## Parameters

| parameter | default | |
|---|---|---|
| `fit_plink` | — | PLINK prefix the models are fitted on (required) |
| `project_plink` | `fit_plink` | PLINK prefix to project |
| `labels` / `colormap` | — | one labels CSV and colormap for both sets, or… |
| `fit_labels`, `project_labels`, `fit_colormap`, `project_colormap` | — | …separate ones per set |
| `geographic` | — | coordinates CSV for the geographic metric |
| `outdir` | `results` | where results are published (a bucket path on Verily) |
| `run_admixture` | `true` | set by the profile |
| `n_pcs`, `pca_backend` | `20`, `python` | |
| `embedding`, `embedding_input`, `n_components` | `phate`, `both`, `2` | |
| `knn`, `t`, `n_landmark`, `random_landmarking` | CLI defaults | use landmarks at biobank scale (e.g. `10000` and `true`): without them PHATE needs dense n × n memory |
| `k_min`, `k_max`, `gpus`, `gpu_type` | `2`, `10`, `0`, `nvidia-tesla-t4` | admixture (`full` only) |
| `extra_args` | `''` | anything else `manifold-genetics pipeline` accepts, verbatim |
| `cpus`, `memory`, `disk`, `time` | `16`, `128 GB`, `500 GB`, `24h` | `full` doubles `time`; an OOM-killed run retries once with twice the memory |
| `container` | `manifold-genetics:latest` | the image to run |

## The container

Runs use a container with the package (and its `admixture` and `interactive`
extras) and plink2, flashpca and plink 1.9 installed at build time, so nothing
is downloaded during a run. Build it from the repository root for `linux/amd64`
(flashpca is x86-64 only) and push it to the workspace's Artifact Registry:

```bash
docker build --platform linux/amd64 -t manifold-genetics:0.3.0 .
docker tag manifold-genetics:0.3.0 us-central1-docker.pkg.dev/<project>/<repo>/manifold-genetics:0.3.0
docker push us-central1-docker.pkg.dev/<project>/<repo>/manifold-genetics:0.3.0
```

then run with `--container us-central1-docker.pkg.dev/<project>/<repo>/manifold-genetics:0.3.0`.
`.dockerignore` is an allowlist (package source only), so a checkout with
controlled-access data in its working tree cannot leak it into the image.

## Execution profiles

Combine with one of the profiles above:

- **Verily Workbench / Google Batch**: the platform supplies the executor and
  work directory; give `--container` and bucket paths.
- `docker`: run locally in the container — `-profile no_admixture,docker`.
- `local`: no container, `manifold-genetics` from the current environment —
  `-profile no_admixture,local`. This is what CI runs on the synthetic cohort.

## Later: one process per step

The work is one process today. It sits inside the `MANIFOLD_GENETICS`
sub-workflow, whose inputs and outputs are the data the steps exchange, so PCA,
admixture, embedding, plotting and metrics can become separate processes — each
with its own resources (admixture on a GPU, PCA with memory, plots small) —
without changing the entry point, the parameters or the profiles.
