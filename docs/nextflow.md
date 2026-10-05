# Nextflow

`main.nf` runs `manifold-genetics pipeline` as a Nextflow workflow, for Verily
Workbench or any Nextflow executor.

```bash
nextflow run main.nf -profile no_admixture \
    --fit_plink data/fit --project_plink data/project \
    --labels labels.csv --colormap colormap.json --outdir results/
```

Two profiles, one workflow:

- `full` runs everything.
- `no_admixture` passes `--skip-admixture`: no admixture step, plots or metrics.

Inputs are PLINK prefixes, labels and a colormap (or `fit_`/`project_` versions
of each), as for `pipeline`. Other options are in `nextflow.config`; anything
else goes through `--extra_args`. At biobank scale set `--n_landmark 10000
--random_landmarking true`.

## Container

Runs use a container with the external tools built in, so nothing is downloaded
during a run. It is published on Docker Hub for `linux/amd64` as
`mattscicluna/manifold-genetics`, tagged by branch (`main`), by commit
(`sha-<commit>`), and by version (`0.3.0`, `latest`) once a release is tagged:

```bash
nextflow run main.nf -profile no_admixture,docker \
    --container mattscicluna/manifold-genetics:sha-97ca2ba ...
```

Prefer a commit or release tag to a branch tag: a registry mirror can keep
serving an old image under a branch name after it has moved. To build your own,
`docker build --platform linux/amd64 -t <registry>/manifold-genetics .`

Add an execution profile to the workflow profile:

- `docker` runs in the container locally (`-profile no_admixture,docker`).
- `google-batch` runs on Google Batch, as on Verily Workbench
  (`-profile no_admixture,google-batch`), with `--container` set to an image
  the workspace can pull.
- `local` uses the current environment.
