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
during a run. Build it for `linux/amd64` and push it to your registry:

```bash
docker build --platform linux/amd64 -t manifold-genetics:0.3.0 .
docker push <registry>/manifold-genetics:0.3.0
```

then pass `--container <registry>/manifold-genetics:0.3.0`.

Add `docker` to the profile to run in the container locally
(`-profile no_admixture,docker`), or `local` to use the current environment.
