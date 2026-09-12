# Running on a cluster

Written for SLURM, which is what this package was developed against. Nothing in
it is tied to a particular cluster: the account comes from `$SLURM_ACCOUNT` or
`--account=NAME`, and paths are derived from the script's own location.

## Login nodes have internet; compute nodes often do not

This trips up every new environment. On most HPC clusters the compute nodes have
no route to the outside world, so anything that downloads must happen first, on
a login node:

```bash
# On the login node
manifold-genetics setup            # plink2, plink, flashpca
uv sync --frozen --extra dev       # resolving needs the index
bash examples/hgdp_1kgp/download_data.sh
```

Then submit the work. A job that fails minutes in with a network timeout is
almost always this.

## Submitting

```bash
export SLURM_ACCOUNT=your-account

bash examples/_shared/submit_batch.sh examples/hgdp_1kgp/config.yaml
bash examples/_shared/submit_batch.sh examples/ukbb/10k_WB_5K_Irish/config.yaml \
    --cpus=16 --mem=128GB
bash examples/_shared/submit_batch.sh examples/aou/10k_WBH/config.yaml \
    --time=48:00:00 -- --skip-admixture
```

SLURM options go before `--`; anything after it is passed to
`manifold-genetics run`. Logs land in `logs/<job>_<jobid>.{out,err}`.

Defaults are 8 CPUs, 32 GB and 24 hours. `--gpus=N` requests GPUs and loads the
CUDA modules, which is only useful with admixture.

## Memory

PCA is where the arithmetic matters. A dense fit needs
`n_samples × n_variants × 8` bytes:

| cohort | dense fit |
|---|---|
| 3,400 × 172k (HGDP+1KGP) | 4.7 GB |
| 60,000 × 121k (UK Biobank sketch) | 58 GB |
| 59,264 × 170k (UK Biobank capped) | 80 GB |

Above `max_fit_memory_gb` (8 GB by default) the backend switches to a streaming
fit, which bounds memory to roughly 110 MB regardless of cohort size — at about
nineteen times the wall clock. It is selected automatically; you do not choose
it. If a run is unexpectedly slow, this is a likely reason, and giving it more
memory is the fix.

Projection chunks over samples and is never the constraint. Embedding a very
large project set can be, which is what `embedding.embed_batch_size` bounds.

Admixture wants a GPU and, critically, `batch_size` — left unset,
neural-admixture batches the whole dataset at once. The package supplies 400;
see [Configuration](configuration.md#admixture).

## Working interactively

An allocation without a shell is convenient for long sessions:

```bash
salloc --account=$SLURM_ACCOUNT -c8 --mem=32G -t3:00:00 --no-shell
# prints a JOBID and stays alive until the walltime

srun --jobid=<JOBID> --overlap --ntasks=1 bash -c 'uv run pytest -m "not slow"'
```

`--overlap` lets the new step share cores with whatever is already running in
the allocation. The compute node still has no internet this way.

Do not run test suites or builds on the login node: it is shared, and often
slower than the compute node you are entitled to.

## Testing on real cohorts

The cohort test suite has its own SLURM entry point, which runs the cheap
preflight checks first and stops if they fail — so a mismatched label file costs
twenty seconds rather than the whole walltime:

```bash
bash tests/integration/submit_cohort_tests.sh ukbb_projection \
    --mem=256GB --time=24:00:00
```

See [Testing against real cohorts](testing-real-cohorts.md).
