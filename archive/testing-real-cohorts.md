# Testing against real cohorts

The unit suite runs in a couple of minutes on mock data and proves the code is
self-consistent. It cannot tell you whether a pipeline run on *your* genotypes
will produce something meaningful. That is what the cohort tests are for.

There are two of them, and they are meant to be used in that order:

| | what it does | cost |
|---|---|---|
| `tests/integration/test_cohort_preflight.py` | cross-checks a cohort's real data against the config that describes it | seconds |
| `tests/integration/test_cohort_pipeline_real.py` | runs the shipped config end to end and asserts the science | minutes to hours |

Both are driven by the registry in `tests/integration/cohorts.py`, which names
every shipped example, whether its data is public, and the environment variable
holding its data root. Cohorts whose data is absent skip; they never fail.

## Preflight: run this first, always

Every bug this project has found in a real run was the same shape: something in
the data did not match what the config claimed, and it surfaced forty minutes in
— or never. A label file left over from a superseded sample selection still
loads, still merges, and silently drops every sample it does not recognise. That
is how a published figure came to colour 40% of its points from the wrong
selection.

None of that needs the pipeline to run:

```bash
pytest tests/integration/test_cohort_preflight.py -v
```

It reads the `.fam` files, the labels and the colormap and checks that they
agree — that every genotyped sample has a label, that the columns the plots name
exist, that the colormap covers the values it will be asked to colour, that each
`.bed` is the size its `.bim` and `.fam` imply. It takes about twenty seconds
across four prepared cohorts.

Controlled-access cohorts also carry the `requires_private_data` marker, so
`-m "not requires_private_data"` deselects them outright.

## The pipeline tests

```bash
# Public cohorts only (HGDP+1KGP) — the default, no flag needed
pytest tests/integration/test_cohort_pipeline_real.py -m "slow and integration"

# One controlled-access cohort, named explicitly
pytest tests/integration/test_cohort_pipeline_real.py -m "slow and integration" \
    --cohort ukbb_projection

# Everything present, including neural admixture
pytest tests/integration/test_cohort_pipeline_real.py -m "slow and integration" \
    --cohort all --cohort-admixture
```

A controlled-access cohort runs **only when named**. Projecting 486,748 samples
is an hour-scale job with a memory footprint to match, and it must not start
because somebody ran `pytest`. Admixture is likewise off unless asked for: it
wants a GPU and hours, and it is not what these assertions are about.

Each cohort is driven through exactly the `config.yaml` it ships, so what is
tested is what a user runs.

### What they assert

The cohorts span 3,400 to 486,748 samples and 7 to 300-odd label groups, so the
statistics are chance-corrected (`tests/science.py`) — a fixed threshold on a raw
number would be vacuous at one end and impossible at the other.

- **PCA covers exactly the genotyped samples**, with the requested number of
  finite components, ordered by the variance they carry.
- **The first ten PCs separate the cohort's label groups** several times better
  than shuffled labels do.
- **The embedding keeps the neighbourhoods PCA found** — the fraction of each
  sample's 30 nearest neighbours in PC space that survive into two dimensions,
  as a multiple of chance. Invariant to rotation and reflection, because an
  embedding is only defined up to those.
- **The embedding separates the label groups**, again against shuffled labels.
- **Geographic preservation** is positive and significant, where the cohort has
  coordinates.
- No stage failed, and every figure the run reported was actually written.

Every measurement is printed, so a batch log records what the run scored rather
than only that it passed. HGDP+1KGP, 2m35s on 8 cores:

```
[hgdp] pc_separation(Population) = 54.29                       floor 3.0
[hgdp] embedding_separation(Population) = 57.53                floor 3.0
[hgdp] neighbourhood_preservation@30 (x chance) = 45.4         floor 10.0
[hgdp] geographic_correlation = 0.5927                         floor 0.4

[ukbb_projection] pc_separation(self_described_ancestry) = 21220
[ukbb_projection] embedding_separation(...) = 19580
[ukbb_projection] neighbourhood_preservation@30 (x chance) = 182.0
```

The thresholds are floors a healthy run clears by more than an order of
magnitude: they are there to catch a pipeline that broke, not one that drifted.

Every threshold is stated against chance, which is what lets one number serve
cohorts spanning 3,400 to 486,748 samples. Neighbourhood preservation was the
exception until 2026-09-12, when it was a raw fraction with a floor of 0.05 —
and UK Biobank's 0.011 failed it while HGDP's 0.333 passed, even though against
chance UK Biobank scores 182x to HGDP's 45x. Chance overlap is `k / (n - 1)`,
which falls by two orders of magnitude across that range, so a raw floor can
only ever be right for one cohort.
Add `-s` to see them locally; `submit_cohort_tests.sh` already does.

## On a SLURM cluster

```bash
export SLURM_ACCOUNT=your-account

bash tests/integration/submit_cohort_tests.sh hgdp
bash tests/integration/submit_cohort_tests.sh ukbb_projection --mem=256GB --time=24:00:00
bash tests/integration/submit_cohort_tests.sh all --admixture --gpus=1
```

The submitted job runs preflight first and stops if it fails, so a mismatched
label file costs twenty seconds rather than the whole walltime.

## Data roots

By default a cohort's data is read from its own example directory
(`examples/aou/10k_WBH/data/`, and so on), which is where `prepare_data.sh`
writes it. Point a cohort somewhere else with its environment variable:

| cohort | variable |
|---|---|
| `hgdp` | `MG_HGDP_DATA` |
| `ukbb_projection` | `MG_UKBB_PROJECTION_DATA` |
| `ukbb_subsample` | `MG_UKBB_SUBSAMPLE_DATA` |
| `ukbb_geosketch` | `MG_UKBB_GEOSKETCH_DATA` |
| `aou_projection` | `MG_AOU_PROJECTION_DATA` |
| `aou_subsample` | `MG_AOU_SUBSAMPLE_DATA` |
| `aou_geosketch` | `MG_AOU_GEOSKETCH_DATA` |

The variable names the directory that *contains* `data/`, mirroring the example
layout. A variable that is set but points nowhere makes the cohort skip rather
than fall back to the in-repo data — quietly testing a different dataset than
you asked for is worse than testing nothing.

Colormaps are tracked in the repo and are not moved by these variables.

## All of Us

The All of Us Researcher Workbench is the one environment none of this can be
rehearsed in beforehand: the data cannot leave it, so the AoU configs were
written by reading the shell scripts they replaced rather than by being run.
Check them before committing to a long run.

```bash
git clone <this repo> && cd manifold_genetics
pip install -e '.[dev]'

bash examples/aou/hgdp_1kgp_proj/prepare_data.sh     # or point MG_AOU_*_DATA at the data

# 1. Does the config say what you think it says?
manifold-genetics run examples/aou/hgdp_1kgp_proj/config.yaml --dry-run

# 2. Does the data agree with it?
pytest tests/integration/test_cohort_preflight.py -v -k aou

# 3. Does the pipeline produce something meaningful?
pytest tests/integration/test_cohort_pipeline_real.py -m "slow and integration" \
    --cohort aou_projection
```

Two things are worth looking at specifically in the `--dry-run` output.

`examples/aou/hgdp_1kgp_proj` must use `hgdp_1kgp_aou_aligned.json` as its fit
colormap, not the plain `hgdp_1kgp.json`. The wrong one produces a plot rather
than an error, which is why preflight checks that the colormap actually covers
the label values it will be asked to colour.

The admixture batch size must be 400. Left unset, neural-admixture batches the
entire dataset at once; 400 is the workaround. It is a package default rather
than something each config repeats, so `--dry-run` prints it as

```
  admix_batch_size  400  (default)
```

even though no config mentions it. `(default)` marks every setting the package
supplied rather than the file.
