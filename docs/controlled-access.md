# Controlled-access data

UK Biobank and All of Us cannot be redistributed, and All of Us cannot leave its
Researcher Workbench at all. So the examples in this repository ship the
*recipe* — the preparation scripts and the config files — and none of the data.

| example | mode | what it does |
|---|---|---|
| `examples/ukbb/hgdp_1kgp_proj` | `projection` | 486,748 UK Biobank samples onto the HGDP+1KGP panel |
| `examples/ukbb/10k_WB_5K_Irish` | `subsample` | majority-capped subset: 10k British + 5k Irish + everyone else |
| `examples/ukbb/geosketch_phate` | `subsample` | 60,000-sample geometric sketch |
| `examples/aou/hgdp_1kgp_proj` | `projection` | All of Us onto the same panel |
| `examples/aou/10k_WBH` | `subsample` | majority-capped All of Us subset |
| `examples/aou/geosketch_phate` | `subsample` | geometric sketch of All of Us |

`examples/generic/` holds two templates to copy for a cohort of your own.

Each example is `prepare_data.sh` followed by `config.yaml`. Preparation is the
part that needs the raw data and `plink`: it intersects variants across cohorts,
handles strand flips, builds the fit and project subsets, and writes the label
files. The pipeline itself only ever sees the prepared subsets.

## Before any long run

Two checks, in this order. Together they take under a minute and between them
catch everything that has actually gone wrong on a real run here.

```bash
# 1. Does the config say what you think it says?
manifold-genetics run examples/ukbb/hgdp_1kgp_proj/config.yaml --dry-run

# 2. Does the data agree with the config?
pytest tests/integration/test_cohort_preflight.py -v
```

Preflight reads the `.fam` files, the labels and the colormap and cross-checks
them: that every genotyped sample has a label, that the columns the plots name
exist, that the colormap covers the values it will be asked to colour, that each
`.bed` is the size its `.bim` and `.fam` imply.

That last set is not hypothetical. A label file left over from a superseded
sample selection still loads, still merges, and silently drops every sample it
does not recognise — which is how one example's published figures came to colour
40% of their points from the wrong selection. The check that catches it takes
twenty seconds.

## Labels

A labels CSV needs a `sample_id` column matching the `.fam`'s second field, plus
any number of label columns. The UK Biobank examples use
`self_described_ancestry` and `Population`; the All of Us ones use `race` and
`race_ethnicity`.

Label files must be **regenerated when the sample selection changes**, not kept
because they exist. The preparation scripts now check this themselves and
regenerate on a shortfall, and `validate_sample_id_overlap` raises below 50%
overlap rather than only at zero.

## All of Us

The Workbench is the one environment none of this can be rehearsed in
beforehand, because the data cannot leave it. The All of Us configs were
therefore written by *reading* the shell scripts they replaced, rather than by
the command-line comparison used for the others — so check them before
committing to a long run.

```bash
git clone https://github.com/MattScicluna/manifold_genetics
cd manifold_genetics
pip install -e '.[dev]'

bash examples/aou/hgdp_1kgp_proj/prepare_data.sh

manifold-genetics run examples/aou/hgdp_1kgp_proj/config.yaml --dry-run
pytest tests/integration/test_cohort_preflight.py -v -k aou
pytest tests/integration/test_cohort_pipeline_real.py -m "slow and integration" \
    --cohort aou_projection
```

Two things to look at specifically in the `--dry-run` output:

- `examples/aou/hgdp_1kgp_proj` must use **`hgdp_1kgp_aou_aligned.json`** as its
  fit colormap, not the plain `hgdp_1kgp.json`. The wrong one produces a plot
  rather than an error, which is why preflight checks colormap coverage.
- `admix_batch_size` must be **400**, shown as `(default)` since no config
  mentions it.

If the data is not inside the example directory, point the cohort at it with its
environment variable — `MG_AOU_PROJECTION_DATA` and the others are listed in
[Testing against real cohorts](testing-real-cohorts.md#data-roots).

## UK Biobank

The cross-projection example — variant intersection, allele consistency, strand
flips — is documented in detail in
[UK Biobank cross-projection](UKBB_README.md).
