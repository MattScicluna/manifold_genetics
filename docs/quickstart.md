# Quickstart

A complete run on real data, from nothing to figures. About fifteen minutes,
most of it downloading.

If you would rather not download anything, the [tutorial](tutorial.ipynb) does
the same thing on a cohort it simulates as it goes, in under a minute.

## 1. Install

```bash
pip install manifold-genetics
```

## 2. Get the example cohort

HGDP+1KGP is public — 4,094 samples across seven genetic regions.

```bash
git clone https://github.com/MattScicluna/manifold_genetics
cd manifold_genetics

bash examples/hgdp_1kgp/download_data.sh   # ~183 MB, needs internet
bash examples/hgdp_1kgp/prepare_data.sh    # needs plink2: manifold-genetics setup
```

`prepare_data.sh` is the only part that needs the external binaries, and it is
the only part specific to this example — it turns a public VCF into the PLINK
triples the pipeline reads. With your own data you start at step 3.

## 3. Look before you run

```bash
manifold-genetics run examples/hgdp_1kgp/config.yaml --dry-run
```

This prints every setting the run will use, marking `(default)` on anything the
package supplied rather than the config file, then exits without doing work. On
a cohort where PCA takes hours, this is the cheapest thing you will ever do.

## 4. Run it

```bash
manifold-genetics run examples/hgdp_1kgp/config.yaml
```

## What goes in

Four kinds of file, named in the [config](configuration.md):

| input | format |
|---|---|
| **genotypes** | a PLINK 1 triple — `<prefix>.bed`, `.bim`, `.fam`, SNP-major |
| **labels** | CSV with a `sample_id` column and one column per grouping |
| **colormap** | JSON mapping a label column to `{value: colour}` |
| **coordinates** *(optional)* | CSV with `sample_id`, `latitude`, `longitude` |

Two genotype sets go in, and which one the model is **fitted** on is the choice
that defines the analysis; the [preset](configuration.md#presets) names that
choice. Labels look like this:

```csv
sample_id,Population,Genetic_region_merged
HG00096,GBR,Europe
HG00097,GBR,Europe
```

A `sample_id` column is required; every other column is a grouping you might
want to colour by. The colormap names which of those columns are worth plotting,
and what colour each value gets:

```json
{
  "Population":            {"GBR": "#4292C6", "YRI": "#C7E9C0"},
  "Genetic_region_merged": {"Europe": "#4292C6", "Africa": "#008000"}
}
```

Every column the colormap names gets its own figure. A column it does not name
is carried through but never plotted.

Geographic coordinates are optional, and only needed for the geographic
preservation metric:

```csv
sample_id,latitude,longitude
HG00096,52.833333,-1.891667
HG00097,52.833333,-1.891667
```

### Genotypes

Give the **prefix only** — the `.bed`, `.bim` and `.fam` suffixes are appended
for you:

```yaml
data:
  fit_plink: data/fit_subset       # the model is fitted on this
  project_plink: data/project_subset   # and applied to this
```

The `.bed` must be SNP-major, which is what PLINK writes by default. If it is
not, the error says so and gives you the `plink --make-bed` command to convert
it. Both cohorts must carry the same variants in the same order — the pipeline
checks this and refuses to project a mismatched set rather than producing
confident nonsense.

!!! warning "The labels must actually match the genotypes"

    A label file left over from a superseded sample selection is the failure
    mode that costs the most, because nothing crashes — you get figures that are
    simply wrong. The pipeline refuses to run unless the overlap covers at least 50% of
    the smaller file, and tells you the fraction it found. Check that number.

## What comes out

One tree, the same shape every run, under `output_dir`:

```
outputs/
├── pca/
│   ├── fit_pca_50.csv           sample_id, dim_1 … dim_50
│   ├── project_pca_50.csv
│   └── flashpca_outputs/        the projectable model: loadings, means, SDs
├── admixture/
│   ├── fit.2.csv … fit.10.csv   ancestry proportions, one file per K
│   ├── project.2.csv … project.10.csv
│   └── checkpoints/
├── embeddings/
│   └── phate_2d.csv             sample_id, dim_1, dim_2
├── figures/
│   ├── pca/                     PC-pair grids
│   ├── embeddings/              one scatter per colormap column
│   └── admixture/               stacked bars, embedding coloured by component
└── metrics/
    ├── geographic.json
    └── admixture.json
```

The file names follow the settings: `fit_pca_50.csv` because `n_pcs: 50`,
`phate_2d.csv` because `method: phate`, one `fit.<K>.csv` per K in range.

The three CSV families look like this:

```csv
# pca/project_pca_50.csv          one row per sample, one column per PC
sample_id,dim_1,dim_2,...,dim_50
HG00096,0.073308,0.212584,...

# embeddings/phate_2d.csv         always two dimensions
sample_id,dim_1,dim_2
HG00096,0.123,-0.456

# admixture/project.2.csv         proportions, summing to 1 per sample
sample_id,component_1,component_2
HG00096,0.9996,0.0004
```

Two details worth knowing:

- **`flashpca_outputs/` is written whichever PCA backend you used.** The name is
  historical; the in-process backend writes the same artefacts, so a model
  fitted by either is readable by the other.
- **In `projection` mode there is also `phate_fit_2d.csv`**, the reference panel's
  own embedding, because that mode embeds both cohorts.

Every CSV has the same shape — a `sample_id` column then `dim_1 … dim_n` — which
is what lets the stages compose, and lets you start from the middle if you
already have principal components.

## Then what

Stages are checkpointed: re-running reuses whatever is already on disk, so an
interrupted run resumes rather than restarting. To force a stage to redo its
work, delete its output.

- [Configuration](configuration.md) — every key a config file accepts
- [Command line](cli.md) — running the stages individually
- [Python API](api.md) — driving it from a notebook instead
