# Formats

Every file the pipeline reads and writes. The one rule that makes the stages
compose: **coordinates are always a `sample_id` column followed by
`dim_1 … dim_n`**, whatever produced them.

## Inputs

Four kinds of file, named in the config:

| input | format |
|---|---|
| **genotypes** | a PLINK 1 triple — `<prefix>.bed`, `.bim`, `.fam`, SNP-major |
| **labels** | CSV with a `sample_id` column and one column per grouping |
| **colormap** | JSON mapping a label column to `{value: colour}` |
| **coordinates** *(optional)* | CSV with `sample_id`, `latitude`, `longitude` |

Which of the two genotype sets the model is **fitted** on is the choice the
preset names. Labels look like this:

```csv
sample_id,Population,Genetic_region_merged
HG00096,GBR,Europe
HG00097,GBR,Europe
```

The colormap says which of those columns to plot, and what colour each value
gets:

```json
{
  "Population":            {"GBR": "#4292C6", "YRI": "#C7E9C0"},
  "Genetic_region_merged": {"Europe": "#4292C6", "Africa": "#008000"}
}
```

Every column the colormap names gets its own figure. A column it does not name
is carried through but never plotted.

Coordinates are needed only for the geographic preservation metric:

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

The `.bed` must be SNP-major, which is what PLINK writes by default; if not,
the error gives you the `plink --make-bed` command to convert it. Both cohorts
must carry the same variants in the same order, and a mismatch is refused rather
than projected.

!!! warning "The labels must actually match the genotypes"

    A stale label file does not crash — it produces figures that are simply
    wrong. The pipeline refuses below 50% overlap and reports the fraction it
    found. Check that number.

## The config file

`manifold-genetics run` takes one YAML file naming the above:

```yaml
preset: whole_cohort          # projection | subsample | whole_cohort

data:
  fit_plink: data/fit_subset
  project_plink: data/project_subset
  labels: data/labels.csv
  colormap: colormap.json
  output_dir: outputs

pca:
  n_pcs: 20
  max_fit_memory_gb: 8        # above this the fit streams
  max_project_memory_gb: 8

embedding:
  method: phate               # phate | umap | tsne | diffusion_map
  n_components: 2             # embedding dimensions; 3 for a rotatable plot-3d figure
  knn: 100                    # method-specific parameters pass straight through
  t: 3

skip:
  admixture: true
```

The preset supplies the embedding settings that follow from it, so most files
are shorter than this. An unknown key is rejected with a suggestion rather than
ignored, and `--dry-run` prints every setting the run will use — including the
ones the package supplied — which is the reliable way to see what a config
resolves to.

## Outputs

One tree, the same shape every run, under `output_dir`. `<n>` is your `n_pcs`:

```
outputs/
├── pca/
│   ├── fit_pca_<n>.csv          sample_id, dim_1 … dim_<n>
│   ├── project_pca_<n>.csv
│   └── flashpca_outputs/        the projectable model: loadings, means, SDs
├── embeddings/
│   └── phate_2d.csv             sample_id, dim_1, dim_2
│                                (phate_3d.csv when n_components: 3)
├── figures/
│   ├── pca/                     PC-pair grids
│   └── embeddings/              one scatter per colormap column
├── admixture/                   only when the admixture stage runs
│   ├── fit.2.csv … fit.10.csv   ancestry proportions, one file per K
│   └── project.2.csv … project.10.csv
└── metrics/                     only when the metrics stage runs
    ├── geographic.json
    └── admixture.json
```

The file names follow the settings: `fit_pca_20.csv` when `n_pcs: 20`,
`phate_2d.csv` because `method: phate` with the default `n_components: 2`
(`phate_3d.csv` when it is 3), one `fit.<K>.csv` per K in range.

Both `acquire` configs skip admixture, because it needs the `admixture` extra
(torch), so a first run produces the first three directories only. Geographic
metrics need a coordinates file, which neither supplies.

The three CSV families look like this:

```csv
# pca/project_pca_20.csv          one row per sample, one column per PC
sample_id,dim_1,dim_2,...,dim_20
HG00096,0.073308,0.212584,...

# embeddings/phate_2d.csv         as many dimensions as n_components
sample_id,dim_1,dim_2
HG00096,0.123,-0.456

# embeddings/phate_3d.csv         the same file with n_components: 3
sample_id,dim_1,dim_2,dim_3
HG00096,0.123,-0.456,0.789

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
- **Three-dimensional embeddings carry the dimensionality in the name**, so a
  3-D run never overwrites or is mistaken for a 2-D one. `plot-3d` turns one
  into a rotatable HTML figure; see [the CLI page](cli.md).

Every CSV has the same shape — a `sample_id` column then `dim_1 … dim_n` — which
is what lets the stages compose, and lets you start from the middle if you
already have principal components.

