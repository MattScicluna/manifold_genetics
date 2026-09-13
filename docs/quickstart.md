# Quickstart

Two commands to a figure, on a cohort the package fetches or simulates for you.
Assumes you have [installed it](install.md).

## 1. Pick a cohort

=== "Simulated — seconds, no download"

    ```bash
    manifold-genetics init synthetic
    manifold-genetics run config.yaml
    ```

    Simulates 2,000 samples along a branching tree and writes the genotypes,
    labels, colormap and config beside each other. Nothing is downloaded and the
    whole run takes under a minute, which makes it the quickest way to confirm an
    installation works.

    It also writes `dla_tree_ground_truth.png`, the tree the cohort was drawn
    along. Compare it with `outputs/figures/embeddings/project_phate_by_branch.png`:
    the eight branches should be recognisable in both, and the branches past an
    unsampled gap edge appear detached in the embedding.

=== "Real — HGDP+1KGP, about 183 MB"

    ```bash
    manifold-genetics init hgdp
    manifold-genetics run config.yaml
    ```

    Downloads the public HGDP+1KGP cohort and prepares it: 4,094 QC-passing
    samples across seven genetic regions, with the model fitted on the 3,400 that
    are also unrelated. Needs internet and `plink2`, which is fetched
    automatically — so on a cluster, run `init` on a login node.

    Expect a few minutes rather than seconds.

Both write into the current directory; pass `--out DIR` to put them somewhere
else. Neither overwrites an existing `config.yaml` without `--force`.

!!! tip "Behind a proxy, or offline"

    If the download fails with a certificate error, `init` falls back to `curl`
    and `wget`, which use the system certificate store. If it still cannot reach
    the network — an HPC compute node, say — fetch the archive by other means and
    point it at the file:

    ```bash
    manifold-genetics init hgdp --archive hgdp_1kgp_full.tar.gz
    ```

## 2. Look before you run

```bash
manifold-genetics run config.yaml --dry-run
```

This prints every setting the run will use, marking `(default)` on anything the
package supplied rather than the config file, then exits without doing work. On
a cohort where PCA takes hours, this is the cheapest thing you will ever do.

## 3. Run it

```bash
manifold-genetics run config.yaml
```

If the run is killed, it is almost certainly memory. `--memory-gb N` sets the
budget: above it the PCA fit streams, which bounds memory at the cost of about
nineteen times the wall clock.

```bash
manifold-genetics run config.yaml --memory-gb 4
```

## Your own data

`init` exists to give you something to run. With your own cohort you skip it and
write a config yourself — the formats are below, and
[`init synthetic`](#1-pick-a-cohort) writes a working example of every one of
them, which is often the fastest way to see what is expected.

## What goes in

Four kinds of file, named in the config:

| input | format |
|---|---|
| **genotypes** | a PLINK 1 triple — `<prefix>.bed`, `.bim`, `.fam`, SNP-major |
| **labels** | CSV with a `sample_id` column and one column per grouping |
| **colormap** | JSON mapping a label column to `{value: colour}` |
| **coordinates** *(optional)* | CSV with `sample_id`, `latitude`, `longitude` |

Two genotype sets go in, and which one the model is **fitted** on is the choice
that defines the analysis; the preset names that choice. Labels look like this:

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

One tree, the same shape every run, under `output_dir`. `<n>` is your `n_pcs`:

```
outputs/
├── pca/
│   ├── fit_pca_<n>.csv          sample_id, dim_1 … dim_<n>
│   ├── project_pca_<n>.csv
│   └── flashpca_outputs/        the projectable model: loadings, means, SDs
├── embeddings/
│   └── phate_2d.csv             sample_id, dim_1, dim_2
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
`phate_2d.csv` because `method: phate`, one `fit.<K>.csv` per K in range.

Both `init` configs skip admixture, because it needs the `admixture` extra
(torch), so a first run produces the first three directories only. Geographic
metrics need a coordinates file, which neither supplies.

The three CSV families look like this:

```csv
# pca/project_pca_20.csv          one row per sample, one column per PC
sample_id,dim_1,dim_2,...,dim_20
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

- [`archive/configuration.md`](https://github.com/MattScicluna/manifold_genetics/blob/main/archive/configuration.md) — every key a config file accepts
- [`archive/cli.md`](https://github.com/MattScicluna/manifold_genetics/blob/main/archive/cli.md) — running the stages individually
- [`archive/api.md`](https://github.com/MattScicluna/manifold_genetics/blob/main/archive/api.md) — driving it from a notebook instead

Those three are not on the site yet; they are being rewritten and reintroduced
one at a time.
