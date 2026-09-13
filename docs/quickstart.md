# Quickstart

Two commands to a figure. Assumes you have [installed it](install.md).

## 1. Pick a cohort

=== "Simulated — seconds, no download"

    ```bash
    manifold-genetics init synthetic
    manifold-genetics run config.yaml
    ```

    2,000 samples simulated along a branching tree, with the genotypes, labels,
    colormap and config written beside each other. Under a minute, nothing
    downloaded.

    It also writes `dla_tree_ground_truth.png`, the tree the cohort was drawn
    along. Compare it with
    `outputs/figures/embeddings/project_phate_by_branch.png`: the eight branches
    should be recognisable in both.

=== "Real — HGDP+1KGP, about 183 MB"

    ```bash
    manifold-genetics init hgdp
    manifold-genetics run config.yaml
    ```

    The public HGDP+1KGP cohort: 4,094 QC-passing samples across seven genetic
    regions, fitted on the 3,400 that are also unrelated. Needs internet and
    `plink2`, which is fetched automatically — on a cluster, run `init` on a
    login node. A few minutes.

Both write into the current directory; `--out DIR` puts them elsewhere, and
neither overwrites an existing `config.yaml` without `--force`.

!!! tip "Behind a proxy, or offline"

    `init` falls back to `curl` and `wget`, which use the system certificate
    store. If there is no route out at all, fetch the archive separately and
    point at it:

    ```bash
    manifold-genetics init hgdp --archive hgdp_1kgp_full.tar.gz
    ```

## 2. Look before you run

```bash
manifold-genetics run config.yaml --dry-run
```

Prints every setting the run will use, marking `(default)` on anything the
package supplied rather than the config file, then exits. Worth doing before any
long run.

## 3. Run it

```bash
manifold-genetics run config.yaml
```

If the run is killed, set a memory budget. Above it the PCA fit streams, which
bounds memory at roughly nineteen times the wall clock.

```bash
manifold-genetics run config.yaml --memory-gb 4
```

## Your own data

`init custom` writes the config and colormap for PLINK files you already have:

```bash
manifold-genetics init custom \
    --fit-plink data/my_cohort \
    --labels data/my_labels.csv \
    --preset whole_cohort
```

`--project-plink` defaults to the fit set. Pass it when the two differ, as in a
projection onto a reference panel.

It generates a colour for every label value, and refuses if fewer than half the
genotyped samples appear in the label file — a mismatch otherwise shows up as a
figure that colours some of its points and looks finished.

The formats are below if you would rather write both files yourself.

## What goes in

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

## All of Us

For readers with access to the controlled data tier, working inside the
Researcher Workbench. The data cannot be reached from outside.

```bash
manifold-genetics init aou
```

This one does not fetch anything — preparing All of Us is workbench-specific
work the package does not reproduce. It checks you are somewhere it could run,
names anything missing, and writes the config matching
`examples/aou/hgdp_1kgp_proj/`. Run that example's `prepare_data.sh` first.

## Then what

Stages are checkpointed: re-running reuses whatever is already on disk, so an
interrupted run resumes rather than restarting. To force a stage to redo its
work, delete its output.

- [`archive/configuration.md`](https://github.com/MattScicluna/manifold_genetics/blob/main/archive/configuration.md) — every key a config file accepts
- [`archive/cli.md`](https://github.com/MattScicluna/manifold_genetics/blob/main/archive/cli.md) — running the stages individually
- [`archive/api.md`](https://github.com/MattScicluna/manifold_genetics/blob/main/archive/api.md) — driving it from a notebook instead

Those three are not on the site yet; they are being rewritten and reintroduced
one at a time.
