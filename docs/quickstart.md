# Quickstart

There are two experiments that you can run immediately. First, make sure you have [installed this package](install.md).

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

## 2. Look before you run

```bash
manifold-genetics run config.yaml --dry-run
```

Prints every setting the run will use, marking `(default)` on anything the
package supplied rather than the config file, then exits. This can help you determine what is going on especially during any complex run.

## 3. Run it

```bash
manifold-genetics run config.yaml
```

If the run is OOM killed, set a memory budget:

```bash
manifold-genetics run config.yaml --memory-gb 4
```

## Your own data

`init custom` writes the config and colormap for PLINK files you already have.

**One cohort**, fitting on a subset of it and embedding that subset:

```bash
manifold-genetics init custom \
    --fit-plink data/fit_subset \
    --project-plink data/project_subset \
    --labels data/labels.csv \
    --preset subsample
```

Both sets come from the same cohort: `fit_subset` is the subset the model is
estimated on, `project_subset` the full cohort the components are computed for.

**Two cohorts**, projecting yours onto a reference panel — separate label files,
because the cohorts need not have the same labels:

```bash
manifold-genetics init custom \
    --fit-plink data/reference_panel \
    --project-plink data/my_cohort \
    --fit-labels data/reference_labels.csv \
    --project-labels data/my_labels.csv \
    --preset projection
```

It generates a colour for every label value, and refuses if fewer than half the
genotyped samples appear in the label file — a mismatch otherwise shows up as a
figure that colours some of its points and looks finished.

See [formats](formats.md) if you would rather write both files yourself.

## All of Us

For readers working in the All of Us Researcher Workbench with access to the
controlled data tier:

```bash
manifold-genetics init aou
```

Running this in the workbench reproduces the All of Us experiments from the
manuscript.
