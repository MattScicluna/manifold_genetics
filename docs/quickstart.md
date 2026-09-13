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

If the run is killed, set a memory budget:

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

See [formats](formats.md) if you would rather write both files yourself.

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
