# manifold-genetics

An end-to-end pipeline from PLINK genotypes to publication figures: PCA →
admixture → manifold embedding → visualisation → metrics.

[Install](install.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }

## What it is for

Human genetic variation is continuous, and our manifold learning framework implemented in this package can produce visualizations that recover that continuity.
For more details see our upcoming publication.
Below are PHATE embeddings for UK Biobank and All of Us cohorts.

<p align="center">
  <img src="https://raw.githubusercontent.com/MattScicluna/manifold_genetics/main/assets/ukbb_phate.png" width="45%" alt="UK Biobank PHATE embedding coloured by self-described ancestry"/>
  <img src="https://raw.githubusercontent.com/MattScicluna/manifold_genetics/main/assets/aou_phate.png" width="45%" alt="All of Us PHATE embedding coloured by ancestry"/>
</p>

## This Package Returns the Following Things

| stage | what it produces |
|---|---|
| **PCA** | components, which are passed to the embedding method |
| **Admixture** | ancestry proportions per K, via neural-admixture |
| **Embedding** | 2-D coordinates from PHATE, UMAP, t-SNE or diffusion maps |
| **Visualisation** | scatter plots by any label column, admixture bar plots |
| **Metrics** | geographic and admixture preservation |

Every stage reads and writes the same kinds of files, so they compose: see
[formats](formats.md) for what each one expects, and the
[command line](cli.md) for running them individually.

Stages are checkpointed: a run that already produced PCA output reuses it. This
matters on cohorts where PCA is the expensive part.

## Inputs

Every run takes two genotype sets, in two roles:

- **fit** — the cohort the model is *estimated* on: the PCA loadings, the
  embedding.
- **project** — the cohort that fitted model is then *applied* to.

They can be the same set, one can contain the other, or they can be different
cohorts entirely. How they relate is the analysis, and each arrangement has a
preset:

| preset | fit | project | for |
|---|---|---|---|
| **`projection`** | a reference panel | your cohort | Visualizing your cohort along axes of genetic variation defined by a reference cohort |
| **`subsample`** | a subset of your cohort | that same subset | Cohorts that are too large—or too strongly dominated by one group—to embed in full, such as UK Biobank or All of Us |
| **`whole_cohort`** | a subset of your cohort | the whole cohort | Cohorts small enough to embed in full, such as the HGDP+1KGP example |

## Where to start

- [Install](install.md) — pip or from source, the extras, and how to check it
  works
- [Quickstart](quickstart.md) — the experiments as commands
- [Tutorial](tutorial.ipynb) — the same run narrated
- [Formats](formats.md) — every file the pipeline reads and writes
- [Command line](cli.md) — every subcommand
- [Python API](api.md) — driving it from a notebook

## Citing

See the repository [README](https://github.com/MattScicluna/manifold_genetics#citation).
