# manifold-genetics

An end-to-end pipeline from PLINK genotypes to publication figures: PCA →
admixture → manifold embedding → visualisation → metrics.

```bash
pip install manifold-genetics
manifold-genetics init synthetic     # a simulated cohort, and a config for it
manifold-genetics run config.yaml
```

Those three commands produce figures from nothing, in about a minute, with no
data of your own and nothing to download. `init hgdp` does the same with the
real HGDP+1KGP cohort instead, fetching it for you.

That is also the whole install. PCA runs in process — it reads PLINK `.bed`
directly and computes a randomized SVD — so there is no binary to fetch and no
platform it only works on. FlashPCA remains available as an opt-in accelerator
and writes the same artefacts, so a model fitted by either is readable by the
other.

[Install](install.md){ .md-button .md-button--primary }
[Tutorial](tutorial.ipynb){ .md-button }

## What it is for

Human genetic variation is continuous. Discrete population labels are a useful
summary and a poor description: individuals of recent mixed ancestry sit between
the groups they descend from, and a method that assumes clusters will put them in
one. Manifold embeddings recover that continuity, which is what these figures
show.

<p align="center">
  <img src="https://raw.githubusercontent.com/MattScicluna/manifold_genetics/main/assets/ukbb_phate.png" width="45%" alt="UK Biobank PHATE embedding coloured by self-described ancestry"/>
  <img src="https://raw.githubusercontent.com/MattScicluna/manifold_genetics/main/assets/aou_phate.png" width="45%" alt="All of Us PHATE embedding coloured by ancestry"/>
</p>

## What it does

| stage | what it produces |
|---|---|
| **PCA** | components and a projectable model, in process or via FlashPCA |
| **Admixture** | ancestry proportions per K, via neural-admixture |
| **Embedding** | 2-D coordinates from PHATE, UMAP, t-SNE or diffusion maps |
| **Visualisation** | scatter plots by any label column, admixture bar plots |
| **Metrics** | geographic and admixture preservation |

Every stage reads and writes one format — a `sample_id` column followed by
`dim_1 … dim_n` — so they compose in whatever order makes sense, and you can
start from the middle if you already have principal components.

Stages are checkpointed: a run that already produced PCA output reuses it. This
matters on cohorts where PCA is the expensive part.

## The shape of a run

Two cohorts go in, and which one the model is *fitted* on is the choice that
defines the analysis:

- **`projection`** — fit on a reference panel, project a target cohort onto it.
  Places your cohort in a frame of reference somebody else defined.
- **`subsample`** — fit on a subset of your own cohort, embed that subset. For
  cohorts too large, or too dominated by one group, to embed whole.
- **`whole_cohort`** — fit on a subset, embed the whole cohort. The default shape
  when the project set contains the fit set, and what `init` writes.

Those three are presets, because the settings that
follow from the choice — particularly landmarking — are easy to get wrong
individually.

## Where to start

- [Installation](install.md) — including the optional extras
- [Quickstart](quickstart.md) — a full run on public data, and the file formats
- [Tutorial](tutorial.ipynb) — a full run on a cohort it simulates as it goes,
  under a minute, nothing to download

Reference material — every config key, every subcommand, the Python API — is
being rewritten and is not on the site yet. Until it is, it lives in the
repository under [`archive/`](https://github.com/MattScicluna/manifold_genetics/blob/main/archive).

## Citing

See the repository [README](https://github.com/MattScicluna/manifold_genetics#citation).
