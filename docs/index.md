# manifold-genetics

An end-to-end pipeline from PLINK genotypes to publication figures: PCA →
admixture → manifold embedding → visualisation → metrics.

```bash
pip install manifold-genetics
manifold-genetics run config.yaml
```

That is the whole install. PCA runs in process — it reads PLINK `.bed` directly
and computes a randomized SVD — so there is no binary to fetch and no platform
it only works on. FlashPCA remains available as an opt-in accelerator and writes
the same artefacts, so a model fitted by either is readable by the other.

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
- **`transform`** — fit on a subset, embed the whole cohort. The default shape
  when the project set contains the fit set.

Those three are [presets](configuration.md#presets), because the settings that
follow from the choice — particularly landmarking — are easy to get wrong
individually. [Concepts](concepts.md) goes through why.

## Where to start

- [Installation](install.md) — including the optional extras
- [Tutorial](tutorial.ipynb) — a full run on a cohort it simulates as it goes,
  under a minute, nothing to download
- [Concepts](concepts.md) — fit versus project, the PCA contract, landmarking
- [Configuration](configuration.md) — every key a config file accepts
- [Running on a cluster](hpc.md) — SLURM, and the memory arithmetic
- [Controlled-access data](controlled-access.md) — UK Biobank, All of Us

## Citing

See the repository [README](https://github.com/MattScicluna/manifold_genetics#citation).
