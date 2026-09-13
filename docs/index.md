# manifold-genetics

An end-to-end pipeline from PLINK genotypes to publication figures: PCA →
admixture → manifold embedding → visualisation → metrics.

PCA runs in process — it reads PLINK `.bed` directly and computes a randomized
SVD — so there is no binary to fetch and no platform it only works on. FlashPCA
remains an opt-in accelerator and writes the same artefacts, so a model fitted
by either is readable by the other.

[Install](install.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }

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

- [Install](install.md) — pip or from source, the optional extras, and how to
  check it works
- [Quickstart](quickstart.md) — the two experiments as commands, and the file
  formats going in and out
- [Tutorial](tutorial.ipynb) — the same run narrated: what each stage produces
  and how to read it

Reference material — every config key, every subcommand, the Python API — is
being rewritten and is not on the site yet. Until it is, it lives in the
repository under [`archive/`](https://github.com/MattScicluna/manifold_genetics/blob/main/archive).

## Citing

See the repository [README](https://github.com/MattScicluna/manifold_genetics#citation).
