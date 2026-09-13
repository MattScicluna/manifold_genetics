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

See XXX for the format of files required for each step. 
Note that there is a CLI available at XXX so you can run individual steps or compose stages in whatever order makes sense for your project.

Stages are checkpointed: a run that already produced PCA output reuses it. This matters on cohorts where PCA is the expensive part.

This part has a problem. we need to describe the 3 modes first I think. whether we have a fit and transform/project or them seperate. somehow this should be introduced here? else the next section makes no sense

## Inputs

The user passes a cohort to  cohorts.  Two cohorts go in, and which one the model is *fitted* on is the choice that
defines the analysis:
- **`fit`** - reference panel ...
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
