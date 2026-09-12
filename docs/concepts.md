# Concepts

Four things are worth understanding before running this on a cohort you care
about. None of them is obvious from the API.

## One word, one meaning

This project used `transform` for two different things, and it caused real
confusion: the second cohort's *dataset role*, and the sklearn-style *method
verb*. The rule now, and it is worth knowing before reading anything else:

| word | means | example |
|---|---|---|
| **`fit`** | estimating a model | `PCA.fit`, the **fit set** |
| **`transform`** | applying a fitted model — the method verb, and nothing else | `PHATE.transform` |
| **`project`** | the second cohort, and its outputs | the **project set**, `project_pca_50.csv` |

So `transform` never names a dataset, and `project` never names an operation.
Preset names follow from that: they describe the *shape of the run*, never a
method. The preset that used to be called `transform` is now `whole_cohort`;
the old name still works and warns.

Output files were renamed to match in an earlier release — see
[the migration note](migrations/2026-09-transform-to-project.md) if you have
result directories predating it.

## Fit and project

Every run takes two PLINK datasets, and the distinction between them is the
analysis, not a detail of the interface.

The **fit** set is what the model is estimated on: PCA's means, standard
deviations and loadings, and admixture's ancestry components. The **project**
set is what those estimates are then applied to. The fit set defines the frame
of reference; the project set is placed inside it.

This matters because PCA on genotypes is not scale-free. The axes it finds are
the directions of greatest variance *in the cohort it was fitted on*. Fit on a
cohort that is 80% one ancestry and the leading components describe variation
within that group. Fit on a globally diverse reference panel and the leading
components describe continental structure, with your cohort's samples placed
along them.

So there are three sensible shapes, and they are the three
[presets](configuration.md#presets).

### `projection` — someone else's frame of reference

Fit on a reference panel (HGDP+1KGP, typically), project your cohort onto it.
Both sets are embedded, so a figure can show the reference and the target
together.

Use it to say where your cohort sits in global variation, and to compare across
cohorts: two biobanks projected onto the same reference are directly comparable,
which they are not if each was analysed on its own axes.

### `subsample` — a fair view of your own cohort

Fit on a subset of your cohort, embed that subset only.

Large biobanks are dominated by one group — UK Biobank is about 80% "British" —
and both PCA and a neighbour graph will spend their capacity describing it. A
majority-capped subset (10,000 British plus everyone else) or a
[geometric sketch](#choosing-a-subset) gives the rest of the cohort room to
appear.

### `whole_cohort` — fit on a subset, embed everything

Fit on a subset, then project and embed the whole cohort. The natural shape when
the project set *contains* the fit set: the fit subset is a computational
convenience, not a different population.

## The PCA contract

The PCA output is not simply "some principal components". It is a specific set
of conventions, because a second cohort projected onto the model must be
standardised exactly as the first was — and getting that wrong produces
projections that look plausible and are wrong.

| property | convention |
|---|---|
| dosage | count of the **A1** allele, the fifth column of the `.bim` |
| mean | over non-missing genotypes only |
| standard deviation | `sqrt(mean * (1 - mean / 2))` — the binomial form, not the sample SD |
| eigenvalues | `S**2 / n_variants` |
| loadings | right singular vectors, unit norm |
| fit coordinates | `U * sqrt(eigenvalues)` |
| projected coordinates | `X_new @ loadings / sqrt(n_variants)`, using the **fit** cohort's mean and SD |

These are FlashPCA's conventions. They were determined empirically, by
regressing recorded `.meansd` values against allele frequencies computed from
the same `.bed`, rather than taken from documentation — and the in-process
backend agrees with FlashPCA to 5e-7 or better on real data. Counting A2 instead
of A1 gives dosages wrong by up to 1.97.

The practical consequence: a model fitted by either backend is usable by the
other, and both write the same artefacts (`.meansd`, `.loadings`, `.eigenval`,
`.eigenvec`, `.PC`, `pve.txt`). Switching backends is a performance decision,
never a scientific one.

## Landmarking, and why it is a preset

Diffusion-based embeddings build a graph over samples. Exactly, that is
quadratic, which is fine at 4,000 samples and impossible at 500,000. Landmarking
approximates it: pick `n_landmark` representatives, diffuse over those, then map
everyone onto the result.

There are two ways to pick them, and the difference is large:

- **spectral** (the default when `n_landmark` is set alone) runs MiniBatch
  k-means on diffusion coordinates. Better landmarks, much more expensive — it
  needs diffusion coordinates first, which is the thing landmarking was supposed
  to avoid.
- **random** draws uniformly and assigns each sample to its nearest landmark in
  PCA space. Cheap.

Random is not merely the affordable option. On a 60,000-sample UK Biobank cohort
at matched settings, random landmarking gave a **31% better** separation of the
Central/South Asian branch than spectral did — density-flattening a
majority-dominated cohort helps, and spectral landmarks follow the density they
should be ignoring.

Setting `n_landmark` without `random_landmarking: true` silently selects the
expensive path. That is exactly the bug the presets exist to prevent, which is
why the two are always set together:

| preset | knn | t | landmarks |
|---|---|---|---|
| `projection` | 100 | 3 | none |
| `whole_cohort` | 100 | 3 | none |
| `subsample` | 500 | 50 | 10,000, random |

Below roughly 50,000 samples, exact diffusion is affordable and better than
approximating it. Above it, use `subsample`'s settings.

## Choosing a subset

Two strategies are used in the examples, and a 2026 comparison at matched
settings found they give the **same branch topology**, differing only within
Europe:

- **majority capping** — take at most N of the dominant group, keep everyone
  else (`examples/ukbb/10k_WB_5K_Irish`). Simple, and the cap is interpretable.
- **geometric sketching** — [geosketch](https://github.com/brianhie/geosketch)
  covers the PCA space evenly rather than proportionally to density
  (`examples/ukbb/geosketch_phate`). No label needed, which matters when
  self-described ancestry is unreliable or absent.

Because they agree, the examples deliberately use *identical* embedding settings
for both: how a subset was chosen should not change how it is embedded.

## Output layout

Every run writes the same tree, which is what makes runs comparable:

```
outputs/
├── pca/
│   ├── fit_pca_<n>.csv            sample_id, dim_1 … dim_n
│   ├── project_pca_<n>.csv
│   └── flashpca_outputs/          the projectable model
├── admixture/                     Q matrices per K
├── embeddings/
│   └── <method>_2d.csv
├── figures/
│   ├── pca/
│   ├── embeddings/
│   └── admixture/
└── metrics/
    ├── geographic.json
    └── admixture.json
```

Which files appear depends on the mode — `subsample` writes no project
embedding, for instance. Stages are idempotent: re-running reuses whatever is
already there, so an interrupted run resumes rather than restarting.
