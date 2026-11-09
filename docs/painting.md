# Haplotype painting

Genotype PCA is driven by allele frequency differences, which accumulate slowly.
Two groups that split a few hundred years ago differ very little in frequency,
so PCA tends to merge them. They do differ in which long haplotype segments
they share, because recent common ancestors leave long shared segments behind.
Haplotype painting measures that sharing directly, and it can separate groups
that genotype PCA cannot, such as regional subpopulations within one country.

The pipeline paints with [pbwt](https://github.com/richarddurbin/pbwt)'s
`-paint` command, which approximates ChromoPainter's all-against-all painting.
ChromoPainter's cost grows with the square of the sample size, while pbwt finds
the matching segments in close to linear time. When painting is enabled, its
coordinates replace the genotype PCA as the input to the embedding. Everything
after that, from the embedding to the figures and metrics, runs unchanged.

## What you need

**Phased VCFs, one per chromosome.** PLINK 1 files do not keep phase, so
painting cannot start from the `.bed` files the rest of the pipeline reads. Each
file must hold exactly one chromosome, because pbwt stops reading a VCF at the
first change of chromosome and would otherwise paint only the first one without
any error. pbwt also does not check that genotypes are phased: an unphased VCF
is painted as if it were phased and gives a result that looks normal and is
wrong. Missing genotypes are read as the reference allele, so impute or filter
them first.

**pbwt itself.** pbwt publishes no binaries, so `setup` builds a pinned version
from source, which takes under a minute and needs a C compiler, `make` and the
zlib headers. On Debian or Ubuntu, including the `python:*-slim` images, install
them first:

```bash
apt-get install gcc make libc6-dev zlib1g-dev
manifold-genetics setup --pbwt
```

On Fedora or RHEL the packages are `gcc make zlib-devel`, and on macOS the
Xcode command line tools provide all three. The build was tested on Linux
x86-64 and aarch64 (Debian 13, `python:3.12-slim`) and on macOS arm64. On Linux
aarch64 there is no plink2 download, so `setup` reports that as an error and
exits non-zero, but it still builds pbwt. On macOS, `setup --pbwt` skips
flashpca, which is published only for Linux x86-64 and which the default PCA
backend does not use, and exits zero.

It goes into the same tool directory as the other binaries. A pbwt already on
`PATH`, or named by `PBWT_PATH`, is used as well. pbwt is licensed under
Apache 2.0.

## Turning it on

Two additions to the config:

```yaml
data:
  # ... the usual fit_plink, project_plink, labels, colormap, output_dir
  phased_vcf: data/chr{chr}.vcf.gz

painting:
  enabled: true
  method: hc               # or finestructure, see "Which to use"
  genotype_weight: 0       # 0 is painting only, 1 is genotype PCA only
```

These three are all most runs need. `genotype_weight` mixes the genotype PCA
back in, as described in [Mixing in the genotype PCA](#mixing-in-the-genotype-pca).
The other settings are under [Advanced settings](#advanced-settings).

When PHATE embeds the painting, its nearest-neighbor graph is built with
`painting: knn` (30 by default), not `embedding: knn`. With `genotype_weight: 0`
the painting is PHATE's only input, so `embedding: knn` has no effect there.
Other methods keep their own settings.

`fit_plink` and `project_plink` are still required. Their `.fam` files say which
painted samples belong to which cohort, and admixture still reads their
genotypes. The number of components comes from `pca: n_pcs`, as it does for the
genotype PCA.

When painting provides the embedding input and the method is PHATE, `t`
defaults to 10 rather than PHATE's automatic choice, which separated the
regions less well on painted HGDP+1KGP. A `t` set in the `embedding` section or
by a preset wins. The `whole_cohort` and `projection` presets set `t: 3`, which
scored lower on every measure there, so with a preset, set `t: 10` in the
`embedding` section:

```yaml
preset: whole_cohort

embedding:
  method: phate
  t: 10
```

The same painting is available as a standalone command, which writes the usual
coordinate CSV for `embed` to read:

```bash
manifold-genetics paint --phased-vcf 'data/chr{chr}.vcf.gz' \
    --fit-plink data/fit --project-plink data/project \
    --fit-output out/fit_painting.csv --output out/project_painting.csv --n-pcs 20
```

### Advanced settings

```yaml
painting:
  enabled: true
  method: hc
  genotype_weight: 0
  knn: 30                  # the painting kernel's PHATE knn (the default)
  chromosomes: 1-22        # substituted for {chr} (the default)
  missing: error           # or drop, see "Fit and project cohorts"
  pbwt_dir: shared/pbwt    # where pbwt's output goes, see "Outputs"
  force: false             # true repaints chromosomes whose pbwt output is cached
```

- `knn` is the number of nearest neighbors in the painting's PHATE kernel. See
  [Mixing in the genotype PCA](#mixing-in-the-genotype-pca).
- `chromosomes` lists the chromosomes substituted for `{chr}`, as a range
  (`1-22`), a list or a single chromosome. `hc` works only on autosomes 1 to 22.
- `missing` says what happens to a cohort sample the phased VCFs lack: `error`
  stops the run, `drop` leaves the sample out with a warning.
- `pbwt_dir` puts pbwt's output outside the output directory, so that runs with
  different settings paint only once. See [Sweeping settings](#sweeping-settings).
- `force` repaints every chromosome even when its pbwt output is cached.

## After preprocess or subsample

`preprocess` and `subsample` write a new config, and both carry `data.phased_vcf`
and the `painting` section into it. An absolute `phased_vcf` is copied as
written. A relative one is rewritten relative to the new config, so it still
names the same files. Neither command filters the phased VCFs: their variants
are not intersected with the other cohort's and get none of the QC, and their
samples are not subset. Painting therefore uses the VCFs as given, and each
command logs a warning saying so. The samples still line up, because painting
takes the fit and project samples from the new `.fam` files, and a VCF sample in
neither acts only as a donor. When `preprocess` is given two configs, the
output takes the first one's `phased_vcf`, and a different one in the second
config is named in a warning and dropped.

## What it computes

Each chromosome is painted every sample against every other:
`pbwt -readVcfGT chr<c>.vcf.gz -writeSamples chr<c>.samples -paint chr<c>`.
pbwt writes one row per recipient and one column per donor, and it has already
summed the two haplotypes of each sample, so the matrix is per individual. It
names samples `IND1 … INDn` in VCF order, which is why the sample names are
written alongside. Every chromosome must list the same samples in the same
order, or the matrices could not be summed, so a mismatch stops the run.

The diagonal of every matrix is zero, because pbwt never lets a sample copy
from itself. How the matrices become PCA features follows one of two published
recipes, chosen by `method`. Neither row-normalizes the matrix, and neither
centers its columns.

### `hc` (the default)

These are the haplotype components of
[Yang, Durbin, Iversen and Lawson (2025)](https://doi.org/10.1038/s41467-025-57601-3),
the paper that describes pbwt's `-paint` and `-paintSparse` and that pbwt's
README points to for their use. They computed them for 406,733 UK Biobank
samples:

> we weighted the chunk length for chromosome i with weight gᵢ/Kᵢ. Then we
> summed up the sparse chunk length matrix for all the chromosomes as matrix A
> \[...\] We performed singular value decomposition (SVD) on the
> log-transformed sparse chunk length matrix log10(A + 1) \[...\] Then we
> extracted the first 150 columns of U√D as the top 150 haplotype principal
> components.

1. **Put chunk lengths in centimorgans.** pbwt measures `chunklengths` in sites,
   so each chromosome's matrix is weighted by its genetic length over its number
   of sites, `gᵢ/Kᵢ`. The genetic lengths are the ones the authors used
   ([SparsePainter painting pipeline](https://github.com/YaolingYang/SparsePainter/tree/main/painting-pipeline)),
   and `Kᵢ` is read from the matrix, whose rows pbwt scales to sum to twice the
   number of sites. Only autosomes 1 to 22 have a genetic length here.
2. **Sum** over chromosomes and take `log10(A + 1)`. The diagonal stays zero.
3. **Decompose** by an SVD without centering and write `U √D`.

The first `hc` component carries the overall level of copying, which is close
to constant across samples, so population structure starts on the second.

### `finestructure`

This is the PCA of the coancestry matrix from fineSTRUCTURE
([Lawson et al. 2012](https://doi.org/10.1371/journal.pgen.1002453)), as the
fineSTRUCTURE R library implements it in `mypca`
([FinestructureLibrary.R](https://people.maths.bris.ac.uk/~madjl/finestructure/finestructureR.html)):

> Normalise for PCA by setting the diagonal to be the average of the rows, then
> zero meaning by substracting the row mean (so the diagonal ends up as zero)

1. **Sum the chunk counts** (`chunkcounts`) over chromosomes. Counts are what
   fineSTRUCTURE models.
2. **Fill the diagonal** with the mean of the row's off-diagonal entries.
3. **Center each row** on its mean.
4. **Decompose** `X Xᵀ`, which `mypca` does with `eigen(xN %*% t(xN))`. The
   pipeline takes the same components from an SVD `X = U D Vᵀ` and writes the
   scores `U D`, which are on the scale of genotype PC scores. `mypca` reports
   `U`, which differs only by one scale factor per component.

Lawson et al. show that in the unlinked limit this matrix is approximately
proportional to the one Eigenstrat decomposes, so the components match genotype
PCA there and add haplotype information when markers are linked.

### Which to use

`hc` is the default. It is the recipe pbwt's own authors give for its output,
and it is the one shown to work at biobank scale. Both recipes were compared
with genotype PCA on the HGDP and 1000 Genomes panel from gnomAD
(`phased_haplotypes_v2`), painted genome-wide: 4,089 samples on the 22
autosomes, kept to biallelic SNPs with a minor allele frequency of at least 1%,
which leaves 11.0 million SNPs. Genotype PCA used the same SNPs after LD
pruning. Every PCA was fit on the 3,398 unrelated samples, which are also the
ones scored. The table gives five-nearest-neighbor accuracy for population
labels, over all regions at 5 and 10 components, and with the PCA fit on one
region at a time at 5 components:

| | all, 5 PCs | all, 10 PCs | AFR | AMR | CSA | EAS | EUR | MID | OCE |
|---|---|---|---|---|---|---|---|---|---|
| genotype PCA | 0.76 | 0.82 | 0.91 | 0.95 | 0.82 | 0.78 | 0.87 | 0.95 | 1.00 |
| `finestructure` | 0.81 | 0.86 | 0.87 | 0.93 | 0.80 | 0.83 | 0.92 | 0.95 | 1.00 |
| `hc` | 0.85 | 0.89 | 0.95 | 0.89 | 0.86 | 0.83 | 0.94 | 0.95 | 0.97 |

`hc` separates populations best, but it is not better on every measure. It
separates the continents less cleanly than genotype PCA does: the silhouette of
the seven regions on five components is 0.57 for `hc` and 0.64 for genotype
PCA. The pipeline's geographic preservation metric is also lower for `hc`, at
0.39, than for `finestructure` (0.54) and genotype PCA (0.53 to 0.57). Paint the
whole genome: on chromosome 22 alone, `hc` reaches only 0.62 over all regions.

Because painting is all against all, a sample's components depend on which
other samples were painted. When 1,000 of these samples were painted on their
own, their components still matched those from the full painting with a mean
canonical correlation of 0.88 for `hc`, against 0.70 for `finestructure`.

Use `finestructure` to match a fineSTRUCTURE analysis, or for chromosomes other
than the autosomes, which `hc` cannot weight.

Either way, the number of components is `n_pcs`, capped at one fewer than the
number of fit samples. Up to 5,000 fit samples the SVD is exact, and above that
it is randomized with 20 power iterations.

## Mixing in the genotype PCA

Painting separates populations well, but it keeps less of the continental
structure that genotype PCA shows. `painting: genotype_weight`, a number w from
0 to 1, gives PHATE a weighted sum of the two views' affinities:

K = w · K_genotype / d_genotype + (1 − w) · K_painting / d_painting

1. **K_genotype** is PHATE's own alpha-decay kernel (built by graphtools, as
   PHATE builds it) on the top 10 components of the genotype PCA, with
   `embedding: knn` neighbors and PHATE's decay of 40.
2. **K_painting** is the same kernel on the painting coordinates, all `n_pcs` of
   them, with `painting: knn` neighbors.
3. Each kernel is divided by its **mean degree** d, the mean of its row sums.
   The two views then enter on the same scale, whatever the density of their
   neighborhoods, so w alone sets the genotype's share. This normalization is
   fixed.
4. The sum is **symmetrized**, (K + Kᵀ) / 2, and PHATE takes it as a
   precomputed affinity. Its diffusion, potential distances and MDS then run
   as usual, with the run's `t` and `gamma`.

At 0, the default, the run is painting PHATE as before, with `painting: knn`,
and no genotype PCA runs. At 1 the run embeds the top 10 genotype PCs (any
method, PHATE with `embedding: knn`) and does not paint. In between, the run
also computes the genotype PCA at its usual paths (`pca/fit_pca_<n>.csv`), so
`fit_plink` and `project_plink` must hold the genotypes.

The two `knn` settings look at the data at different scales. `painting: knn`
is the painting's local view: painting resolves fine structure, and a small
neighborhood keeps close relatives of a population together without blurring
neighboring populations into it. `embedding: knn` is the genotype's global
view: a larger neighborhood ties each sample to its continent and keeps the
regions in place. So `painting: knn` is usually the smaller of the two.

Between 0 and 1 the run is refused, with a configuration error before any
step runs, when it asks for something the mixed kernel cannot do:

- **Methods other than PHATE** (`umap`, `tsne`, `diffusion_map`): the sum is
  of PHATE kernels, and the other methods build their own graphs. Use a weight
  of 0 or 1 with them.
- **`input_mode: both`, and so the `projection` preset**: the kernel is built
  over the samples being embedded, and nothing extends it to new samples, so
  the run cannot fit on one cohort and place another. Use `input_mode: fit` or
  `project`, or the `whole_cohort` preset.
- **Landmarks** (`n_landmark` other than none, and so the `subsample` preset):
  PHATE's landmarks are computed from its own graph, not from a precomputed
  affinity, so the run is exact.
- **Grids**: every `method_grid` panel must be `phate`, and no panel of either
  grid may set `n_landmark`. The PHATE panels of both grids mix the same two
  kernels with the run's `genotype_weight` and `painting: knn`. A `phate_grid`
  axis over `knn` varies `embedding: knn`, which is the genotype kernel's
  between 0 and 1, and the painting kernel's at 0.

Exact PHATE holds dense n × n matrices, so like painting itself it is
practical up to about 10,000 to 20,000 samples.

On the HGDP and 1000 Genomes panel described above (4,089 samples painted
genome-wide with `n_pcs: 20`, genotype PCA on the 172,000 SNPs `acquire hgdp`
provides), with exact PHATE on all samples and `embedding: knn: 100`, scored
on the 3,398 unrelated samples in 2-D. Each cell gives the silhouette of the
seven regions, five-nearest-neighbor accuracy for populations and the
pipeline's geographic preservation metric, in that order:

| `genotype_weight` | `hc`, knn 10 | `hc`, knn 30 | `finestructure`, knn 10 | `finestructure`, knn 30 |
|---|---|---|---|---|
| 0 (painting only) | -0.02 / 0.71 / 0.16 | 0.17 / 0.84 / 0.45 | 0.19 / 0.72 / 0.37 | 0.08 / 0.78 / 0.46 |
| 0.25 | 0.45 / 0.78 / 0.66 | **0.47 / 0.78 / 0.65** | 0.47 / 0.73 / 0.66 | 0.24 / 0.77 / 0.56 |
| 0.5 | 0.44 / 0.76 / 0.67 | 0.45 / 0.74 / 0.66 | 0.46 / 0.66 / 0.66 | 0.29 / 0.75 / 0.57 |
| 0.75 | 0.41 / 0.73 / 0.68 | 0.44 / 0.71 / 0.67 | 0.45 / 0.71 / 0.67 | 0.32 / 0.60 / 0.58 |
| 1 (genotype PCA only) | 0.37 / 0.69 / 0.66 | 0.37 / 0.69 / 0.66 | 0.37 / 0.69 / 0.66 | 0.37 / 0.69 / 0.66 |

Each column's knn is `painting: knn`, and `t` is 10. Automatic `t` gave nearly the same
scores (`hc`, knn 30, weight 0.25: 0.47 / 0.78 / 0.67) and took seven times as
long (50 seconds against 7 per embedding on an Apple M2 Max).

With `hc`, every weight from 0.25 to 0.75 separated the regions better than
either view alone (0.41 to 0.47, against 0.37 for the genotype PCA and at most
0.17 for the painting), preserved geography as well as the genotype PCA (0.65
to 0.68, against 0.66) and kept more population accuracy than it. 0.25 kept the
most, and `painting: knn` mattered little once the genotype was mixed in. With
`finestructure`, knn 10 did as well as `hc`, and knn 30 much worse. Painting alone at knn 30 had the best population accuracy (0.84 with
`hc`) but the weakest regions. As starting values, use `method: hc`,
`genotype_weight: 0.25`, `painting: knn: 30`, `embedding: knn: 100` and
`t: 10`, with `input_mode: project` (or the `whole_cohort` preset with `t: 10`).
The best values depend on the data and on which structure matters most, so
sweep them (see [Sweeping settings](#sweeping-settings)).

## Fit and project cohorts

Painting is all against all within the painted set, so it has no model to
project a separate cohort through in the way genotype PCA does. What carries
over is the shape of the data. Every row is one recipient's copying profile
over the same donors, the whole painted set. So the fit cohort's rows fit the
PCA, and the project cohort's rows are projected through it, as a project
cohort's genotypes are projected through loadings fitted elsewhere.

This means every sample of both cohorts must be in the phased VCFs, and by
default the run stops if one is not, rather than quietly shrink a cohort. The
check reads only the VCF headers and the `.fam` files, so it runs before any
painting and fails in seconds. Sample names in the VCF may be either the `.fam`
IID or `FID_IID`, which covers the usual ways a PLINK export names samples.
Coordinates are written under the IID, which is what the labels file uses.

The error tells two cases apart. When the names differ only in format, for
example a `.fam` IID of `FAM1_HG00096` against a VCF sample `HG00096`, it shows
a few pairs, and one side has to be renamed (for example with
`bcftools reheader --samples`). When samples are absent from the VCF, it names a
few. A public phased panel often lacks a handful of a cohort's samples: the
gnomAD HGDP and 1000 Genomes haplotypes hold 4,089 of the 4,094 samples that
`acquire hgdp` writes. To paint without them, set `painting: missing: drop`
(`--missing drop` for `paint`). The missing samples are then left out of their
cohort, with a warning that counts them, and the coordinates have no row for
them. A sample in the VCF that is in neither cohort is still painted and
still serves as a donor, which is how a reference panel added to the VCF
sharpens the painting without appearing in the figures.

The presets work as they do with genotype PCA. `whole_cohort` embeds the project
cohort's painting coordinates, `subsample` embeds the fit cohort's, and
`projection` fits the embedding on the fit cohort and applies it to the project
cohort. With a `genotype_weight` between 0 and 1, only `whole_cohort` works
(see [Mixing in the genotype PCA](#mixing-in-the-genotype-pca)).

## Outputs

```
outputs/painting/
├── pbwt/
│   ├── chr1.chunkcounts.out          pbwt's own output, reused by a rerun
│   ├── chr1.chunklengths.out
│   ├── chr1.samples                  sample names, in VCF order
│   └── chr1.log                      pbwt's log
├── coancestry_<method>.csv                 summed matrix, recipients by donors
├── fit_painting_<method>_pca_<n>.csv        sample_id, dim_1 … dim_<n>
├── project_painting_<method>_pca_<n>.csv
├── fit_genotype_pca_top10.csv               with genotype_weight 1: the top
└── project_genotype_pca_top10.csv           10 genotype PCs the run embeds
```

A chromosome is repainted only when its outputs are missing, older than its VCF,
or `painting: force: true` is set. With `skip: pca: true` the painting is
skipped too, and the cached coordinates of the configured method are embedded.
The method is in each file name, so switching between `hc` and `finestructure`
in one output directory keeps both sets, and `skip: pca` never picks up the
other method's coordinates.

pbwt's output is the same for both methods, and it is by far the slow part.
`painting: pbwt_dir` (`--pbwt-dir` for `paint`) puts it somewhere other than
`painting/pbwt/` in the output directory, so that runs with different methods
or settings in separate output directories paint only once. A relative
`pbwt_dir` is resolved against the config file's directory. The PC-pair grids in
`figures/pca/` show the painting components. The comparison grids
(`visualization: phate_grid` and `method_grid`) embed the same coordinates as
the run: the painting components, or the top 10 genotype PCs when
`genotype_weight` is 1. Between 0 and 1 their PHATE panels mix the same two
kernels as the run.


## Sweeping settings

pbwt is by far the slow part, and its output is the same for both methods and
every `genotype_weight` and `knn`. Give every run the same `pbwt_dir` and its
own output directory, and pbwt runs once: the first run paints and the others
reuse its output. Start from one config, here `config.yaml`:

```yaml
data:
  fit_plink: data/fit
  project_plink: data/project
  labels: data/labels.csv
  colormap: data/colormap.json
  output_dir: outputs
  phased_vcf: data/chr{chr}.vcf.gz

painting:
  enabled: true
  method: hc
  genotype_weight: 0
  knn: 30
  pbwt_dir: pbwt

pca:
  n_pcs: 20

embedding:
  method: phate
  input_mode: project
  knn: 100
  t: 10

visualization:
  phate_grid:
    t: [3, 10, auto]

skip:
  admixture: true
```

`input_mode: project` embeds the project cohort exactly, which every weight
can do. Then write one config per setting next to it, so that its relative
paths, `pbwt_dir` included, resolve to the same places, and run each into its
own output directory. The third `sed` expression changes `knn` only in the
`painting` section, from `painting:` to the next blank line:

```bash
for method in hc finestructure; do
  for w in 0 0.25 0.5 0.75; do
    for k in 10 30; do
      name="${method}_w${w}_k${k}"
      sed -e "s/method: hc/method: $method/" \
          -e "s/genotype_weight: .*/genotype_weight: $w/" \
          -e "/^painting:/,/^$/s/knn: .*/knn: $k/" \
          config.yaml > "config_${name}.yaml"
      manifold-genetics run "config_${name}.yaml" --output "sweep/${name}"
    done
  done
done
```

A weight of 1 depends on neither the method nor `painting: knn`, so run it
once:

```bash
sed -e "s/genotype_weight: .*/genotype_weight: 1/" config.yaml > config_genotype.yaml
manifold-genetics run config_genotype.yaml --output sweep/genotype
```

Run them one after another, as the loop does. Runs started together would all
find `pbwt_dir` empty and paint at the same time. Each run's
`figures/embeddings/` holds its embedding, and `phate_grid` adds a figure of
PHATE over `t` on the same input to `figures/embeddings/grids/`, so one sweep
covers the method, the weight, both `knn` settings and `t`. With
`geographic_coords` set in `data`, each run's `metrics/geographic.json` gives
its geographic preservation score.

## Limits

pbwt paints one chromosome at a time, so the time for a genome is the sum over
chromosomes, and peak memory is that of the largest one. Measured on an Apple
M2 Max for chromosome 22 of the 1000 Genomes high coverage panel
(190,184 biallelic SNPs with MAF of at least 1%):

| samples | `-paint` time | `-paint` peak memory |
|---|---|---|
| 500 | 5 s | 0.19 GB |
| 1,000 | 11 s | 0.38 GB |
| 2,000 | 24 s | 0.79 GB |
| 3,202 | 41 s | 1.37 GB |

Time grows close to linearly over this range, and in proportion to the number
of sites. Genome-wide, on the gnomAD HGDP and 1000 Genomes panel described
above (4,089 samples, 11.0 million SNPs) and on the same machine, the whole
`hc` run took 44 minutes, nearly all of it in pbwt. Peak memory was 4.4 GB, for
chromosome 2. A `finestructure` run that reused that pbwt output took 51
seconds. Memory has two
parts: the matches pbwt stores, about 0.34 MB per sample here (linear in the
samples and in the chromosome's sites), and the four
dense n × n matrices of doubles `-paint` writes (32 n² bytes). The second
dominates past about 10,000 samples: extrapolated, `-paint` needs about 6 GB at
10,000 samples, 80 GB at 50,000 and 300 GB at 100,000. The pipeline then holds
the summed matrix, its features and the SVD, about 15 copies of an n × n matrix
at 3,202 samples. Painting a whole biobank with `-paint` is therefore out of
reach. Paint a subsample, or the cohort you want to resolve at fine scale,
alongside a modest reference panel.

pbwt's `-paintSparse` writes the same values (identical on the benchmark above)
as a list of nonzero entries and never holds the n × n matrices, so its memory
stays linear in the samples. It is what Yang et al. used for the UK Biobank.
The pipeline does not use it yet.

Painting is not yet part of the WDL workflows, and the container image does not
include pbwt.
