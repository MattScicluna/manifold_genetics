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
exits non-zero, but it still builds pbwt.

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
  method: hc               # or finestructure, see below
  chromosomes: 1-22        # substituted for {chr} (the default)
```

`fit_plink` and `project_plink` are still required. Their `.fam` files say which
painted samples belong to which cohort, and admixture still reads their
genotypes. The number of components comes from `pca: n_pcs`, as it does for the
genotype PCA.

The same thing is available as a standalone command, which writes the usual
coordinate CSV for `embed` to read:

```bash
manifold-genetics paint --phased-vcf 'data/chr{chr}.vcf.gz' \
    --fit-plink data/fit --project-plink data/project \
    --fit-output out/fit_painting.csv --output out/project_painting.csv --n-pcs 20
```

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
it is the one shown to work at biobank scale, and on the 1000 Genomes high
coverage panel (chromosome 22 only, 3,202 samples) it separated populations
within each continent better than `finestructure` did. Five-nearest-neighbor
accuracy for population labels on the first five components, with the PCA fit
on one continent at a time:

| | AFR | AMR | EAS | EUR | SAS |
|---|---|---|---|---|---|
| genotype PCA | 0.57 | 0.56 | 0.48 | 0.45 | 0.27 |
| `finestructure` | 0.74 | 0.69 | 0.84 | 0.81 | 0.44 |
| `hc` | 0.83 | 0.95 | 0.89 | 0.89 | 0.70 |

`hc` was also the more stable of the two when the painted set changed. Use
`finestructure` to match a fineSTRUCTURE analysis, or for chromosomes other
than the autosomes, which `hc` cannot weight.

Either way, the number of components is `n_pcs`, capped at one fewer than the
number of fit samples. Up to 5,000 fit samples the SVD is exact, and above that
it is randomized with 20 power iterations.

## Fit and project cohorts

Painting is all against all within the painted set, so it has no model to
project a separate cohort through in the way genotype PCA does. What carries
over is the shape of the data. Every row is one recipient's copying profile
over the same donors, the whole painted set. So the fit cohort's rows fit the
PCA, and the project cohort's rows are projected through it, as a project
cohort's genotypes are projected through loadings fitted elsewhere.

This means every sample of both cohorts must be in the phased VCFs, and the run
stops if one is not, rather than quietly shrink a cohort. Sample names in the
VCF may be either the `.fam` IID or `FID_IID`, which covers the usual ways a
PLINK export names samples. Coordinates are written under the IID, which is what the labels
file uses. A sample in the VCF that is in neither cohort is still painted and
still serves as a donor, which is how a reference panel added to the VCF
sharpens the painting without appearing in the figures.

The presets work as they do with genotype PCA. `whole_cohort` embeds the project
cohort's painting coordinates, `subsample` embeds the fit cohort's, and
`projection` fits the embedding on the fit cohort and applies it to the project
cohort.

## Outputs

```
outputs/painting/
├── pbwt/
│   ├── chr1.chunkcounts.out          pbwt's own output, reused by a rerun
│   ├── chr1.chunklengths.out
│   ├── chr1.samples                  sample names, in VCF order
│   └── chr1.log                      pbwt's log
├── coancestry_<method>.csv          summed matrix, recipients by donors
├── fit_painting_pca_<n>.csv          sample_id, dim_1 … dim_<n>
└── project_painting_pca_<n>.csv
```

A chromosome is repainted only when its outputs are missing, older than its VCF,
or `painting: force: true` is set. With `skip: pca: true` the painting is
skipped too, and the cached coordinates are embedded. The PC-pair grids in
`figures/pca/` show the painting components, since those are what the embedding
read.

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
of sites. Chromosome 22 is about 1.5% of the autosomal sequence, so the 22
autosomes take roughly 60 to 70 times as long at this density. Memory has two
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
