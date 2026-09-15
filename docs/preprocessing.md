# Preprocessing

Optional. If your genotypes are already filtered to the SNPs you want, skip
this page: `acquire custom` and `run` are enough.

If you are starting from raw biobank PLINK files, two commands sit between
`acquire` and `run`. Each reads a cohort directory and writes a new one, so
they compose and can be re-run:

```
acquire  →  preprocess  →  subsample  →  run
            filter SNPs    choose the fit samples
```

`preprocess` filters SNPs and never removes a sample; `subsample` chooses
samples and never touches a SNP. Every published flow runs them in that order,
so the MAF filter sees the whole cohort before any subset is drawn.

## The cohort directory

What `acquire` writes and what `preprocess`, `subsample` and `run` read:

```
<dir>/
  config.yaml            names every file below, relative to <dir>
  colormap.json          whole_cohort   (or colormap_fit.json + colormap_project.json for projection)
  data/
    fit_subset.{bed,bim,fam}
    project_subset.{bed,bim,fam}
    labels.csv           whole_cohort   (or fit_labels.csv + project_labels.csv for projection)
```

A `geographic_coords` file, when the input names one, is carried through both
commands -- filtered to the output's project `.fam` -- since neither removes a
project-side sample; a carried `embedding` section has its `input_mode` and
landmarking keys (`knn`, `t`, `n_landmark`, `random_landmarking`) dropped, with
a warning per key, whenever the output's preset differs from the input's, so
they cannot silently override what the new preset sets.

`config.yaml` is the entry point, which is why `preprocess` and `subsample`
take configs rather than PLINK prefixes: the config says where the labels and
colormaps are. A label file must cover at least half of the `.fam` it is
paired with — the rule `acquire` applies — and samples without a row are
warned about and drawn grey; extra rows are fine. Both commands check this
before doing any work, so a label file that describes the wrong cohort fails
in seconds, not after the shell has run. Each command rewrites the labels in
its output directory from its input, so they always describe the genotypes
beside them.

## preprocess — filter SNPs

```bash
manifold-genetics preprocess cohort/config.yaml --out filtered/
manifold-genetics preprocess ref/config.yaml biobank/config.yaml --out proj/ --preset harmonise
```

One config filters a cohort: its own fit and project sets are the two sides.
Two configs intersect two cohorts: the first supplies the fit set and its
labels, the second the project set, and the output config has
`preset: projection`. Either way the result is a new cohort directory, ready
for `run`, `subsample`, or another `preprocess`.

Under the hood this runs `preprocess_cross_projection.sh`, the shell script
that produced the published figures, shipped inside the package. It needs
`bash`, `plink2` and `plink` v1.9 — both binaries are fetched by
`manifold-genetics setup`, or on first use where there is internet.

### Presets

| preset | what runs | reproduces |
|---|---|---|
| `--preset intersect-only` | indel removal, missingness (`--geno`), position intersection | the `examples/generic` flows |
| `--preset harmonise` | WRayner/TOPMed allele checks, GIAB difficult regions, HLA, deduplication, MAF on the reference side only (`--skip-project-maf`), LD pruning, intersection; `--cleanup` | the All of Us flow |
| *(none)* | the shell's defaults: every step on, MAF on both sides; add `--skip-*` flags | the UK Biobank flow is `--skip-wrayner --skip-project-maf` — `harmonise` minus WRayner, without `--cleanup` |

Every step has a `--skip-*` flag (`--skip-wrayner`, `--skip-giab`,
`--skip-hla`, `--skip-ld-prune`, `--skip-dedup`, `--skip-maf`, `--skip-geno`,
`--skip-project-maf`), and the thresholds are `--maf`, `--geno`,
`--ld-window`, `--ld-step`, `--ld-r2`. A skip flag added to a preset is
honoured on top of it.

`--skip-geno` exists for a specific reason. The public HGDP+1KGP panel that
`acquire hgdp` fetches is already filtered and LD-pruned, so preprocessing it
should change nothing — but `intersect-only` still recomputes missingness on
the fit and project subsets, and 747 of its 172,152 SNPs (0.43%) newly exceed
5% on those smaller sets. `--preset intersect-only --skip-geno` on that panel
is exactly lossless: same 172,152 SNPs, same samples on both sides. The same
is true of the intersected UK Biobank cohort.

Note that the public panel is **not** a drop-in reference for a harmonised
cross-projection: the UK Biobank flag set on it removes 41% of its SNPs
(GIAB and HLA exclusion, then LD re-pruning of what is left). The published UK
Biobank and All of Us runs each used a different HGDP+1KGP file — the
a private gnomAD-derived panel and the workbench's own archive respectively —
and a reproduction has to start from the same one.

### What needs internet

`intersect-only` needs nothing beyond `plink2`. `harmonise` needs three
references — the GIAB difficult-regions bed, the WRayner
`HRC-1000G-check-bim.pl` checker, and the TOPMed reference panel (about 2 GB
together, most of it TOPMed) — which the shell downloads into the tool cache
when they are missing. On a cluster whose compute nodes have no internet,
prefetch them on a login node:

```bash
manifold-genetics setup --preprocessing
```

**The WRayner download currently fails, everywhere.** Its upstream URL
(`https://www.chg.ox.ac.uk/~wrayner/tools/HRC-1000G-check-bim-v4.3.0.zip`)
returns 404 as of this release. `setup --preprocessing` attempts all three
references regardless, so today it places GIAB and TOPMed, then exits non-zero
with a message naming the WRayner URL and the exact path where a hand-placed
checker goes. The shell itself fetches the same URL when it finds no checker,
so `--preset harmonise` dies at its WRayner step on any machine, internet or
not, until the file is in place. Until upstream is back:

1. Place a copy of `HRC-1000G-check-bim.pl` (MIT-licensed; it circulates in
   many imputation pipelines) at `<tools-dir>/wrayner/HRC-1000G-check-bim.pl`,
   where `<tools-dir>` is the tool cache's `preprocessing/` subdirectory:

    ```bash
    python -c "from manifold_genetics.preprocessing.references import default_tools_dir; print(default_tools_dir())"
    ```

    Comment out its `--recode vcf` line — the one beginning
    `print SH "$plink --bfile $newfile --real-ref-alleles --recode vcf` — as
    the fetcher does, so it does not write a VCF nobody reads.

2. Re-run `manifold-genetics setup --preprocessing`. It is idempotent: it
   leaves GIAB, TOPMed and the hand-placed checker alone and fetches only
   what is still missing.

Or pass `--tools-dir DIR` to `preprocess` with a directory laid out the same
way. `--skip-wrayner` sidesteps the whole step, which is what the UK Biobank
flow does.

### Resources

Set `--threads` (default: `SLURM_CPUS_PER_TASK`, else 4) and `--memory` in MB
(default 100000). Reproducing the UK Biobank projection — 486,748 samples
against a 120,849-SNP panel — peaked at 85 GB, so run that on a node with at
least 128 GB and leave `--memory` at its default. `--temp-dir` moves the
scratch space (default `OUT/data/temp`); `--cleanup` deletes intermediates as
they are consumed, which matters for a biobank-sized `.bed`.

### Troubleshooting

`preprocess` resumes: a step whose output already exists and is complete is
skipped. After an out-of-memory kill, a truncated `.bed` under
`OUT/data/temp/` is not mistaken for a finished output — an incomplete
intermediate is detected and redone on the next run. `--force` still only
rewrites the config and labels: to recompute with different flags, use a new
`--out` or delete `OUT/data/temp`, because the shell otherwise reuses every
complete intermediate that exists.

If the two cohorts share fewer than 50,000 SNPs the run aborts rather than
producing a projection nobody should trust; `--min-common-snps` changes the
threshold, and usually the right fix is upstream (different builds, or a
`chr` prefix on one side — see `--fit-has-chr-prefix`).

## subsample — choose the fit samples

```bash
manifold-genetics subsample proj/config.yaml --out 10k/ \
    --group "self_described_ancestry=^British$:10000" \
    --group "self_described_ancestry=^Irish$:5000"
```

Reads a cohort directory and writes one whose fit set is a chosen subset of its
project set. The project set is linked, not copied, the labels are filtered to
match, and the output config uses the `subsample` preset: fit on the subset,
embed the subset, with random landmarking. Given a `projection` config, the
input's fit side (the reference panel) is dropped — the output fits on a subset
of the biobank.

Choose the samples by exactly one of:

- `--group COLUMN=PATTERN:COUNT`, repeatable. `PATTERN` is a case-insensitive
  regex *searched for* in `COLUMN` of the label file — a substring match, so
  `British` also takes `Black or Black British`; anchor it (`^British$`) to
  mean the whole value. `COUNT` samples are drawn from the matches with
  `--seed` (default 42), or all of them if there are fewer. A sample is taken at most once across groups. `--include-rest` appends
  every sample no group matched. For All of Us:

    ```bash
    manifold-genetics subsample proj/config.yaml --out 10k_wbh/ \
        --group "race_ethnicity=White|European:10000" \
        --group "race_ethnicity=Black or African American:10000" \
        --group "race_ethnicity=Hispanic or Latino:10000" \
        --include-rest --seed 42
    ```

- `--fit-samples FILE`: a `FID IID` list chosen elsewhere.

    ```bash
    manifold-genetics subsample proj/config.yaml --fit-samples fit_samples.txt --out 10k/
    ```

- `--geosketch N --pca CSV`: geometric sketching (Hie et al. 2019) on the PCA
  coordinates of an earlier `run`, restricted to the project `.fam` first.
  `--n-pcs` limits how many columns of the CSV are used (default: all). Needs
  the `geosketch` extra: `pip install 'manifold-genetics[geosketch]'`.

    ```bash
    manifold-genetics subsample proj/config.yaml --geosketch 50000 \
        --pca 10k/outputs/pca/project_pca_20.csv --out sketch/
    ```

## Worked examples

### UK Biobank projected onto HGDP+1KGP

UK Biobank is `acquire custom` with your own paths — there is nothing for the
package to fetch and no label schema to derive; the label CSV needs a
`sample_id` column plus one column per grouping (see
[Formats](formats.md#inputs)). Then intersect the two, skipping
WRayner (no internet on the compute node) and MAF filtering on the biobank side:

```bash
manifold-genetics acquire custom --fit-plink /path/to/hgdp_tgp --labels hgdp_metadata.csv --out ref/
manifold-genetics acquire custom --fit-plink /path/to/ukb_array --labels ukbb_labels.csv --out ukbb/
manifold-genetics preprocess ref/config.yaml ukbb/config.yaml --skip-wrayner --skip-project-maf --out proj/
manifold-genetics run proj/config.yaml
```

This reproduces the published cohort exactly from the raw inputs: 3,340 × 120,849
on the fit side and 486,748 × 120,849 on the project side. Peak memory was
85 GB; use a node with at least 128 GB.

### All of Us, in the Researcher Workbench

Four commands, each resumable, in a terminal or a notebook cell prefixed with
`!`. The workbench has internet, so `harmonise` can fetch GIAB and TOPMed as it
goes — but not WRayner, whose upstream URL is down (see
[What needs internet](#what-needs-internet)): place `HRC-1000G-check-bim.pl`
by hand or pass `--tools-dir` before running `preprocess`, or the run dies at
that step. `acquire hgdp` writes the workbench's HGDP+1KGP panel with `chr`-prefixed
chromosome names, as the published flow did, hence `--fit-has-chr-prefix`.

```bash
manifold-genetics acquire hgdp --archive gs://fc-secure-47ccf5a8-b9ba-460a-aa03-dea8d260953b/Data/1KGPHGDP.tar.gz --out ref/
manifold-genetics acquire aou --out aou/
manifold-genetics preprocess ref/config.yaml aou/config.yaml --preset harmonise --fit-has-chr-prefix --out proj/
manifold-genetics run proj/config.yaml
```

`acquire aou` needs the `aou` extra and checks the workbench environment first
(`GOOGLE_PROJECT`, `WORKSPACE_CDR`, `gsutil`, `bq`), naming everything missing
at once. Two honest caveats: it was written against the shell script that
produced the published figures but has not yet been run inside the workbench
(issue #124); and because that archive carries no relatedness metadata,
`acquire hgdp --archive` fits on all of its samples, where the public panel
fits on the 3,400 unrelated.

### A biobank subsample

Subsetting comes after preprocessing, so every filter sees the whole cohort:

```bash
manifold-genetics preprocess ref/config.yaml ukbb/config.yaml --skip-wrayner --skip-project-maf --out proj/
manifold-genetics subsample proj/config.yaml --out 10k/ \
    --group "self_described_ancestry=^British$:10000" \
    --group "self_described_ancestry=^Irish$:5000"
manifold-genetics run 10k/config.yaml
```
