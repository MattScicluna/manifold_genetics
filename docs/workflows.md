# Workflows (WDL)

The same pipeline as WDL workflows, for Cromwell (as on Verily Workbench) or
miniwdl. Each runs `manifold-genetics` in the [published
container](nextflow.md#container).

| file | what it does |
|---|---|
| `manifold_genetics.wdl` | `manifold-genetics pipeline` as one task, for a cohort you already have |
| `aou_prepare.wdl` | All of Us, step 1: fetch a release and intersect it with HGDP+1KGP |
| `prepare.wdl` | any other biobank, step 1: the same, from PLINK files you already have |
| `analyse.wdl` | step 2, any prepared cohort: subsample, PCA, then 2-D PHATE, UMAP and PCA figures |
| `analyse_3d.wdl` | after `analyse`: 3-D PHATE, UMAP and PCA (HTML and MP4) from its PCA |
| `admixture.wdl` | after `analyse`, optional: Neural Admixture per experiment, and its figures on the published embeddings |

Example inputs are in `wdl/`.

## All of Us (Verily Workbench)

You need a controlled-tier workspace with the release attached, and a bucket
of your own for results (Resources → New → Cloud Storage bucket).

1. **Add the workflows.** Workflows → Add workflow → WDL →
   `https://github.com/MattScicluna/manifold_genetics`, then pick
   `aou_prepare.wdl`; add again for `analyse.wdl`, `analyse_3d.wdl` and
   `admixture.wdl`.
2. **Run `aou_prepare` once per release.** It writes the prepared cohort to
   `<output_dir>/prepared/`.

    | input | value |
    |---|---|
    | `output_dir` | e.g. `gs://<bucket>/aou_v9` |
    | `workspace_cdr` | `$WORKSPACE_CDR` in any app |
    | `cdr_storage_path` | `$CDR_STORAGE_PATH` (default: v9) |
    | `reference_bed/bim/fam` | the HGDP+1KGP reference, population in the FID |
    | `topmed_reference` | optional `bravo-dbsnp-all.hrc_format.tab.gz`; turns on the WRayner check against TOPMed, as the published preprocessing ran it |

3. **Run `analyse`** with `cohort_dir` = `<output_dir>/prepared` and the same
   `output_dir`. Each experiment writes `<output_dir>/<name>/`: PCA, embeddings,
   2-D figures, metrics and logs.
4. **Run `analyse_3d`** with the same `output_dir` for the 3-D figures. It
   reads each experiment's PCA, so `experiments` must name those folders.
5. **Run `admixture`** (optional) with the same `cohort_dir` and `output_dir`.
   It writes `outputs/admixture/` (Q matrices for K = 2 to 10) and colours
   every embedding already in `outputs/embeddings/` by admixture.

The default experiments are the manuscript's: `balanced` (10,000 each of the
four largest groups plus everyone else) and `geosketch_90k`. Each is a name and
the arguments to [`subsample`](cli.md#subsample); a list of your own replaces
them. Keep the geosketch size close to the balanced set's, so the two fit sets
are comparable.

The embeddings are their own task, so with call caching a change to their
settings reruns only them and the figures. PHATE and UMAP are fit on the fit
set's 20 PCs, and the PCA figures show its first two (`analyse`) or three
(`analyse_3d`).

Workflow machines pull the image through the All of Us Docker Hub mirror
(`us-central1-docker.pkg.dev/all-of-us-rw-prod/aou-rw-gar-remote-repo-docker-prod/mattscicluna/manifold-genetics:<tag>`).
Set `docker` to a `sha-<commit>` or version tag: the mirror can keep serving
a stale image under a branch tag.

Follow the All of Us dissemination rules before sharing figures or counts.

!!! note "The reference panel"
    `aou_prepare` needs an HGDP+1KGP reference restricted to the array's
    positions. The public `acquire hgdp` panel is LD-pruned before
    intersection and keeps too few array SNPs, and the one used so far is an
    internal extract, so for now the step cannot be reproduced from outside
    ([#172](https://github.com/MattScicluna/manifold_genetics/issues/172)).

## Any biobank (UK Biobank, your own)

`prepare.wdl` replaces step 1 for a cohort you already have as GRCh38 PLINK
files, with chromosomes as `chr1` or `1`. It runs the same `preprocess` step as
`aou_prepare`, writes the same `<output_dir>/prepared/`, and `analyse.wdl` then
runs on it unchanged.

| input | value |
|---|---|
| `output_dir` | a bucket or a local path |
| `cohort_bed/bim/fam` | the biobank's genotypes |
| `cohort_labels` | CSV: `sample_id` plus the columns to colour by |
| `reference_bed/bim/fam` | HGDP+1KGP, restricted to the cohort's positions (below) |
| `reference_labels` | optional CSV: `sample_id`, `Population`, ...; default: population from the FID |
| `reference_has_chr_prefix` | `true` if the reference's chromosomes are `chr1` |
| `topmed_reference` | optional `bravo-dbsnp-all.hrc_format.tab.gz`: turns on the WRayner check against TOPMed, as in `aou_prepare`; without it WRayner is skipped, as in the published UK Biobank preprocessing |
| `cohort_colormap`, `reference_colormap` | optional colormap JSONs that take precedence |

Labels with published colours (HGDP+1KGP populations and regions, UK Biobank
`self_described_ancestry` and superpopulations, All of Us race and ethnicity)
are drawn in them; any other value gets a generated colour.

### The reference

The reference is the one input that differs between biobanks: the same
HGDP+1KGP samples, restricted to the positions on your array **before** any LD
pruning, so that pruning happens after the intersection. A panel pruned on its
own first keeps only a sample of positions, and few of them are on any given
array: CARTaGENE's GSA array shares 18,796 positions with the pruned panel,
below the 50,000 `preprocess` requires. Restrict an unpruned panel by position,
since variant IDs often differ (`chr1:...` against `1:...`):

```bash
awk '{ sub(/^chr/, "", $1); print $1, $4, $4, $2 }' cohort.bim > positions.txt
plink --bfile hgdp1kgp_unpruned --extract range positions.txt --make-bed --out reference
```

A public build of that panel is
[#172](https://github.com/MattScicluna/manifold_genetics/issues/172).

### UK Biobank, end to end

The published UK Biobank figures, rerun with these two workflows (miniwdl +
Apptainer on a cluster, 486,748 samples):

- **`prepare`**: the UK Biobank array with HGDP+1KGP at its positions, 5
  minutes, giving 3,340 reference samples and 120,849 SNPs, as in the
  manuscript.
- **`analyse`**: the experiments in `wdl/analyse.biobank.inputs.json`:
  `balanced` (10,000 British, 5,000 Irish and everyone else, 59,264 samples) and
  `geosketch_60k` to match. 3 h 20 on 16 CPUs and 128 GB, with
  `pca_memory_gb` 100 and `embed_memory_gb` 32; peak memory 118 GB while both
  experiments ran their PCA at once. `geosketch` takes longer because it runs
  PCA twice: once on a random 100,000 to choose the sketch, then on the sketch.

```json
"analyse.experiments": [
  {"name": "balanced", "subsample_args": "--group 'self_described_ancestry=^British$:10000' --group 'self_described_ancestry=^Irish$:5000' --include-rest --seed 42"},
  {"name": "geosketch_60k", "subsample_args": "--geosketch 60000 --seed 42"}
]
```

Each experiment's folder has PCA, the 2-D PHATE and UMAP embeddings and their
figures; `analyse_3d` with `wdl/analyse_3d.biobank.inputs.json` adds the 3-D
ones and the rotating videos (`outputs/figures/embeddings_3d/*.mp4`).

## Outside Verily

Any WDL runner works. `output_dir` may be a local path, and the tasks then copy
results there instead of to a bucket:

```bash
miniwdl run prepare.wdl -i prepare.json
miniwdl run analyse.wdl -i analyse.json
miniwdl run analyse_3d.wdl -i analyse_3d.json
```

On a cluster without Docker, `wdl/miniwdl.apptainer.cfg` runs the same image
with Apptainer. Set its paths, and pull the image once on a node with internet,
as its comments show. Two of its settings matter:

- `allow_any_input = true`: `analyse.wdl` reads files under `cohort_dir`
  (and `analyse_3d.wdl` under `output_dir`),
  which miniwdl otherwise refuses.
- `cpu_max` and `memory_max`: set them to the job's allocation. miniwdl
  otherwise schedules tasks against the whole node.

Then run both steps in one batch job, for example with SLURM:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
export MINIWDL_CFG=wdl/miniwdl.apptainer.cfg
miniwdl run prepare.wdl -i prepare.json
miniwdl run analyse.wdl -i analyse.json
miniwdl run analyse_3d.wdl -i analyse_3d.json
```
