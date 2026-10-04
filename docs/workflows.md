# Workflows (WDL)

The same pipeline as WDL workflows, for Cromwell (as on Verily Workbench) or
miniwdl. Each runs `manifold-genetics` in the [published
container](nextflow.md#container).

| file | what it does |
|---|---|
| `manifold_genetics.wdl` | `manifold-genetics pipeline` as one task, for a cohort you already have |
| `aou_prepare.wdl` | All of Us, step 1: fetch a release and intersect it with HGDP+1KGP |
| `analyse.wdl` | step 2, any prepared cohort: subsample, PCA, PHATE, 2-D and 3-D figures |

Example inputs are in `wdl/`.

## All of Us (Verily Workbench)

You need a controlled-tier workspace with the release attached, and a bucket
of your own for results (Resources → New → Cloud Storage bucket).

1. **Add the workflows.** Workflows → Add workflow → WDL →
   `https://github.com/MattScicluna/manifold_genetics`, then pick
   `aou_prepare.wdl`; add again for `analyse.wdl`.
2. **Run `aou_prepare` once per release.** It writes the prepared cohort to
   `<output_dir>/prepared/`.

    | input | value |
    |---|---|
    | `output_dir` | e.g. `gs://<bucket>/aou_v9` |
    | `workspace_cdr` | `$WORKSPACE_CDR` in any app |
    | `cdr_storage_path` | `$CDR_STORAGE_PATH` (default: v9) |
    | `reference_bed/bim/fam` | the HGDP+1KGP reference, population in the FID |

3. **Run `analyse`** with `cohort_dir` = `<output_dir>/prepared` and the same
   `output_dir`. Each experiment writes `<output_dir>/<name>/`: PCA, embeddings,
   2-D figures, the 3-D HTML and MP4, metrics and logs.

The default experiments are the manuscript's: `balanced` (10,000 each of the
four largest groups plus everyone else) and `geosketch_90k`. Each is a name and
the arguments to [`subsample`](cli.md#subsample); a list of your own replaces
them. The geosketch size matches the balanced set's, so UK Biobank, whose
balanced set is 59,264, uses 60,000:

```json
"analyse.experiments": [
  {"name": "geosketch_60k", "subsample_args": "--geosketch 60000 --seed 42"}
]
```

PHATE is its own task, so with call caching a change to `knn`, `t` or
`n_landmark` reruns only PHATE and the figures.

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

## Outside Verily

Any WDL runner works. `output_dir` may be a local path, and the tasks then copy
results there instead of to a bucket:

```bash
miniwdl run analyse.wdl -i inputs.json
```

On a cluster without Docker, miniwdl's Singularity/Apptainer backend runs the
same image (`[scheduler] container_backend = singularity`). Pull it once on a
node with internet, as miniwdl's cache expects it:

```bash
apptainer pull <cache>/docker___mattscicluna_manifold-genetics_<tag>.sif \
    docker://mattscicluna/manifold-genetics:<tag>
```

and bind the filesystems holding inputs and `output_dir` with
`run_options = ["--containall", "--bind", "/lustre06,/lustre07"]` (your paths).
