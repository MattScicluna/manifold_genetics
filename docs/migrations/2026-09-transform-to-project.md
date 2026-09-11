# Output filename migration: `transform_*` → `project_*`

Branch `refactor/consistent-project-naming` renames the second-cohort dataset role
from **transform** to **project** everywhere in the code, CLI, and pipeline output
filenames. Existing run outputs on disk keep the old names, so the pipeline's
checkpoint/idempotency logic will not recognise them and will recompute those
steps. Rename them in place to skip the recompute.

Applies to output directories produced before the `transform` -> `project` rename.
Outputs created after it already use the new names.

## Filename map

| Old | New |
|-----|-----|
| `<out>/pca/transform_pca_<N>.csv` | `<out>/pca/project_pca_<N>.csv` |
| `<out>/pca/flashpca_outputs/transform_<name>.*` | `<out>/pca/flashpca_outputs/project_<name>.*` |
| `<out>/admixture/transform.<K>.csv` | `<out>/admixture/project.<K>.csv` |
| `<out>/figures/admixture/transform_bars.png` | `<out>/figures/admixture/project_bars.png` |
| `<out>/figures/admixture/transform_admixture_colored_embedding.png` | `<out>/figures/admixture/project_admixture_colored_embedding.png` |
| `<out>/figures/embeddings/transform_<method>_*.png` | `<out>/figures/embeddings/project_<method>_*.png` |

Unchanged: `fit_pca_<N>.csv`, `fit.<K>.csv`, `<method>_2d.csv`, `<method>_fit_2d.csv`,
`<method>_project_2d.csv`, everything under `metrics/`.

## One-liner per output dir

Run from inside each pipeline output directory (e.g. `examples/hgdp_1kgp/outputs/`):

```bash
find . -depth -name 'transform*' | while read -r p; do
  mv "$p" "$(dirname "$p")/$(basename "$p" | sed 's/^transform/project/')"
done
```

Dry run first:

```bash
find . -depth -name 'transform*' -print
```

## Example dirs likely affected on this machine

- `examples/hgdp_1kgp/outputs/`
- `examples/ukbb/10k_WB_5K_Irish/outputs/`  (+ `outputs_old/`)
- `examples/ukbb/hgdp_1kgp_proj/outputs/`   (+ `outputs_old/`)
- `examples/aou/10k_WBH/outputs/`
- `examples/aou/hgdp_1kgp_proj/outputs/`

## Downstream scripts that read these paths (already updated in the branch)

- `examples/_shared/select_samples_geosketch.py`
- `examples/{aou,ukbb}/geosketch_phate/prepare_data.sh`
- `examples/{aou,ukbb}/60k_random/prepare_data.py`
- `examples/aou/shared/copy_to_manifoldGenetics.sh`

## Out of scope for this PR (separate decisions)

- the `transform` preset (now `manifold_genetics.pipeline.configfile.PRESETS`) — this is the
  "fit_transform-only, no cross-projection" mode name, not the dataset role. Left as-is.
- `transform_subset.bed` input PLINK filename in `examples/hgdp_1kgp/download_data.sh`
  — externally hosted data; the integration test already expects `project_subset`,
  so there is a separate inconsistency to untangle in the data-prep scripts.
