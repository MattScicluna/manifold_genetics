# manifold-genetics

A lightweight, batteries-included Python package for genetic analysis with dimensionality reduction and visualization.

## Features

- **PCA**: FlashPCA wrapper for fast principal component analysis
- **Admixture**: Neural admixture analysis
- **Embeddings**: PHATE, UMAP, t-SNE, and Diffusion Maps for manifold learning
- **Visualization**: Publication-ready plots with customizable colormaps
- **Metrics**: Geographic and admixture preservation metrics
- **Pipeline**: End-to-end orchestration from PLINK files to visualizations
- **Auto-downloads tools**: Automatically downloads plink2 and flashPCA on first use

## Quick Start

### Step 1: Installation (Login Node)

**IMPORTANT**: Run setup.sh before running any of this code.

```bash
cd /path/to/manifold_genetics
bash setup.sh
```

This will:
- Create virtual environment
- Install all Python dependencies
- **Download plink2 and flashPCA** (~22MB total)

Note that this step requires internet access.

### Step 2: Run HGDP+1KGP Example

The package includes a complete example using HGDP+1000 Genomes Project data.

```bash
# download data
bash examples/hgdp_1kgp/download_data.sh
bash examples/hgdp_1kgp/prepare_data.sh

# Run pipeline
cd /path/to/manifold_genetics/
source .venv/bin/activate
bash examples/hgdp_1kgp/run_pipeline.sh
```

Results saved to `examples/hgdp_1kgp/outputs`:
- `pca/fit_pca_50.csv` - PCA coordinates for fit subset (3,452 samples)
- `pca/transform_pca_50.csv` - PCA coordinates for project subset (4,094 samples)
- `admixture/*.Q` - Admixture files for K=2 to 10
- `embeddings/phate_2d.csv` - PHATE 2D embedding
- `figures/*.png` - Visualization plots

### About the Example Data

The HGDP+1KGP example includes:
- **Fit subset:** 3,452 unrelated samples (for model training)
- **Project subset:** 4,094 QC-passing samples (for model application)
- **172,152 SNPs** (LD-pruned, MAF ≥0.01)
- **7 genetic regions:** Africa, Americas, Central/South Asia, East Asia, Europe, Middle East, Oceania

For detailed data processing information, see `docs/details_for_paper.md`.

## Command-Line Interface

### Pipeline (recommended)

Run from the repository root. Outputs land under `examples/hgdp_1kgp/outputs/` in subfolders (`pca/`, `admixture/`, `embeddings/`, `figures/`), and metrics are computed from the pipeline run.

```bash
manifold-genetics pipeline \
    --fit-plink examples/hgdp_1kgp/data/fit_subset \
    --project-plink examples/hgdp_1kgp/data/project_subset \
    --labels examples/hgdp_1kgp/data/hgdp_project_labels.csv \
    --colormap examples/hgdp_1kgp/data/colormap.json \
    --output examples/hgdp_1kgp/outputs \
    --n-pcs 50 \
    --k-min 2 --k-max 5 \
    --embedding phate --knn 100 --t 3 \
    --threads 8
# Optional: --num-gpus 1
```

Skip steps as needed:
```bash
manifold-genetics pipeline ... --skip-pca --skip-admixture --skip-metrics
```

### Equivalent Individual Commands (same outputs as pipeline)

Run from repo root; paths below match the pipeline output layout. Metrics are produced by the pipeline; individual commands do not compute metrics (use the pipeline to get `outputs/metrics/*.json`).

```bash
# 1) PCA: fit on fit_subset, project on project_subset
manifold-genetics pca \
    --fit-plink examples/hgdp_1kgp/data/fit_subset \
    --project-plink examples/hgdp_1kgp/data/project_subset \
    --fit-output examples/hgdp_1kgp/outputs/pca/fit_pca_50.csv \
    --project-output examples/hgdp_1kgp/outputs/pca/transform_pca_50.csv \
    --flashpca-output-dir examples/hgdp_1kgp/outputs/pca/flashpca_outputs \
    --n-pcs 50

# 2) Admixture: fit on fit_subset, project on project_subset
manifold-genetics admixture \
    --fit-plink examples/hgdp_1kgp/data/fit_subset \
    --project-plink examples/hgdp_1kgp/data/project_subset \
    --neuraladmixture-output-dir examples/hgdp_1kgp/outputs/admixture/checkpoints \
    --fit-output examples/hgdp_1kgp/outputs/admixture/fit \
    --project-output examples/hgdp_1kgp/outputs/admixture/transform \
    --k-min 2 --k-max 5 --threads 8
# Outputs (per K): examples/hgdp_1kgp/outputs/admixture/fit.{K}.csv and transform.{K}.csv

# 3) Embedding (PHATE): run on transform PCA coordinates
manifold-genetics embed \
    --method phate \
    --input examples/hgdp_1kgp/outputs/pca/transform_pca_50.csv \
    --project-output examples/hgdp_1kgp/outputs/embeddings/phate_2d.csv \
    --knn 100 --t 3 --n-landmark None

# 4) Visualization

# PCA
manifold-genetics plot-pca \
    --input examples/hgdp_1kgp/outputs/pca/transform_pca_50.csv \
    --labels examples/hgdp_1kgp/data/hgdp_project_labels.csv \
    --colormap examples/hgdp_1kgp/data/colormap.json \
    --output examples/hgdp_1kgp/outputs/figures/pca \
    --n-pcs 50

# PHATE
manifold-genetics plot \
    --input examples/hgdp_1kgp/outputs/embeddings/phate_2d.csv \
    --labels examples/hgdp_1kgp/data/hgdp_project_labels.csv \
    --colormap examples/hgdp_1kgp/data/colormap.json \
    --output examples/hgdp_1kgp/outputs/figures/embeddings/phate.png

# Admixture barplots (stacked bars per K)
manifold-genetics plot-admixture \
    --q-prefix examples/hgdp_1kgp/outputs/admixture/transform \
    --labels examples/hgdp_1kgp/data/hgdp_project_labels.csv \
    --group-column Genetic_region_merged \
    --colormap examples/hgdp_1kgp/data/colormap.json \
    --k-min 2 --k-max 5 \
    --output examples/hgdp_1kgp/outputs/figures/admixture/transform_bars.png

# Admixture embedding grid (seismic colormap per component)
manifold-genetics plot-admixture-embedding \
    --embedding examples/hgdp_1kgp/outputs/embeddings/phate_2d.csv \
    --q-prefix examples/hgdp_1kgp/outputs/admixture/transform \
    --k-min 2 --k-max 5 \
    --output examples/hgdp_1kgp/outputs/figures/admixture/transform_embedding.png


# 5) Metrics (optional, standalone)
manifold-genetics metrics-geographic \
    --embedding examples/hgdp_1kgp/outputs/embeddings/phate_2d.csv \
    --geographic examples/hgdp_1kgp/data/hgdp_project_geographic.csv \
    --output examples/hgdp_1kgp/outputs/metrics/geographic.json \
    --num-dists-sampled 50000

manifold-genetics metrics-admixture \
    --embedding examples/hgdp_1kgp/outputs/embeddings/phate_2d.csv \
    --q-dir examples/hgdp_1kgp/outputs/admixture \
    --output examples/hgdp_1kgp/outputs/metrics/admixture.json \
    --k-min 2 --k-max 5 \
    --num-dists-sampled 50000
```

## Data Formats

### Input

**PLINK files**: Binary format (`.bed`, `.bim`, `.fam`). Specify the prefix:
```bash
--fit-plink examples/hgdp_1kgp/data/fit_subset       # PCA/admixture reference set (training)
--project-plink examples/hgdp_1kgp/data/project_subset  # projection/application set
```

**Labels CSV**: Must have `sample_id` column + label columns:
```csv
sample_id,Population,Genetic_region
HGDP00001,Yoruba,Africa
HGDP00002,Yoruba,Africa
HGDP00003,Han,EastAsia
```

**Colormap JSON**: Maps labels to hex colors:
```json
{
  "Population": {
    "Yoruba": "#FF0000",
    "Han": "#00FF00",
    "French": "#0000FF"
  },
  "Genetic_region": {
    "Africa": "#FF6B6B",
    "EastAsia": "#4ECDC4",
    "Europe": "#45B7D1"
  }
}
```

### Output

All embeddings use "manylatents" format:
```csv
sample_id,dim_1,dim_2,...,dim_N
HGDP00001,0.123,-0.456,...
HGDP00002,0.234,-0.567,...
```

## Embedding Methods

- **PHATE**: `--embedding phate --knn 100`
- **UMAP**: `--embedding umap --n-neighbors 15 --min-dist 0.1`
- **t-SNE**: `--embedding tsne --perplexity 30`
- **Diffusion Maps**: `--embedding diffusion_map --knn 100`

## Requirements

### Python Dependencies (Auto-installed)
- numpy, pandas, scipy, scikit-learn
- matplotlib, seaborn
- phate, umap-learn
- torch, neural-admixture

### External Tools (Auto-downloaded)
- **plink2**: Downloaded to `bin/plink2` (~20MB)
- **flashPCA**: Downloaded to `bin/flashpca` (~2MB)

No manual installation needed!

## Troubleshooting

**Virtual environment not activated**:
```bash
source .venv/bin/activate
```

**Import errors after installation**:
```bash
source .venv/bin/activate
pip install -e . --force-reinstall --no-deps
```

## Documentation

- `docs/INSTALL.md` - Detailed installation instructions
- `docs/IMPLEMENTATION_SUMMARY.md` - Technical overview and architecture
- `docs/details_for_paper.md` - Complete data processing pipeline for reproducibility
- `examples/test_pipeline.py` - Working examples

## License

BSD 3-Clause License (see LICENSE file)

## Citation

If you use this package in your research, please cite:

```
[Your citation here]
```
