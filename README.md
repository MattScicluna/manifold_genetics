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

### Step 1: Installation

**IMPORTANT**: Run `setup.sh` before using this package. This requires **internet access** (only needed once).

```bash
cd /path/to/manifold_genetics
bash setup.sh
```

This will:
- Create virtual environment in `.venv/`
- Install all Python dependencies
- **Download plink2 and flashPCA** to `bin/` (~22MB total)

After setup, always activate the virtual environment:
```bash
source .venv/bin/activate
```

### Step 2: Verify Installation (Optional but Recommended)

Run the test suite tso confirm everything is working:

```bash
source .venv/bin/activate
pytest -m "not slow and not network"
```

Expected: **45 tests passing** in ~90 seconds. See [Testing](#testing) section for details.

### Step 3: Run HGDP+1KGP Example

The package includes a complete working example using HGDP+1000 Genomes Project data.

#### Download and prepare data (first time only):

```bash
cd /path/to/manifold_genetics
source .venv/bin/activate

# Download data (~500MB, requires internet)
bash examples/hgdp_1kgp/download_data.sh

# Prepare data for analysis
bash examples/hgdp_1kgp/prepare_data.sh
```

#### Run the full pipeline:

```bash
# From repository root, with virtual environment activated
bash examples/hgdp_1kgp/run_pipeline.sh
```

**Runtime:** ~30 minutes on CPU (faster with GPU for admixture)

**Outputs** saved to `examples/hgdp_1kgp/outputs/`:
- `pca/` - PCA coordinates (fit: 3,400 samples, project: 4,094 samples)
- `admixture/` - Ancestry proportions for K=2 to 10
- `embeddings/` - PHATE 2D embedding (4,094 samples)
- `figures/` - All visualization plots
- `metrics/` - Geographic and admixture preservation metrics

#### About the Example Data

- **Fit subset:** 3,400 unrelated samples (for model training)
- **Project subset:** 4,094 QC-passing samples (for model application)
- **172,152 SNPs** (LD-pruned, MAF ≥0.01)
- **7 genetic regions:** Africa, Americas, Central/South Asia, East Asia, Europe, Middle East, Oceania

## Command-Line Interface

### Pipeline (recommended)

Run from the repository root. Outputs land under `examples/hgdp_1kgp/outputs/` in subfolders (`pca/`, `admixture/`, `embeddings/`, `figures/`), and metrics are computed from the pipeline run.

```bash
manifold-genetics pipeline \
    --fit-plink examples/hgdp_1kgp/data/fit_subset \
    --project-plink examples/hgdp_1kgp/data/project_subset \
    --labels examples/hgdp_1kgp/data/hgdp_project_labels.csv \
    --colormap examples/colormaps/hgdp_1kgp.json \
    --output examples/hgdp_1kgp/outputs \
    --n-pcs 50 \
    --k-min 2 --k-max 5 \
    --embedding phate --knn 100 --t 3 \
    --embedding-input project \
    --threads 8
# Optional: --num-gpus 1
```

Skip steps as needed:
```bash
manifold-genetics pipeline ... --skip-pca --skip-admixture --skip-metrics
```

### Equivalent Individual Commands (same outputs as pipeline)

Run from repo root; paths below match the pipeline output layout.

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

# 3) Embedding (PHATE): fit and transform on projected (transform) PCA coordinates
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
    --colormap examples/colormaps/hgdp_1kgp.json \
    --output examples/hgdp_1kgp/outputs/figures/pca \
    --n-pcs 50

# PHATE
manifold-genetics plot \
    --input examples/hgdp_1kgp/outputs/embeddings/phate_2d.csv \
    --labels examples/hgdp_1kgp/data/hgdp_project_labels.csv \
    --colormap examples/colormaps/hgdp_1kgp.json \
    --output examples/hgdp_1kgp/outputs/figures/embeddings/phate.png

# Admixture barplots (stacked bars per K)
manifold-genetics plot-admixture \
    --q-prefix examples/hgdp_1kgp/outputs/admixture/transform \
    --labels examples/hgdp_1kgp/data/hgdp_project_labels.csv \
    --group-column Genetic_region_merged \
    --colormap examples/colormaps/hgdp_1kgp.json \
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

## Running on Your Own Data

### Quick Start (3 Steps)

1. **Prepare your files:**
   - PLINK files (`.bed/.bim/.fam`) - binary genotype data
   - Labels CSV - sample metadata with `sample_id` column
   - Colormap JSON - hex colors for each label value

2. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Run the pipeline:**
   ```bash
   manifold-genetics pipeline \
       --fit-plink data/your_data \
       --project-plink data/your_data \
       --labels data/labels.csv \
       --colormap data/colormap.json \
       --output results/
   ```

**That's it!** Results (PCA, admixture, embeddings, figures, metrics) saved to `results/`.

### File Format Requirements

**Labels CSV** - Must have `sample_id` column matching PLINK `.fam` file:
```csv
sample_id,Population,Region
SAMPLE001,Han,EastAsia
SAMPLE002,Yoruba,Africa
SAMPLE003,French,Europe
```

**Colormap JSON** - Maps label values to hex colors:
```json
{
  "Population": {
    "Han": "#E74C3C",
    "Yoruba": "#3498DB",
    "French": "#2ECC71"
  },
  "Region": {
    "EastAsia": "#FF6B6B",
    "Africa": "#4ECDC4",
    "Europe": "#95E1D3"
  }
}
```

**PLINK files** - Specify prefix only (tool finds `.bed/.bim/.fam`):
```bash
--fit-plink data/my_data  # Looks for my_data.bed, my_data.bim, my_data.fam
```

**Geographic coordinates** (optional, for metrics):
```csv
sample_id,latitude,longitude
SAMPLE001,39.9042,116.4074
SAMPLE002,6.5244,3.3792
```

### Common Options

**Adjust parameters:**
```bash
manifold-genetics pipeline \
    --fit-plink data/your_data \
    --project-plink data/your_data \
    --labels data/labels.csv \
    --colormap data/colormap.json \
    --output results/ \
    --n-pcs 50 \                    # Number of PCA components (default: 50)
    --k-min 2 --k-max 10 \          # Admixture K range (default: 2-10)
    --embedding phate \              # Method: phate, umap, tsne, diffusion_map
    --knn 100 --t 3 \               # Embedding parameters
    --threads 8 \                   # CPU threads
    --num-gpus 1                    # Use GPU for admixture
```

**For large datasets (>10K samples):**
```bash
# Use landmarking for computational efficiency
manifold-genetics pipeline ... \
    --n-landmark 10000 \
    --random-landmarking \
    --neuraladmixture-batch-size 400
```

**Skip steps:**
```bash
manifold-genetics pipeline ... \
    --skip-admixture \      # Skip ancestry analysis
    --skip-metrics          # Skip preservation metrics
```

### Data Preparation Tips

**Before running the pipeline:**
- LD-prune your SNPs: `plink2 --indep-pairwise 50 5 0.2`
- Filter by MAF: `--maf 0.01`
- Remove related individuals from fit subset
- Apply standard QC filters

**For large cohorts:**
- Use ~3,000 unrelated samples for fit subset
- Project remaining samples onto fit models
- Expected runtimes: PCA (minutes), Admixture (hours), Embeddings (minutes-hours)

## Embedding Methods

- **PHATE**: `--embedding phate --knn 100` (recommended for population structure)
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

## Testing

Verify your installation with the test suite:

```bash
source .venv/bin/activate
pytest -m "not slow and not network"
```

**Expected:** ~45 tests passing in ~90 seconds (excludes slow admixture tests and network-dependent tests).

Run all tests:
```bash
pytest -v
```

## Additional Examples

Beyond the HGDP+1KGP example, this repository includes:

- **`examples/generic/`** - Template scripts for running on your own data (copy and customize)
- **`examples/ukbb/`** - UK Biobank pipeline scripts (requires UKBB access)
- **`examples/aou/`** - All of Us pipeline scripts (requires AoU access)

These examples demonstrate the DRY architecture where biobank-specific wrappers call shared generic templates.

## License

BSD 3-Clause License (see LICENSE file)

## Citation

If you use this package in your research, please cite:

```
[Your citation here]
```
