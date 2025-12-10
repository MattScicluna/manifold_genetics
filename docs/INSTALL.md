# Installation and Quick Start Guide

## Installation

### On Narval (Compute Canada)

```bash
# Navigate to the package directory
cd /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install the package in editable mode
uv pip install -e .
```

### On other systems

```bash
# Clone the repository (or copy the directory)
cd /path/to/manifold_genetics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate  # On Windows

# Install the package
pip install -e .
```

## Environment Setup (Narval)

Before running the package, load required modules:

```bash
# Load plink2 (required for PCA on PLINK files)
module load StdEnv/2023 plink/2.00a6.7

# Activate virtual environment
source /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/.venv/bin/activate
```

**Note**: FlashPCA will be automatically downloaded on first use. Neural-admixture is installed via pip.

## Quick Start

### 1. Test Installation

```bash
# Run the test script (on a compute node!)
python examples/test_pipeline.py --output ./test_results
```

This will:
- Run PCA on test data
- Train neural admixture models
- Compute PHATE embeddings
- Generate visualizations

### 2. Python API Usage

```python
from manifold_genetics import PCA, NeuralAdmixture, PHATE, visualize

# Step 1: Compute PCA
pca = PCA(n_components=50)
pca_coords = pca.fit_transform(
    plink_prefix="data/hgdp",
    output_path="results/pca_50.csv"
)

# Step 2: Neural Admixture
admix = NeuralAdmixture(k_min=2, k_max=10)
q_files = admix.fit_transform(
    plink_prefix="data/hgdp",
    output_dir="results/admixture"
)

# Step 3: PHATE Embedding
phate = PHATE(n_components=2, knn=25)
embedding = phate.fit_transform(
    "results/pca_50.csv",
    output_path="results/phate_2d.csv"
)

# Step 4: Visualize
figure_paths = visualize(
    embedding="results/phate_2d.csv",
    labels="data/labels.csv",
    colormap="data/colormap.json",
    output_dir="results/figures"
)
```

### 3. Command-Line Interface

#### Individual Commands

```bash
# Run PCA
manifold-genetics pca \
    --input data/hgdp \
    --output results/pca_50.csv \
    --n-pcs 50

# Run admixture
manifold-genetics admixture \
    --input data/hgdp \
    --output results/admixture \
    --k-min 2 --k-max 10

# Run PHATE embedding
manifold-genetics embed \
    --input results/pca_50.csv \
    --output results/phate_2d.csv \
    --method phate --knn 25

# Generate plots
manifold-genetics plot \
    --input results/phate_2d.csv \
    --labels data/labels.csv \
    --colormap data/colormap.json \
    --output results/figures/phate.png
```

#### Full Pipeline

```bash
manifold-genetics pipeline \
    --plink data/hgdp \
    --labels data/labels.csv \
    --colormap data/colormap.json \
    --output results/ \
    --n-pcs 50 \
    --k-min 2 --k-max 10 \
    --embedding phate --knn 25
```

## Data Format Requirements

### PLINK Files

Binary PLINK format (`.bed`, `.bim`, `.fam`). Specify the prefix (without extension):

```bash
# If you have: data/hgdp.bed, data/hgdp.bim, data/hgdp.fam
# Use: --plink data/hgdp
```

### Labels CSV

CSV file with `sample_id` column + label columns:

```csv
sample_id,Population,Genetic_region
HGDP00001,Yoruba,Africa
HGDP00002,Yoruba,Africa
HGDP00003,Han,EastAsia
...
```

### Colormap JSON

JSON mapping label columns to color dictionaries:

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

## Running on Compute Nodes (Narval)

**IMPORTANT**: Never run computationally intensive tasks on login nodes!

### Interactive Session

```bash
# Request an interactive session
salloc --account=ctb-hussinju --cpus-per-task=4 --mem=16GB --time=2:00:00

# Once on compute node, load modules and activate environment
module load StdEnv/2023 plink/2.00a6.7
source /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/.venv/bin/activate

# Run your analysis
python examples/test_pipeline.py
```

### Batch Job

Create a script `run_analysis.sh`:

```bash
#!/bin/bash
#SBATCH --account=ctb-hussinju
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --job-name=manifold_genetics
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load modules
module load StdEnv/2023 plink/2.00a6.7

# Activate environment
source /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/.venv/bin/activate

# Run analysis
manifold-genetics pipeline \
    --plink /lustre06/project/6065672/sciclun4/ActiveProjects/manyGenomes/data_new/hgdp/standalone/genotypes/fit_subset \
    --labels /lustre06/project/6065672/sciclun4/ActiveProjects/manyGenomes/data_new/hgdp/hgdp_fit_labels.csv \
    --colormap /lustre06/project/6065672/sciclun4/ActiveProjects/manyGenomes/data_new/hgdp/colormap.json \
    --output results/ \
    --n-pcs 50 \
    --k-min 2 --k-max 10 \
    --embedding phate --knn 25
```

Submit:

```bash
mkdir -p logs
sbatch run_analysis.sh
```

## Troubleshooting

### plink2 not found

```bash
# Make sure plink module is loaded
module load StdEnv/2023 plink/2.00a6.7

# Or set environment variable
export PLINK_PATH=/path/to/plink2
```

### neural-admixture not found

```bash
# Reinstall in virtual environment
pip install neural-admixture
```

### Import errors

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall package
pip install -e .
```

### Memory errors

For large datasets, request more memory:

```bash
# Interactive
salloc --mem=32GB ...

# Batch job
#SBATCH --mem=32GB
```

## Example with Real Data

Using HGDP test data:

```bash
# On compute node
manifold-genetics pipeline \
    --plink /lustre06/project/6065672/sciclun4/ActiveProjects/manyGenomes/data_new/hgdp/standalone/genotypes/fit_subset \
    --labels /lustre06/project/6065672/sciclun4/ActiveProjects/manyGenomes/data_new/hgdp/hgdp_fit_labels.csv \
    --colormap /lustre06/project/6065672/sciclun4/ActiveProjects/manyGenomes/data_new/hgdp/colormap.json \
    --output hgdp_results/ \
    --n-pcs 50 \
    --k-min 2 --k-max 10 \
    --embedding phate --knn 25 \
    --verbose
```

This will create:
- `hgdp_results/pca/pca_50.csv` - PCA coordinates
- `hgdp_results/admixture/*.Q` - Admixture files
- `hgdp_results/embeddings/phate_2d.csv` - PHATE coordinates
- `hgdp_results/figures/*.png` - Visualization plots

## Next Steps

- See `README.md` for detailed API documentation
- Check `examples/test_pipeline.py` for usage examples
- Consult CLAUDE.md for integration with ExperimentStash
