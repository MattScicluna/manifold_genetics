# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`manifold-genetics` is a batteries-included Python package for genetic analysis combining PCA, neural admixture, and manifold learning (PHATE, UMAP, t-SNE, Diffusion Maps) with publication-ready visualization.

**Key Principle**: Tools are auto-downloaded during setup because Narval compute nodes lack internet access.

## Environment Context

This project runs on **Narval** (Compute Canada/Alliance cluster with Slurm):
- **Login nodes**: Have internet access, used for setup/installation
- **Compute nodes**: No internet access, used for all computations
- Must pre-download all external tools (`plink2`, `flashpca`) during setup on login node
- Always use `virtualenv` (never conda)

## Setup and Installation

### Initial Setup (Login Node Only)
```bash
cd /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics
bash setup.sh
```

This script:
1. Creates virtual environment in `.venv/`
2. Installs Python package with `pip install -e .`
3. Downloads `plink2` and `flashpca` to `bin/` (~22MB total)

### Development Installation
```bash
source .venv/bin/activate
pip install -e .
```

## Running Jobs on Narval

### Request Compute Node
```bash
salloc --account=ctb-hussinju --cpus-per-task=4 --mem=16GB --time=1:00:00
```

### Batch Job Template
```bash
#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=manifold_job
#SBATCH --output=/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/logs/job_%j.out
#SBATCH --error=/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/logs/job_%j.err

source /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/.venv/bin/activate
python your_script.py
```

## Architecture

### Module Structure
```
src/manifold_genetics/
├── pca/flashpca.py          # FlashPCA wrapper with fit/project API
├── admixture/neural.py      # Neural admixture wrapper (K=k_min to k_max)
├── embeddings/              # Manifold learning methods
│   ├── base.py             # Abstract base class with shared I/O
│   ├── phate.py            # PHATE implementation
│   ├── umap.py             # UMAP implementation
│   ├── tsne.py             # t-SNE implementation
│   └── diffusion_map.py    # Diffusion Maps implementation
├── visualization/plotting.py # Publication-ready plots with custom colormaps
├── metrics/                 # Evaluation metrics
│   ├── geographic.py       # Geographic distance preservation
│   └── admixture.py        # Admixture proportion preservation
├── pipeline/orchestrator.py # End-to-end workflow coordination
├── utils/                   # Utilities
│   ├── tools.py            # Tool path resolution
│   └── io.py               # File I/O (PLINK, CSV, labels, colormaps)
└── cli.py                   # Command-line interface
```

### Design Patterns

1. **Unified fit/project API**: All methods (PCA, embeddings) follow scikit-learn pattern
   ```python
   model = PCA(n_components=50)
   coords = model.fit_transform(input_path, output_path="output.csv")
   ```

2. **Input flexibility**: All methods accept:
   - PLINK files (`.bed/.bim/.fam` prefix)
   - CSV files (path as string or Path)
   - pandas DataFrames
   - numpy arrays

3. **Standardized output format** ("manylatents"):
   ```csv
   sample_id,dim_1,dim_2,...,dim_N
   HGDP00001,0.123,-0.456,...
   ```

4. **EmbeddingBase abstract class**: All embedding methods inherit from `embeddings/base.py`:
   - Defines abstract methods: `fit()`, `transform()`, `fit_transform()`
   - Provides shared utilities: `_load_input_data()`, `_format_output()`
   - Ensures consistent I/O across all manifold learning methods

5. **Pipeline orchestrator**: Coordinates all steps with modular skip flags
   - PCA → Admixture → Embedding → Visualization → Metrics
   - Can skip any step with `--skip-{pca,admixture,embedding,visualization,metrics}`

## Common Commands

### CLI Usage
```bash
# Full pipeline
manifold-genetics pipeline \
    --fit-plink data/fit_subset \
    --project-plink data/project_subset \
    --labels labels.csv \
    --colormap colormap.json \
    --output results/ \
    --n-pcs 50 \
    --k-min 2 --k-max 10 \
    --embedding phate --knn 25

# Individual steps
manifold-genetics pca --input data.plink --output pca_50.csv --n-pcs 50
manifold-genetics admixture --input data.plink --output admixture/ --k-min 2 --k-max 10
manifold-genetics embed --input pca_50.csv --output phate_2d.csv --method phate --knn 25
manifold-genetics plot --input phate_2d.csv --labels labels.csv --colormap colormap.json
```

### Python API
```python
from manifold_genetics import PCA, NeuralAdmixture, PHATE, visualize

# PCA
pca = PCA(n_components=50)
pca_coords = pca.fit_transform("data/hgdp.plink", output_path="pca_50.csv")
pca.fit("data/hgdp_fit.plink")
projected = pca.project("data/hgdp_transform.plink", output_path="pca_transform_50.csv")

# Admixture
admix = NeuralAdmixture(k_min=2, k_max=10)
q_files = admix.fit_transform("data/hgdp.plink", output_dir="admixture/")

# Embedding
phate = PHATE(n_components=2, knn=25)
embedding = phate.fit_transform("pca_50.csv", output_path="phate_2d.csv")

# Visualization
visualize("phate_2d.csv", "labels.csv", "colormap.json", output_dir="figures/")
```

### Example Pipeline
```bash
# Run HGDP+1000 Genomes example
sbatch examples/hgdp_1kgp/run_pipeline_batch.sh

# Interactive on compute node
salloc --account=ctb-hussinju --cpus-per-task=4 --mem=16GB --time=1:00:00
cd /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics
source .venv/bin/activate
bash examples/hgdp_1kgp/run_pipeline.sh
```

## Development Guidelines

### Adding a New Embedding Method

1. Create new file in `src/manifold_genetics/embeddings/`
2. Inherit from `EmbeddingBase`
3. Implement required methods: `fit()`, `transform()`, `fit_transform()`
4. Use `_load_input_data()` and `_format_output()` for I/O
5. Add to `embeddings/__init__.py` exports
6. Add CLI support in `cli.py` (both `embed` and `pipeline` commands)

Example skeleton:
```python
from .base import EmbeddingBase

class NewMethod(EmbeddingBase):
    def __init__(self, n_components=2, param1=default, **kwargs):
        super().__init__(n_components, **kwargs)
        self.param1 = param1

    def fit(self, X):
        X_array, sample_ids = self._load_input_data(X)
        # Fit logic here
        self._is_fitted = True
        return self

    def fit_transform(self, X, output_path=None):
        X_array, sample_ids = self._load_input_data(X)
        # Transform logic here
        return self._format_output(embedding, sample_ids, output_path)
```

### Testing

**Test Suite Location**: `tests/` directory with pytest framework

**Run all tests**:
```bash
source .venv/bin/activate
pytest
```

**Run specific test files**:
```bash
pytest tests/test_embeddings.py -v
pytest tests/test_io.py -v
pytest tests/test_metrics.py -v
pytest tests/test_visualization.py -v
```

**Run with coverage**:
```bash
pytest --cov=manifold_genetics tests/
```

**Test development**: Use `conftest.py` fixtures for dummy data generation. Tests use small synthetic datasets (50 samples) for speed and avoid expensive external tools in unit tests.

### Code Style

- Line length: 100 characters (Black)
- Type hints encouraged for public APIs
- Docstrings: NumPy style for classes/functions
- Logging: Use `logging` module (not print statements)
- Error handling: Descriptive error messages with context

**Format code**:
```bash
black src/ tests/
isort src/ tests/
```

**Lint code**:
```bash
flake8 src/ tests/
```

**Install dev dependencies**:
```bash
pip install -e .[dev]
```

## File Formats

### Input Formats

**PLINK files**: Binary format (`.bed`, `.bim`, `.fam`)
```bash
--fit-plink data/fit_subset        # reference (training) set
--project-plink data/project_subset  # projection/application set
```

**Labels CSV**: Must have `sample_id` column
```csv
sample_id,Population,Genetic_region
HGDP00001,Yoruba,Africa
HGDP00002,Han,EastAsia
```

**Colormap JSON**: Maps label values to hex colors
```json
{
  "Population": {
    "Yoruba": "#FF0000",
    "Han": "#00FF00"
  },
  "Genetic_region": {
    "Africa": "#FF6B6B",
    "EastAsia": "#4ECDC4"
  }
}
```

### Output Format

All embeddings use standardized "manylatents" CSV:
```csv
sample_id,dim_1,dim_2,...,dim_N
HGDP00001,0.123,-0.456,...
```

## Troubleshooting

**Import errors after installation**:
```bash
source .venv/bin/activate
pip install -e . --force-reinstall --no-deps
```

**Missing tools (plink2/flashpca)**:
- Must run `setup.sh` on login node (has internet)
- Tools auto-downloaded to `bin/` directory
- Set paths: `export PLINK_PATH=$(pwd)/bin/plink2`

**Memory errors**:
- Request more memory: `--mem=32GB` or `--mem=64GB`
- Reduce PCA components: `--n-pcs 20`
- Subsample data before analysis

**Job debugging**:
- Check SLURM logs in output directory: `job_%j.out`, `job_%j.err`
- Monitor running jobs: `squeue -u $USER`
- Cancel jobs: `scancel JOBID`

**Tool path issues**:
- Verify tools exist: `ls -la bin/`
- Check tool permissions: `chmod +x bin/plink2 bin/flashpca`
- Manual path export: `export PLINK_PATH=$(pwd)/bin/plink2`

## Documentation

- `README.md` - User documentation and quick start
- `docs/INSTALL.md` - Detailed installation instructions
- `docs/IMPLEMENTATION_SUMMARY.md` - Technical architecture overview
- `docs/details_for_paper.md` - Data processing pipeline for reproducibility
- `examples/` - Working examples with real data

## Dependencies

**Core Python packages** (auto-installed):
- Scientific: numpy, pandas, scipy, scikit-learn
- Visualization: matplotlib, seaborn
- Embeddings: phate, umap-learn
- ML: torch, neural-admixture

**External tools** (auto-downloaded to `bin/`):
- `plink2` - Genetic data processing (~20MB)
- `flashpca` - Fast PCA computation (~2MB)

## Key Implementation Details

1. **Tool resolution** (`utils/tools.py`):
   - Checks environment variables (`PLINK_PATH`, `FLASHPCA_PATH`)
   - Falls back to `bin/` directory
   - Provides helpful error messages if tools missing

2. **Checkpointing**:
   - PCA and admixture check for existing output files
   - Use `force=True` to recompute

3. **Pipeline orchestrator** (`pipeline/orchestrator.py`):
   - Creates output directory structure automatically
   - Passes data between steps (e.g., PCA coords to embedding)
   - Aggregates all results and metrics

4. **Metrics** (`metrics/`):
   - Geographic preservation: Spearman correlation between geographic and embedding distances
   - Admixture preservation: Correlation of admixture proportions with embedding distances

## Key Data Processing Patterns

### Typical Workflow
1. **Data Input**: PLINK files (`.bed/.bim/.fam`) or CSV matrices
2. **PCA**: Dimensionality reduction (typically 50 components)
3. **Admixture**: Ancestry inference for K=2 to K=10
4. **Embedding**: 2D manifold learning (PHATE/UMAP/t-SNE/DM)
5. **Visualization**: Scatter plots colored by population/geography
6. **Metrics**: Quantify preservation of structure

### Data Flow Between Steps
- **PCA output** → used as input for embeddings (reduces noise)
- **Admixture Q files** → used for admixture preservation metrics
- **Labels CSV** → used for coloring visualizations and geographic metrics
- **Colormap JSON** → defines colors for each population/region

### Output Directory Structure
```
results/
├── pca/pca_50.csv                    # PCA coordinates (samples × 50 dims)
├── admixture/fit_K{2..10}.Q          # Ancestry proportions per K
├── embeddings/phate_2d.csv           # 2D embedding coordinates
├── figures/phate_2d_{label}.png      # Plots per label category
└── metrics/{geo,admix}_preservation.json  # Quantitative metrics
```
