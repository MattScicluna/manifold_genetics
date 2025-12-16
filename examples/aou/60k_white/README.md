# AoU 60K White/European Subset Analysis

This example demonstrates within-AoU genetic analysis using a 60,000-sample subset of white/European ancestry participants.

## Dataset Design

**Fit Subset** (for PCA/admixture training):
- 60,000 white/European ancestry samples (randomly selected)

**Project Subset** (for projection):
- Remaining white/European ancestry samples from AoU

**Configuration**:
- 20 principal components
- Neural Admixture: K=2 to K=10
- Embedding: PHATE (knn=500, t=50, 10K landmarks)

## Rationale

This design allows:
1. **Within-ancestry variation**: Focus on fine-scale genetic structure within European ancestries
2. **Scalable training**: 60K samples provide robust statistics while remaining computationally manageable
3. **Full cohort analysis**: Project remaining samples to analyze entire white/European subset
4. **Substructure detection**: Identify regional/ethnic variation within European populations

## Setup

### 1. Prepare Data

The `prepare_data.sh` script handles all data preparation:

```bash
cd examples/aou/60k_white
bash prepare_data.sh
```

This script:
1. Prepares AoU genotype data
2. Filters for white/European ancestry samples (based on self-reported ancestry or genetic clustering)
3. Randomly selects 60,000 samples for fit subset
4. Uses remaining samples for project subset
5. Creates PLINK files:
   - `data/fit_subset.{bed,bim,fam}` - 60K training samples
   - `data/project_subset.{bed,bim,fam}` - Remaining samples
6. Generates labels:
   - `data/fit_labels.csv` - Labels for fit samples
   - `data/project_labels.csv` - Labels for project samples
7. Creates colormap:
   - `data/colormap.json` - Color scheme for populations/regions

**Note**: You will need to customize this script with:
- Path to your AoU PLINK data
- Logic for filtering white/European ancestry samples
- Metadata for creating meaningful labels (e.g., self-reported ethnicity, geographic region)

### 2. Sample Selection Criteria

Define "white/European ancestry" based on:
- Self-reported ancestry metadata
- Genetic ancestry inference from previous analyses
- Principal component analysis clustering

Random sampling ensures:
- Unbiased representation in training set
- Fixed random seed (42) for reproducibility

### 3. Verify Required Files

After running `prepare_data.sh`, verify these files exist:

```bash
data/
├── fit_subset.{bed,bim,fam}      # 60K training samples
├── project_subset.{bed,bim,fam}  # Remaining samples
├── fit_labels.csv                # Fit sample labels
├── project_labels.csv            # Project sample labels
└── colormap.json                 # Color scheme
```

## Running the Pipeline

### Interactive (on compute node)

```bash
# Request compute node with GPU
salloc --account=ctb-hussinju --cpus-per-task=8 --mem=128GB --gres=gpu:1 --time=8:00:00

# Activate environment
source .venv/bin/activate

# Run pipeline
bash examples/aou/60k_white/run_pipeline.sh
```

### Batch Job

```bash
sbatch examples/aou/60k_white/run_pipeline_batch.sh
```

## Outputs

Results are saved to `outputs/`:

```
outputs/
├── pca/
│   ├── fit_pca_20.csv             # PCA coordinates (fit)
│   └── transform_pca_20.csv       # Projected PCA coordinates
├── admixture/
│   ├── fit.{2..10}.csv           # Admixture proportions (fit)
│   └── transform.{2..10}.csv     # Admixture proportions (projected)
├── embeddings/
│   └── phate_2d.csv              # 2D PHATE embedding
└── figures/
    ├── pca_2d_*.png              # PCA visualizations
    ├── phate_2d_*.png            # PHATE visualizations
    ├── fit_admixture.png         # Admixture barplots (fit)
    └── transform_admixture.png   # Admixture barplots (projected)
```

## Sample Sizes

Expected sample counts (depends on AoU cohort):
- **White/European available**: Variable (e.g., ~200K)
- **Fit subset**: 60,000 samples
- **Project subset**: Remaining (e.g., ~140K)

## Notes

- Random seed is fixed (42) for reproducible sampling
- If fewer than 60K white/European samples available, all samples are used in fit subset
- PHATE uses 10K landmarks for scalability with large cohorts
- Neural Admixture uses batch size of 400 for GPU efficiency
- Metrics computation is skipped by default
- GPU recommended for Neural Admixture (batch script requests 1 GPU)
- Adjust memory/time in batch script based on your actual cohort size

## Interpretation

This within-ancestry analysis allows you to:
1. **Fine-scale structure**: Detect regional European subpopulations (e.g., Northern vs. Southern European)
2. **Admixture patterns**: Identify gradients of genetic ancestry within European populations
3. **Outlier detection**: Find samples with unexpected ancestry or data quality issues
4. **Method validation**: Compare fit and project sets to evaluate model generalization

## Expected Patterns

Within white/European ancestry, you may observe:
- North-South European gradient (PC1 typically)
- East-West European gradient (PC2 typically)
- Distinct clusters for specific ethnic groups (e.g., Ashkenazi Jewish, Finnish)
- Admixture with non-European ancestries in some individuals
- Family structure and relatedness patterns
