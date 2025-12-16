# AoU-HGDP Cross-Projection Analysis

This example demonstrates cross-population genetic analysis by projecting All of Us (AoU) samples onto a reference frame built from HGDP+1000 Genomes data.

## Dataset Design

**Fit Subset** (reference for PCA/admixture training):
- HGDP samples (~938 samples)
- 1000 Genomes samples (~2,504 samples)
- Total: ~3,442 samples representing global genetic diversity

**Project Subset** (for projection):
- All of Us (AoU) cohort samples

**Configuration**:
- 20 principal components
- Neural Admixture: K=2 to K=5
- Embedding: PHATE (knn=500, t=50, 10K landmarks)
- Separate labels and colormaps for HGDP and AoU populations

## Rationale

This design allows:
1. **Reference-based projection**: Use well-characterized global reference (HGDP+1KGP) to interpret AoU diversity
2. **Cross-population comparison**: Visualize where AoU samples fall relative to global populations
3. **Ancestry inference**: Understand AoU sample ancestries in context of worldwide genetic variation
4. **Separate visualizations**: Generate plots for HGDP (reference) and AoU (projected) separately for clarity

## Setup

### 1. Prepare Data

The `prepare_data.sh` script handles all data preparation:

```bash
cd examples/aou/hgdp_1kgp_proj
bash prepare_data.sh
```

This script:
1. Downloads/prepares HGDP+1000 Genomes reference data
2. Prepares AoU genotype data
3. Finds common SNPs between datasets
4. Creates fit_subset (HGDP+1KGP) and project_subset (AoU) PLINK files
5. Generates labels:
   - `data/hgdp_labels.csv` - Labels for HGDP+1KGP samples
   - `data/aou_labels.csv` - Labels for AoU samples
6. Creates colormaps:
   - `data/hgdp_colormap.json` - Colors for HGDP populations
   - `data/aou_colormap.json` - Colors for AoU populations

**Note**: You will need to customize this script with:
- Path to your AoU PLINK data
- Path to HGDP+1KGP reference data (or download instructions)
- Logic for creating AoU labels based on available metadata

### 2. Verify Required Files

After running `prepare_data.sh`, verify these files exist:

```bash
data/
├── fit_subset.{bed,bim,fam}      # HGDP+1KGP reference
├── project_subset.{bed,bim,fam}  # AoU samples
├── hgdp_labels.csv               # HGDP+1KGP population labels
├── aou_labels.csv                # AoU sample labels
├── hgdp_colormap.json            # HGDP color scheme
└── aou_colormap.json             # AoU color scheme
```

## Running the Pipeline

### Interactive (on compute node)

```bash
# Request compute node with GPU
salloc --account=ctb-hussinju --cpus-per-task=8 --mem=128GB --gres=gpu:1 --time=8:00:00

# Activate environment
source .venv/bin/activate

# Run pipeline
bash examples/aou/hgdp_1kgp_proj/run_pipeline.sh
```

### Batch Job

```bash
sbatch examples/aou/hgdp_1kgp_proj/run_pipeline_batch.sh
```

## Outputs

Results are saved to `outputs/`:

```
outputs/
├── pca/
│   ├── fit_pca_20.csv             # PCA coordinates (HGDP fit)
│   └── transform_pca_20.csv       # Projected PCA coordinates (AoU)
├── admixture/
│   ├── fit.{2..5}.csv            # Admixture proportions (HGDP fit)
│   └── transform.{2..5}.csv      # Admixture proportions (AoU projected)
├── embeddings/
│   └── phate_2d.csv              # Combined 2D PHATE embedding
└── figures/
    └── embeddings/
        ├── phate_by_*.png        # Separate plots by label category
        └── ...
```

## Sample Sizes

Expected sample counts (approximate):
- **HGDP**: ~938 samples (52 populations worldwide)
- **1000 Genomes**: ~2,504 samples (26 populations)
- **Total fit subset**: ~3,442 samples
- **AoU project subset**: Variable (depends on your AoU cohort size)

## Notes

- The pipeline uses `--fit-labels` and `--project-labels` for separate label files
- The pipeline uses `--fit-colormap` and `--project-colormap` for separate color schemes
- Common SNPs are identified during data preparation (typically 100K-500K SNPs)
- PHATE uses 10K landmarks for scalability with large AoU cohorts
- Metrics are skipped by default (use `--skip-metrics` to disable)
- GPU recommended for Neural Admixture (batch script requests 1 GPU)
- Adjust memory/time in batch script based on AoU cohort size

## Interpretation

The cross-projection allows you to:
1. **Visualize AoU diversity**: See where AoU samples cluster relative to global populations
2. **Identify ancestries**: Match AoU samples to reference populations
3. **Detect admixture**: Observe samples with mixed ancestry between reference groups
4. **Compare fit vs. project**: Evaluate how well the HGDP reference generalizes to AoU
