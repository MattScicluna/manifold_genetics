# UKBB-HGDP Cross-Projection Example

This example demonstrates cross-cohort projection analysis using UKBB and HGDP datasets, where HGDP serves as the diverse reference population for training models that are then projected onto UKBB samples.

## Overview

**Analysis Type**: Cross-cohort projection (HGDP reference → UKBB target)

**Key Features**:
- SNP intersection between UKBB and HGDP datasets with allele consistency checking
- HGDP populations as diverse reference for model training (fit dataset)
- UKBB samples projected into HGDP-trained PCA/admixture space (project dataset)
- Separate population visualizations for each cohort
- Automatic strand flip handling for compatible allele mismatches

## Data Requirements

### Raw Data Paths (configured in `data/mappings.json`)
- **UKBB PLINK files**: `ukb_v3.woWithrawalSamples.liftOverCommon.GRCh38.onlySNPs.sorted.common1000G*`
- **HGDP PLINK files**: `gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned150kb_1_0.05.noHLA.unrelated*`
- **UKBB Metadata**: `UKBB_metadata.csv` with sample ancestry information

### Expected Data Statistics
- **UKBB**: ~500K samples, ~500K SNPs (before filtering)
- **HGDP**: ~4K samples, ~200K SNPs (LD-pruned)
- **Common SNPs**: Expected ~140-160K SNPs after intersection

## Quick Start

### Step 1: Run Data Preparation
```bash
cd $REPO_ROOT/examples/ukbb/hgdp_1kgp_proj/
bash prepare_data.sh
```

This step:
1. Reads raw UKBB and HGDP PLINK files
2. Finds SNP intersection and checks allele consistency
3. Auto-flips compatible strand mismatches (A/T ↔ T/A, C/G ↔ G/C)
4. Creates intersected datasets with common SNPs
5. Generates sample subsets and label files

### Step 2: Run Cross-Projection Pipeline
```bash
# Interactive (on compute node)
salloc --account=<your-slurm-account> --cpus-per-task=8 --mem=64GB --time=4:00:00
manifold-genetics run config.yaml

# Or batch job
sbatch run_pipeline_batch.sh
```

## Output Structure

```
outputs/
├── pca/
│   ├── fit_pca_50.csv              # HGDP PCA coordinates (reference)
│   ├── transform_pca_50.csv        # UKBB PCA coordinates (projected)
│   └── figures/pca/
│       └── pca_pairs_by_*.png      # UKBB ancestry visualizations
│
├── admixture/
│   ├── fit.{2..5}.csv              # HGDP Q proportions (reference)
│   ├── transform.{2..5}.csv        # UKBB Q proportions (projected)
│   └── checkpoints/                # Trained neural admixture models
│
├── embeddings/
│   └── phate_2d.csv                # UKBB 2D PHATE embedding
│
├── figures/
│   ├── admixture/
│   │   ├── transform_bars.png                           # UKBB ancestry bars
│   │   └── transform_admixture_colored_embedding.png    # PHATE colored by admixture
│   └── embeddings/
│       └── phate_by_*.png          # UKBB ancestry plots
│
└── metrics/
    ├── geographic_preservation.json  # Geographic structure preservation
    └── admixture_preservation.json   # Admixture structure preservation
```

## Key Differences from Same-Cohort Analysis

### 1. SNP Intersection Process
- **Same-cohort**: Assumes identical SNP sets
- **Cross-cohort**: Explicit SNP intersection with allele validation

### 2. Population Labels and Colors
- **HGDP labels**: Population-based (Yoruba, Han, French, etc.)
- **UKBB labels**: Ancestry-based (British, African, Chinese, etc.)
- **Separate colormaps**: Different color schemes for each cohort

### 3. Visualization Strategy
- **PCA plots**: Show UKBB ancestry structure (project dataset)
- **Admixture plots**: Show UKBB ancestry structure (project dataset)
- **Embedding plots**: Show UKBB ancestry structure (project dataset)
- **Fit dataset**: Used for training but not directly visualized

### 4. Pipeline Command
```bash
# Cross-cohort pipeline with separate labels/colormaps
manifold-genetics pipeline \
    --fit-plink data/fit_subset \           # HGDP
    --project-plink data/project_subset \   # UKBB
    --labels data/hgdp_labels.csv \         # Primary (unused in this case)
    --colormap data/hgdp_colormap.json \    # Primary (unused in this case)
    --fit-labels data/hgdp_labels.csv \     # HGDP populations
    --project-labels data/ukbb_labels.csv \ # UKBB ancestries
    --fit-colormap data/hgdp_colormap.json \
    --project-colormap data/ukbb_colormap.json \
    --output outputs/
```

## Scientific Interpretation

### Model Training (Fit Phase)
- **PCA**: Principal components learned from HGDP genetic diversity
- **Admixture**: Ancestry components (K=2-5) learned from HGDP populations
- **Result**: Models capture global human genetic structure

### Projection (Transform Phase)
- **PCA**: UKBB samples projected into HGDP PC space
- **Admixture**: UKBB samples assigned ancestry proportions based on HGDP-trained models
- **Embedding**: 2D visualization of UKBB samples in reduced space

### Expected Insights
- How UKBB ancestry groups map to global genetic structure
- Ancestry proportions of UKBB samples relative to HGDP populations
- Population structure preservation across projection steps
- Effectiveness of HGDP as reference for UKBB population analysis

## Troubleshooting

### Low SNP Count After Intersection
- **Cause**: Different genome builds, array types, or QC filtering
- **Solution**: Check raw data compatibility and filtering parameters

### Allele Mismatches
- **Cause**: Different reference genomes or strand conventions
- **Solution**: Review `data/temp/exclude_list.txt` for problematic SNPs

### Memory Issues
- **Solution**: Increase memory allocation: `--mem=128GB`
- **Alternative**: Subset samples for testing

### Long Runtime
- **Cause**: Large datasets, especially UKBB
- **Solution**: Increase CPU allocation or subset samples for initial testing

## Files Created

### Data Preparation Outputs
- `data/fit_subset.{bed,bim,fam}` - HGDP intersected data
- `data/project_subset.{bed,bim,fam}` - UKBB intersected data
- `data/hgdp_labels.csv` - HGDP sample metadata
- `data/ukbb_labels.csv` - UKBB sample metadata
- `data/temp/common_snps.txt` - SNP intersection results
- `data/temp/flip_list.txt` - SNPs requiring strand flip

### Pipeline Outputs
- Standard manifold-genetics outputs with cross-cohort visualization
- Metrics evaluating projection quality and structure preservation

This example demonstrates the power of cross-cohort projection for understanding population structure and ancestry in the context of global genetic diversity.