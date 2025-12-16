#!/bin/bash
#
# Prepare AoU 60K White Subset Data
# This script integrates the aou_pipeline steps to:
# 1. Download/prepare AoU genotype data (step3)
# 2. Query AoU metadata (step4)
# 3. Filter for white/European ancestry samples
# 4. Randomly select 60K for fit subset
# 5. Use remaining for project subset
# 6. Create labels and colormaps
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Configuration
CPU_CORES="${SLURM_CPUS_PER_TASK:-4}"
GOOGLE_PROJECT="${GOOGLE_PROJECT:-}"
CDR_VERSION="${WORKSPACE_CDR:-}"

# Shared AoU data directories
SHARED_AOU_DIR="${SCRIPT_DIR}/../shared/data/AllofUs_V8"
SHARED_META_DIR="${SCRIPT_DIR}/../shared/data/Metadata"

echo "=========================================="
echo "  AoU 60K White Subset Data Preparation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  CPU cores: ${CPU_CORES}"
echo "  Google project: ${GOOGLE_PROJECT:-'Not set'}"
echo "  CDR version: ${CDR_VERSION:-'Not set'}"
echo ""

# Create directories
mkdir -p "${DATA_DIR}"

# =============================================================================
# STEP 3-4: All of Us Data Download (using shared script)
# =============================================================================
echo "=========================================="
echo "Step 3-4: All of Us Data Download"
echo "=========================================="
echo ""

# Check if shared AoU data is complete
AOU_DATA_COMPLETE=true

# Check for main files
for ext in bed bim fam; do
    if [[ ! -f "${SHARED_AOU_DIR}/extractedChrAll.${ext}" ]]; then
        AOU_DATA_COMPLETE=false
        break
    fi
done

# Check for all chromosome splits (1-22)
if [[ "$AOU_DATA_COMPLETE" == "true" ]]; then
    for chr in {1..22}; do
        if [[ ! -f "${SHARED_AOU_DIR}/extractedChr${chr}.bed" ]]; then
            AOU_DATA_COMPLETE=false
            break
        fi
    done
fi

if [[ "$AOU_DATA_COMPLETE" == "true" ]]; then
    echo "✓ Shared AoU data already downloaded at: ${SHARED_AOU_DIR}"
    echo "  Skipping download (shared across all AoU experiments)"
else
    echo "Downloading AoU data to shared location..."
    echo "  This will be cached for all AoU experiments"
    bash "${SCRIPT_DIR}/../shared/download_aou_data.sh"
fi
echo ""

# =============================================================================
# STEP 5: Filter for White/European Ancestry
# =============================================================================
echo "=========================================="
echo "Step 5: Filter for White/European Ancestry"
echo "=========================================="

echo "TODO: Filter AoU samples for white/European ancestry"
echo ""
echo "Steps needed:"
echo "  1. Load ${SHARED_META_DIR}/DemographicData.tsv"
echo "  2. Filter for white/European race (race == 'White' or similar)"
echo "  3. Create sample ID list: ${DATA_DIR}/white_samples.txt"
echo "  4. Use PLINK to extract white samples:"
echo ""
echo "     plink --bfile ${SHARED_AOU_DIR}/extractedChrAll \\"
echo "           --keep ${DATA_DIR}/white_samples.txt \\"
echo "           --make-bed \\"
echo "           --out ${DATA_DIR}/white_all"
echo ""

# =============================================================================
# STEP 6: Create Fit Subset (60K samples)
# =============================================================================
echo "=========================================="
echo "Step 6: Create Fit Subset (60K samples)"
echo "=========================================="

echo "TODO: Randomly select 60K white samples for training"
echo ""
echo "Steps needed:"
echo "  1. Use PLINK to randomly thin to 60K samples:"
echo ""
echo "     plink2 --bfile ${DATA_DIR}/white_all \\"
echo "            --thin-indiv-count 60000 \\"
echo "            --seed 42 \\"
echo "            --make-bed \\"
echo "            --out ${DATA_DIR}/fit_subset"
echo ""
echo "     OR manually select with shuf:"
echo ""
echo "     shuf -n 60000 ${DATA_DIR}/white_all.fam | awk '{print \$1\"\\t\"\$2}' > ${DATA_DIR}/fit_samples.txt"
echo "     plink --bfile ${DATA_DIR}/white_all \\"
echo "           --keep ${DATA_DIR}/fit_samples.txt \\"
echo "           --make-bed \\"
echo "           --out ${DATA_DIR}/fit_subset"
echo ""

# =============================================================================
# STEP 7: Create Project Subset (remaining samples)
# =============================================================================
echo "=========================================="
echo "Step 7: Create Project Subset (remaining)"
echo "=========================================="

echo "TODO: Create project subset with remaining white samples"
echo ""
echo "Steps needed:"
echo "  1. Extract sample IDs from fit subset"
echo "  2. Remove those from white_all to get project subset:"
echo ""
echo "     plink --bfile ${DATA_DIR}/white_all \\"
echo "           --remove ${DATA_DIR}/fit_subset.fam \\"
echo "           --make-bed \\"
echo "           --out ${DATA_DIR}/project_subset"
echo ""

# =============================================================================
# STEP 8: Create Labels and Colormaps
# =============================================================================
echo "=========================================="
echo "Step 8: Labels and Colormaps"
echo "=========================================="

echo "TODO: Create labels and colormaps"
echo ""
echo "Files to create:"
echo "  1. fit_labels.csv - Labels for fit samples (from DemographicData.tsv)"
echo "  2. project_labels.csv - Labels for project samples (from DemographicData.tsv)"
echo "  3. colormap.json - Color scheme for populations (can reuse UKBB colormap)"
echo ""
echo "Label format (CSV):"
echo "  sample_id,race,ethnicity,..."
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Data Preparation Summary"
echo "=========================================="
echo ""
echo "Completed:"
echo "  ✓ Step 3: AoU genotype data downloaded and split"
echo "  ✓ Step 4: Metadata extraction (if CDR_VERSION set)"
echo ""
echo "TODO:"
echo "  - Step 5: Filter for white/European ancestry"
echo "  - Step 6: Create fit subset (60K samples)"
echo "  - Step 7: Create project subset (remaining)"
echo "  - Step 8: Create labels and colormaps"
echo ""
echo "Shared AoU data location:"
echo "  ${SHARED_AOU_DIR}/"
echo "  ${SHARED_META_DIR}/DemographicData.tsv"
echo "  (This is shared across all AoU experiments)"
echo ""
echo "Expected final outputs in ${DATA_DIR}:"
echo "  - white_all.{bed,bim,fam}      (All white samples)"
echo "  - fit_subset.{bed,bim,fam}     (60K white samples for training)"
echo "  - project_subset.{bed,bim,fam} (Remaining white samples)"
echo "  - fit_labels.csv"
echo "  - project_labels.csv"
echo ""
echo "To check progress:"
echo "  Fit samples: \$(wc -l < ${DATA_DIR}/fit_subset.fam 2>/dev/null || echo 'N/A')"
echo "  Project samples: \$(wc -l < ${DATA_DIR}/project_subset.fam 2>/dev/null || echo 'N/A')"
echo "  SNPs: \$(wc -l < ${DATA_DIR}/fit_subset.bim 2>/dev/null || echo 'N/A')"
echo ""
