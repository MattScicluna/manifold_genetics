#!/bin/bash
#
# Prepare AoU-HGDP cross-projection data
# This script integrates the aou_pipeline steps to:
# 1. Download/prepare HGDP+1KGP reference data (step2)
# 2. Download/prepare AoU genotype data (step3)
# 3. Query AoU metadata (step4)
# 4. Find common SNPs and create fit/project subsets
# 5. Create labels and colormaps
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Configuration
CPU_CORES="${SLURM_CPUS_PER_TASK:-4}"
REF_DATA_URL="gs://fc-secure-47ccf5a8-b9ba-460a-aa03-dea8d260953b/Data/1KGPHGDP.tar.gz"
GOOGLE_PROJECT="${GOOGLE_PROJECT:-}"
CDR_VERSION="${WORKSPACE_CDR:-}"

# Directories
REF_DIR="${DATA_DIR}/1KGPHGDP"

echo "=========================================="
echo "  AoU-HGDP Data Preparation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  CPU cores: ${CPU_CORES}"
echo "  Google project: ${GOOGLE_PROJECT:-'Not set'}"
echo "  CDR version: ${CDR_VERSION:-'Not set'}"
echo ""

# Create directories
mkdir -p "${DATA_DIR}" "${REF_DIR}"

# =============================================================================
# STEP 2: Reference Data Processing (HGDP+1KGP)
# =============================================================================
echo "=========================================="
echo "Step 2: HGDP+1KGP Reference Data"
echo "=========================================="

TAR_PATH="${DATA_DIR}/1KGPHGDP.tar.gz"

# 1. Download
if [[ ! -f "${TAR_PATH}" ]]; then
    echo "Downloading HGDP+1KGP reference data..."
    echo "  Source: ${REF_DATA_URL}"
    gsutil cp "${REF_DATA_URL}" "${DATA_DIR}/"
else
    echo "  Reference data archive already exists"
fi

# 2. Extract
if [[ ! -d "${REF_DIR}/extractedChrAllUnpruned.bed" ]]; then
    echo "Extracting reference data..."
    tar -xvf "${TAR_PATH}" --directory "${DATA_DIR}/"
else
    echo "  Reference data already extracted"
fi

# 3. Fix BIM file (add 'chr' prefix)
BIM_PATH="${REF_DIR}/extractedChrAllUnpruned.bim"
ORIG_BIM_PATH="${REF_DIR}/extractedChrAllUnpruned.Original.bim"

if [[ ! -f "${ORIG_BIM_PATH}" ]]; then
    echo "Fixing BIM file chromosome prefixes..."
    mv "${BIM_PATH}" "${ORIG_BIM_PATH}"
    awk '{print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}' "${ORIG_BIM_PATH}" > "${BIM_PATH}"
else
    echo "  BIM file already fixed"
fi

# 4. Split by chromosome (1-22)
echo "Splitting reference data by chromosome (using ${CPU_CORES} cores)..."
if [[ ! -f "${REF_DIR}/extractedChr22.bed" ]]; then
    seq 1 22 | xargs -I {} -P "${CPU_CORES}" sh -c \
        "plink --bfile ${REF_DIR}/extractedChrAllUnpruned \
               --keep-allele-order --allow-no-sex --chr {} \
               --make-bed --out ${REF_DIR}/extractedChr{}"
else
    echo "  Reference data already split by chromosome"
fi

echo "  ✓ HGDP+1KGP reference data ready"
echo ""

# =============================================================================
# STEP 3-4: All of Us Data Download (using shared script)
# =============================================================================
echo "=========================================="
echo "Step 3-4: All of Us Data Download"
echo "=========================================="
echo ""
echo "Using shared AoU download script..."
bash "${SCRIPT_DIR}/../shared/download_aou_data.sh" "${DATA_DIR}"
echo ""

# =============================================================================
# STEP 5: Find Common SNPs and Create Subsets
# =============================================================================
echo "=========================================="
echo "Step 5: Common SNPs & Subset Creation"
echo "=========================================="

echo "TODO: Implement common SNP finding and subset creation"
echo ""
echo "Steps needed:"
echo "  1. Merge chromosomes for HGDP and AoU"
echo "  2. Find SNPs in common (use plink --bmerge or extract common from .bim files)"
echo "  3. Create fit_subset.{bed,bim,fam} from HGDP+1KGP (all samples)"
echo "  4. Create project_subset.{bed,bim,fam} from AoU (all samples)"
echo "  5. Both subsets should have same SNPs (common set)"
echo ""

# =============================================================================
# STEP 6: Create Labels and Colormaps
# =============================================================================
echo "=========================================="
echo "Step 6: Labels and Colormaps"
echo "=========================================="

echo "TODO: Create labels and colormaps"
echo ""
echo "Files to create:"
echo "  1. hgdp_labels.csv - HGDP population labels (can copy from examples/hgdp_1kgp)"
echo "  2. aou_labels.csv - AoU sample labels (from DemographicData.tsv)"
echo "  3. hgdp_colormap.json - HGDP color scheme (can copy from examples/hgdp_1kgp)"
echo "  4. aou_colormap.json - AoU color scheme"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Data Preparation Summary"
echo "=========================================="
echo ""
echo "Completed:"
echo "  ✓ Step 2: HGDP+1KGP reference data downloaded and split"
echo "  ✓ Step 3: AoU genotype data downloaded and split"
echo "  ✓ Step 4: Metadata extraction (if CDR_VERSION set)"
echo ""
echo "TODO:"
echo "  - Step 5: Find common SNPs and create fit/project subsets"
echo "  - Step 6: Create labels and colormaps"
echo ""
echo "Expected final outputs in ${DATA_DIR}:"
echo "  - fit_subset.{bed,bim,fam}     (HGDP+1KGP reference)"
echo "  - project_subset.{bed,bim,fam} (AoU samples)"
echo "  - hgdp_labels.csv"
echo "  - aou_labels.csv"
echo "  - hgdp_colormap.json"
echo "  - aou_colormap.json"
echo ""
