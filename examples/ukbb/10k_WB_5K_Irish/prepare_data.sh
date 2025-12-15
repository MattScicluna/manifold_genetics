#!/bin/bash
#
# Data preparation for UKBB 10K WB + 5K Irish + Others analysis
#
# PREREQUISITES:
# You must first create the sample lists and labels. For internal use, run:
#   bash prepare_data_create_labels.sh
#
# For external users: manually create these files:
#   - data/fit_samples.txt (FID IID format, one sample per line)
#   - data/fit_labels.csv (sample_id, self_described_ancestry, Population)
#   - data/project_labels.csv (sample_id, self_described_ancestry, Population)
#
# This script:
# 1. Reads UKBB PLINK files from mappings.json
# 2. Uses existing sample lists to create PLINK subsets
# 3. Creates fit subset (10K WB + 5K Irish + others) and project subset (all samples)
#
# Usage:
#   cd /path/to/manifold_genetics/examples/ukbb/10k_WB_5K_Irish/
#   bash prepare_data.sh
#

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ANSI colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Print header
echo ""
echo "========================================="
echo "  UKBB 10K WB + 5K Irish Data Prep"
echo "========================================="
echo ""

# Paths
DATA_DIR="${SCRIPT_DIR}/data"
MAPPINGS_FILE="${DATA_DIR}/mappings.json"

# Step 1: Check for sample lists
print_status "Checking for required sample lists and labels..."

REQUIRED_LISTS=(
    "${DATA_DIR}/fit_samples.txt"
    "${DATA_DIR}/fit_labels.csv"
    "${DATA_DIR}/project_labels.csv"
)

MISSING_LISTS=()
for file in "${REQUIRED_LISTS[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_LISTS+=("$file")
    fi
done

if [[ ${#MISSING_LISTS[@]} -gt 0 ]]; then
    print_error "Missing required sample lists:"
    for file in "${MISSING_LISTS[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please create these files first:"
    echo "  For internal use: bash prepare_data_create_labels.sh"
    echo "  For external use: Create sample lists manually"
    echo ""
    echo "Required files:"
    echo "  - fit_samples.txt: FID IID format (tab-separated, no header)"
    echo "  - fit_labels.csv: sample_id,self_described_ancestry,Population"
    echo "  - project_labels.csv: sample_id,self_described_ancestry,Population"
    echo ""
    exit 1
fi

print_success "All required sample lists found"

# Count samples in lists
FIT_LIST_COUNT=$(wc -l < "${DATA_DIR}/fit_samples.txt")
FIT_LABELS_COUNT=$(($(wc -l < "${DATA_DIR}/fit_labels.csv") - 1))  # Subtract header
PROJECT_LABELS_COUNT=$(($(wc -l < "${DATA_DIR}/project_labels.csv") - 1))

echo "  fit_samples.txt: $FIT_LIST_COUNT samples"
echo "  fit_labels.csv: $FIT_LABELS_COUNT samples"
echo "  project_labels.csv: $PROJECT_LABELS_COUNT samples"

# Step 2: Read mappings file
print_status "Reading data paths from mappings.json..."

if [[ ! -f "$MAPPINGS_FILE" ]]; then
    print_error "Mappings file not found: $MAPPINGS_FILE"
    echo "Please create mappings_private.json with UKBB PLINK path"
    exit 1
fi

# Extract paths using Python
python3 << EOF > "${DATA_DIR}/.paths.sh"
import json
from pathlib import Path

with open('$MAPPINGS_FILE') as f:
    mappings = json.load(f)

project_root = Path('$PROJECT_ROOT')

for key, value in mappings.items():
    if key == 'ukbb_plink':  # Only need plink path, not metadata
        if value.startswith('/'):
            print(f"UKBB_PLINK='{value}'")
        else:
            resolved = project_root / value
            print(f"UKBB_PLINK='{resolved}'")
EOF

source "${DATA_DIR}/.paths.sh"
rm -f "${DATA_DIR}/.paths.sh"

print_success "Loaded UKBB PLINK path: $UKBB_PLINK"

# Step 3: Verify PLINK files exist
print_status "Verifying PLINK files exist..."

REQUIRED_FILES=(
    "${UKBB_PLINK}.bed"
    "${UKBB_PLINK}.bim"
    "${UKBB_PLINK}.fam"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    print_error "Missing required PLINK files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

print_success "All required PLINK files found"

# Step 4: Find plink2
print_status "Looking for plink2..."

PLINK2=""

if [[ -f "${PROJECT_ROOT}/bin/plink2" ]]; then
    PLINK2="${PROJECT_ROOT}/bin/plink2"
    print_success "Using plink2 from bin/plink2"
elif module list 2>&1 | grep -q plink2; then
    PLINK2="plink2"
    print_success "Using plink2 from loaded module"
elif command -v plink2 &> /dev/null; then
    PLINK2="plink2"
    print_success "Using plink2 from PATH"
else
    print_error "plink2 not found!"
    echo ""
    echo "Please ensure plink2 is available via one of:"
    echo "  1. Run setup.sh to download to bin/plink2 (recommended)"
    echo "  2. Load plink2 module: module load plink2"
    echo "  3. Add plink2 to your PATH"
    echo ""
    exit 1
fi

if ! ${PLINK2} --version &> /dev/null; then
    print_error "plink2 found but not executable!"
    exit 1
fi

# Step 5: Create PLINK subsets
print_status "Creating PLINK subsets..."

# Create fit subset
print_status "  Creating fit subset (10K WB + 5K Irish + others)..."
${PLINK2} --bfile ${UKBB_PLINK} \
    --keep ${DATA_DIR}/fit_samples.txt \
    --make-bed \
    --out ${DATA_DIR}/fit_subset

# Create project subset (all samples)
print_status "  Creating project subset (all UKBB samples)..."
cp "${UKBB_PLINK}.bed" "${DATA_DIR}/project_subset.bed"
cp "${UKBB_PLINK}.bim" "${DATA_DIR}/project_subset.bim"
cp "${UKBB_PLINK}.fam" "${DATA_DIR}/project_subset.fam"

print_success "PLINK subsets created"

# Step 6: Summary
FIT_SAMPLES=$(wc -l < "${DATA_DIR}/fit_subset.fam")
PROJECT_SAMPLES=$(wc -l < "${DATA_DIR}/project_subset.fam")
TOTAL_SNPS=$(wc -l < "${DATA_DIR}/fit_subset.bim")

echo ""
echo "========================================="
print_success "Data preparation complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  📊 Processed PLINK data:"
echo "    - data/fit_subset.{bed,bim,fam}     ($FIT_SAMPLES samples, $TOTAL_SNPS SNPs)"
echo "    - data/project_subset.{bed,bim,fam} ($PROJECT_SAMPLES samples, $TOTAL_SNPS SNPs)"
echo ""
echo "  🏷️ Sample labels (from prerequisites):"
echo "    - data/fit_labels.csv     ($FIT_LABELS_COUNT samples)"
echo "    - data/project_labels.csv ($PROJECT_LABELS_COUNT samples)"
echo ""
echo "You can now run the analysis pipeline:"
echo "  bash run_pipeline.sh"
echo ""
