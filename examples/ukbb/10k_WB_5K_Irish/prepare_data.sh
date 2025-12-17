#!/bin/bash
#
# UKBB 10K WB + 5K Irish Subset Data Preparation
#
# This is a wrapper that calls the generic subset script.
# It reads UKBB PLINK path from mappings.json.
#
# INTERNAL NOTE: This experiment uses pre-selected samples:
# - fit_samples.txt: 10K British + 5K Irish + others (random selection, seed=42)
# - project_samples: All UKBB samples
# See _internal/subset_creation_logic.md for selection details
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
# Usage:
#   bash examples/ukbb/10k_WB_5K_Irish/prepare_data.sh
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

# Step 1: Check for sample lists and labels
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

# Step 2: Read mappings file
print_status "Reading data paths from mappings.json..."

if [[ ! -f "$MAPPINGS_FILE" ]]; then
    print_error "Mappings file not found: $MAPPINGS_FILE"
    echo "Please create mappings.json with UKBB PLINK path"
    exit 1
fi

# Extract paths using Python
TEMP_DIR="${DATA_DIR}/temp"
mkdir -p "$TEMP_DIR"

python3 << EOF > "${TEMP_DIR}/paths.sh"
import json
from pathlib import Path

with open('$MAPPINGS_FILE') as f:
    mappings = json.load(f)

project_root = Path('$PROJECT_ROOT')

for key, value in mappings.items():
    if key == 'ukbb_plink':  # Only need plink path
        if value.startswith('/'):
            print(f"UKBB_PLINK='{value}'")
        else:
            resolved = project_root / value
            print(f"UKBB_PLINK='{resolved}'")
EOF

source "${TEMP_DIR}/paths.sh"

print_success "Loaded UKBB PLINK path: $UKBB_PLINK"

# Step 3: Call generic subset prepare script
print_status "Calling generic subset prepare script..."
echo ""

# Determine threads from SLURM or default
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Note: project_samples not provided means use all samples from PLINK
bash "${PROJECT_ROOT}/examples/generic/subset/prepare_data.sh" \
    --plink "$UKBB_PLINK" \
    --fit-samples "${DATA_DIR}/fit_samples.txt" \
    --metadata "${DATA_DIR}/ukbb_metadata.csv" \
    --output-dir "$DATA_DIR" \
    --fit-labels-out "${DATA_DIR}/fit_labels_generated.csv" \
    --project-labels-out "${DATA_DIR}/project_labels_generated.csv" \
    --memory 100000 \
    --threads "$THREADS"

# Note: We already have manually created fit_labels.csv and project_labels.csv,
# so we don't overwrite them. The generic script creates *_generated.csv files
# which can be compared for verification.

# Cleanup
rm -f "${TEMP_DIR}/paths.sh"

echo ""
print_success "UKBB 10K WB + 5K Irish data preparation complete!"
echo ""
echo "Note: Using existing fit_labels.csv and project_labels.csv"
echo "      Generated labels saved as *_labels_generated.csv for comparison"
echo ""
