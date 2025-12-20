#!/bin/bash
#
# UKBB 10K WB + 5K Irish Subset Data Preparation
#
# This wrapper script:
# 1. Reads data paths from mappings_private.json
# 2. Calls the generic subset script to create PLINK subsets
# 3. Creates fit_labels.csv and project_labels.csv by filtering full labels
#
# INTERNAL NOTE: This experiment uses pre-selected samples:
# - fit_samples.txt: 10K British + 5K Irish (random selection, seed=42)
# See _internal/subset_creation_logic.md for selection details
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
MAPPINGS_FILE="${DATA_DIR}/mappings_private.json"
TEMP_DIR="${DATA_DIR}/temp"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$TEMP_DIR"

# ============================================================================
# Step 1: Read mappings file
# ============================================================================
print_status "Reading data paths from mappings_private.json..."

if [[ ! -f "$MAPPINGS_FILE" ]]; then
    print_error "Mappings file not found: $MAPPINGS_FILE"
    echo "Please create mappings_private.json with data paths"
    exit 1
fi

# Extract paths using Python (resolve relative paths from PROJECT_ROOT)
python3 << EOF > "${TEMP_DIR}/paths.sh"
import json
from pathlib import Path

with open('$MAPPINGS_FILE') as f:
    mappings = json.load(f)

project_root = Path('$PROJECT_ROOT')

# Resolve paths (relative ones get resolved from project root)
for key, value in mappings.items():
    if value.startswith('/'):
        # Absolute path
        print(f"{key.upper()}='{value}'")
    else:
        # Relative path - resolve from project root
        resolved = project_root / value
        print(f"{key.upper()}='{resolved}'")
EOF

source "${TEMP_DIR}/paths.sh"

print_success "Loaded paths:"
echo "  UKBB PLINK: $UKBB_PLINK"
echo "  Labels (source): $LABELS"
echo "  Fit samples: $FIT_SAMPLES"

# ============================================================================
# Step 2: Verify required files exist
# ============================================================================
print_status "Verifying required files..."

REQUIRED_FILES=(
    "$FIT_SAMPLES"
    "$LABELS"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    print_error "Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure:"
    echo "  - fit_samples.txt exists (FID IID format)"
    echo "  - ukbb_labels.csv exists (full UKBB labels with sample_id column)"
    echo ""
    exit 1
fi

print_success "All required files found"

# ============================================================================
# Step 3: Call generic subset prepare script
# ============================================================================
print_status "Calling generic subset prepare script..."
echo ""

# Determine threads from SLURM or default
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Note: project_samples not provided means use all samples from PLINK
bash "${PROJECT_ROOT}/examples/generic/subset/prepare_data.sh" \
    --plink "$UKBB_PLINK" \
    --fit-samples "$FIT_SAMPLES" \
    --output-dir "$DATA_DIR" \
    --memory 100000 \
    --threads "$THREADS"

# ============================================================================
# Step 4: Create fit_labels.csv and project_labels.csv from full labels
# ============================================================================
print_status "Creating label files by filtering full labels..."

python3 << PYTHON_SCRIPT
import pandas as pd
import sys

# Read .fam files to get sample IDs
print("  Reading sample IDs from PLINK subsets...")
fit_fam = pd.read_csv('${DATA_DIR}/fit_subset.fam', sep=r'\s+', header=None,
                      names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
project_fam = pd.read_csv('${DATA_DIR}/project_subset.fam', sep=r'\s+', header=None,
                          names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])

fit_samples = set(fit_fam['IID'].astype(str))
project_samples = set(project_fam['IID'].astype(str))

print(f"    Fit samples: {len(fit_samples)}")
print(f"    Project samples: {len(project_samples)}")

# Read full labels file
print("  Reading full UKBB labels...")
labels = pd.read_csv('${LABELS}')

if 'sample_id' not in labels.columns:
    print("  Error: 'sample_id' column not found in labels file")
    sys.exit(1)

labels['sample_id'] = labels['sample_id'].astype(str)

# Filter to fit and project samples
print("  Filtering labels to fit samples...")
fit_labels = labels[labels['sample_id'].isin(fit_samples)].copy()
fit_labels.to_csv('${DATA_DIR}/fit_labels.csv', index=False)

print("  Filtering labels to project samples...")
project_labels = labels[labels['sample_id'].isin(project_samples)].copy()
project_labels.to_csv('${DATA_DIR}/project_labels.csv', index=False)

print(f"  ✓ Created fit_labels.csv: {len(fit_labels)} samples, {len(fit_labels.columns)} columns")
print(f"  ✓ Created project_labels.csv: {len(project_labels)} samples, {len(project_labels.columns)} columns")
PYTHON_SCRIPT

print_success "Label files created"

# Cleanup
rm -f "${TEMP_DIR}/paths.sh"

echo ""
print_success "UKBB 10K WB + 5K Irish data preparation complete!"
echo ""
echo "Generated files:"
echo "  📊 PLINK subsets:"
echo "    - ${DATA_DIR}/fit_subset.{bed,bim,fam}"
echo "    - ${DATA_DIR}/project_subset.{bed,bim,fam}"
echo ""
echo "  🏷️ Label files:"
echo "    - ${DATA_DIR}/fit_labels.csv"
echo "    - ${DATA_DIR}/project_labels.csv"
echo ""
